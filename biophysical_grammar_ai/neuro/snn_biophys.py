from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import cupy as cp

# -------------------------- Utilities --------------------------

def _to_gpu(x, dtype=None):
    if isinstance(x, cp.ndarray):
        return x.astype(dtype) if dtype is not None else x
    return cp.asarray(x, dtype=dtype)

def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b

# -------------------------- Neuron Population --------------------------

@dataclass
class NeuronParams:
    C: float = 200e-12            # capacitance (F)
    gL: float = 10e-9             # leak conductance (S)
    EL: float = -70e-3            # leak reversal (V)
    VT0: float = -50e-3           # base threshold (V)
    DeltaT: float = 2e-3          # slope factor (V)
    Vreset: float = -58e-3        # reset (V)
    Vspike: float = 20e-3         # spike detect level (V)
    tau_ref: float = 2e-3         # refractory (s)
    # Adaptation (soma AHP)
    a: float = 2e-9               # subthreshold adaptation (S)
    b: float = 0.0                # spike-triggered adaptation increment (A)
    tau_w: float = 100e-3         # adaptation time constant (s)
    # AIS threshold dynamics
    tau_T: float = 30e-3          # threshold relaxation (s)
    dVT_spike: float = 2e-3       # threshold increment per spike (V)
    # Reversal potentials
    E_exc: float = 0.0            # AMPA/NMDA reversal (V)
    E_inh: float = -70e-3         # GABA_A reversal (V)
    # NMDA magnesium block params (Jahr-Stevens)
    nmda_eta: float = 0.062       # 1/V factor in exp
    nmda_Mg: float = 1.0          # mM
    nmda_gamma: float = 3.57      # 1/mM

class NeuronPopulation:
    """
    Adaptive exponential integrate-and-fire neurons with AIS dynamics.
    """
    def __init__(self, N: int, dt: float = 1e-3, params: Optional[NeuronParams] = None, name: str = "neurons"):
        assert N > 0
        self.N = N
        self.name = name
        self.dt = float(dt)
        self.p = params or NeuronParams()

        # State (GPU)
        self.V = cp.full(N, self.p.EL, dtype=cp.float32)          # membrane voltage
        self.w = cp.zeros(N, dtype=cp.float32)                    # adaptation current
        self.VT = cp.full(N, self.p.VT0, dtype=cp.float32)        # AIS threshold
        self.ref_count = cp.zeros(N, dtype=cp.int32)              # refractory steps left
        # Conductance accumulators (per-step, cleared each step)
        self.gE = cp.zeros(N, dtype=cp.float32)                   # AMPA total conductance
        self.gI = cp.zeros(N, dtype=cp.float32)                   # GABA total conductance
        self.gN = cp.zeros(N, dtype=cp.float32)                   # NMDA total conductance
        # Spike output buffer
        self.spike = cp.zeros(N, dtype=cp.uint8)

        # Precompute decay factors
        self._exp_dt_tau_w = math.exp(-self.dt / self.p.tau_w)
        self._exp_dt_tau_T = math.exp(-self.dt / self.p.tau_T)
        self._ref_steps = max(1, int(round(self.p.tau_ref / self.dt)))

        # Kernels
        self._neuron_step_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void neuron_step(
            const int N,
            const float dt,
            const float C, const float gL, const float EL,
            const float VT0, const float DeltaT, const float Vreset, const float Vspike,
            const float a, const float b,
            const float E_exc, const float E_inh,
            const float nmda_eta, const float nmda_Mg, const float nmda_gamma,
            const float exp_dt_tau_w, const float exp_dt_tau_T,
            const int ref_steps, const float dVT_spike,
            float* __restrict__ V,
            float* __restrict__ w,
            float* __restrict__ VT,
            int*   __restrict__ ref_count,
            float* __restrict__ gE,
            float* __restrict__ gI,
            float* __restrict__ gN,
            const float* __restrict__ Iext,
            unsigned char* __restrict__ spike_out
        ) {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i >= N) return;

            float v = V[i];
            float w_i = w[i];
            float vt = VT[i];
            int refc = ref_count[i];

            unsigned char s = 0;

            if (refc > 0) {
                // In refractory: clamp V to reset, decay adaptation and threshold
                v = Vreset;
                w_i = w_i * exp_dt_tau_w; // small decay during refractory
                vt = VT0 + (vt - VT0) * exp_dt_tau_T;
                refc -= 1;
            } else {
                // Synaptic conductances
                float gE_i = gE[i];
                float gI_i = gI[i];
                float gN_i = gN[i];

                // NMDA magnesium block factor: 1 / (1 + gamma * Mg * exp(-eta * V))
                float B = 1.0f / (1.0f + nmda_gamma * nmda_Mg * expf(-nmda_eta * v));
                float Isyn = gE_i * (E_exc - v) + gI_i * (E_inh - v) + (gN_i * B) * (E_exc - v);

                float Iext_i = Iext ? Iext[i] : 0.0f;

                // AdEx dynamics
                float dv = ( -gL * (v - EL) + gL * DeltaT * expf((v - vt) / DeltaT) - w_i + Isyn + Iext_i ) / C;
                v += dt * dv;

                // Adaptation
                w_i = w_i * exp_dt_tau_w + dt * a * (v - EL) * (1.0f - exp_dt_tau_w);

                // AIS threshold relaxation
                vt = VT0 + (vt - VT0) * exp_dt_tau_T;

                // Spike condition
                if (v >= Vspike) {
                    s = 1;
                    v = Vreset;
                    w_i += b;
                    vt += dVT_spike;
                    refc = ref_steps;
                }
            }

            // Store back
            V[i] = v;
            w[i] = w_i;
            VT[i] = vt;
            ref_count[i] = refc;
            spike_out[i] = s;

            // Clear conductances for next step
            gE[i] = 0.0f;
            gI[i] = 0.0f;
            gN[i] = 0.0f;
        }
        ''', 'neuron_step')

    def step(self, Iext: Optional[cp.ndarray] = None) -> cp.ndarray:
        """
        Advance neurons by one dt. Iext is an optional external current vector (A).
        Returns spike vector (uint8).
        """
        N = self.N
        if Iext is None:
            Iext = cp.zeros(N, dtype=cp.float32)
        else:
            Iext = _to_gpu(Iext, cp.float32)
            assert Iext.size == N

        threads = 256
        blocks = _ceil_div(N, threads)
        self._neuron_step_kernel((blocks,), (threads,), (
            int(N),
            float(self.dt),
            self.p.C, self.p.gL, self.p.EL,
            self.p.VT0, self.p.DeltaT, self.p.Vreset, self.p.Vspike,
            self.p.a, self.p.b,
            self.p.E_exc, self.p.E_inh,
            self.p.nmda_eta, self.p.nmda_Mg, self.p.nmda_gamma,
            float(self._exp_dt_tau_w), float(self._exp_dt_tau_T),
            int(self._ref_steps), float(self.p.dVT_spike),
            self.V, self.w, self.VT, self.ref_count,
            self.gE, self.gI, self.gN,
            Iext, self.spike
        ))
        return self.spike

# -------------------------- Synapses --------------------------

class SynapseGroup:
    """
    Sparse, conductance-based chemical synapses with delay lines and plasticity (STDP + BCM LTP).
    Connections stored as edges (COO-style): E edges from pre -> post.
    """
    RECEPTOR_AMPA = 0
    RECEPTOR_GABA = 1
    RECEPTOR_NMDA = 2

    def __init__(
        self,
        pre: NeuronPopulation,
        post: NeuronPopulation,
        receptor_types: str = 'AMPA',
        density: float = 0.1,
        w_init: Tuple[float, float] = (1e-9, 1e-9),
        myelin: float = 1.0,
        axon_length: float = 0.01,
        distance_jitter: float = 0.0,
        seed: Optional[int] = None,
        stdp_params: Optional[Dict] = None,
        bcm_params: Optional[Dict] = None,
        name: str = "synapses"
    ):
        """
        Args:
            receptor_types: 'AMPA', 'GABA', 'NMDA', or combination like 'AMPA+NMDA' or 'AMPA+GABA'.
            density: probability of connection for Erdos-Renyi graph.
            w_init: (min, max) initial weights (S).
            myelin: 0 (unmyelinated) to 1 (fully myelinated) controls conduction speed.
            axon_length: meters; used to compute delay = length / velocity.
            distance_jitter: fraction of axon_length as random variation per connection.
        """
        self.pre = pre
        self.post = post
        self.name = name
        self.dt = pre.dt
        assert math.isclose(pre.dt, post.dt), "pre/post dt mismatch"

        # Receptor composition
        valid = {'AMPA', 'GABA', 'NMDA'}
        comps = [c.strip() for c in receptor_types.split('+')]
        for c in comps:
            if c not in valid:
                raise ValueError(f"Invalid receptor type: {c}")
        self._receptors = comps

        rng = cp.random.RandomState(seed or 1234)

        # Build random connections
        Npre, Npost = pre.N, post.N
        p_conn = float(density)
        mask = rng.rand(Npre, Npost) < p_conn
        idx_pre, idx_post = cp.nonzero(mask)

        # Number of edges
        self.E = int(idx_pre.size)
        if self.E == 0:
            raise RuntimeError("No synapses created; increase density or sizes.")

        # Edge arrays
        self.pre_idx = idx_pre.astype(cp.int32)
        self.post_idx = idx_post.astype(cp.int32)

        # Receptor type per edge (round-robin over requested comps)
        type_map = {'AMPA': 0, 'GABA': 1, 'NMDA': 2}
        type_ids = []
        for e in range(self.E):
            t = comps[e % len(comps)]
            type_ids.append(type_map[t])
        self.type_id = cp.asarray(type_ids, dtype=cp.int32)

        # Initial weights (conductance scale per edge)
        w0 = rng.uniform(w_init[0], w_init[1], size=self.E).astype(cp.float32)
        self.w = w0

        # Synaptic state per edge: release r, gating s
        self.r = cp.zeros(self.E, dtype=cp.float32)  # transmitter release
        self.s = cp.zeros(self.E, dtype=cp.float32)  # open channel fraction

        # Plasticity traces (STDP)
        self.Apre = cp.zeros(self.E, dtype=cp.float32)
        self.Apost = cp.zeros(self.E, dtype=cp.float32)
        self.pre_event_flag = cp.zeros(self.E, dtype=cp.uint8)

        # Rates for BCM (running averages per neuron)
        self.pre_rate = cp.zeros(Npre, dtype=cp.float32)
        self.post_rate = cp.zeros(Npost, dtype=cp.float32)

        # Synapse kinetics per receptor
        # tau_r: release decay, tau_s: channel decay, alpha: coupling r->s, p_release: on pre-event
        self.tau_r = cp.where(self.type_id == 0, 1e-3, cp.where(self.type_id == 1, 1e-3, 10e-3)).astype(cp.float32)
        self.tau_s = cp.where(self.type_id == 0, 5e-3, cp.where(self.type_id == 1, 10e-3, 100e-3)).astype(cp.float32)
        self.alpha = cp.where(self.type_id == 0, 1.0, cp.where(self.type_id == 1, 1.0, 0.5)).astype(cp.float32)
        self.p_release = cp.where(self.type_id == 0, 0.6, cp.where(self.type_id == 1, 0.5, 0.4)).astype(cp.float32)

        # Decay factors per edge for r and s
        self.decay_r = cp.exp(-self.dt / self.tau_r).astype(cp.float32)
        self.decay_s = cp.exp(-self.dt / self.tau_s).astype(cp.float32)

        # STDP parameters
        sp = {
            "tau_pre": 20e-3, "tau_post": 20e-3,
            "A_plus": 1e-10, "A_minus": 1e-10,
            "w_min": 0.0, "w_max": 1e-7
        }
        if stdp_params:
            sp.update(stdp_params)
        self.tau_pre = float(sp["tau_pre"]); self.tau_post = float(sp["tau_post"])
        self.A_plus = float(sp["A_plus"]); self.A_minus = float(sp["A_minus"])
        self.w_min = float(sp["w_min"]); self.w_max = float(sp["w_max"])
        self.decay_pre = math.exp(-self.dt / self.tau_pre)
        self.decay_post = math.exp(-self.dt / self.tau_post)

        # BCM-like LTP parameters
        bp = {
            "tau_rate": 200e-3,   # running average window
            "eta": 1e-9,          # learning rate
            "theta_factor": 1.0   # threshold = theta_factor * post_rate^2
        }
        if bcm_params:
            bp.update(bcm_params)
        self.tau_rate = float(bp["tau_rate"])
        self.bcm_eta = float(bp["eta"])
        self.theta_factor = float(bp["theta_factor"])
        self.rate_decay = math.exp(-self.dt / self.tau_rate)

        # Conduction delays via myelination
        # velocity (m/s): interpolate unmyelinated ~ 0.5 m/s to myelinated ~ 10 m/s
        v_un = 0.5
        v_my = 10.0
        myelin = float(myelin)
        base_len = float(axon_length)
        if distance_jitter > 0.0:
            jitter = (rng.rand(self.E).astype(cp.float32) - 0.5) * (2.0 * distance_jitter) * base_len
        else:
            jitter = cp.zeros(self.E, dtype=cp.float32)
        self.distance = cp.full(self.E, base_len, dtype=cp.float32) + jitter
        vel = v_un * (1.0 - myelin) + v_my * myelin
        delay_s = self.distance / float(vel)
        delay_steps = cp.maximum(1, cp.asarray(cp.rint(delay_s / self.dt), dtype=cp.int32))
        self.delay_steps = delay_steps
        self.max_delay = int(cp.asnumpy(delay_steps.max()))

        # Spike history ring buffer for pre neurons
        self._buf_len = self.max_delay + 3  # small guard
        self._hist = cp.zeros((self._buf_len, Npre), dtype=cp.uint8)
        self._hist_idx = 0

        # Accumulator work arrays
        self._gE = self.post.gE
        self._gI = self.post.gI
        self._gN = self.post.gN

        # Kernels
        self._syn_phase1 = cp.RawKernel(r'''
        extern "C" __global__
        void syn_phase1(
            const int E,
            const int Npre,
            const int buf_len,
            const int hist_idx,
            const float dt,
            const float* __restrict__ decay_r,
            const float* __restrict__ decay_s,
            const float* __restrict__ alpha,
            const float* __restrict__ p_release,
            const int*   __restrict__ pre_idx,
            const int*   __restrict__ post_idx,
            const int*   __restrict__ type_id,
            const int*   __restrict__ delay_steps,
            const unsigned char* __restrict__ hist,  // shape [buf_len, Npre] flattened
            float* __restrict__ r,     // E
            float* __restrict__ s,     // E
            float* __restrict__ w,     // E
            unsigned char* __restrict__ pre_event_flag, // E
            float* __restrict__ gE,    // Npost
            float* __restrict__ gI,    // Npost
            float* __restrict__ gN     // Npost
        ) {
            int e = blockDim.x * blockIdx.x + threadIdx.x;
            if (e >= E) return;

            int pre = pre_idx[e];
            int post = post_idx[e];
            int d = delay_steps[e];
            int t_idx = hist_idx - d;
            while (t_idx < 0) t_idx += buf_len;
            int idx = t_idx * Npre + pre;
            unsigned char pre_d = hist[idx];

            // Decay release and channel gating
            float r_e = r[e] * decay_r[e];
            float s_e = s[e] * decay_s[e];

            // Pre event updates
            unsigned char flag = 0;
            if (pre_d) {
                float pr = p_release[e];
                r_e = r_e + pr * (1.0f - r_e);
                flag = 1;
            }

            // Channel opening driven by release
            float al = alpha[e];
            s_e = s_e + dt * al * r_e * (1.0f - s_e);

            // Accumulate conductance into postsynaptic bins
            float g = s_e * w[e];
            int ty = type_id[e];
            if (ty == 0) { atomicAdd(&gE[post], g); }
            else if (ty == 1) { atomicAdd(&gI[post], g); }
            else { atomicAdd(&gN[post], g); }

            // Store back
            r[e] = r_e;
            s[e] = s_e;
            pre_event_flag[e] = flag;
        }
        ''', 'syn_phase1')

        self._stdp_pre = cp.RawKernel(r'''
        extern "C" __global__
        void stdp_pre(
            const int E,
            const float decay_pre,
            const float A_plus,
            const float w_min, const float w_max,
            const unsigned char* __restrict__ pre_event_flag,
            float* __restrict__ Apre,
            const float* __restrict__ Apost,
            float* __restrict__ w
        ) {
            int e = blockDim.x * blockIdx.x + threadIdx.x;
            if (e >= E) return;
            float a_pre = Apre[e] * decay_pre;
            if (pre_event_flag[e]) {
                a_pre += 1.0f;
                // Potentiation on pre event proportional to recent post activity
                float dw = A_plus * Apost[e];
                float w_new = w[e] + dw;
                if (w_new < w_min) w_new = w_min;
                if (w_new > w_max) w_new = w_max;
                w[e] = w_new;
            }
            Apre[e] = a_pre;
        }
        ''', 'stdp_pre')

        self._stdp_post = cp.RawKernel(r'''
        extern "C" __global__
        void stdp_post(
            const int E,
            const int Npost,
            const float decay_post,
            const float A_minus,
            const float w_min, const float w_max,
            const int*   __restrict__ post_idx,
            const unsigned char* __restrict__ post_spike,
            float* __restrict__ Apost,
            const float* __restrict__ Apre,
            float* __restrict__ w
        ) {
            int e = blockDim.x * blockIdx.x + threadIdx.x;
            if (e >= E) return;
            int post = post_idx[e];
            float a_post = Apost[e] * decay_post;
            if (post_spike[post]) {
                a_post += 1.0f;
                // Depression on post event proportional to recent pre activity
                float dw = -A_minus * Apre[e];
                float w_new = w[e] + dw;
                if (w_new < w_min) w_new = w_min;
                if (w_new > w_max) w_new = w_max;
                w[e] = w_new;
            }
            Apost[e] = a_post;
        }
        ''', 'stdp_post')

        self._bcm = cp.RawKernel(r'''
        extern "C" __global__
        void bcm_edge_update(
            const int E,
            const float eta,
            const float theta_factor,
            const int* __restrict__ pre_idx,
            const int* __restrict__ post_idx,
            const float* __restrict__ pre_rate,
            const float* __restrict__ post_rate,
            const float w_min, const float w_max,
            float* __restrict__ w
        ) {
            int e = blockDim.x * blockIdx.x + threadIdx.x;
            if (e >= E) return;
            int pre = pre_idx[e];
            int post = post_idx[e];
            float rp = pre_rate[pre];
            float rq = post_rate[post];
            float theta = theta_factor * rq * rq;
            float dw = eta * rp * (rq - theta);
            float w_new = w[e] + dw;
            if (w_new < w_min) w_new = w_min;
            if (w_new > w_max) w_new = w_max;
            w[e] = w_new;
        }
        ''', 'bcm_edge_update')

        # Elementwise kernels for rate updates
        self._rate_update_pre = cp.ElementwiseKernel(
            in_params='float32 rate, uint8 s, float32 decay',
            out_params='float32 rate_out',
            operation='rate_out = rate * decay + (1.0f - decay) * (float)s;',
            name='rate_update_pre'
        )
        self._rate_update_post = cp.ElementwiseKernel(
            in_params='float32 rate, uint8 s, float32 decay',
            out_params='float32 rate_out',
            operation='rate_out = rate * decay + (1.0f - decay) * (float)s;',
            name='rate_update_post'
        )

    def push_pre_spikes(self, pre_spike: cp.ndarray):
        """Insert pre spike vector into ring buffer (called each global step)."""
        assert pre_spike.size == self.pre.N
        self._hist[self._hist_idx, :] = pre_spike
        self._hist_idx += 1
        if self._hist_idx >= self._buf_len:
            self._hist_idx = 0

    def phase1_and_plasticity(self, post_spike: cp.ndarray):
        """
        Execute synaptic phase1 (conductances) using delayed pre spikes;
        then apply STDP and BCM updates with current post spikes.
        """
        # Phase 1: update r, s, accumulate conductances, record pre_event flags
        threads = 256
        blocks = _ceil_div(self.E, threads)

        self._syn_phase1((blocks,), (threads,), (
            int(self.E),
            int(self.pre.N),
            int(self._buf_len),
            int(self._hist_idx),
            float(self.dt),
            self.decay_r, self.decay_s, self.alpha, self.p_release,
            self.pre_idx, self.post_idx, self.type_id,
            self.delay_steps,
            self._hist.ravel(),
            self.r, self.s, self.w, self.pre_event_flag,
            self._gE, self._gI, self._gN
        ))

        # STDP pre-side (uses pre_event_flag and Apost trace)
        self._stdp_pre((blocks,), (threads,), (
            int(self.E),
            float(self.decay_pre),
            float(self.A_plus),
            float(self.w_min), float(self.w_max),
            self.pre_event_flag,
            self.Apre,
            self.Apost,
            self.w
        ))

        # STDP post-side (needs post spikes)
        # Update Apost traces and apply LTD on post events
        self._stdp_post((blocks,), (threads,), (
            int(self.E),
            int(self.post.N),
            float(self.decay_post),
            float(self.A_minus),
            float(self.w_min), float(self.w_max),
            self.post_idx,
            post_spike,
            self.Apost,
            self.Apre,
            self.w
        ))

        # BCM LTP edge-level update (slow)
        self._bcm((blocks,), (threads,), (
            int(self.E),
            float(self.bcm_eta),
            float(self.theta_factor),
            self.pre_idx, self.post_idx,
            self.pre_rate, self.post_rate,
            float(self.w_min), float(self.w_max),
            self.w
        ))

        # Update running rates
        self.pre_rate = self._rate_update_pre(self.pre_rate, self._hist[(self._hist_idx - 1) % self._buf_len], cp.float32(self.rate_decay))
        self.post_rate = self._rate_update_post(self.post_rate, post_spike, cp.float32(self.rate_decay))

    @classmethod
    def from_connections(cls, pre: NeuronPopulation, post: NeuronPopulation, connections: List[Tuple[int, int, float]], receptor_types: str = 'AMPA', stdp_params: Optional[Dict] = None, bcm_params: Optional[Dict] = None, name: str = "synapses_custom"):
        """
        Creates a SynapseGroup from an explicit list of connections.
        Args:
            connections: A list of (pre_idx, post_idx, weight) tuples.
        """
        # 1. Create an empty instance by calling __new__
        syn_group = cls.__new__(cls)

        # 2. Basic setup
        syn_group.pre = pre
        syn_group.post = post
        syn_group.name = name
        syn_group.dt = pre.dt
        assert math.isclose(pre.dt, post.dt), "pre/post dt mismatch"

        # 3. Process connections list
        if not connections:
            raise ValueError("Connection list cannot be empty for from_connections.")
        
        pre_idx_list, post_idx_list, w_init_list = zip(*connections)
        
        syn_group.E = len(pre_idx_list)
        syn_group.pre_idx = _to_gpu(pre_idx_list, cp.int32)
        syn_group.post_idx = _to_gpu(post_idx_list, cp.int32)
        syn_group.w = _to_gpu(w_init_list, cp.float32)

        # 4. Set up all other attributes that __init__ would have handled
        valid = {'AMPA', 'GABA', 'NMDA'}
        comps = [c.strip() for c in receptor_types.split('+')]
        for c in comps:
            if c not in valid:
                raise ValueError(f"Invalid receptor type: {c}")
        syn_group._receptors = comps
        
        type_map = {'AMPA': 0, 'GABA': 1, 'NMDA': 2}
        type_ids = [type_map[comps[e % len(comps)]] for e in range(syn_group.E)]
        syn_group.type_id = _to_gpu(type_ids, cp.int32)

        syn_group.r = cp.zeros(syn_group.E, dtype=cp.float32)
        syn_group.s = cp.zeros(syn_group.E, dtype=cp.float32)
        syn_group.Apre = cp.zeros(syn_group.E, dtype=cp.float32)
        syn_group.Apost = cp.zeros(syn_group.E, dtype=cp.float32)
        syn_group.pre_event_flag = cp.zeros(syn_group.E, dtype=cp.uint8)
        syn_group.pre_rate = cp.zeros(pre.N, dtype=cp.float32)
        syn_group.post_rate = cp.zeros(post.N, dtype=cp.float32)

        syn_group.tau_r = cp.where(syn_group.type_id == 0, 1e-3, cp.where(syn_group.type_id == 1, 1e-3, 10e-3)).astype(cp.float32)
        syn_group.tau_s = cp.where(syn_group.type_id == 0, 5e-3, cp.where(syn_group.type_id == 1, 10e-3, 100e-3)).astype(cp.float32)
        syn_group.alpha = cp.where(syn_group.type_id == 0, 1.0, cp.where(syn_group.type_id == 1, 1.0, 0.5)).astype(cp.float32)
        syn_group.p_release = cp.where(syn_group.type_id == 0, 0.6, cp.where(syn_group.type_id == 1, 0.5, 0.4)).astype(cp.float32)
        syn_group.decay_r = cp.exp(-syn_group.dt / syn_group.tau_r)
        syn_group.decay_s = cp.exp(-syn_group.dt / syn_group.tau_s)

        sp = {"tau_pre": 20e-3, "tau_post": 20e-3, "A_plus": 1e-10, "A_minus": 1e-10, "w_min": 0.0, "w_max": 1e-7}
        if stdp_params: sp.update(stdp_params)
        syn_group.tau_pre = float(sp["tau_pre"]); syn_group.tau_post = float(sp["tau_post"])
        syn_group.A_plus = float(sp["A_plus"]); syn_group.A_minus = float(sp["A_minus"])
        syn_group.w_min = float(sp["w_min"]); syn_group.w_max = float(sp["w_max"])
        syn_group.decay_pre = math.exp(-syn_group.dt / syn_group.tau_pre)
        syn_group.decay_post = math.exp(-syn_group.dt / syn_group.tau_post)

        bp = {"tau_rate": 200e-3, "eta": 1e-9, "theta_factor": 1.0}
        if bcm_params: bp.update(bcm_params)
        syn_group.tau_rate = float(bp["tau_rate"])
        syn_group.bcm_eta = float(bp["eta"])
        syn_group.theta_factor = float(bp["theta_factor"])
        syn_group.rate_decay = math.exp(-syn_group.dt / syn_group.tau_rate)

        syn_group.distance = cp.zeros(syn_group.E, dtype=cp.float32)
        syn_group.delay_steps = cp.ones(syn_group.E, dtype=cp.int32)
        syn_group.max_delay = 1
        syn_group._buf_len = 4
        syn_group._hist = cp.zeros((syn_group._buf_len, pre.N), dtype=cp.uint8)
        syn_group._hist_idx = 0

        syn_group._gE = post.gE
        syn_group._gI = post.gI
        syn_group._gN = post.gN
        
        # Copy kernels from a dummy instance
        dummy = cls(pre, post, density=0.00001, seed=1) # Create a dummy to get kernels
        syn_group._syn_phase1 = dummy._syn_phase1
        syn_group._stdp_pre = dummy._stdp_pre
        syn_group._stdp_post = dummy._stdp_post
        syn_group._bcm = dummy._bcm
        syn_group._rate_update_pre = dummy._rate_update_pre
        syn_group._rate_update_post = dummy._rate_update_post

        return syn_group

# -------------------------- Network orchestrator --------------------------

class SNN:
    """
    Container managing multiple neuron populations and synapse groups.
    Call step() each dt to advance the whole SNN.
    """
    def __init__(self, neurons: List[NeuronPopulation], synapses: List[SynapseGroup]):
        self.neurons = neurons
        self.synapses = synapses
        # Map pre->syn groups for pushing spike histories
        self._pre_to_syn = {}
        for syn in synapses:
            self._pre_to_syn.setdefault(syn.pre, []).append(syn)

    def step(self, external_currents: Optional[Dict[NeuronPopulation, cp.ndarray]] = None) -> Dict[NeuronPopulation, cp.ndarray]:
        """
        One full dt update:
        1) Push pre spikes into each outgoing synapse's delay buffer.
        2) For each synapse, update conductances (phase1) and plasticity (STDP+BCM) using delayed pre and current post spikes.
        3) Update neurons with accumulated conductances.
        Returns dict mapping neuron group -> spike vector (uint8).
        """
        if external_currents is None:
            external_currents = {}
        # 1) Push spikes from previous step into delay lines
        for pre, syn_list in self._pre_to_syn.items():
            for syn in syn_list:
                syn.push_pre_spikes(pre.spike)

        # 2) For each synapse, compute conductances and plasticity using current post spikes (from previous step)
        for syn in self.synapses:
            syn.phase1_and_plasticity(syn.post.spike)

        # 3) Update neurons (integration + spike generation). Conductances already accumulated into gE/gI/gN.
        spikes_out = {}
        for grp in self.neurons:
            Iext = external_currents.get(grp, None)
            s = grp.step(Iext)
            spikes_out[grp] = s

        return spikes_out

# -------------------------- Convenience constructors --------------------------

def fully_connected(pre: NeuronPopulation, post: NeuronPopulation, receptor_types='AMPA', w_init=(1e-9, 1e-9), **kw) -> SynapseGroup:
    sg = SynapseGroup(pre, post, receptor_types=receptor_types, density=1.0, w_init=w_init, **kw)
    return sg

# -------------------------- Self-test (optional) --------------------------

if __name__ == "__main__":
    import cupy as cp
    pre = NeuronPopulation(64, dt=1e-3, name="pre")
    post = NeuronPopulation(64, dt=1e-3, name="post")
    syn = SynapseGroup(pre, post, receptor_types='AMPA+NMDA', density=0.2, myelin=1.0, axon_length=0.02, distance_jitter=0.2)
    net = SNN([pre, post], [syn])

    # Drive pre with random external current to elicit spikes
    T = 500
    rng = cp.random.RandomState(0)
    I_pre = cp.zeros((T, pre.N), dtype=cp.float32)
    I_post = cp.zeros((T, post.N), dtype=cp.float32)
    I_pre[:] = (rng.rand(T, pre.N) < 0.05).astype(cp.float32) * 3e-9  # pulse

    spikes_pre = []
    spikes_post = []
    for t in range(T):
        ext = {pre: I_pre[t], post: I_post[t]}
        sout = net.step(ext)
        spikes_pre.append(cp.asnumpy(sout[pre]).copy())
        spikes_post.append(cp.asnumpy(sout[post]).copy())

    print("Sim finished. Example spike counts:", int(sum(map(sum, spikes_pre))), int(sum(map(sum, spikes_post))))

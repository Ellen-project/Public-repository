from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Callable
import math
import numpy as np
import heapq
import enum
import random

# ----------------------------- Utilities & Constants -----------------------------

def exp_clip(x: float) -> float:
    """Numerically stable exp for rate equations (clip extreme voltages)."""
    x = max(min(x, 50.0), -50.0)
    return math.exp(x)


def sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def vtrap(x: float, y: float) -> float:
    """Stable x / (1 - exp(-x / y)) with the correct limit at x == 0."""
    z = x / y
    if abs(z) < 1e-7:
        return y * (1.0 + 0.5 * z)
    if z < -50.0:
        return -x * math.exp(z)
    if z > 50.0:
        return x
    return x / (1.0 - math.exp(-z))


# Physiological defaults (mV, ms, uF/cm^2, mS/cm^2)
E_Na = 50.0     # Sodium reversal (mV)
E_K  = -77.0    # Potassium reversal (mV)
E_L  = -54.4    # Leak reversal (mV) classical HH
E_Cl = -70.0    # GABA_A (approx) (mV)
E_AMPA = 0.0
E_NMDA = 0.0

DEFAULT_TEMP_C = 36.0  # Temperature for Q10 scaling if extended


# ----------------------------- Event Scheduler -----------------------------

class EventQueue:
    """Min-heap based event queue for scheduled (time, callable) tasks."""
    def __init__(self):
        self._heap: List[Tuple[float, int, Callable[[], None]]] = []
        self._counter = 0

    def schedule(self, t: float, fn: Callable[[], None]) -> None:
        heapq.heappush(self._heap, (t, self._counter, fn))
        self._counter += 1

    def pop_ready(self, t: float) -> List[Callable[[], None]]:
        ready = []
        while self._heap and self._heap[0][0] <= t:
            _, _, fn = heapq.heappop(self._heap)
            ready.append(fn)
        return ready

    def clear(self) -> None:
        self._heap.clear()
        self._counter = 0


# ----------------------------- Ion Channel Mechanisms -----------------------------

class Mechanism:
    """Base class for ionic mechanisms attached to a compartment."""
    def step(self, V: float, dt: float) -> float:
        """
        Return ionic current (positive outward) contributed by this mechanism
        and internally update gating variables using V and dt.
        """
        raise NotImplementedError()


@dataclass
class HHNaKParams:
    gNa: float = 120.0  # mS/cm^2
    gK: float  = 36.0   # mS/cm^2
    gLeak: float = 0.3  # mS/cm^2
    ENa: float = E_Na
    EK: float  = E_K
    ELeak: float = E_L


class HodgkinHuxleyNaK(Mechanism):
    """
    Classic Hodgkin-Huxley (1952) Na/K/Leak mechanism.
    V in mV, dt in ms, currents in uA/cm^2.
    """
    def __init__(self, params: HHNaKParams):
        self.p = params
        self.reset()

    def reset(self, V0: float = -65.0):
        """Reset gating variables to their steady state at V0."""
        self.m = self._alpha_m(V0) / (self._alpha_m(V0) + self._beta_m(V0))
        self.h = self._alpha_h(V0) / (self._alpha_h(V0) + self._beta_h(V0))
        self.n = self._alpha_n(V0) / (self._alpha_n(V0) + self._beta_n(V0))

    # Rate functions (mV)
    def _alpha_m(self, V): return 0.1 * vtrap(V + 40.0, 10.0)
    def _beta_m(self, V):  return 4.0 * exp_clip(-(V + 65.0) / 18.0)
    def _alpha_h(self, V): return 0.07 * exp_clip(-(V + 65.0) / 20.0)
    def _beta_h(self, V):  return sigmoid((V + 35.0) / 10.0)
    def _alpha_n(self, V): return 0.01 * vtrap(V + 55.0, 10.0)
    def _beta_n(self, V):  return 0.125 * exp_clip(-(V + 65.0) / 80.0)

    def step(self, V: float, dt: float) -> float:
        am = self._alpha_m(V); bm = self._beta_m(V)
        ah = self._alpha_h(V); bh = self._beta_h(V)
        an = self._alpha_n(V); bn = self._beta_n(V)
        dm = dt * (am * (1.0 - self.m) - bm * self.m)
        dh = dt * (ah * (1.0 - self.h) - bh * self.h)
        dn = dt * (an * (1.0 - self.n) - bn * self.n)
        self.m += dm; self.h += dh; self.n += dn
        # Currents (uA/cm^2)
        INa = self.p.gNa * (self.m**3) * self.h * (V - self.p.ENa)
        IK  = self.p.gK  * (self.n**4) * (V - self.p.EK)
        IL  = self.p.gLeak * (V - self.p.ELeak)
        return INa + IK + IL


# ----------------------------- Synapses & Plasticity -----------------------------

class SynapseKind(enum.Enum):
    AMPA_NMDA = 1
    GABA_A = 2


@dataclass
class STDPParams:
    tau_pre: float = 20.0   # ms
    tau_post: float = 20.0  # ms
    A_plus: float = 0.01    # LTP amplitude
    A_minus: float = 0.012  # LTD amplitude
    eta: float = 1.0        # global learning rate scaling


@dataclass
class LTPParams:
    tau_Ca: float = 50.0     # ms, Ca decay
    k_Ca: float = 0.005      # Ca influx scaling per NMDA current
    theta_pot: float = 1.2   # Ca threshold (arbitrary units)
    eta_pot: float = 0.0005  # potentiation rate
    # Optional weak LTD (when Ca moderately elevated)
    theta_dep_low: float = 0.3
    theta_dep_high: float = 0.8
    eta_dep: float = 0.0002


@dataclass
class Synapse:
    """Biophysical synapse attached to a specific postsynaptic compartment."""
    pre: 'SpikingSource'
    post_compartment: 'Compartment'
    kind: SynapseKind
    weight: float
    gmax: float = 0.001    # mS (peak conductance scaling)
    tau_rise: float = 0.5  # ms
    tau_decay: float = 3.0 # ms (AMPA) / 8-100 for NMDA; we'll separate internally
    nmda_fraction: float = 0.25  # fraction of g that is NMDA for excitatory
    mg_conc: float = 1.0   # mM Mg2+ for NMDA block
    stdp: STDPParams = field(default_factory=STDPParams)
    ltp: LTPParams = field(default_factory=LTPParams)
    w_min: float = 0.05
    w_max: float = 2.5

    # Internal state (CPU path; GPU path holds states in arrays)
    r_fast: float = 0.0
    r_slow: float = 0.0
    x_pre: float = 0.0
    x_post: float = 0.0
    Ca: float = 0.0
    last_pre_spike: Optional[float] = None
    last_post_spike: Optional[float] = None

    # GPU index (assigned by CuPyEngine)
    _gpu_index: Optional[int] = None

    def __post_init__(self):
        # Register with presynaptic source
        self.pre._register_outgoing(self)

    def schedule_pre_spike(self, t: float):
        """Schedule a presynaptic arrival on the active CPU or GPU execution path."""
        self.post_compartment.net.events.schedule(t, lambda: self._deliver_pre_spike(t))

    def _deliver_pre_spike(self, t: float):
        engine = getattr(self.post_compartment.net, "_active_gpu_engine", None)
        if engine is not None and self._gpu_index is not None:
            engine._buffer_pre_event(self._gpu_index, t)
        else:
            self._on_pre_spike(t)

    def _on_pre_spike(self, t: float):
        # Conductance transient (bi-exponential via two states)
        self.r_fast += 1.0
        self.r_slow += 1.0
        # STDP: nearest-neighbor (pre event uses current post-trace)
        if self.stdp is not None:
            dw = self.stdp.eta * self.stdp.A_plus * self.x_post
            self.weight = float(np.clip(self.weight + dw, self.w_min, self.w_max))
            self.x_pre += 1.0
        self.last_pre_spike = t

    def on_post_spike(self, t: float):
        # STDP: post event uses current pre-trace
        if self.stdp is not None:
            dw = - self.stdp.eta * self.stdp.A_minus * self.x_pre
            self.weight = float(np.clip(self.weight + dw, self.w_min, self.w_max))
            self.x_post += 1.0
        self.last_post_spike = t

    def reset_state(self):
        self.r_fast = 0.0
        self.r_slow = 0.0
        self.x_pre = 0.0
        self.x_post = 0.0
        self.Ca = 0.0
        self.last_pre_spike = None
        self.last_post_spike = None

    def step(self, V_post: float, dt: float) -> float:
        """Advance synapse state and return synaptic current (positive outward from post)."""
        # Update traces for STDP
        if self.stdp is not None:
            self.x_pre  += -dt * self.x_pre  / self.stdp.tau_pre
            self.x_post += -dt * self.x_post / self.stdp.tau_post

        # Update conductance states (bi-exponential shape via two first-order filters)
        tau_r = max(1e-6, self.tau_rise)
        tau_d_fast = max(self.tau_decay, self.tau_rise + 0.1)
        tau_d_slow = max(40.0, self.tau_decay * 12.0)  # NMDA slower

        self.r_fast *= math.exp(-dt / tau_d_fast)
        self.r_slow *= math.exp(-dt / tau_d_slow)

        g_total = self.gmax * self.weight
        if self.kind == SynapseKind.AMPA_NMDA:
            g_ampa = g_total * (1.0 - self.nmda_fraction) * (1.0 - math.exp(-dt / tau_r)) * self.r_fast
            g_nmda_base = g_total * self.nmda_fraction * (1.0 - math.exp(-dt / tau_r)) * self.r_slow
            B = 1.0 / (1.0 + self.mg_conc * math.exp(-0.062 * V_post) / 3.57)
            g_nmda = g_nmda_base * B
            I_ampa = g_ampa * (V_post - E_AMPA)
            I_nmda = g_nmda * (V_post - E_NMDA)
            I_syn = I_ampa + I_nmda
            self.Ca += dt * (-self.ltp.k_Ca * I_nmda) - dt * (self.Ca / self.ltp.tau_Ca)

            if self.Ca > self.ltp.theta_pot:
                dw = self.ltp.eta_pot * (1.0 - (self.weight / self.w_max)) * (self.Ca - self.ltp.theta_pot) / (self.Ca + self.ltp.theta_pot)
                self.weight = float(np.clip(self.weight + dw * dt, self.w_min, self.w_max))
            elif self.ltp.theta_dep_low < self.Ca < self.ltp.theta_dep_high:
                dw = - self.ltp.eta_dep * (self.weight / self.w_max) * (self.ltp.theta_dep_high - self.Ca) / self.ltp.theta_dep_high
                self.weight = float(np.clip(self.weight + dw * dt, self.w_min, self.w_max))
            return I_syn
        elif self.kind == SynapseKind.GABA_A:
            g_gaba = g_total * (1.0 - math.exp(-dt / tau_r)) * self.r_fast
            I_gaba = g_gaba * (V_post - E_Cl)
            return I_gaba
        else:
            return 0.0


# ----------------------------- Compartments -----------------------------

@dataclass(eq=False)
class Compartment:
    name: str
    net: 'Network'
    area_cm2: float = 1e-6  # typical small area; used for scaling currents to volts
    Cm_uF_per_cm2: float = 1.0
    Gleak_mS_per_cm2: float = 0.1
    Eleak_mV: float = -65.0
    mechanisms: List[Mechanism] = field(default_factory=list)
    synapses: List[Synapse] = field(default_factory=list)
    neighbors: List[Tuple['Compartment', float]] = field(default_factory=list)  # (neighbor, g_axial mS)
    I_ext_uA: float = 0.0

    V_mV: float = -65.0

    def _mark_topology_changed(self):
        marker = getattr(self.net, "_mark_topology_changed", None)
        if marker is not None:
            marker()

    def connect_axial(self, other: 'Compartment', g_axial_mS: float):
        self.neighbors.append((other, g_axial_mS))
        self._mark_topology_changed()

    def add_mechanism(self, mech: Mechanism):
        self.mechanisms.append(mech)
        self._mark_topology_changed()

    def add_synapse(self, syn: Synapse):
        self.synapses.append(syn)
        self._mark_topology_changed()

    def reset_state(self, V_mV: Optional[float] = None, clear_current: bool = True):
        if V_mV is not None:
            self.V_mV = float(V_mV)
        if clear_current:
            self.I_ext_uA = 0.0
        for mech in self.mechanisms:
            reset = getattr(mech, "reset", None)
            if reset is not None:
                reset(self.V_mV)
        for syn in self.synapses:
            syn.reset_state()

    def step_current(self, dt: float) -> float:
        """CPU path: Compute net outward current density (uA/cm^2) for this timestep."""
        # Leak
        I_leak = self.Gleak_mS_per_cm2 * (self.V_mV - self.Eleak_mV)

        # Ionic mechanisms (e.g., HH)
        I_ion = 0.0
        for mech in self.mechanisms:
            I_ion += mech.step(self.V_mV, dt)

        # Axial coupling (Ohmic): sum g*(V_i - V_j) as outward current from i
        I_axial = 0.0
        for nbr, g in self.neighbors:
            I_axial += g * (self.V_mV - nbr.V_mV)

        # Synaptic currents (already sign-correct as outward)
        I_syn = 0.0
        for syn in self.synapses:
            I_syn += syn.step(self.V_mV, dt)

        # External current (uA); convert to density by dividing area
        I_ext_density = self.I_ext_uA / max(1e-12, self.area_cm2)

        # Net outward density
        I_out = I_leak + I_ion + I_axial + I_syn - I_ext_density
        return I_out

    def integrate_voltage(self, I_out_density: float, dt: float):
        """Forward Euler update V using dV/dt = - I_out / C (outward hyperpolarizes)."""
        Cm = self.Cm_uF_per_cm2
        dV = - (I_out_density / max(1e-12, Cm)) * dt
        self.V_mV += dV


# ----------------------------- Spiking Sources & External Generators -----------------------------

class SpikingSource:
    """Abstract base for any object that can emit spikes and has outgoing synapses."""
    def __init__(self, name: str, net: 'Network'):
        self.name = name
        self.net = net
        self._outgoing: List[Synapse] = []

    def _register_outgoing(self, syn: Synapse):
        if syn not in self._outgoing:
            self._outgoing.append(syn)

    def fire(self, t: float, branches: Optional[List['AxonBranch']] = None):
        """
        Emit a spike at time t. If branches provided, use branch routing to schedule arrivals.
        Otherwise deliver immediately (for non-axonal sources).
        """
        if branches is None:
            for syn in self._outgoing:
                syn.schedule_pre_spike(t)
        else:
            for br in branches:
                br.route_and_deliver(spike_time=t)


class ExternalSpikeGenerator(SpikingSource):
    """Poisson spike generator as an external 'neuron' (A, B, C, etc.)."""
    def __init__(self, name: str, net: 'Network', rate_hz: float, weight_scale: float = 1.0):
        if rate_hz < 0.0:
            raise ValueError("rate_hz must be non-negative")
        super().__init__(name, net)
        self.rate_hz = rate_hz
        self.weight_scale = weight_scale

    def connect_to(self, neuron: 'PyramidalNeuron', target_region: str, segment_index: int,
                   kind: SynapseKind, weight: float) -> Synapse:
        return neuron.attach_synapse_from(self, target_region, segment_index, kind, weight * self.weight_scale)

    def step(self, t: float, dt: float):
        # Thinning method for Poisson with small dt
        p = min(1.0, self.rate_hz * dt / 1000.0)
        if random.random() < p:
            self.fire(t)


# ----------------------------- Axon Branching & Routing -----------------------------

@dataclass
class AxonBranch:
    """Models deterministic spike routing along a branch using safety factor (SF)."""
    name: str
    parent_neuron: 'PyramidalNeuron'
    length_um: float
    diameter_um: float
    myelin_factor: float  # 0 (unmyelinated) ... 1 (heavily myelinated)
    channel_density: float  # relative Na channel density (1.0 baseline)
    terminals: List[Synapse] = field(default_factory=list)

    # Routing parameters
    v_base_m_per_s: float = 0.5
    delay_noise_sd_ms: float = 0.05
    sf_threshold: float = 1.0
    freq_sensitivity: float = 0.8
    last_spike_time: Optional[float] = None
    last_arrival_time: Optional[float] = None

    def _conduction_velocity(self) -> float:
        v = self.v_base_m_per_s * math.sqrt(max(1e-6, self.diameter_um)) * (1.0 + 4.0 * self.myelin_factor)
        return max(0.1, v)

    def _prop_delay_ms(self) -> float:
        L_m = self.length_um * 1e-6
        v = self._conduction_velocity()
        delay = 1000.0 * (L_m / v)  # ms
        if self.delay_noise_sd_ms > 0.0:
            delay += random.gauss(0.0, self.delay_noise_sd_ms)
        return max(0.0, delay)

    def _safety_factor(self, spike_amplitude_mV: float, isi_ms: float) -> float:
        freq_hz = 1000.0 / max(isi_ms, 1e-3)
        geom = math.sqrt(max(1e-6, self.diameter_um)) * (1.0 + 2.0 * self.myelin_factor)
        excit = self.channel_density * (1.0 + 0.01 * max(0.0, spike_amplitude_mV))
        freq_penalty = 1.0 + self.freq_sensitivity * (freq_hz / 100.0)
        sf = (geom * excit) / freq_penalty
        return sf

    def route_and_deliver(self, spike_time: float):
        last = self.last_spike_time
        isi = float('inf') if last is None else (spike_time - last)
        self.last_spike_time = spike_time
        Vamp = self.parent_neuron.ais_spike_amplitude_mV
        sf = self._safety_factor(Vamp, isi)
        if sf >= self.sf_threshold:
            delay = self._prop_delay_ms()
            t_arrival = spike_time + delay
            for syn in self.terminals:
                syn.schedule_pre_spike(t_arrival)
            self.last_arrival_time = t_arrival
        else:
            pass


# ----------------------------- Pyramidal Neuron -----------------------------

class PyramidalNeuron(SpikingSource):
    """
    Multi-compartment pyramidal neuron with basal/apical/tuft dendrites, soma, AIS, and axon branching.
    Soma/AIS carry HH Na/K mechanisms; dendrites are passive by default but accept synapses and axial coupling.
    """
    def __init__(self, name: str, net: 'Network',
                 n_basal: int = 10, n_apical: int = 10, n_tuft: int = 6):
        super().__init__(name, net)
        self.traces: Dict[str, List[float]] = {"t": [], "soma_V": [], "ais_V": []}
        self.record_traces: bool = True
        self.trace_max_len: Optional[int] = None
        self.spike_times: List[float] = []
        self.spike_times_max_len: Optional[int] = None
        self._spike_count_total: int = 0
        self.last_ais_spike_time: Optional[float] = None
        self.ais_spike_amplitude_mV: float = 100.0

        # Build compartments
        self.basal: List[Compartment] = [Compartment(f"{name}_basal_{i}", net) for i in range(n_basal)]
        self.apical: List[Compartment] = [Compartment(f"{name}_apical_{i}", net) for i in range(n_apical)]
        self.tuft:   List[Compartment] = [Compartment(f"{name}_tuft_{i}",   net) for i in range(n_tuft)]
        self.soma = Compartment(f"{name}_soma", net)
        self.ais  = Compartment(f"{name}_ais",  net)

        # Mechanisms
        soma_mech = HodgkinHuxleyNaK(HHNaKParams(gNa=120.0, gK=36.0, gLeak=0.3))
        ais_mech  = HodgkinHuxleyNaK(HHNaKParams(gNa=200.0, gK=40.0, gLeak=0.2))
        self.soma.add_mechanism(soma_mech)
        self.ais.add_mechanism(ais_mech)

        # Passive dendrites
        for comp in self.basal + self.apical + self.tuft:
            comp.Cm_uF_per_cm2 = 1.0
            comp.Gleak_mS_per_cm2 = 0.12
            comp.Eleak_mV = -65.0

        # Axial connections: chains and soma hub
        def connect_chain(lst: List[Compartment], g_axial):
            for i in range(len(lst) - 1):
                lst[i].connect_axial(lst[i+1], g_axial)
                lst[i+1].connect_axial(lst[i], g_axial)
        connect_chain(self.basal, g_axial=0.3)
        connect_chain(self.apical, g_axial=0.3)
        connect_chain(self.tuft,   g_axial=0.25)

        if self.basal:
            self.basal[0].connect_axial(self.soma, 0.6)
            self.soma.connect_axial(self.basal[0], 0.6)
        if self.apical:
            self.apical[0].connect_axial(self.soma, 0.6)
            self.soma.connect_axial(self.apical[0], 0.6)
        if self.tuft and self.apical:
            self.apical[-1].connect_axial(self.tuft[0], 0.4)
            self.tuft[0].connect_axial(self.apical[-1], 0.4)

        # soma ↔ AIS coupling
        self.soma.connect_axial(self.ais, 0.8)
        self.ais.connect_axial(self.soma, 0.8)

        # AIS spike detection params
        self.ais_threshold_mV = -50.0
        self.ais_reset_mV = -60.0
        self.use_ais_reset: bool = True
        self.ais_refrac_ms = 2.0
        self._ais_last_spike: Optional[float] = None

        # Axon branches
        self.main_axon = AxonBranch(
            name=f"{name}_main",
            parent_neuron=self,
            length_um=800.0, diameter_um=1.2, myelin_factor=0.9, channel_density=1.2
        )
        self.collateral_axon = AxonBranch(
            name=f"{name}_collateral",
            parent_neuron=self,
            length_um=300.0, diameter_um=0.6, myelin_factor=0.2, channel_density=0.9
        )

        net.register_neuron(self)

    # ----------------- Connectivity -----------------

    def attach_synapse_from(self, pre: SpikingSource, target_region: str, segment_index: int,
                            kind: SynapseKind, weight: float) -> Synapse:
        comp = self._resolve_region_segment(target_region, segment_index)
        syn = Synapse(pre=pre, post_compartment=comp, kind=kind, weight=weight)
        comp.add_synapse(syn)
        return syn

    def connect_axon_to(self, target: 'PyramidalNeuron', branch: str, target_region: str,
                        segment_index: int, kind: SynapseKind, weight: float) -> Synapse:
        if branch == "main":
            axon = self.main_axon
        elif branch == "collateral":
            axon = self.collateral_axon
        else:
            raise ValueError("branch must be 'main' or 'collateral'")

        comp = target._resolve_region_segment(target_region, segment_index)
        syn = Synapse(pre=self, post_compartment=comp, kind=kind, weight=weight)
        comp.add_synapse(syn)
        axon.terminals.append(syn)
        return syn

    def _resolve_region_segment(self, region: str, idx: int) -> Compartment:
        region = region.lower()
        if region == "soma":
            if idx != 0:
                raise IndexError("segment_index must be 0 for region soma")
            return self.soma
        if region == "ais":
            if idx != 0:
                raise IndexError("segment_index must be 0 for region ais")
            return self.ais
        arr = {"basal": self.basal, "apical": self.apical, "tuft": self.tuft}.get(region)
        if arr is None:
            raise ValueError(f"Unknown region {region}")
        if not (0 <= idx < len(arr)):
            raise IndexError(f"segment_index {idx} out of bounds for region {region}")
        return arr[idx]

    # ----------------- CPU Simulation Support -----------------

    def step(self, t: float, dt: float):
        comps = self.basal + self.apical + self.tuft + [self.soma, self.ais]
        prev_ais_V = self.ais.V_mV
        Iouts = [c.step_current(dt) for c in comps]
        for c, Iout in zip(comps, Iouts):
            c.integrate_voltage(Iout, dt)

        can_spike = True
        if self._ais_last_spike is not None and (t - self._ais_last_spike) < self.ais_refrac_ms:
            can_spike = False

        spike_detected = can_spike and self.ais.V_mV >= self.ais_threshold_mV and prev_ais_V < self.ais_threshold_mV
        if spike_detected:
            self._ais_last_spike = t
            self.last_ais_spike_time = t
            self._spike_count_total += 1
            self.spike_times.append(t)
            if self.spike_times_max_len is not None:
                if self.spike_times_max_len == 0:
                    self.spike_times = []
                elif len(self.spike_times) > self.spike_times_max_len:
                    self.spike_times = self.spike_times[-self.spike_times_max_len:]

            for comp in comps:
                for syn in comp.synapses:
                    syn.on_post_spike(t)

            self.fire(t, branches=[self.main_axon, self.collateral_axon])

            if self.use_ais_reset:
                self.ais.V_mV = self.ais_reset_mV

        if self.record_traces:
            self.traces.setdefault("t", []).append(t)
            self.traces.setdefault("soma_V", []).append(self.soma.V_mV)
            self.traces.setdefault("ais_V", []).append(self.ais.V_mV)
            if self.trace_max_len is not None:
                self.truncate_traces(self.trace_max_len)

    def reset_traces(self):
        self.traces = {"t": [], "soma_V": [], "ais_V": []}

    def truncate_traces(self, max_len: int = 1000):
        if max_len < 0:
            raise ValueError("max_len must be non-negative")
        for k in list(self.traces):
            if max_len == 0:
                self.traces[k] = []
            elif len(self.traces[k]) > max_len:
                self.traces[k] = self.traces[k][-max_len:]

    def spike_count(self) -> int:
        return self._spike_count_total

    def spiked_since(self, count_before: int) -> bool:
        return self._spike_count_total > count_before

    def set_region_current(self, region: str, value_uA: float, segment_index: int = 0):
        comp = self._resolve_region_segment(region, segment_index)
        comp.I_ext_uA = float(value_uA)

    def clear_all_currents(self):
        for c in self.basal + self.apical + self.tuft + [self.soma, self.ais]:
            c.I_ext_uA = 0.0

    def reset_state(
        self,
        V_mV: float = -65.0,
        clear_spikes: bool = True,
        reset_traces: bool = True,
        clear_currents: bool = True,
    ):
        for comp in self.basal + self.apical + self.tuft + [self.soma, self.ais]:
            comp.reset_state(V_mV=V_mV, clear_current=clear_currents)

        if clear_spikes:
            self.spike_times = []
            self._spike_count_total = 0
            self.last_ais_spike_time = None
            self._ais_last_spike = None

        self.main_axon.last_spike_time = None
        self.main_axon.last_arrival_time = None
        self.collateral_axon.last_spike_time = None
        self.collateral_axon.last_arrival_time = None

        if reset_traces:
            self.reset_traces()

    def get_trace(self, key: str) -> np.ndarray:
        return np.array(self.traces.get(key, []), dtype=float)


# ----------------------------- Network Simulator (CPU) -----------------------------

class Network:
    """Manages time stepping, event scheduling, and neuron/external source registry."""
    def __init__(self, dt: float = 0.025, seed: Optional[int] = None):
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.dt = dt
        self.t = 0.0
        self.events = EventQueue()
        self.neurons: List[PyramidalNeuron] = []
        self.generators: List[ExternalSpikeGenerator] = []
        if seed is not None:
            np.random.seed(seed); random.seed(seed)
        # GPU engine (lazy)
        self._gpu_engine: Optional['CuPyEngine'] = None
        self._active_gpu_engine: Optional['CuPyEngine'] = None
        self._topology_version = 0

    def register_neuron(self, n: PyramidalNeuron):
        if n not in self.neurons:
            self.neurons.append(n)
            self._mark_topology_changed()

    def register_generator(self, g: ExternalSpikeGenerator):
        if g not in self.generators:
            self.generators.append(g)

    def _mark_topology_changed(self):
        self._topology_version += 1
        self._gpu_engine = None

    def add(self, obj):
        if isinstance(obj, PyramidalNeuron):
            self.register_neuron(obj)
        elif isinstance(obj, ExternalSpikeGenerator):
            self.register_generator(obj)
        else:
            raise ValueError("Unsupported object type")

    # ---------------- CPU run ----------------
    def run(self, T_ms: float, use_gpu: bool = False):
        if T_ms < 0.0:
            raise ValueError("T_ms must be non-negative")
        if use_gpu:
            if self._gpu_engine is None or self._gpu_engine.topology_version != self._topology_version:
                self._gpu_engine = CuPyEngine(self)
            previous_engine = self._active_gpu_engine
            self._active_gpu_engine = self._gpu_engine
            try:
                self._gpu_engine.run(T_ms)
            finally:
                self._active_gpu_engine = previous_engine
            return

        n_steps = int(math.ceil(T_ms / self.dt))
        t0 = self.t

        for step in range(n_steps):
            self.t = t0 + step * self.dt
            for fn in self.events.pop_ready(self.t):
                fn()

            for g in self.generators:
                g.step(self.t, self.dt)

            for n in self.neurons:
                n.step(self.t, self.dt)

        self.t = t0 + n_steps * self.dt

    def run_window(self, T_ms: float, use_gpu: bool = False):
        self.run(T_ms, use_gpu=use_gpu)

    def reset_state(
        self,
        V_mV: float = -65.0,
        reset_traces: bool = True,
        clear_spikes: bool = True,
        clear_currents: bool = True,
    ):
        self.t = 0.0
        self.events.clear()
        self._active_gpu_engine = None
        self._gpu_engine = None
        for neuron in self.neurons:
            neuron.reset_state(
                V_mV=V_mV,
                clear_spikes=clear_spikes,
                reset_traces=reset_traces,
                clear_currents=clear_currents,
            )

    def get_time_vector(self, absolute: bool = False) -> np.ndarray:
        n = 0
        if self.neurons:
            n = len(self.neurons[0].traces.get("soma_V", []))
            if absolute and "t" in self.neurons[0].traces:
                return np.array(self.neurons[0].traces["t"], dtype=float)
        return np.arange(n, dtype=float) * self.dt


# ----------------------------- CuPy Engine -----------------------------

class CuPyEngine:
    """
    Consolidated CuPy/CUDA execution engine.
    Builds dense/sparse structures once and runs fully vectorized timesteps.
    """
    def __init__(self, net: Network):
        try:
            import cupy as cp
            from cupyx.scipy import sparse as cpx_sparse  # noqa: F401
        except Exception as e:
            raise RuntimeError("CuPy is required for GPU execution. Please install cupy (e.g., cupy-cuda12x).") from e

        self.cp = cp
        from cupyx.scipy import sparse as cpx_sparse
        self.cpx_sparse = cpx_sparse
        self.net = net
        self.topology_version = net._topology_version

        self._build_indices()
        self._build_sparse_laplacian()
        self._build_synapse_arrays()

        # Buffers used during run
        self._buffer_pre_events: List[Tuple[int, float]] = []  # (syn index, arrival time)

    # ---------- Build topology ----------
    def _build_indices(self):
        cp = self.cp
        self.comp_list: List[Compartment] = []
        self.comp_to_idx: Dict[Compartment, int] = {}
        self.neuron_to_comp_idx: List[List[int]] = []  # per neuron list of comp indices
        self.soma_idx: List[int] = []
        self.ais_idx: List[int] = []
        # Map compartments to neuron ownership
        for n in self.net.neurons:
            comps = n.basal + n.apical + n.tuft + [n.soma, n.ais]
            idxs = []
            for c in comps:
                self.comp_to_idx[c] = len(self.comp_list)
                self.comp_list.append(c)
                idxs.append(self.comp_to_idx[c])
            self.neuron_to_comp_idx.append(idxs)
            self.soma_idx.append(self.comp_to_idx[n.soma])
            self.ais_idx.append(self.comp_to_idx[n.ais])

        self.n_comp = len(self.comp_list)

        # Per-comp arrays
        self.V = cp.asarray([c.V_mV for c in self.comp_list], dtype=cp.float64)
        self.Cm = cp.asarray([c.Cm_uF_per_cm2 for c in self.comp_list], dtype=cp.float64)
        self.Gleak = cp.asarray([c.Gleak_mS_per_cm2 for c in self.comp_list], dtype=cp.float64)
        self.Eleak = cp.asarray([c.Eleak_mV for c in self.comp_list], dtype=cp.float64)
        self.Area = cp.asarray([c.area_cm2 for c in self.comp_list], dtype=cp.float64)
        self.Iext_uA = cp.asarray([c.I_ext_uA for c in self.comp_list], dtype=cp.float64)

        # Mechanisms (HH) indices and params
        hh_idx = []
        gNa = []; gK = []; gL_mech = []; ENa = []; EK = []; ELeak = []
        m0 = []; h0 = []; n0 = []
        for i, c in enumerate(self.comp_list):
            has_hh = False
            for mech in c.mechanisms:
                if isinstance(mech, HodgkinHuxleyNaK):
                    has_hh = True
                    p = mech.p
                    gNa.append(p.gNa); gK.append(p.gK); gL_mech.append(p.gLeak)
                    ENa.append(p.ENa);  EK.append(p.EK); ELeak.append(p.ELeak)
                    m0.append(mech.m); h0.append(mech.h); n0.append(mech.n)
                    break
            if has_hh:
                hh_idx.append(i)
        self.hh_idx = self.cp.asarray(hh_idx, dtype=self.cp.int32)
        self.n_hh = int(len(hh_idx))
        if self.n_hh > 0:
            self.hh_gNa = cp.asarray(gNa, dtype=cp.float32)
            self.hh_gK  = cp.asarray(gK,  dtype=cp.float32)
            self.hh_gL  = cp.asarray(gL_mech, dtype=cp.float32)
            self.hh_ENa = cp.asarray(ENa, dtype=cp.float32)
            self.hh_EK  = cp.asarray(EK,  dtype=cp.float32)
            self.hh_ELeak = cp.asarray(ELeak, dtype=cp.float32)
            self.hh_m = cp.asarray(m0, dtype=cp.float32)
            self.hh_h = cp.asarray(h0, dtype=cp.float32)
            self.hh_n = cp.asarray(n0, dtype=cp.float32)
        else:
            # allocate minimal to avoid None checks
            self.hh_gNa = cp.zeros((0,), dtype=cp.float32)
            self.hh_gK  = cp.zeros((0,), dtype=cp.float32)
            self.hh_gL  = cp.zeros((0,), dtype=cp.float32)
            self.hh_ENa = cp.zeros((0,), dtype=cp.float32)
            self.hh_EK  = cp.zeros((0,), dtype=cp.float32)
            self.hh_ELeak = cp.zeros((0,), dtype=cp.float32)
            self.hh_m = cp.zeros((0,), dtype=cp.float32)
            self.hh_h = cp.zeros((0,), dtype=cp.float32)
            self.hh_n = cp.zeros((0,), dtype=cp.float32)

        # Spike detection arrays (per neuron)
        self.n_neurons = len(self.net.neurons)
        self.ais_thresh = cp.asarray([n.ais_threshold_mV for n in self.net.neurons], dtype=cp.float32)
        self.ais_refrac = cp.asarray([n.ais_refrac_ms for n in self.net.neurons], dtype=cp.float32)
        self.ais_prev_V = cp.asarray([n.ais.V_mV for n in self.net.neurons], dtype=cp.float32)
        self.last_spike_time = cp.asarray([
            n._ais_last_spike if n._ais_last_spike is not None else -1e9
            for n in self.net.neurons
        ], dtype=cp.float32)

    def _build_sparse_laplacian(self):
        # Build axial Laplacian L such that I_axial = L @ V
        cp = self.cp; cpx = self.cpx_sparse
        rows = []; cols = []; data = []
        for i, c in enumerate(self.comp_list):
            diag_accum = 0.0
            for (nbr, g) in c.neighbors:
                j = self.comp_to_idx[nbr]
                rows.append(i); cols.append(j); data.append(-g)
                diag_accum += g
            rows.append(i); cols.append(i); data.append(diag_accum)
        if len(rows) == 0:
            self.L = cpx.csr_matrix((self.n_comp, self.n_comp), dtype=cp.float32)
        else:
            self.L = cpx.csr_matrix((cp.asarray(data, dtype=cp.float32),
                                     (cp.asarray(rows, dtype=cp.int32),
                                      cp.asarray(cols, dtype=cp.int32))),
                                    shape=(self.n_comp, self.n_comp))

    def _build_synapse_arrays(self):
        cp = self.cp
        self.syn_list: List[Synapse] = []
        post_idx = []; kind = []; weight = []
        gmax = []; tau_rise = []; tau_decay = []; nmda_fraction = []; mg_conc = []
        # STDP/LTP params
        tau_pre = []; tau_post = []; A_plus = []; A_minus = []; eta = []
        tau_Ca = []; k_Ca = []; theta_pot = []; eta_pot = []
        theta_dep_low = []; theta_dep_high = []; eta_dep = []
        w_min = []; w_max = []
        # States
        r_fast = []; r_slow = []; x_pre = []; x_post = []; Ca = []
        # Mapping synapses to neuron by postsyn target
        self.syn_indices_by_neuron: List[List[int]] = [[] for _ in range(self.n_neurons)]

        # Collect synapses across all compartments
        for c in self.comp_list:
            for syn in c.synapses:
                idx = len(self.syn_list)
                self.syn_list.append(syn)
                post_idx.append(self.comp_to_idx[syn.post_compartment])
                kind.append(1 if syn.kind == SynapseKind.AMPA_NMDA else 2)
                weight.append(syn.weight)
                gmax.append(syn.gmax); tau_rise.append(max(1e-6, syn.tau_rise)); tau_decay.append(syn.tau_decay)
                nmda_fraction.append(syn.nmda_fraction); mg_conc.append(syn.mg_conc)
                tau_pre.append(syn.stdp.tau_pre if syn.stdp else 20.0)
                tau_post.append(syn.stdp.tau_post if syn.stdp else 20.0)
                A_plus.append(syn.stdp.A_plus if syn.stdp else 0.0)
                A_minus.append(syn.stdp.A_minus if syn.stdp else 0.0)
                eta.append(syn.stdp.eta if syn.stdp else 0.0)
                tau_Ca.append(syn.ltp.tau_Ca if syn.ltp else 50.0)
                k_Ca.append(syn.ltp.k_Ca if syn.ltp else 0.0)
                theta_pot.append(syn.ltp.theta_pot if syn.ltp else 1e9)  # disable if None
                eta_pot.append(syn.ltp.eta_pot if syn.ltp else 0.0)
                theta_dep_low.append(syn.ltp.theta_dep_low if syn.ltp else 1e9)
                theta_dep_high.append(syn.ltp.theta_dep_high if syn.ltp else -1e9)
                eta_dep.append(syn.ltp.eta_dep if syn.ltp else 0.0)
                w_min.append(syn.w_min); w_max.append(syn.w_max)

                r_fast.append(syn.r_fast); r_slow.append(syn.r_slow)
                x_pre.append(syn.x_pre);   x_post.append(syn.x_post); Ca.append(syn.Ca)

        self.n_syn = len(self.syn_list)
        self.post_idx = cp.asarray(post_idx, dtype=cp.int32) if self.n_syn else cp.zeros((0,), dtype=cp.int32)
        self.kind = cp.asarray(kind, dtype=cp.int8) if self.n_syn else cp.zeros((0,), dtype=cp.int8)
        self.weight = cp.asarray(weight, dtype=cp.float32) if self.n_syn else cp.zeros((0,), dtype=cp.float32)
        self.gmax = cp.asarray(gmax, dtype=cp.float32) if self.n_syn else cp.zeros((0,), dtype=cp.float32)
        self.tau_rise = cp.asarray(tau_rise, dtype=cp.float32) if self.n_syn else cp.zeros((0,), dtype=cp.float32)
        self.tau_decay = cp.asarray(tau_decay, dtype=cp.float32) if self.n_syn else cp.zeros((0,), dtype=cp.float32)
        self.nmda_fraction = cp.asarray(nmda_fraction, dtype=cp.float32) if self.n_syn else cp.zeros((0,), dtype=cp.float32)
        self.mg_conc = cp.asarray(mg_conc, dtype=cp.float32) if self.n_syn else cp.zeros((0,), dtype=cp.float32)

        self.tau_pre = cp.asarray(tau_pre, dtype=cp.float32) if self.n_syn else cp.zeros((0,), dtype=cp.float32)
        self.tau_post = cp.asarray(tau_post, dtype=cp.float32) if self.n_syn else cp.zeros((0,), dtype=cp.float32)
        self.A_plus = cp.asarray(A_plus, dtype=cp.float32) if self.n_syn else cp.zeros((0,), dtype=cp.float32)
        self.A_minus = cp.asarray(A_minus, dtype=cp.float32) if self.n_syn else cp.zeros((0,), dtype=cp.float32)
        self.eta = cp.asarray(eta, dtype=cp.float32) if self.n_syn else cp.zeros((0,), dtype=cp.float32)

        self.tau_Ca = cp.asarray(tau_Ca, dtype=cp.float32) if self.n_syn else cp.zeros((0,), dtype=cp.float32)
        self.k_Ca = cp.asarray(k_Ca, dtype=cp.float32) if self.n_syn else cp.zeros((0,), dtype=cp.float32)
        self.theta_pot = cp.asarray(theta_pot, dtype=cp.float32) if self.n_syn else cp.zeros((0,), dtype=cp.float32)
        self.eta_pot = cp.asarray(eta_pot, dtype=cp.float32) if self.n_syn else cp.zeros((0,), dtype=cp.float32)
        self.theta_dep_low = cp.asarray(theta_dep_low, dtype=cp.float32) if self.n_syn else cp.zeros((0,), dtype=cp.float32)
        self.theta_dep_high = cp.asarray(theta_dep_high, dtype=cp.float32) if self.n_syn else cp.zeros((0,), dtype=cp.float32)
        self.eta_dep = cp.asarray(eta_dep, dtype=cp.float32) if self.n_syn else cp.zeros((0,), dtype=cp.float32)
        self.w_min = cp.asarray(w_min, dtype=cp.float32) if self.n_syn else cp.zeros((0,), dtype=cp.float32)
        self.w_max = cp.asarray(w_max, dtype=cp.float32) if self.n_syn else cp.zeros((0,), dtype=cp.float32)

        self.r_fast = cp.asarray(r_fast, dtype=cp.float32) if self.n_syn else cp.zeros((0,), dtype=cp.float32)
        self.r_slow = cp.asarray(r_slow, dtype=cp.float32) if self.n_syn else cp.zeros((0,), dtype=cp.float32)
        self.x_pre = cp.asarray(x_pre, dtype=cp.float32) if self.n_syn else cp.zeros((0,), dtype=cp.float32)
        self.x_post = cp.asarray(x_post, dtype=cp.float32) if self.n_syn else cp.zeros((0,), dtype=cp.float32)
        self.Ca = cp.asarray(Ca, dtype=cp.float32) if self.n_syn else cp.zeros((0,), dtype=cp.float32)

        # Precompute decay constants per synapse
        if self.n_syn:
            self.tau_d_fast = self.cp.maximum(self.tau_decay, self.tau_rise + self.cp.float32(0.1))
            self.tau_d_slow = self.cp.maximum(self.cp.float32(40.0), self.tau_decay * self.cp.float32(12.0))
        else:
            self.tau_d_fast = cp.zeros((0,), dtype=cp.float32)
            self.tau_d_slow = cp.zeros((0,), dtype=cp.float32)

        # Build postsynaptic-neuron mapping (for post-spike STDP)
        self.syn_indices_by_neuron = [[] for _ in range(self.n_neurons)]
        comp_owner = self.cp.full((self.n_comp,), -1, dtype=self.cp.int32)
        # Map compartment -> neuron idx
        for ni, idxs in enumerate(self.neuron_to_comp_idx):
            for ci in idxs:
                comp_owner[ci] = ni
        if self.n_syn:
            # compute neuron index for each synapse's post comp
            owner_np = self.cp.asnumpy(comp_owner[self.post_idx])
            for s_idx, n_idx in enumerate(owner_np.tolist()):
                if n_idx >= 0:
                    self.syn_indices_by_neuron[n_idx].append(s_idx)

        # Assign GPU index to each Synapse object for scheduling
        for i, syn in enumerate(self.syn_list):
            syn._gpu_index = i

    def _buffer_pre_event(self, syn_index: int, t: Optional[float] = None):
        # push into CPU list; will be batch-applied on GPU inside the step
        if syn_index is not None:
            arrival_t = self.net.t if t is None else t
            self._buffer_pre_events.append((int(syn_index), float(arrival_t)))

    def _refresh_state_from_objects(self):
        cp = self.cp
        self.V = cp.asarray([c.V_mV for c in self.comp_list], dtype=cp.float64)
        self.Cm = cp.asarray([c.Cm_uF_per_cm2 for c in self.comp_list], dtype=cp.float64)
        self.Gleak = cp.asarray([c.Gleak_mS_per_cm2 for c in self.comp_list], dtype=cp.float64)
        self.Eleak = cp.asarray([c.Eleak_mV for c in self.comp_list], dtype=cp.float64)
        self.Area = cp.asarray([c.area_cm2 for c in self.comp_list], dtype=cp.float64)
        self.Iext_uA = cp.asarray([c.I_ext_uA for c in self.comp_list], dtype=cp.float64)
        self.ais_prev_V = cp.asarray([n.ais.V_mV for n in self.net.neurons], dtype=cp.float32)
        self.last_spike_time = cp.asarray([
            n._ais_last_spike if n._ais_last_spike is not None else -1e9
            for n in self.net.neurons
        ], dtype=cp.float32)

        if self.n_hh:
            m0 = []
            h0 = []
            n0 = []
            for comp_idx in cp.asnumpy(self.hh_idx).tolist():
                mech = next(m for m in self.comp_list[comp_idx].mechanisms if isinstance(m, HodgkinHuxleyNaK))
                m0.append(mech.m)
                h0.append(mech.h)
                n0.append(mech.n)
            self.hh_m = cp.asarray(m0, dtype=cp.float32)
            self.hh_h = cp.asarray(h0, dtype=cp.float32)
            self.hh_n = cp.asarray(n0, dtype=cp.float32)

        if self.n_syn:
            self.weight = cp.asarray([s.weight for s in self.syn_list], dtype=cp.float32)
            self.r_fast = cp.asarray([s.r_fast for s in self.syn_list], dtype=cp.float32)
            self.r_slow = cp.asarray([s.r_slow for s in self.syn_list], dtype=cp.float32)
            self.x_pre = cp.asarray([s.x_pre for s in self.syn_list], dtype=cp.float32)
            self.x_post = cp.asarray([s.x_post for s in self.syn_list], dtype=cp.float32)
            self.Ca = cp.asarray([s.Ca for s in self.syn_list], dtype=cp.float32)

    def _sync_state_to_objects(self):
        if self.n_comp:
            V_final = self.cp.asnumpy(self.V)
            for i, c in enumerate(self.comp_list):
                c.V_mV = float(V_final[i])

        if self.n_hh:
            hh_indices = self.cp.asnumpy(self.hh_idx).tolist()
            hh_m = self.cp.asnumpy(self.hh_m)
            hh_h = self.cp.asnumpy(self.hh_h)
            hh_n = self.cp.asnumpy(self.hh_n)
            for k, comp_idx in enumerate(hh_indices):
                for mech in self.comp_list[comp_idx].mechanisms:
                    if isinstance(mech, HodgkinHuxleyNaK):
                        mech.m = float(hh_m[k])
                        mech.h = float(hh_h[k])
                        mech.n = float(hh_n[k])
                        break

        if self.n_syn:
            weights = self.cp.asnumpy(self.weight)
            r_fast = self.cp.asnumpy(self.r_fast)
            r_slow = self.cp.asnumpy(self.r_slow)
            x_pre = self.cp.asnumpy(self.x_pre)
            x_post = self.cp.asnumpy(self.x_post)
            ca = self.cp.asnumpy(self.Ca)
            for i, syn in enumerate(self.syn_list):
                syn.weight = float(weights[i])
                syn.r_fast = float(r_fast[i])
                syn.r_slow = float(r_slow[i])
                syn.x_pre = float(x_pre[i])
                syn.x_post = float(x_post[i])
                syn.Ca = float(ca[i])

    # ---------- GPU run ----------
    def run(self, T_ms: float):
        cp = self.cp
        dt = float(self.net.dt)
        n_steps = int(math.ceil(T_ms / dt))
        self._buffer_pre_events.clear()
        self._refresh_state_from_objects()
        t0 = self.net.t

        for step in range(n_steps):
            t = t0 + step * dt
            self.net.t = t

            # 1) Run due events: they will buffer syn indices for this step
            for fn in self.net.events.pop_ready(t):
                fn()

            # 2) External generators (events scheduled at same t will be applied next step, like CPU path)
            for g in self.net.generators:
                g.step(t, dt)

            # 3) Apply batched pre-synaptic arrivals
            if self._buffer_pre_events:
                pending_pre_events = self._buffer_pre_events
                idx = cp.asarray([event[0] for event in pending_pre_events], dtype=cp.int32)
                for syn_index, arrival_t in pending_pre_events:
                    self.syn_list[syn_index].last_pre_spike = arrival_t
                # r increments
                if idx.size > 0:
                    cp.add.at(self.r_fast, idx, cp.float32(1.0))
                    cp.add.at(self.r_slow, idx, cp.float32(1.0))
                    # STDP pre: dw = eta*A_plus*x_post[idx]; x_pre += 1
                    dw = self.eta[idx] * self.A_plus[idx] * self.x_post[idx]
                    self.weight[idx] = cp.clip(self.weight[idx] + dw, self.w_min[idx], self.w_max[idx])
                    self.x_pre[idx] = self.x_pre[idx] + cp.float32(1.0)
                self._buffer_pre_events.clear()

            # 4) Decay STDP traces
            if self.n_syn:
                self.x_pre  = self.x_pre  + (-dt) * self.x_pre  / self.tau_pre
                self.x_post = self.x_post + (-dt) * self.x_post / self.tau_post

            # 5) Decay synaptic states
            if self.n_syn:
                self.r_fast = self.r_fast * cp.exp((-dt) / self.tau_d_fast)
                self.r_slow = self.r_slow * cp.exp((-dt) / self.tau_d_slow)

            # 6) Synaptic currents accumulation per compartment
            I_syn_comp = cp.zeros((self.n_comp,), dtype=cp.float32)
            if self.n_syn:
                rise_fac = (cp.float32(1.0) - cp.exp((-dt) / self.tau_rise))
                g_total = self.gmax * self.weight
                V_post_syn = self.V[self.post_idx]

                # Excitatory (AMPA+NMDA)
                ex_mask = (self.kind == 1)
                if cp.any(ex_mask):
                    idx = cp.where(ex_mask)[0]
                    g_ampa = g_total[idx] * (cp.float32(1.0) - self.nmda_fraction[idx]) * rise_fac[idx] * self.r_fast[idx]
                    g_nmda_base = g_total[idx] * self.nmda_fraction[idx] * rise_fac[idx] * self.r_slow[idx]
                    B = cp.float32(1.0) / (cp.float32(1.0) + self.mg_conc[idx] * cp.exp(-0.062 * V_post_syn[idx]) / 3.57)
                    g_nmda = g_nmda_base * B
                    I_ampa = g_ampa * (V_post_syn[idx] - cp.float32(E_AMPA))
                    I_nmda = g_nmda * (V_post_syn[idx] - cp.float32(E_NMDA))
                    I_ex = I_ampa + I_nmda
                    # Accumulate to compartments
                    I_comp_ex = cp.bincount(self.post_idx[idx], weights=I_ex, minlength=self.n_comp).astype(cp.float32, copy=False)
                    I_syn_comp = I_syn_comp + I_comp_ex
                    # Ca dynamics (only excitatory, driven by NMDA inward current magnitude)
                    self.Ca[idx] = self.Ca[idx] + dt * (-self.k_Ca[idx] * I_nmda) - dt * (self.Ca[idx] / self.tau_Ca[idx])
                    # Ca-based LTP/LTD
                    pot_mask = (self.Ca[idx] > self.theta_pot[idx])
                    if cp.any(pot_mask):
                        idp = idx[pot_mask]
                        dw = self.eta_pot[idp] * (cp.float32(1.0) - (self.weight[idp] / self.w_max[idp])) * \
                             ((self.Ca[idp] - self.theta_pot[idp]) / (self.Ca[idp] + self.theta_pot[idp]))
                        self.weight[idp] = cp.clip(self.weight[idp] + dw * dt, self.w_min[idp], self.w_max[idp])
                    ltd_mask = (self.Ca[idx] > self.theta_dep_low[idx]) & (self.Ca[idx] < self.theta_dep_high[idx])
                    if cp.any(ltd_mask):
                        idd = idx[ltd_mask]
                        dw = - self.eta_dep[idd] * (self.weight[idd] / self.w_max[idd]) * \
                             ((self.theta_dep_high[idd] - self.Ca[idd]) / self.theta_dep_high[idd])
                        self.weight[idd] = cp.clip(self.weight[idd] + dw * dt, self.w_min[idd], self.w_max[idd])

                # Inhibitory (GABA_A)
                inh_mask = (self.kind == 2)
                if cp.any(inh_mask):
                    idx = cp.where(inh_mask)[0]
                    g_gaba = g_total[idx] * rise_fac[idx] * self.r_fast[idx]
                    I_inh = g_gaba * (V_post_syn[idx] - cp.float32(E_Cl))
                    I_comp_inh = cp.bincount(self.post_idx[idx], weights=I_inh, minlength=self.n_comp).astype(cp.float32, copy=False)
                    I_syn_comp = I_syn_comp + I_comp_inh

            # 7) Ionic currents (HH) per compartment (scatter from HH subset)
            I_ion_comp = cp.zeros((self.n_comp,), dtype=cp.float32)
            if self.n_hh > 0:
                V_hh = self.V[self.hh_idx]

                # Rate functions mirror the CPU vtrap/exp clipping logic.
                def cp_vtrap(x, y):
                    z = x / y
                    eps = cp.float32(1e-7)
                    safe_z = cp.where(cp.abs(z) < eps, eps, cp.clip(z, -50.0, 50.0))
                    raw = x / (cp.float32(1.0) - cp.exp(-safe_z))
                    small = y * (cp.float32(1.0) + cp.float32(0.5) * z)
                    low = -x * cp.exp(cp.minimum(z, cp.float32(50.0)))
                    return cp.where(
                        cp.abs(z) < eps,
                        small,
                        cp.where(z < -50.0, low, cp.where(z > 50.0, x, raw)),
                    )

                am = 0.1 * cp_vtrap(V_hh + 40.0, cp.float32(10.0))
                bm = 4.0 * cp.exp(cp.clip(-(V_hh + 65.0) / 18.0, -50.0, 50.0))
                ah = 0.07 * cp.exp(cp.clip(-(V_hh + 65.0) / 20.0, -50.0, 50.0))
                bh = cp.float32(1.0) / (cp.float32(1.0) + cp.exp(cp.clip(-(V_hh + 35.0) / 10.0, -50.0, 50.0)))
                an = 0.01 * cp_vtrap(V_hh + 55.0, cp.float32(10.0))
                bn = 0.125 * cp.exp(cp.clip(-(V_hh + 65.0) / 80.0, -50.0, 50.0))

                self.hh_m = self.hh_m + dt * (am * (1.0) - (am + bm) * self.hh_m)
                self.hh_h = self.hh_h + dt * (ah * (1.0) - (ah + bh) * self.hh_h)
                self.hh_n = self.hh_n + dt * (an * (1.0) - (an + bn) * self.hh_n)

                INa = self.hh_gNa * (self.hh_m**3) * self.hh_h * (V_hh - self.hh_ENa)
                IK  = self.hh_gK  * (self.hh_n**4) * (V_hh - self.hh_EK)
                ILm = self.hh_gL  * (V_hh - self.hh_ELeak)
                I_hh = INa + IK + ILm

                I_comp_hh = cp.bincount(self.hh_idx, weights=I_hh, minlength=self.n_comp).astype(cp.float32, copy=False)
                I_ion_comp = I_ion_comp + I_comp_hh

            # 8) Leak current per compartment
            I_leak = self.Gleak * (self.V - self.Eleak)

            # 9) Axial current via Laplacian
            I_axial = self.L @ self.V  # (uA) as in CPU path

            # 10) External current density
            I_ext_density = self.Iext_uA / self.Area

            # 11) Net outward density and integrate V
            I_out = I_leak + I_ion_comp + I_axial + I_syn_comp - I_ext_density
            self.V = self.V + (-(I_out / self.Cm)) * dt

            # 12) Spike detection on GPU
            ais_V = self.V[self.cp.asarray(self.ais_idx, dtype=self.cp.int32)]
            can_spike = (t - self.last_spike_time) >= self.ais_refrac
            crossed = (ais_V >= self.ais_thresh) & (self.ais_prev_V < self.ais_thresh)
            new_spikes_mask = can_spike & crossed
            if cp.any(new_spikes_mask):
                spike_neurons = cp.where(new_spikes_mask)[0]
                # Update last-spike times
                self.last_spike_time[spike_neurons] = cp.float32(t)

                # CPU-side bookkeeping: traces & routing & STDP(post)
                spike_idx_list = self.cp.asnumpy(spike_neurons).tolist()
                for ni in spike_idx_list:
                    neuron = self.net.neurons[ni]
                    neuron._ais_last_spike = t
                    neuron.last_ais_spike_time = t
                    neuron._spike_count_total += 1
                    neuron.spike_times.append(t)
                    if neuron.spike_times_max_len is not None:
                        if neuron.spike_times_max_len == 0:
                            neuron.spike_times = []
                        elif len(neuron.spike_times) > neuron.spike_times_max_len:
                            neuron.spike_times = neuron.spike_times[-neuron.spike_times_max_len:]
                    # Post-STDP for all synapses targeting this neuron
                    s_list = self.syn_indices_by_neuron[ni]
                    if s_list:
                        s_idx = self.cp.asarray(s_list, dtype=self.cp.int32)
                        dw = - self.eta[s_idx] * self.A_minus[s_idx] * self.x_pre[s_idx]
                        self.weight[s_idx] = self.cp.clip(self.weight[s_idx] + dw, self.w_min[s_idx], self.w_max[s_idx])
                        self.x_post[s_idx] = self.x_post[s_idx] + self.cp.float32(1.0)
                        for syn_index in s_list:
                            self.syn_list[syn_index].last_post_spike = t
                    # Route along axon branches (events get queued and batch-applied in the next step)
                    neuron.fire(t, branches=[neuron.main_axon, neuron.collateral_axon])
                    if neuron.use_ais_reset:
                        self.V[self.ais_idx[ni]] = self.cp.float32(neuron.ais_reset_mV)

            # 13) Update previous AIS V and record traces (CPU lists)
            ais_V = self.V[self.cp.asarray(self.ais_idx, dtype=self.cp.int32)]
            self.ais_prev_V = ais_V  # gpu array
            # Append soma/ais traces to Python lists
            # (small host transfer; cost negligible for modest neuron counts)
            for ni, neuron in enumerate(self.net.neurons):
                if neuron.record_traces:
                    v_soma = float(self.V[self.soma_idx[ni]].get())
                    v_ais  = float(self.V[self.ais_idx[ni]].get())
                    neuron.traces.setdefault("t", []).append(t)
                    neuron.traces.setdefault("soma_V", []).append(v_soma)
                    neuron.traces.setdefault("ais_V", []).append(v_ais)
                    if neuron.trace_max_len is not None:
                        neuron.truncate_traces(neuron.trace_max_len)

        # After run: write final GPU state back into CPU objects.
        self._sync_state_to_objects()
        self.net.t = t0 + n_steps * dt

    # (no separate finish; state already synced)
# ----------------------------- Convenience API -----------------------------

__all__ = [
    "Network", "PyramidalNeuron", "ExternalSpikeGenerator", "SynapseKind",
    "HHNaKParams", "HodgkinHuxleyNaK", "STDPParams", "LTPParams"
]


if __name__ == "__main__":
    # Minimal smoke test: runs on GPU if available, otherwise raises at engine init
    net = Network(dt=0.025, seed=1)
    pyr = PyramidalNeuron("P1", net)
    genA = ExternalSpikeGenerator("A", net, rate_hz=12.0)
    genB = ExternalSpikeGenerator("B", net, rate_hz=15.0)
    genC = ExternalSpikeGenerator("C", net, rate_hz=10.0)
    net.register_generator(genA); net.register_generator(genB); net.register_generator(genC)

    for i in range(5):
        pyr.attach_synapse_from(genA, "basal", i, SynapseKind.AMPA_NMDA, weight=0.5)
        pyr.attach_synapse_from(genB, "apical", i, SynapseKind.AMPA_NMDA, weight=0.4)
        pyr.attach_synapse_from(genC, "tuft",  i, SynapseKind.AMPA_NMDA, weight=0.35)

    genI = ExternalSpikeGenerator("I", net, rate_hz=20.0)
    net.register_generator(genI)
    pyr.attach_synapse_from(genI, "ais", 0, SynapseKind.GABA_A, weight=1.2)

    targets = [PyramidalNeuron(f"T{i}", net) for i in range(3)]
    for post in targets:
        pyr.connect_axon_to(post, "main", "basal", 0, SynapseKind.AMPA_NMDA, weight=0.5)
        pyr.connect_axon_to(post, "collateral", "apical", 0, SynapseKind.AMPA_NMDA, weight=0.35)

    net.run(150.0, use_gpu=False)
    print(f"CPU P1 spikes: {len(pyr.spike_times)}")

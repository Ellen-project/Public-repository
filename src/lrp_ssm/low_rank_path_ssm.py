from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from snn.pyramidal_neuron import Network, PyramidalNeuron


class FullSNNPathRouter:
    """
    Event-driven path router for LowRankPathSSM-A.

    One PyramidalNeuron owns one low-rank path. A path gate is 1.0 when the
    corresponding neuron's AIS spikes during the SNN window, otherwise 0.0.
    The router is intentionally outside torch autograd in v0.
    """

    def __init__(
        self,
        num_paths: int,
        input_dim: int,
        window_ms: float = 1.0,
        dt: float = 0.025,
        seed: int = 1,
        current_scale: float = 1e-9,
        current_clip: float = 1e-6,
        gate_ema_rho: float = 0.9,
        n_basal: int = 10,
        n_apical: int = 10,
        n_tuft: int = 6,
        record_traces: bool = False,
        trace_max_len: Optional[int] = 1000,
        spike_times_max_len: Optional[int] = 1000,
        use_ais_reset: bool = True,
    ):
        if num_paths <= 0:
            raise ValueError("num_paths must be positive")
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if window_ms <= 0.0:
            raise ValueError("window_ms must be positive")
        if current_scale < 0.0:
            raise ValueError("current_scale must be non-negative")
        if current_clip <= 0.0:
            raise ValueError("current_clip must be positive")
        if not 0.0 <= gate_ema_rho < 1.0:
            raise ValueError("gate_ema_rho must be in [0, 1)")

        self.num_paths = int(num_paths)
        self.input_dim = int(input_dim)
        self.window_ms = float(window_ms)
        self.dt = float(dt)
        self.seed = int(seed)
        self.current_scale = float(current_scale)
        self.current_clip = float(current_clip)
        self.gate_ema_rho = float(gate_ema_rho)
        self.n_basal = int(n_basal)
        self.n_apical = int(n_apical)
        self.n_tuft = int(n_tuft)
        self.record_traces = bool(record_traces)
        self.trace_max_len = trace_max_len
        self.spike_times_max_len = spike_times_max_len
        self.use_ais_reset = bool(use_ais_reset)

        self.net = Network(dt=dt, seed=seed)
        self.neurons: List[PyramidalNeuron] = [
            PyramidalNeuron(
                f"path_{p}",
                self.net,
                n_basal=n_basal,
                n_apical=n_apical,
                n_tuft=n_tuft,
            )
            for p in range(self.num_paths)
        ]

        for neuron in self.neurons:
            neuron.record_traces = record_traces
            neuron.trace_max_len = trace_max_len
            neuron.spike_times_max_len = spike_times_max_len
            neuron.use_ais_reset = use_ais_reset
            neuron.main_axon.delay_noise_sd_ms = 0.0
            neuron.collateral_axon.delay_noise_sd_ms = 0.0

        rng = np.random.default_rng(seed)
        self.W_basal = (rng.standard_normal((self.num_paths, self.input_dim)) * 0.01).astype(np.float32)
        self.W_apical = (rng.standard_normal((self.num_paths, self.input_dim)) * 0.01).astype(np.float32)
        self.W_tuft = (rng.standard_normal((self.num_paths, self.input_dim)) * 0.01).astype(np.float32)

        self.gate_ema = np.zeros(self.num_paths, dtype=np.float32)
        self.firing_rate = np.zeros(self.num_paths, dtype=np.float32)
        self.gate_counts = np.zeros(self.num_paths, dtype=np.float32)
        self.window_count = 0
        self.last_gate = np.zeros(self.num_paths, dtype=np.float32)
        self.last_returned_gate = np.zeros(self.num_paths, dtype=np.float32)
        self.last_currents = np.zeros((self.num_paths, 3), dtype=np.float32)

    def clear_currents(self):
        for neuron in self.neurons:
            neuron.clear_all_currents()

    def reset_statistics(self):
        self.gate_ema.fill(0.0)
        self.firing_rate.fill(0.0)
        self.gate_counts.fill(0.0)
        self.window_count = 0
        self.last_gate.fill(0.0)
        self.last_returned_gate.fill(0.0)
        self.last_currents.fill(0.0)

    def reset_traces(self):
        for neuron in self.neurons:
            neuron.reset_traces()

    def reset_state(
        self,
        reset_statistics: bool = True,
        reset_traces: bool = True,
        clear_spikes: bool = True,
    ):
        self.net.reset_state(
            reset_traces=reset_traces,
            clear_spikes=clear_spikes,
            clear_currents=True,
        )
        self.clear_currents()
        if reset_statistics:
            self.reset_statistics()

    def get_config(self) -> Dict[str, Any]:
        return {
            "num_paths": self.num_paths,
            "input_dim": self.input_dim,
            "window_ms": self.window_ms,
            "dt": self.dt,
            "seed": self.seed,
            "current_scale": self.current_scale,
            "current_clip": self.current_clip,
            "gate_ema_rho": self.gate_ema_rho,
            "n_basal": self.n_basal,
            "n_apical": self.n_apical,
            "n_tuft": self.n_tuft,
            "record_traces": self.record_traces,
            "trace_max_len": self.trace_max_len,
            "spike_times_max_len": self.spike_times_max_len,
            "use_ais_reset": self.use_ais_reset,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FullSNNPathRouter":
        kwargs = dict(config)
        weights = {name: kwargs.pop(name) for name in ("W_basal", "W_apical", "W_tuft") if name in kwargs}
        router = cls(**kwargs)
        for name, value in weights.items():
            arr = np.asarray(value, dtype=np.float32)
            expected = (router.num_paths, router.input_dim)
            if arr.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {arr.shape}")
            setattr(router, name, arr.copy())
        return router

    def save_preset(self, path: str):
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        def encode_optional(value: Optional[int]) -> np.ndarray:
            return np.array(-1 if value is None else int(value), dtype=np.int64)

        np.savez(
            path_obj,
            num_paths=np.array(self.num_paths, dtype=np.int64),
            input_dim=np.array(self.input_dim, dtype=np.int64),
            window_ms=np.array(self.window_ms, dtype=np.float64),
            dt=np.array(self.dt, dtype=np.float64),
            seed=np.array(self.seed, dtype=np.int64),
            current_scale=np.array(self.current_scale, dtype=np.float64),
            current_clip=np.array(self.current_clip, dtype=np.float64),
            gate_ema_rho=np.array(self.gate_ema_rho, dtype=np.float64),
            n_basal=np.array(self.n_basal, dtype=np.int64),
            n_apical=np.array(self.n_apical, dtype=np.int64),
            n_tuft=np.array(self.n_tuft, dtype=np.int64),
            record_traces=np.array(self.record_traces, dtype=np.bool_),
            trace_max_len=encode_optional(self.trace_max_len),
            spike_times_max_len=encode_optional(self.spike_times_max_len),
            use_ais_reset=np.array(self.use_ais_reset, dtype=np.bool_),
            W_basal=self.W_basal.astype(np.float32, copy=True),
            W_apical=self.W_apical.astype(np.float32, copy=True),
            W_tuft=self.W_tuft.astype(np.float32, copy=True),
        )
        return path_obj

    @classmethod
    def load_preset(cls, path: str, **override_kwargs) -> "FullSNNPathRouter":
        with np.load(path, allow_pickle=False) as data:
            def optional_int(name: str) -> Optional[int]:
                value = int(data[name])
                return None if value < 0 else value

            config = {
                "num_paths": int(data["num_paths"]),
                "input_dim": int(data["input_dim"]),
                "window_ms": float(data["window_ms"]),
                "dt": float(data["dt"]),
                "seed": int(data["seed"]) if "seed" in data.files else 1,
                "current_scale": float(data["current_scale"]),
                "current_clip": float(data["current_clip"]),
                "gate_ema_rho": float(data["gate_ema_rho"]),
                "n_basal": int(data["n_basal"]),
                "n_apical": int(data["n_apical"]),
                "n_tuft": int(data["n_tuft"]),
                "record_traces": bool(data["record_traces"]),
                "trace_max_len": optional_int("trace_max_len"),
                "spike_times_max_len": optional_int("spike_times_max_len"),
                "use_ais_reset": bool(data["use_ais_reset"]),
                "W_basal": data["W_basal"].astype(np.float32),
                "W_apical": data["W_apical"].astype(np.float32),
                "W_tuft": data["W_tuft"].astype(np.float32),
            }

        config.update(override_kwargs)
        return cls.from_config(config)

    def _as_input_vector(self, x_t_np: np.ndarray) -> np.ndarray:
        x = np.asarray(x_t_np, dtype=np.float32)
        if x.shape != (self.input_dim,):
            raise ValueError(f"x_t_np must have shape ({self.input_dim},), got {x.shape}")
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def _encode_currents(self, x: np.ndarray) -> np.ndarray:
        i_b = np.maximum(self.W_basal @ x, 0.0) * self.current_scale
        i_a = np.maximum(self.W_apical @ x, 0.0) * self.current_scale
        i_t = np.maximum(self.W_tuft @ x, 0.0) * self.current_scale
        currents = np.stack([i_b, i_a, i_t], axis=1)
        currents = np.clip(currents, 0.0, self.current_clip)
        return currents.astype(np.float32, copy=False)

    def inject_input(self, x_t_np: np.ndarray):
        x = self._as_input_vector(x_t_np)
        self.clear_currents()
        currents = self._encode_currents(x)
        self.last_currents = currents

        for p, neuron in enumerate(self.neurons):
            if neuron.basal:
                neuron.set_region_current("basal", float(currents[p, 0]), 0)
            if neuron.apical:
                neuron.set_region_current("apical", float(currents[p, 1]), 0)
            if neuron.tuft:
                neuron.set_region_current("tuft", float(currents[p, 2]), 0)

    def step(self, x_t_np: np.ndarray, use_ema: bool = False, use_gpu: bool = False) -> np.ndarray:
        before = np.array([neuron.spike_count() for neuron in self.neurons], dtype=np.int64)

        self.inject_input(x_t_np)
        try:
            self.net.run_window(self.window_ms, use_gpu=use_gpu)
        finally:
            self.clear_currents()

        after = np.array([neuron.spike_count() for neuron in self.neurons], dtype=np.int64)
        gate = (after > before).astype(np.float32)

        self.window_count += 1
        self.gate_counts += gate
        self.firing_rate = self.gate_counts / max(1, self.window_count)
        self.gate_ema = self.gate_ema_rho * self.gate_ema + (1.0 - self.gate_ema_rho) * gate
        self.last_gate = gate
        self.last_returned_gate = self.gate_ema.astype(np.float32) if use_ema else gate
        return self.last_returned_gate.astype(np.float32, copy=True)

    def get_diagnostics(self) -> Dict[str, np.ndarray | float | int]:
        spike_counts = np.array([neuron.spike_count() for neuron in self.neurons], dtype=np.int64)
        ais_v = np.array([neuron.ais.V_mV for neuron in self.neurons], dtype=np.float32)
        return {
            "firing_rate": self.firing_rate.copy(),
            "gate_ema": self.gate_ema.copy(),
            "last_gate": self.last_gate.copy(),
            "last_returned_gate": self.last_returned_gate.copy(),
            "last_currents": self.last_currents.copy(),
            "net_time": float(self.net.t),
            "spike_counts": spike_counts,
            "active_paths": int(self.last_gate.sum()),
            "ais_voltage_min": float(ais_v.min()) if ais_v.size else 0.0,
            "ais_voltage_max": float(ais_v.max()) if ais_v.size else 0.0,
        }


class LowRankPathSSMCore(nn.Module):
    """
    Low-rank path SSM core.

    h_t = alpha * h_{t-1} + B x_t + gamma * sum_p gate_p U_p phi(V_p h + E_p x)
    y_t = C h_t + D x_t
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        num_paths: int,
        rank: int,
        output_dim: Optional[int] = None,
        gamma_init: float = 0.1,
        init_scale: float = 0.02,
        use_direct_output: bool = True,
        normalize_state: bool = True,
        learnable_gamma: bool = True,
        gamma_min: float = 0.0,
        gamma_max: float = 2.0,
        path_residual_scale: float = 1.0,
        use_path_norm: bool = True,
        path_dropout: float = 0.0,
        force_min_active_paths: int = 0,
        topk_fallback: int = 0,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if num_paths <= 0:
            raise ValueError("num_paths must be positive")
        if rank <= 0:
            raise ValueError("rank must be positive")

        self.input_dim = int(input_dim)
        self.state_dim = int(state_dim)
        self.num_paths = int(num_paths)
        self.rank = int(rank)
        self.output_dim = int(output_dim or input_dim)
        self.use_direct_output = bool(use_direct_output)
        self.normalize_state = bool(normalize_state)
        self.learnable_gamma = bool(learnable_gamma)
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)
        self.path_residual_scale = float(path_residual_scale)
        self.use_path_norm = bool(use_path_norm)
        self.force_min_active_paths = int(force_min_active_paths)
        self.topk_fallback = int(topk_fallback)
        if self.gamma_max <= self.gamma_min:
            raise ValueError("gamma_max must be greater than gamma_min")
        if self.force_min_active_paths < 0:
            raise ValueError("force_min_active_paths must be non-negative")
        if self.topk_fallback < 0:
            raise ValueError("topk_fallback must be non-negative")

        self.U = nn.Parameter(torch.randn(num_paths, state_dim, rank) * init_scale)
        self.V = nn.Parameter(torch.randn(num_paths, rank, state_dim) * init_scale)
        self.E = nn.Parameter(torch.randn(num_paths, rank, input_dim) * init_scale)

        self.B = nn.Linear(input_dim, state_dim, bias=False)
        self.C = nn.Linear(state_dim, self.output_dim, bias=False)
        self.D = nn.Linear(input_dim, self.output_dim, bias=False) if use_direct_output else None

        self.log_decay = nn.Parameter(torch.zeros(state_dim))
        gamma_frac = (float(gamma_init) - self.gamma_min) / (self.gamma_max - self.gamma_min)
        gamma_frac = min(max(gamma_frac, 1e-6), 1.0 - 1e-6)
        raw_gamma = math.log(gamma_frac / (1.0 - gamma_frac))
        if self.learnable_gamma:
            self.raw_gamma = nn.Parameter(torch.tensor(float(raw_gamma)))
        else:
            self.register_buffer("raw_gamma", torch.tensor(float(raw_gamma)))
        self.norm = nn.LayerNorm(state_dim) if normalize_state else nn.Identity()
        self.path_norm = nn.LayerNorm(state_dim) if use_path_norm else nn.Identity()
        self.path_dropout = nn.Dropout(float(path_dropout))

    @property
    def gamma(self) -> torch.Tensor:
        return self.gamma_min + (self.gamma_max - self.gamma_min) * torch.sigmoid(self.raw_gamma)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        old_gamma_key = prefix + "gamma"
        raw_gamma_key = prefix + "raw_gamma"
        if old_gamma_key in state_dict and raw_gamma_key not in state_dict:
            gamma_value = state_dict.pop(old_gamma_key).detach().float()
            gamma_value = torch.clamp(gamma_value, self.gamma_min + 1e-6, self.gamma_max - 1e-6)
            gamma_frac = (gamma_value - self.gamma_min) / (self.gamma_max - self.gamma_min)
            state_dict[raw_gamma_key] = torch.log(gamma_frac / (1.0 - gamma_frac))
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        for key in (prefix + "path_norm.weight", prefix + "path_norm.bias"):
            if key in missing_keys:
                missing_keys.remove(key)

    def initial_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        param = self.U
        return torch.zeros(
            batch_size,
            self.state_dim,
            device=device or param.device,
            dtype=dtype or param.dtype,
        )

    def step(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        gate_t: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]]:
        if x_t.ndim != 2 or x_t.shape[-1] != self.input_dim:
            raise ValueError(f"x_t must have shape [B, {self.input_dim}]")
        if h.ndim != 2 or h.shape[-1] != self.state_dim:
            raise ValueError(f"h must have shape [B, {self.state_dim}]")
        if gate_t.ndim != 2 or gate_t.shape[-1] != self.num_paths:
            raise ValueError(f"gate_t must have shape [B, {self.num_paths}]")

        alpha = torch.exp(-F.softplus(self.log_decay))

        q_state = torch.einsum("bn,prn->bpr", h, self.V)
        q_input = torch.einsum("bd,prd->bpr", x_t, self.E)
        q_pre = F.silu(q_state + q_input)
        effective_gate = gate_t
        min_active = max(self.force_min_active_paths, self.topk_fallback)
        if min_active > 0:
            min_active = min(min_active, self.num_paths)
            active = effective_gate.detach().sum(dim=-1)
            needs_extra = active < float(min_active)
            if bool(needs_extra.any()):
                q_score = q_pre.detach().abs().mean(dim=-1)
                topk_idx = torch.topk(q_score, k=min_active, dim=-1).indices
                fallback = torch.zeros_like(effective_gate).scatter(1, topk_idx, 1.0)
                effective_gate = torch.where(needs_extra.unsqueeze(-1), torch.maximum(effective_gate, fallback), effective_gate)
        q = q_pre * effective_gate.unsqueeze(-1)

        raw_delta = torch.einsum("bpr,pnr->bn", q, self.U)
        raw_delta = raw_delta / math.sqrt(max(1, self.num_paths * self.rank))
        delta = self.path_norm(raw_delta)
        delta = self.path_dropout(delta)

        base_update = alpha * h + self.B(x_t)
        scaled_delta = self.gamma * self.path_residual_scale * delta
        h_next = base_update + scaled_delta
        h_next = self.norm(h_next)

        y = self.C(h_next)
        if self.D is not None:
            y = y + self.D(x_t)

        if return_diagnostics:
            raw_delta_norm = raw_delta.detach().norm(dim=-1)
            scaled_delta_norm = scaled_delta.detach().norm(dim=-1)
            base_update_norm = base_update.detach().norm(dim=-1)
            diagnostics = {
                "alpha": alpha.detach(),
                "gamma": self.gamma.detach().expand(x_t.shape[0]),
                "raw_delta_norm": raw_delta_norm,
                "scaled_delta_norm": scaled_delta_norm,
                "base_update_norm": base_update_norm,
                "path_to_base_ratio": scaled_delta_norm / (base_update_norm + 1e-6),
                "delta_norm": scaled_delta_norm,
                "h_norm": h_next.detach().norm(dim=-1),
                "active_paths": effective_gate.detach().sum(dim=-1),
            }
            return y, h_next, diagnostics
        return y, h_next


RouterLike = Union[FullSNNPathRouter, Sequence[FullSNNPathRouter]]


class LowRankPathSSMModel(nn.Module):
    """
    Full v0 model: fixed non-differentiable SNN router + trainable SSM core.

    Batch safety:
        A single router instance is stateful, so B > 1 is rejected unless a
        separate router is provided for each batch element.
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: int = 128,
        num_paths: int = 8,
        rank: int = 4,
        output_dim: Optional[int] = None,
        router: Optional[RouterLike] = None,
        router_kwargs: Optional[Dict[str, Any]] = None,
        use_router_ema: bool = False,
        use_gpu_router: bool = False,
        reset_router_state_on_forward: bool = True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.state_dim = int(state_dim)
        self.num_paths = int(num_paths)
        self.rank = int(rank)
        self.output_dim = int(output_dim or input_dim)
        self.use_router_ema = bool(use_router_ema)
        self.use_gpu_router = bool(use_gpu_router)
        self.reset_router_state_on_forward = bool(reset_router_state_on_forward)

        if router is None:
            kwargs = dict(router_kwargs or {})
            kwargs.pop("num_paths", None)
            kwargs.pop("input_dim", None)
            router = FullSNNPathRouter(num_paths=num_paths, input_dim=input_dim, **kwargs)
        self.router = router

        self.ssm_core = LowRankPathSSMCore(
            input_dim=input_dim,
            state_dim=state_dim,
            num_paths=num_paths,
            rank=rank,
            output_dim=self.output_dim,
        )

    def _routers_for_batch(self, batch_size: int) -> List[FullSNNPathRouter]:
        if isinstance(self.router, FullSNNPathRouter):
            if batch_size != 1:
                raise ValueError(
                    "FullSNNPathRouter is stateful. Use batch size 1, or pass one router per batch element."
                )
            return [self.router]

        routers = list(self.router)
        if len(routers) != batch_size:
            raise ValueError(f"Expected {batch_size} routers, got {len(routers)}")
        if any(not isinstance(router, FullSNNPathRouter) for router in routers):
            raise TypeError("All routers must be FullSNNPathRouter instances")
        return routers

    def reset_router_state(self):
        routers = [self.router] if isinstance(self.router, FullSNNPathRouter) else list(self.router)
        for router in routers:
            router.reset_state()

    @staticmethod
    def _stack_step_diagnostics(step_diagnostics: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not step_diagnostics:
            return {}
        keys = step_diagnostics[0].keys()
        stacked = {}
        for key in keys:
            values = [diag[key] for diag in step_diagnostics]
            if values[0].ndim == 1:
                stacked[key] = torch.stack(values, dim=1)
            else:
                stacked[key] = torch.stack(values, dim=0)
        return stacked

    def forward(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        return_gates: bool = False,
        return_diagnostics: bool = False,
    ):
        if x.ndim != 3 or x.shape[-1] != self.input_dim:
            raise ValueError(f"x must have shape [B, T, {self.input_dim}]")

        batch_size, seq_len, _ = x.shape
        routers = self._routers_for_batch(batch_size)
        if self.reset_router_state_on_forward:
            for router in routers:
                router.reset_state()
        h = h0 if h0 is not None else self.ssm_core.initial_state(batch_size, x.device, x.dtype)
        if h.shape != (batch_size, self.state_dim):
            raise ValueError(f"h0 must have shape [{batch_size}, {self.state_dim}]")

        ys: List[torch.Tensor] = []
        gates: List[torch.Tensor] = []
        step_diagnostics: List[Dict[str, torch.Tensor]] = []

        for t in range(seq_len):
            x_t = x[:, t, :]
            gate_np = []
            for b, router in enumerate(routers):
                x_np = x_t[b].detach().to(dtype=torch.float32, device="cpu").numpy()
                gate_np.append(router.step(x_np, use_ema=self.use_router_ema, use_gpu=self.use_gpu_router))

            gate_t = torch.as_tensor(np.stack(gate_np, axis=0), dtype=x.dtype, device=x.device)

            if return_diagnostics:
                y_t, h, diag = self.ssm_core.step(x_t, h, gate_t, return_diagnostics=True)
                step_diagnostics.append(diag)
            else:
                y_t, h = self.ssm_core.step(x_t, h, gate_t)

            ys.append(y_t)
            if return_gates:
                gates.append(gate_t)

        y = torch.stack(ys, dim=1)
        result: Any = y
        gate_seq = torch.stack(gates, dim=1) if return_gates else None
        if return_gates:
            result = (result, gate_seq)
        if return_diagnostics:
            router_diags = [router.get_diagnostics() for router in routers]
            diagnostics = self._stack_step_diagnostics(step_diagnostics)
            diagnostics["router"] = router_diags
            result = (*result, diagnostics) if isinstance(result, tuple) else (result, diagnostics)
        return result


class CachedGateLowRankPathSSM(nn.Module):
    """Run the SSM core with precomputed gates, without executing the SNN router."""

    def __init__(self, ssm_core: LowRankPathSSMCore):
        super().__init__()
        self.ssm_core = ssm_core

    def forward(
        self,
        x: torch.Tensor,
        gates: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ):
        if x.ndim != 3 or x.shape[-1] != self.ssm_core.input_dim:
            raise ValueError(f"x must have shape [B, T, {self.ssm_core.input_dim}]")
        if gates.ndim != 3 or gates.shape[:2] != x.shape[:2] or gates.shape[-1] != self.ssm_core.num_paths:
            raise ValueError(f"gates must have shape [B, T, {self.ssm_core.num_paths}]")

        batch_size, seq_len, _ = x.shape
        h = h0 if h0 is not None else self.ssm_core.initial_state(batch_size, x.device, x.dtype)
        ys: List[torch.Tensor] = []
        step_diagnostics: List[Dict[str, torch.Tensor]] = []

        for t in range(seq_len):
            if return_diagnostics:
                y_t, h, diag = self.ssm_core.step(x[:, t, :], h, gates[:, t, :], return_diagnostics=True)
                step_diagnostics.append(diag)
            else:
                y_t, h = self.ssm_core.step(x[:, t, :], h, gates[:, t, :])
            ys.append(y_t)

        y = torch.stack(ys, dim=1)
        if return_diagnostics:
            return y, LowRankPathSSMModel._stack_step_diagnostics(step_diagnostics)
        return y


class LightweightRouter(nn.Module):
    """Small torch router for distilling FullSNNPathRouter cached gates."""

    def __init__(self, input_dim: int, num_paths: int, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_paths = int(num_paths)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.num_paths),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"x last dim must be {self.input_dim}")
        return self.net(x)


def save_ssm_checkpoint(model: nn.Module, path: str):
    if isinstance(model, LowRankPathSSMModel):
        core = model.ssm_core
        payload = {
            "input_dim": model.input_dim,
            "state_dim": model.state_dim,
            "num_paths": model.num_paths,
            "rank": model.rank,
            "output_dim": model.output_dim,
            "use_router_ema": model.use_router_ema,
            "use_gpu_router": model.use_gpu_router,
            "reset_router_state_on_forward": model.reset_router_state_on_forward,
            "ssm_core_state_dict": core.state_dict(),
        }
    elif isinstance(model, CachedGateLowRankPathSSM):
        core = model.ssm_core
        payload = {
            "input_dim": core.input_dim,
            "state_dim": core.state_dim,
            "num_paths": core.num_paths,
            "rank": core.rank,
            "output_dim": core.output_dim,
            "use_router_ema": False,
            "use_gpu_router": False,
            "reset_router_state_on_forward": True,
            "ssm_core_state_dict": core.state_dict(),
        }
    else:
        raise TypeError("model must be LowRankPathSSMModel or CachedGateLowRankPathSSM")

    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path_obj)
    return path_obj


def load_ssm_checkpoint(
    path: str,
    router: Optional[RouterLike] = None,
    router_kwargs: Optional[Dict[str, Any]] = None,
    map_location: str = "cpu",
) -> LowRankPathSSMModel:
    payload = torch.load(path, map_location=map_location)
    model = LowRankPathSSMModel(
        input_dim=int(payload["input_dim"]),
        state_dim=int(payload["state_dim"]),
        num_paths=int(payload["num_paths"]),
        rank=int(payload["rank"]),
        output_dim=int(payload["output_dim"]),
        router=router,
        router_kwargs=router_kwargs,
        use_router_ema=bool(payload.get("use_router_ema", False)),
        use_gpu_router=bool(payload.get("use_gpu_router", False)),
        reset_router_state_on_forward=bool(payload.get("reset_router_state_on_forward", True)),
    )
    model.ssm_core.load_state_dict(payload["ssm_core_state_dict"])
    return model


def _as_numpy_sequence_dataset(x_dataset: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(x_dataset, torch.Tensor):
        x_np = x_dataset.detach().cpu().numpy()
    else:
        x_np = np.asarray(x_dataset)
    if x_np.ndim != 3:
        raise ValueError("x_dataset must have shape [num_samples, seq_len, input_dim]")
    return x_np.astype(np.float32, copy=False)


def build_gate_cache(
    router: FullSNNPathRouter,
    x_dataset: Union[np.ndarray, torch.Tensor],
    use_ema: bool = False,
    use_gpu: bool = False,
    reset_each_sample: bool = True,
) -> np.ndarray:
    x_np = _as_numpy_sequence_dataset(x_dataset)
    num_samples, seq_len, input_dim = x_np.shape
    if input_dim != router.input_dim:
        raise ValueError(f"dataset input_dim {input_dim} != router input_dim {router.input_dim}")

    gates = np.zeros((num_samples, seq_len, router.num_paths), dtype=np.float32)
    for sample_idx in range(num_samples):
        if reset_each_sample:
            router.reset_state()
        for t in range(seq_len):
            gates[sample_idx, t] = router.step(x_np[sample_idx, t], use_ema=use_ema, use_gpu=use_gpu)
    return gates


def gate_cache_metadata(
    gates: np.ndarray,
    input_dim: int,
    router: FullSNNPathRouter,
    use_ema: bool = False,
    use_gpu: bool = False,
    router_preset_path: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "num_samples": int(gates.shape[0]),
        "seq_len": int(gates.shape[1]),
        "input_dim": int(input_dim),
        "num_paths": int(gates.shape[2]),
        "use_ema": bool(use_ema),
        "use_gpu": bool(use_gpu),
        "router_preset_path": router_preset_path,
        "current_scale": router.current_scale,
        "current_clip": router.current_clip,
        "window_ms": router.window_ms,
        "dt": router.dt,
    }


def _metadata_sidecar_path(path_obj: Path) -> Path:
    return path_obj.with_suffix(".metadata.json")


def save_gate_cache(gates: np.ndarray, path: str, metadata: Dict[str, Any]):
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    gates = np.asarray(gates, dtype=np.float32)
    if path_obj.suffix == ".pt":
        torch.save({"gates": torch.from_numpy(gates), "metadata": metadata}, path_obj)
    elif path_obj.suffix == ".npy":
        np.save(path_obj, gates)
        _metadata_sidecar_path(path_obj).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    else:
        raise ValueError("gate cache path must end with .npy or .pt")
    return path_obj


def load_gate_cache(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    path_obj = Path(path)
    if path_obj.suffix == ".pt":
        payload = torch.load(path_obj, map_location="cpu")
        gates = payload["gates"].detach().cpu().numpy().astype(np.float32, copy=False)
        metadata = dict(payload.get("metadata", {}))
    elif path_obj.suffix == ".npy":
        gates = np.load(path_obj).astype(np.float32, copy=False)
        sidecar = _metadata_sidecar_path(path_obj)
        metadata = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
    else:
        raise ValueError("gate cache path must end with .npy or .pt")
    return gates, metadata


def save_distill_dataset(
    x_dataset: Union[np.ndarray, torch.Tensor],
    teacher_gate_cache: Union[np.ndarray, torch.Tensor],
    path: str,
    metadata: Optional[Dict[str, Any]] = None,
):
    x_tensor = torch.as_tensor(_as_numpy_sequence_dataset(x_dataset), dtype=torch.float32)
    gate_tensor = torch.as_tensor(teacher_gate_cache, dtype=torch.float32)
    if gate_tensor.ndim != 3 or gate_tensor.shape[:2] != x_tensor.shape[:2]:
        raise ValueError("teacher_gate_cache must have shape [num_samples, seq_len, num_paths]")
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"x": x_tensor, "gates": gate_tensor, "metadata": metadata or {}}, path_obj)
    return path_obj


def save_lightweight_router(router: LightweightRouter, path: str):
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "input_dim": router.input_dim,
            "num_paths": router.num_paths,
            "hidden_dim": router.hidden_dim,
            "state_dict": router.state_dict(),
        },
        path_obj,
    )
    return path_obj


def load_lightweight_router(path: str, map_location: str = "cpu") -> LightweightRouter:
    payload = torch.load(path, map_location=map_location)
    router = LightweightRouter(
        input_dim=int(payload["input_dim"]),
        num_paths=int(payload["num_paths"]),
        hidden_dim=int(payload["hidden_dim"]),
    )
    router.load_state_dict(payload["state_dict"])
    return router


def build_low_rank_path_ssm_a(
    input_dim: int,
    output_dim: Optional[int] = None,
    num_paths: int = 8,
    rank: int = 4,
    state_dim: int = 128,
    router_kwargs: Optional[Dict[str, Any]] = None,
) -> LowRankPathSSMModel:
    return LowRankPathSSMModel(
        input_dim=input_dim,
        output_dim=output_dim,
        num_paths=num_paths,
        rank=rank,
        state_dim=state_dim,
        router_kwargs=router_kwargs,
    )


__all__ = [
    "FullSNNPathRouter",
    "LowRankPathSSMCore",
    "LowRankPathSSMModel",
    "CachedGateLowRankPathSSM",
    "LightweightRouter",
    "save_ssm_checkpoint",
    "load_ssm_checkpoint",
    "build_gate_cache",
    "gate_cache_metadata",
    "save_gate_cache",
    "load_gate_cache",
    "save_distill_dataset",
    "save_lightweight_router",
    "load_lightweight_router",
    "build_low_rank_path_ssm_a",
]

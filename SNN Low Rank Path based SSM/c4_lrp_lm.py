from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Low_Rank_Path_SSM import (
    CachedGateLowRankPathSSM,
    FullSNNPathRouter,
    LightweightRouter,
    LowRankPathSSMCore,
)


RouterLike = Union[FullSNNPathRouter, Sequence[FullSNNPathRouter]]


def torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def safe_perplexity(loss: float) -> float:
    if not math.isfinite(loss):
        return float("inf")
    return float(math.exp(loss)) if loss < 50.0 else float("inf")


def gate_statistics(gates: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        gate_f = gates.detach().float()
        active = gate_f.sum(dim=-1)
        num_paths = gate_f.shape[-1]
        return {
            "mean_active_paths": float(active.mean().item()) if active.numel() else 0.0,
            "zero_gate_ratio": float((active <= 1e-6).float().mean().item()) if active.numel() else 0.0,
            "all_on_gate_ratio": float((active >= num_paths - 1e-6).float().mean().item()) if active.numel() else 0.0,
        }


def create_gate_feature_encoder(
    vocab_size: int,
    model_dim: int,
    seed: int = 1,
    device: str | torch.device = "cpu",
) -> nn.Embedding:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    encoder = nn.Embedding(int(vocab_size), int(model_dim))
    with torch.no_grad():
        encoder.weight.copy_(torch.randn((int(vocab_size), int(model_dim)), generator=generator))
    encoder.requires_grad_(False)
    return encoder.to(device)


def save_gate_feature_encoder(
    encoder: nn.Embedding,
    path: str | Path,
    tokenizer_name: str,
    seed: int,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tokenizer_name": tokenizer_name,
        "vocab_size": int(encoder.num_embeddings),
        "model_dim": int(encoder.embedding_dim),
        "state_dict": encoder.state_dict(),
        "seed": int(seed),
        "metadata": metadata or {},
    }
    torch.save(payload, path_obj)
    return path_obj


def load_gate_feature_encoder(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> Tuple[nn.Embedding, Dict[str, Any]]:
    payload = torch_load(path, map_location=map_location)
    encoder = nn.Embedding(int(payload["vocab_size"]), int(payload["model_dim"]))
    encoder.load_state_dict(payload["state_dict"])
    encoder.requires_grad_(False)
    encoder.eval()
    return encoder.to(map_location), payload


def _resize_router_matrix(matrix: np.ndarray, rows: int, cols: int) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    row_reps = int(math.ceil(rows / matrix.shape[0]))
    col_reps = int(math.ceil(cols / matrix.shape[1]))
    resized = np.tile(matrix, (row_reps, col_reps))[:rows, :cols].astype(np.float32, copy=True)
    resized *= math.sqrt(float(matrix.shape[1]) / float(cols))
    return resized


def load_full_snn_router_for_dimensions(
    router_preset: Optional[str],
    input_dim: int,
    num_paths: int,
    seed: int = 1,
    fallback_current_scale: float = 0.01,
    fallback_current_clip: float = 1e-4,
    strict_input_dim: bool = True,
    allow_adapt_preset: bool = False,
) -> Tuple[FullSNNPathRouter, Dict[str, Any]]:
    metadata: Dict[str, Any] = {
        "router_preset": router_preset,
        "router_preset_loaded": False,
        "router_adapted": False,
    }
    if router_preset and Path(router_preset).exists():
        base = FullSNNPathRouter.load_preset(router_preset)
        metadata.update(
            {
                "router_preset_loaded": True,
                "preset_input_dim": base.input_dim,
                "preset_num_paths": base.num_paths,
            }
        )
        if base.input_dim == int(input_dim) and base.num_paths == int(num_paths):
            return base, metadata
        if strict_input_dim and not allow_adapt_preset:
            raise ValueError(
                f"Router preset shape mismatch: preset input_dim={base.input_dim}, num_paths={base.num_paths}; "
                f"requested input_dim={input_dim}, num_paths={num_paths}. Recalibrate the router or pass --allow-adapt-preset."
            )

        config = base.get_config()
        config["input_dim"] = int(input_dim)
        config["num_paths"] = int(num_paths)
        config["record_traces"] = False
        config["spike_times_max_len"] = 0
        router = FullSNNPathRouter(**config)
        router.W_basal = _resize_router_matrix(base.W_basal, int(num_paths), int(input_dim))
        router.W_apical = _resize_router_matrix(base.W_apical, int(num_paths), int(input_dim))
        router.W_tuft = _resize_router_matrix(base.W_tuft, int(num_paths), int(input_dim))
        metadata["router_adapted"] = True
        metadata["adapted_from_input_dim"] = int(base.input_dim)
        metadata["adapted_to_input_dim"] = int(input_dim)
        metadata["adapted_from_num_paths"] = int(base.num_paths)
        metadata["adapted_to_num_paths"] = int(num_paths)
        return router, metadata

    router = FullSNNPathRouter(
        num_paths=int(num_paths),
        input_dim=int(input_dim),
        window_ms=1.0,
        dt=0.025,
        seed=int(seed),
        current_scale=float(fallback_current_scale),
        current_clip=float(fallback_current_clip),
        n_basal=1,
        n_apical=1,
        n_tuft=1,
        record_traces=False,
        spike_times_max_len=0,
    )
    metadata["router_preset_missing_fallback"] = True
    return router, metadata


class LearnedPathRouter(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_paths: int,
        hidden_dim: int = 128,
        mode: str = "sigmoid",
        target_active_paths: float = 1.5,
        temperature: float = 1.0,
        topk: int = 2,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_paths = int(num_paths)
        self.hidden_dim = int(hidden_dim)
        self.mode = str(mode)
        self.target_active_paths = float(target_active_paths)
        self.temperature = float(temperature)
        self.topk = int(topk)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.num_paths),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.net(x) / max(self.temperature, 1e-6)
        if self.mode == "topk_st":
            probs = torch.sigmoid(logits)
            k = max(1, min(self.topk, self.num_paths))
            idx = torch.topk(probs, k=k, dim=-1).indices
            hard = torch.zeros_like(probs).scatter(-1, idx, 1.0)
            gates = hard.detach() - probs.detach() + probs
        else:
            probs = torch.sigmoid(logits)
            threshold = min(max(self.target_active_paths / max(1, self.num_paths), 0.05), 0.95)
            hard = (probs > threshold).float()
            if self.mode == "sigmoid":
                gates = probs
            else:
                gates = hard.detach() - probs.detach() + probs
        return gates, logits, probs


def learned_gate_losses(
    probs: torch.Tensor,
    target_active_paths: float,
    num_paths: int,
) -> Dict[str, torch.Tensor]:
    target_rate = float(target_active_paths) / max(1, int(num_paths))
    mean_gate = probs.mean()
    rate_loss = (mean_gate - target_rate) ** 2
    path_rates = probs.mean(dim=(0, 1))
    balance_loss = path_rates.var(unbiased=False)
    entropy = -(probs.clamp(1e-6, 1 - 1e-6) * torch.log(probs.clamp(1e-6, 1 - 1e-6))).mean()
    return {
        "gate_rate_loss": rate_loss,
        "gate_balance_loss": balance_loss,
        "gate_entropy": entropy,
    }


def generate_snn_gates_for_input_ids(
    input_ids: torch.Tensor,
    gate_feature_encoder: nn.Embedding,
    router: FullSNNPathRouter,
    use_ema: bool = False,
    use_gpu: bool = False,
) -> torch.Tensor:
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [B, T]")
    batch_size, seq_len = input_ids.shape
    gates = torch.zeros(batch_size, seq_len, router.num_paths, dtype=torch.float32)
    encoder_device = next(gate_feature_encoder.parameters()).device
    with torch.no_grad():
        features = gate_feature_encoder(input_ids.to(encoder_device)).detach().cpu()
        for b in range(batch_size):
            router.reset_state()
            for t in range(seq_len):
                gates[b, t] = torch.from_numpy(
                    router.step(features[b, t].numpy(), use_ema=use_ema, use_gpu=use_gpu)
                )
    return gates


def load_c4_lightweight_router(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> Tuple[LightweightRouter, Dict[str, Any]]:
    payload = torch_load(path, map_location=map_location)
    input_dim = int(payload.get("input_dim", payload.get("model_dim")))
    router = LightweightRouter(
        input_dim=input_dim,
        num_paths=int(payload["num_paths"]),
        hidden_dim=int(payload["hidden_dim"]),
    )
    router.load_state_dict(payload["state_dict"])
    router.to(map_location)
    router.eval()
    return router, payload


def generate_distilled_gates_for_input_ids(
    input_ids: torch.Tensor,
    gate_feature_encoder: nn.Embedding,
    lightweight_router: nn.Module,
    threshold: float = 0.5,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    with torch.no_grad():
        gate_feature_encoder = gate_feature_encoder.to(device)
        lightweight_router = lightweight_router.to(device)
        features = gate_feature_encoder(input_ids.to(device))
        logits = lightweight_router(features)
        gates = (torch.sigmoid(logits) > float(threshold)).float()
    return gates.detach().cpu()


class C4LRPSSMLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        model_dim: int = 128,
        state_dim: int = 128,
        num_paths: int = 8,
        rank: int = 4,
        dropout: float = 0.0,
        tie_weights: bool = False,
        pad_token_id: Optional[int] = None,
        num_layers: int = 1,
        gamma_init: float = 0.1,
        learnable_gamma: bool = True,
        gamma_min: float = 0.0,
        gamma_max: float = 2.0,
        path_residual_scale: float = 1.0,
        use_path_norm: bool = True,
        path_dropout: float = 0.0,
        force_min_active_paths: int = 0,
        topk_fallback: int = 0,
        gate_mode: str = "cached_snn",
        router_hidden_dim: int = 128,
        router_temperature: float = 1.0,
        router_topk: int = 2,
        target_active_paths: float = 1.5,
        share_gates_across_layers: bool = True,
        lrp_residual: bool = True,
        **unused_kwargs: Any,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.model_dim = int(model_dim)
        self.state_dim = int(state_dim)
        self.num_paths = int(num_paths)
        self.rank = int(rank)
        self.dropout_p = float(dropout)
        self.tie_weights = bool(tie_weights)
        self.pad_token_id = None if pad_token_id is None else int(pad_token_id)
        self.num_layers = int(num_layers)
        self.gamma_init = float(gamma_init)
        self.learnable_gamma = bool(learnable_gamma)
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)
        self.path_residual_scale = float(path_residual_scale)
        self.use_path_norm = bool(use_path_norm)
        self.path_dropout_p = float(path_dropout)
        self.force_min_active_paths = int(force_min_active_paths)
        self.topk_fallback = int(topk_fallback)
        self.gate_mode = str(gate_mode)
        self.router_hidden_dim = int(router_hidden_dim)
        self.router_temperature = float(router_temperature)
        self.router_topk = int(router_topk)
        self.target_active_paths = float(target_active_paths)
        self.share_gates_across_layers = bool(share_gates_across_layers)
        self.lrp_residual = bool(lrp_residual)
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.token_emb = nn.Embedding(self.vocab_size, self.model_dim)
        self.emb_norm = nn.LayerNorm(self.model_dim)
        self.dropout = nn.Dropout(self.dropout_p)
        core_kwargs = {
            "input_dim": self.model_dim,
            "state_dim": self.state_dim,
            "num_paths": self.num_paths,
            "rank": self.rank,
            "output_dim": self.model_dim,
            "gamma_init": self.gamma_init,
            "learnable_gamma": self.learnable_gamma,
            "gamma_min": self.gamma_min,
            "gamma_max": self.gamma_max,
            "path_residual_scale": self.path_residual_scale,
            "use_path_norm": self.use_path_norm,
            "path_dropout": self.path_dropout_p,
            "force_min_active_paths": self.force_min_active_paths,
            "topk_fallback": self.topk_fallback,
        }
        self.ssm_layers = nn.ModuleList([LowRankPathSSMCore(**core_kwargs) for _ in range(self.num_layers)])
        self.ssm_core = self.ssm_layers[0]
        self.cached_ssm = CachedGateLowRankPathSSM(self.ssm_core)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(self.model_dim) for _ in range(self.num_layers)])
        self.layer_dropouts = nn.ModuleList([nn.Dropout(self.dropout_p) for _ in range(self.num_layers)])
        if self.gate_mode in ("learned_sigmoid", "learned_topk_st", "hybrid_cached_plus_learned"):
            router_mode = "topk_st" if self.gate_mode == "learned_topk_st" else "sigmoid"
            self.learned_router = LearnedPathRouter(
                input_dim=self.model_dim,
                num_paths=self.num_paths,
                hidden_dim=self.router_hidden_dim,
                mode=router_mode,
                target_active_paths=self.target_active_paths,
                temperature=self.router_temperature,
                topk=self.router_topk,
            )
        else:
            self.learned_router = None
        self.lm_head = nn.Linear(self.model_dim, self.vocab_size, bias=False)
        if self.tie_weights:
            self.lm_head.weight = self.token_emb.weight

    def get_config(self) -> Dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "model_dim": self.model_dim,
            "state_dim": self.state_dim,
            "num_paths": self.num_paths,
            "rank": self.rank,
            "dropout": self.dropout_p,
            "tie_weights": self.tie_weights,
            "pad_token_id": self.pad_token_id,
            "num_layers": self.num_layers,
            "gamma_init": self.gamma_init,
            "learnable_gamma": self.learnable_gamma,
            "gamma_min": self.gamma_min,
            "gamma_max": self.gamma_max,
            "path_residual_scale": self.path_residual_scale,
            "use_path_norm": self.use_path_norm,
            "path_dropout": self.path_dropout_p,
            "force_min_active_paths": self.force_min_active_paths,
            "topk_fallback": self.topk_fallback,
            "gate_mode": self.gate_mode,
            "router_hidden_dim": self.router_hidden_dim,
            "router_temperature": self.router_temperature,
            "router_topk": self.router_topk,
            "target_active_paths": self.target_active_paths,
            "share_gates_across_layers": self.share_gates_across_layers,
            "lrp_residual": self.lrp_residual,
        }

    @staticmethod
    def _routers_for_batch(router: RouterLike, batch_size: int) -> List[FullSNNPathRouter]:
        if isinstance(router, FullSNNPathRouter):
            if batch_size != 1:
                raise ValueError("A single FullSNNPathRouter supports batch_size=1 only.")
            return [router]
        routers = list(router)
        if len(routers) != batch_size:
            raise ValueError(f"Expected {batch_size} routers, got {len(routers)}")
        return routers

    def _run_router(
        self,
        gate_features: torch.Tensor,
        router: RouterLike,
        use_router_ema: bool,
        use_gpu_router: bool,
    ) -> torch.Tensor:
        batch_size, seq_len, feature_dim = gate_features.shape
        routers = self._routers_for_batch(router, batch_size)
        gates = torch.zeros(batch_size, seq_len, self.num_paths, dtype=gate_features.dtype, device=gate_features.device)

        with torch.no_grad():
            for b, batch_router in enumerate(routers):
                if batch_router.input_dim != feature_dim:
                    raise ValueError(
                        f"router input_dim {batch_router.input_dim} != gate feature dim {feature_dim}"
                    )
                if batch_router.num_paths != self.num_paths:
                    raise ValueError(f"router num_paths {batch_router.num_paths} != model num_paths {self.num_paths}")
                batch_router.reset_state()
                for t in range(seq_len):
                    feature_np = gate_features[b, t].detach().to(dtype=torch.float32, device="cpu").numpy()
                    gate_np = batch_router.step(feature_np, use_ema=use_router_ema, use_gpu=use_gpu_router)
                    gates[b, t] = torch.as_tensor(gate_np, dtype=gate_features.dtype, device=gate_features.device)
        return gates.detach()

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        gates: Optional[torch.Tensor] = None,
        router: Optional[RouterLike] = None,
        gate_features: Optional[torch.Tensor] = None,
        use_router_ema: bool = False,
        use_gpu_router: bool = False,
        h0: Optional[torch.Tensor] = None,
        return_gates: bool = False,
        return_diagnostics: bool = False,
    ):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B, T]")
        batch_size, seq_len = input_ids.shape

        x = self.token_emb(input_ids)
        x = self.emb_norm(x)
        x = self.dropout(x)

        learned_gate = None
        learned_logits = None
        learned_probs = None
        cached_gates = None
        if gates is None and self.gate_mode in ("cached_snn", "hybrid_cached_plus_learned"):
            if router is None:
                if self.gate_mode == "cached_snn":
                    raise ValueError("router is required when gates is None")
            if gate_features is None:
                # Debug fallback only: cached C4 gates are generated from a fixed
                # feature encoder, while token_emb changes during LM training.
                gate_features = x.detach()
            if router is not None:
                if gate_features.shape[:2] != input_ids.shape:
                    raise ValueError("gate_features must have shape [B, T, D]")
                gates = self._run_router(gate_features.detach(), router, use_router_ema, use_gpu_router)

        if gates is not None:
            gates = gates.detach().to(device=x.device, dtype=x.dtype)
            gates.requires_grad_(False)
            cached_gates = gates

        if self.learned_router is not None:
            learned_gate, learned_logits, learned_probs = self.learned_router(x)
            if self.gate_mode == "hybrid_cached_plus_learned":
                if cached_gates is None:
                    gates = learned_gate
                else:
                    gates = torch.maximum(cached_gates, learned_gate)
            else:
                gates = learned_gate

        if gates is None:
            raise ValueError(f"gate_mode={self.gate_mode} requires gates, router, or learned_router")

        gates = gates.to(device=x.device, dtype=x.dtype)
        if gates.ndim == 3:
            if gates.shape != (batch_size, seq_len, self.num_paths):
                raise ValueError(f"gates must have shape [{batch_size}, {seq_len}, {self.num_paths}]")
            gate_seq = gates
        elif gates.ndim == 4:
            if gates.shape != (batch_size, seq_len, self.num_layers, self.num_paths):
                raise ValueError(f"layer gates must have shape [{batch_size}, {seq_len}, {self.num_layers}, {self.num_paths}]")
            gate_seq = gates
        else:
            raise ValueError("gates must have shape [B,T,P] or [B,T,L,P]")

        if h0 is not None:
            if isinstance(h0, (list, tuple)):
                states = list(h0)
            else:
                states = [h0] + [
                    layer.initial_state(batch_size, x.device, x.dtype)
                    for layer in self.ssm_layers[1:]
                ]
        else:
            states = [layer.initial_state(batch_size, x.device, x.dtype) for layer in self.ssm_layers]
        outputs: List[torch.Tensor] = []
        h_norms: List[torch.Tensor] = []
        raw_delta_norms: List[torch.Tensor] = []
        scaled_delta_norms: List[torch.Tensor] = []
        base_update_norms: List[torch.Tensor] = []
        path_to_base_ratios: List[torch.Tensor] = []
        gammas: List[torch.Tensor] = []
        active_paths: List[torch.Tensor] = []

        for t in range(seq_len):
            z = x[:, t]
            for layer_idx, layer in enumerate(self.ssm_layers):
                gate_t = gate_seq[:, t, layer_idx] if gate_seq.ndim == 4 else gate_seq[:, t]
                if return_diagnostics:
                    y_l, states[layer_idx], diag = layer.step(z, states[layer_idx], gate_t, return_diagnostics=True)
                    h_norms.append(diag["h_norm"])
                    raw_delta_norms.append(diag["raw_delta_norm"])
                    scaled_delta_norms.append(diag["scaled_delta_norm"])
                    base_update_norms.append(diag["base_update_norm"])
                    path_to_base_ratios.append(diag["path_to_base_ratio"])
                    gammas.append(diag["gamma"])
                    active_paths.append(diag["active_paths"])
                else:
                    y_l, states[layer_idx] = layer.step(z, states[layer_idx], gate_t)
                if self.lrp_residual:
                    z = self.layer_norms[layer_idx](z + self.layer_dropouts[layer_idx](y_l))
                else:
                    z = y_l
            outputs.append(z)

        hidden = torch.stack(outputs, dim=1)
        logits = self.lm_head(hidden)
        if labels is None:
            return logits

        ignore_index = self.pad_token_id if self.pad_token_id is not None else -100
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            labels.reshape(-1),
            ignore_index=ignore_index,
        )
        result: Dict[str, Any] = {"loss": loss, "logits": logits}
        if return_gates:
            result["gates"] = gates.detach()
        if return_diagnostics:
            stats_gate = gates if gates.ndim == 3 else gates.mean(dim=2)
            stats = gate_statistics(stats_gate)
            if self.pad_token_id is None:
                num_tokens = int(labels.numel())
            else:
                num_tokens = int((labels != self.pad_token_id).sum().item())
            diagnostics = {
                **stats,
                "h_norm_mean": float(torch.cat(h_norms).mean().item()) if h_norms else 0.0,
                "delta_norm_mean": float(torch.cat(scaled_delta_norms).mean().item()) if scaled_delta_norms else 0.0,
                "raw_delta_norm_mean": float(torch.cat(raw_delta_norms).mean().item()) if raw_delta_norms else 0.0,
                "scaled_delta_norm_mean": float(torch.cat(scaled_delta_norms).mean().item()) if scaled_delta_norms else 0.0,
                "base_update_norm_mean": float(torch.cat(base_update_norms).mean().item()) if base_update_norms else 0.0,
                "path_to_base_ratio_mean": float(torch.cat(path_to_base_ratios).mean().item()) if path_to_base_ratios else 0.0,
                "gamma": float(torch.cat(gammas).mean().item()) if gammas else 0.0,
                "active_paths_diagnostic_mean": float(torch.cat(active_paths).mean().item()) if active_paths else 0.0,
                "batch_size": int(batch_size),
                "seq_len": int(seq_len),
                "num_tokens": num_tokens,
            }
            if learned_probs is not None:
                losses = learned_gate_losses(learned_probs, self.target_active_paths, self.num_paths)
                diagnostics["learned_mean_active_paths"] = float(learned_gate.detach().sum(dim=-1).mean().item())
                diagnostics["learned_zero_gate_ratio"] = float((learned_gate.detach().sum(dim=-1) <= 1e-6).float().mean().item())
                diagnostics["gate_rate_loss"] = float(losses["gate_rate_loss"].detach().cpu())
                diagnostics["gate_balance_loss"] = float(losses["gate_balance_loss"].detach().cpu())
                result["gate_loss"] = losses["gate_rate_loss"] + losses["gate_balance_loss"]
                result["gate_losses"] = losses
            if cached_gates is not None and learned_gate is not None:
                diagnostics.update({f"cached_{k}": v for k, v in gate_statistics(cached_gates).items()})
            result["diagnostics"] = diagnostics
            result["hidden"] = hidden.detach()
        return result


def save_c4_lm_checkpoint(
    model: C4LRPSSMLanguageModel,
    path: str | Path,
    tokenizer_name: str,
    model_config: Dict[str, Any],
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    step: int = 0,
    metrics: Optional[Dict[str, Any]] = None,
) -> Path:
    model_to_save = getattr(model, "_orig_mod", model)
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "model_config": dict(model_config),
        "tokenizer_name": tokenizer_name,
        "step": int(step),
        "metrics": metrics or {},
        "token_emb_state_dict": model_to_save.token_emb.state_dict(),
        "emb_norm_state_dict": model_to_save.emb_norm.state_dict(),
        "ssm_layers_state_dict": model_to_save.ssm_layers.state_dict(),
        "ssm_core_state_dict": model_to_save.ssm_core.state_dict(),
        "layer_norms_state_dict": model_to_save.layer_norms.state_dict(),
        "lm_head_state_dict": model_to_save.lm_head.state_dict(),
    }
    if getattr(model_to_save, "learned_router", None) is not None:
        payload["learned_router_state_dict"] = model_to_save.learned_router.state_dict()
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, path_obj)
    return path_obj


def load_c4_lm_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
    load_optimizer: bool = False,
) -> Dict[str, Any]:
    payload = torch_load(path, map_location=map_location)
    model_config = dict(payload["model_config"])
    model = C4LRPSSMLanguageModel(**model_config)
    model.token_emb.load_state_dict(payload["token_emb_state_dict"])
    if "emb_norm_state_dict" in payload:
        model.emb_norm.load_state_dict(payload["emb_norm_state_dict"])
    if "ssm_layers_state_dict" in payload:
        model.ssm_layers.load_state_dict(payload["ssm_layers_state_dict"], strict=False)
    elif "ssm_core_state_dict" in payload:
        model.ssm_core.load_state_dict(payload["ssm_core_state_dict"], strict=False)
    if "layer_norms_state_dict" in payload:
        model.layer_norms.load_state_dict(payload["layer_norms_state_dict"], strict=False)
    if getattr(model, "learned_router", None) is not None and "learned_router_state_dict" in payload:
        model.learned_router.load_state_dict(payload["learned_router_state_dict"], strict=False)
    model.lm_head.load_state_dict(payload["lm_head_state_dict"])
    result: Dict[str, Any] = {
        "model": model.to(map_location),
        "model_config": model_config,
        "tokenizer_name": payload.get("tokenizer_name"),
        "step": int(payload.get("step", 0)),
        "metrics": payload.get("metrics", {}),
        "payload": payload,
    }
    if load_optimizer:
        result["optimizer_state_dict"] = payload.get("optimizer_state_dict")
        result["scheduler_state_dict"] = payload.get("scheduler_state_dict")
    return result


__all__ = [
    "C4LRPSSMLanguageModel",
    "LearnedPathRouter",
    "create_gate_feature_encoder",
    "save_gate_feature_encoder",
    "load_gate_feature_encoder",
    "load_full_snn_router_for_dimensions",
    "generate_snn_gates_for_input_ids",
    "load_c4_lightweight_router",
    "generate_distilled_gates_for_input_ids",
    "save_c4_lm_checkpoint",
    "load_c4_lm_checkpoint",
    "safe_perplexity",
    "gate_statistics",
    "torch_load",
]

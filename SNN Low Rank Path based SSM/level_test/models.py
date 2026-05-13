from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


LEVEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LEVEL_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def lm_loss(logits: torch.Tensor, labels: torch.Tensor, pad_token_id: Optional[int] = None) -> torch.Tensor:
    ignore_index = int(pad_token_id) if pad_token_id is not None else -100
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=ignore_index)


def perplexity(loss: float) -> float:
    return float(math.exp(loss)) if math.isfinite(loss) and loss < 50.0 else float("inf")


def count_parameters(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


def gate_diagnostics(gates: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        gates_f = gates.detach().float()
        active = gates_f.sum(dim=-1)
        num_paths = gates_f.shape[-1]
        return {
            "mean_active_paths": float(active.mean().item()) if active.numel() else 0.0,
            "zero_gate_ratio": float((active <= 1e-6).float().mean().item()) if active.numel() else 0.0,
            "all_on_gate_ratio": float((active >= num_paths - 1e-6).float().mean().item()) if active.numel() else 0.0,
        }


class LRPSSMWrapper(nn.Module):
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
        max_seq_len: int = 128,
        **extra: Any,
    ):
        super().__init__()
        try:
            from c4_lrp_lm import C4LRPSSMLanguageModel

            self.model = C4LRPSSMLanguageModel(
                vocab_size=vocab_size,
                model_dim=model_dim,
                state_dim=state_dim,
                num_paths=num_paths,
                rank=rank,
                dropout=dropout,
                tie_weights=tie_weights,
                pad_token_id=pad_token_id,
                **extra,
            )
            self.uses_root_c4_model = True
        except Exception:
            from Low_Rank_Path_SSM import LowRankPathSSMCore

            class FallbackLRP(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.token_emb = nn.Embedding(vocab_size, model_dim)
                    self.norm = nn.LayerNorm(model_dim)
                    self.dropout = nn.Dropout(dropout)
                    self.ssm_core = LowRankPathSSMCore(
                        input_dim=model_dim,
                        state_dim=state_dim,
                        num_paths=num_paths,
                        rank=rank,
                        output_dim=model_dim,
                    )
                    self.lm_head = nn.Linear(model_dim, vocab_size, bias=False)

                def forward(self, input_ids, labels=None, gates=None):
                    if gates is None:
                        raise ValueError("LRPSSMWrapper requires cached gates")
                    x = self.dropout(self.norm(self.token_emb(input_ids)))
                    h = self.ssm_core.initial_state(input_ids.shape[0], x.device, x.dtype)
                    ys = []
                    for t in range(input_ids.shape[1]):
                        y_t, h = self.ssm_core.step(x[:, t], h, gates[:, t].detach().to(x.device, x.dtype))
                        ys.append(y_t)
                    logits = self.lm_head(torch.stack(ys, dim=1))
                    result = {"logits": logits, "diagnostics": gate_diagnostics(gates)}
                    if labels is not None:
                        result["loss"] = lm_loss(logits, labels, pad_token_id)
                    return result

            self.model = FallbackLRP()
            self.uses_root_c4_model = False
        self.num_paths = int(num_paths)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None, gates: Optional[torch.Tensor] = None):
        gate_mode = getattr(self.model, "gate_mode", "cached_snn")
        if gates is None and gate_mode in ("cached_snn", "hybrid_cached_plus_learned"):
            raise ValueError("lrp_ssm requires cached gates")
        if gates is not None:
            gates = gates.detach()
            gates.requires_grad_(False)
        if self.uses_root_c4_model:
            if labels is None:
                logits = self.model(input_ids, gates=gates)
                return {"logits": logits, "diagnostics": gate_diagnostics(gates)}
            output = self.model(
                input_ids,
                labels=labels,
                gates=gates,
                return_gates=False,
                return_diagnostics=True,
            )
            if "diagnostics" not in output:
                output["diagnostics"] = gate_diagnostics(gates)
            return output
        return self.model(input_ids, labels=labels, gates=gates)


class CausalSelfAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.0, local_window: Optional[int] = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout, batch_first=True)
        self.local_window = local_window

    def mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        idx = torch.arange(seq_len, device=device)
        query = idx[:, None]
        key = idx[None, :]
        mask = key > query
        if self.local_window is not None:
            mask = mask | (key < query - int(self.local_window) + 1)
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_mask = self.mask(x.shape[1], x.device)
        y, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        return y


class TransformerBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        local_window: Optional[int] = None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(model_dim)
        self.attn = CausalSelfAttention(model_dim, num_heads, dropout=dropout, local_window=local_window)
        self.norm2 = nn.LayerNorm(model_dim)
        hidden = int(model_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, model_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        model_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_seq_len: int = 128,
        pad_token_id: Optional[int] = None,
        **_: Any,
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.token_emb = nn.Embedding(vocab_size, model_dim)
        self.pos_emb = nn.Embedding(max_seq_len, model_dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(model_dim, num_heads, mlp_ratio, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(model_dim)
        self.lm_head = nn.Linear(model_dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None, **_: Any):
        batch_size, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))
        result = {"logits": logits, "diagnostics": {}}
        if labels is not None:
            result["loss"] = lm_loss(logits, labels, self.pad_token_id)
        return result


class LocalAttentionLM(TransformerLM):
    def __init__(
        self,
        vocab_size: int,
        model_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_seq_len: int = 128,
        window_size: int = 64,
        pad_token_id: Optional[int] = None,
        **_: Any,
    ):
        nn.Module.__init__(self)
        self.pad_token_id = pad_token_id
        self.token_emb = nn.Embedding(vocab_size, model_dim)
        self.pos_emb = nn.Embedding(max_seq_len, model_dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(model_dim, num_heads, mlp_ratio, dropout, local_window=window_size) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(model_dim)
        self.lm_head = nn.Linear(model_dim, vocab_size, bias=False)
        self.window_size = int(window_size)


class LinearAttentionBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0, eps: float = 1e-6):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_dim = int(model_dim // num_heads)
        self.eps = float(eps)
        self.norm1 = nn.LayerNorm(model_dim)
        self.qkv = nn.Linear(model_dim, 3 * model_dim, bias=False)
        self.out_proj = nn.Linear(model_dim, model_dim)
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(model_dim)
        hidden = int(model_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, model_dim),
            nn.Dropout(dropout),
        )

    @staticmethod
    def phi(x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def linear_attention(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, model_dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.phi(q)
        k = self.phi(k)
        kv = k.unsqueeze(-1) * v.unsqueeze(-2)
        kv_prefix = kv.cumsum(dim=2)
        k_prefix = k.cumsum(dim=2)
        numerator = torch.einsum("bhtd,bhtde->bhte", q, kv_prefix)
        denominator = (q * k_prefix).sum(dim=-1, keepdim=True).clamp_min(self.eps)
        y = numerator / denominator
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, model_dim)
        return self.out_proj(y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.linear_attention(self.norm1(x)))
        x = x + self.mlp(self.norm2(x))
        return x


class LinearAttentionLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        model_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_seq_len: int = 128,
        pad_token_id: Optional[int] = None,
        **_: Any,
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.token_emb = nn.Embedding(vocab_size, model_dim)
        self.pos_emb = nn.Embedding(max_seq_len, model_dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [LinearAttentionBlock(model_dim, num_heads, mlp_ratio, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(model_dim)
        self.lm_head = nn.Linear(model_dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None, **_: Any):
        batch_size, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device)
        x = self.drop(self.token_emb(input_ids) + self.pos_emb(pos)[None, :, :])
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))
        result = {"logits": logits, "diagnostics": {}}
        if labels is not None:
            result["loss"] = lm_loss(logits, labels, self.pad_token_id)
        return result


class GRULM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        model_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
        max_seq_len: int = 128,
        pad_token_id: Optional[int] = None,
        **_: Any,
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.token_emb = nn.Embedding(vocab_size, model_dim)
        self.gru = nn.GRU(
            input_size=model_dim,
            hidden_size=model_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.lm_head = nn.Linear(model_dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None, **_: Any):
        x = self.token_emb(input_ids)
        y, _ = self.gru(x)
        logits = self.lm_head(y)
        result = {"logits": logits, "diagnostics": {}}
        if labels is not None:
            result["loss"] = lm_loss(logits, labels, self.pad_token_id)
        return result


def model_factory(name: str, config: Dict[str, Any], vocab_size: int) -> nn.Module:
    cfg = dict(config)
    cfg["vocab_size"] = int(vocab_size)
    if name in ("lrp_ssm", "lrp_ssm_fixed_calibrated", "lrp_ssm_learned_router", "lrp_ssm_hybrid"):
        return LRPSSMWrapper(**cfg)
    if name == "transformer":
        return TransformerLM(**cfg)
    if name == "linear_attention":
        return LinearAttentionLM(**cfg)
    if name == "local_attention":
        return LocalAttentionLM(**cfg)
    if name == "gru":
        return GRULM(**cfg)
    raise ValueError(f"Unknown model: {name}")

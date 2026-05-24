from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch


LEVEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LEVEL_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import load_gate_cache, load_token_cache, make_dataloader, resolve_project_path, split_cache
from models import count_parameters, gate_diagnostics, model_factory, perplexity


LRP_MODELS = {
    "lrp_ssm",
    "lrp_ssm_fixed_calibrated",
    "lrp_ssm_learned_router",
    "lrp_ssm_hybrid",
    "lrp_ssm_strong_path_bias_decay",
}
GATE_RATE_WEIGHT = 0.01
GATE_BALANCE_WEIGHT = 0.001
GATE_ENTROPY_WEIGHT = 0.001
GATE_COMMITMENT_WEIGHT = 0.001
GATE_TEACHER_WEIGHT = 0.0


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def level_path(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        resolved = path_obj.resolve()
    elif path_obj.parts and path_obj.parts[0] in {"experiments", "level_test"}:
        resolved = (PROJECT_ROOT / path_obj).resolve()
    else:
        resolved = (LEVEL_DIR / path_obj).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"Refusing to write outside project root: {path}") from exc
    return resolved


def autocast_context(device: torch.device, enabled: bool):
    active = bool(enabled and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=active)
    return torch.cuda.amp.autocast(enabled=active)


def make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def cuda_sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))


def parse_args():
    parser = argparse.ArgumentParser(description="Train one experiment model.")
    parser.add_argument(
        "--model",
        choices=[
            "lrp_ssm",
            "lrp_ssm_fixed_calibrated",
            "lrp_ssm_learned_router",
            "lrp_ssm_hybrid",
            "lrp_ssm_strong_path_bias_decay",
            "transformer",
            "linear_attention",
            "local_attention",
            "gru",
        ],
        required=True,
    )
    parser.add_argument("--token-cache", type=str, default="../c4_token_cache.pt")
    parser.add_argument("--gate-cache", type=str, default="../c4_gate_cache.pt")
    parser.add_argument("--output-dir", type=str, default="experiments/runs")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-train-steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--max-eval-batches", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--gate-conditioned-decay", action="store_true")
    parser.add_argument("--gate-decay-scale", type=float, default=1.0)
    parser.add_argument("--gate-conditioned-gamma", action="store_true")
    parser.add_argument("--gate-gamma-scale", type=float, default=1.0)
    parser.add_argument("--gate-gamma-min", type=float, default=0.1)
    parser.add_argument("--gate-gamma-max", type=float, default=4.0)
    parser.add_argument("--gate-floor", type=float, default=0.0)
    parser.add_argument("--gate-temperature", type=float, default=1.0)
    parser.add_argument("--gate-dropout", type=float, default=0.0)
    parser.add_argument("--use-path-bias", action="store_true")
    return parser.parse_args()


def load_json_config(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return {}
    path_obj = resolve_project_path(path)
    return json.loads(path_obj.read_text(encoding="utf-8"))


def base_config(model_name: str, token_metadata: dict, args) -> Dict[str, Any]:
    block_size = int(args.block_size or token_metadata.get("block_size", 128))
    vocab_size = int(token_metadata.get("vocab_size", 50257))
    pad_token_id = token_metadata.get("pad_token_id")
    cfg: Dict[str, Any] = {
        "vocab_size": vocab_size,
        "model_dim": 128,
        "num_layers": 4,
        "num_heads": 4,
        "mlp_ratio": 4.0,
        "dropout": 0.0,
        "max_seq_len": block_size,
        "pad_token_id": pad_token_id,
    }
    if model_name in LRP_MODELS:
        cfg.update({"state_dim": 128, "num_paths": 8, "rank": 4, "tie_weights": False})
        cfg.update({"num_layers": cfg.get("num_layers", 4), "gamma_init": 0.1})
        cfg.update(
            {
                "gate_conditioned_decay": bool(args.gate_conditioned_decay),
                "gate_decay_scale": float(args.gate_decay_scale),
                "gate_conditioned_gamma": bool(args.gate_conditioned_gamma),
                "gate_gamma_scale": float(args.gate_gamma_scale),
                "gate_gamma_min": float(args.gate_gamma_min),
                "gate_gamma_max": float(args.gate_gamma_max),
                "gate_floor": float(args.gate_floor),
                "gate_temperature": float(args.gate_temperature),
                "gate_dropout": float(args.gate_dropout),
                "use_path_bias": bool(args.use_path_bias),
            }
        )
    if model_name == "lrp_ssm_fixed_calibrated":
        cfg.update({"gate_mode": "cached_snn", "force_min_active_paths": 1, "topk_fallback": 1})
    if model_name == "lrp_ssm_learned_router":
        cfg.update({"gate_mode": "learned_topk_st", "target_active_paths": 1.5, "router_topk": 2})
    if model_name == "lrp_ssm_hybrid":
        cfg.update({"gate_mode": "hybrid_cached_plus_learned", "target_active_paths": 1.5, "router_topk": 2, "force_min_active_paths": 1})
    if model_name == "lrp_ssm_strong_path_bias_decay":
        cfg.update(
            {
                "gate_mode": "cached_snn",
                "gate_conditioned_decay": True,
                "use_path_bias": True,
            }
        )
    if model_name == "local_attention":
        cfg["window_size"] = min(64, block_size)
    if model_name == "gru":
        cfg.update({"num_layers": 2})
    return cfg


def crop_cache(input_ids: torch.Tensor, labels: torch.Tensor, gates: Optional[torch.Tensor], block_size: Optional[int]):
    if block_size is None:
        return input_ids, labels, gates
    if block_size > input_ids.shape[1]:
        raise ValueError(f"--block-size {block_size} exceeds cache block size {input_ids.shape[1]}")
    input_ids = input_ids[:, :block_size].contiguous()
    labels = labels[:, :block_size].contiguous()
    gates = None if gates is None else gates[:, :block_size].contiguous()
    return input_ids, labels, gates


def endless(loader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def evaluate_model(model, loader, model_name: str, device: torch.device, amp: bool, max_batches: int) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    gate_stats = []
    start = time.perf_counter()
    for idx, batch in enumerate(loader):
        if idx >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        kwargs = {}
        if model_name in LRP_MODELS and "learned_router" not in model_name:
            kwargs["gates"] = batch["gates"].to(device).float().detach()
        cuda_sync(device)
        t0 = time.perf_counter()
        with autocast_context(device, amp):
            output = model(input_ids, labels=labels, **kwargs)
        cuda_sync(device)
        tokens = int(labels.numel())
        total_loss += float(output["loss"].detach().cpu()) * tokens
        total_tokens += tokens
        if output.get("diagnostics"):
            gate_stats.append(output["diagnostics"])
    elapsed = max(time.perf_counter() - start, 1e-9)
    loss = total_loss / max(1, total_tokens)
    metrics = {
        "eval_loss": loss,
        "eval_ppl": perplexity(loss),
        "eval_tokens_per_sec": total_tokens / elapsed,
    }
    if gate_stats:
        for key in ("mean_active_paths", "zero_gate_ratio", "all_on_gate_ratio"):
            metrics[key] = float(sum(float(item.get(key, 0.0)) for item in gate_stats) / len(gate_stats))
    model.train()
    return metrics


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_checkpoint(path: Path, model, optimizer, step: int, config: dict, summary: dict):
    torch.save(
        {
            "model_state_dict": getattr(model, "_orig_mod", model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "step": int(step),
            "config": config,
            "summary": summary,
        },
        path,
    )


def failed_summary(run_dir: Path, args, status: str, error: str) -> dict:
    summary = {
        "model": args.model,
        "status": status,
        "error": error,
        "parameter_count": None,
        "best_eval_loss": None,
        "best_eval_ppl": None,
        "final_train_loss": None,
        "final_train_ppl": None,
        "avg_train_tokens_per_sec": 0.0,
        "avg_eval_tokens_per_sec": 0.0,
        "peak_gpu_memory_mb": 0.0,
        "best_checkpoint": None,
        "latest_checkpoint": None,
        "run_dir": str(run_dir),
        "report_generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(run_dir / "summary.json", summary)
    return summary


def train_model(args) -> dict:
    torch.manual_seed(int(args.seed))
    output_root = level_path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{args.model}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    try:
        input_ids, labels, token_metadata = load_token_cache(args.token_cache)
        gates = None
        gate_metadata = {}
        if args.model in LRP_MODELS and args.model != "lrp_ssm_learned_router":
            gates, gate_metadata = load_gate_cache(args.gate_cache)
        input_ids, labels, gates = crop_cache(input_ids, labels, gates, args.block_size)
        train_split, eval_split = split_cache(input_ids, labels, gates, val_ratio=args.val_ratio, seed=args.seed)
        train_loader = make_dataloader(*train_split, batch_size=args.batch_size, shuffle=True)
        eval_loader = make_dataloader(*eval_split, batch_size=args.batch_size, shuffle=False)

        config = base_config(args.model, token_metadata, args)
        config.update(load_json_config(args.config))
        config["vocab_size"] = int(token_metadata.get("vocab_size", config.get("vocab_size", 50257)))
        config["max_seq_len"] = max(int(config.get("max_seq_len", input_ids.shape[1])), int(input_ids.shape[1]))
        config.setdefault("pad_token_id", token_metadata.get("pad_token_id"))

        model = model_factory(args.model, config, vocab_size=int(config["vocab_size"])).to(device)
        if args.compile and hasattr(torch, "compile"):
            model = torch.compile(model)
        param_count = count_parameters(getattr(model, "_orig_mod", model))
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scaler = make_grad_scaler(args.amp and device.type == "cuda")
        train_iter = endless(train_loader)
        metrics_path = run_dir / "metrics.jsonl"
        write_json(
            run_dir / "config.json",
            {
                "model": args.model,
                "config": config,
                "train_args": vars(args),
                "token_metadata": token_metadata,
                "gate_metadata": gate_metadata,
            },
        )

        best_eval_loss = float("inf")
        best_eval_ppl = float("inf")
        train_tps_values = []
        eval_tps_values = []
        final_train_loss = None
        total_tokens_seen = 0
        convergence_step_to_threshold = None

        for step in range(1, int(args.max_train_steps) + 1):
            model.train()
            batch = next(train_iter)
            input_ids_b = batch["input_ids"].to(device)
            labels_b = batch["labels"].to(device)
            kwargs = {}
            if args.model in LRP_MODELS and "gates" in batch:
                gates_b = batch["gates"].to(device).float().detach()
                gates_b.requires_grad_(False)
                kwargs["gates"] = gates_b
            optimizer.zero_grad(set_to_none=True)
            cuda_sync(device)
            start = time.perf_counter()
            with autocast_context(device, args.amp):
                output = model(input_ids_b, labels=labels_b, **kwargs)
                loss = output["loss"]
                if "gate_losses" in output:
                    gate_losses = output["gate_losses"]
                    loss = loss + GATE_RATE_WEIGHT * gate_losses["gate_rate_loss"]
                    loss = loss + GATE_BALANCE_WEIGHT * gate_losses["gate_balance_loss"]
                    loss = loss + GATE_ENTROPY_WEIGHT * gate_losses["gate_entropy"]
                    loss = loss + GATE_COMMITMENT_WEIGHT * gate_losses["gate_commitment_loss"]
                    if "gate_teacher_bce_loss" in gate_losses:
                        loss = loss + GATE_TEACHER_WEIGHT * gate_losses["gate_teacher_bce_loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step}: {float(loss.detach().cpu())}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            cuda_sync(device)
            elapsed = max(time.perf_counter() - start, 1e-9)
            tokens = int(labels_b.numel())
            total_tokens_seen += tokens
            train_tps = tokens / elapsed
            train_tps_values.append(train_tps)
            final_train_loss = float(loss.detach().cpu())
            metrics = {
                "step": step,
                "train_loss": final_train_loss,
                "train_ppl": perplexity(final_train_loss),
                "train_tokens_per_sec": train_tps,
                "tokens_seen": total_tokens_seen,
                "grad_norm": float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm),
                "peak_gpu_memory_mb": peak_memory_mb(device),
            }
            if output.get("diagnostics"):
                metrics.update(output["diagnostics"])
            if args.eval_every > 0 and step % int(args.eval_every) == 0:
                eval_metrics = evaluate_model(model, eval_loader, args.model, device, args.amp, args.max_eval_batches)
                metrics.update(eval_metrics)
                eval_tps_values.append(eval_metrics["eval_tokens_per_sec"])
                if eval_metrics["eval_loss"] < best_eval_loss:
                    best_eval_loss = float(eval_metrics["eval_loss"])
                    best_eval_ppl = float(eval_metrics["eval_ppl"])
                    if convergence_step_to_threshold is None and best_eval_ppl < 1000.0:
                        convergence_step_to_threshold = step
                    save_checkpoint(run_dir / "best.pt", model, optimizer, step, config, metrics)
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics) + "\n")

        latest_metrics = {
            "step": int(args.max_train_steps),
            "final_train_loss": final_train_loss,
            "final_train_ppl": perplexity(final_train_loss or float("inf")),
        }
        save_checkpoint(run_dir / "latest.pt", model, optimizer, int(args.max_train_steps), config, latest_metrics)
        if not (run_dir / "best.pt").exists():
            save_checkpoint(run_dir / "best.pt", model, optimizer, int(args.max_train_steps), config, latest_metrics)

        summary = {
            "model": args.model,
            "status": "pass",
            "config": config,
            "parameter_count": param_count,
            "best_eval_loss": best_eval_loss if math.isfinite(best_eval_loss) else None,
            "best_eval_ppl": best_eval_ppl if math.isfinite(best_eval_ppl) else None,
            "final_train_loss": final_train_loss,
            "final_train_ppl": perplexity(final_train_loss or float("inf")),
            "max_train_steps": int(args.max_train_steps),
            "total_tokens_seen": int(total_tokens_seen),
            "avg_train_tokens_per_sec": float(sum(train_tps_values) / max(1, len(train_tps_values))),
            "avg_eval_tokens_per_sec": float(sum(eval_tps_values) / max(1, len(eval_tps_values))),
            "peak_gpu_memory_mb": peak_memory_mb(device),
            "best_checkpoint": str(run_dir / "best.pt"),
            "latest_checkpoint": str(run_dir / "latest.pt"),
            "convergence_step_to_threshold": convergence_step_to_threshold,
            "run_dir": str(run_dir),
            "report_generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        if metrics_path.exists():
            last_line = metrics_path.read_text(encoding="utf-8").strip().splitlines()[-1]
            last_metrics = json.loads(last_line)
            for key in (
                "mean_active_paths",
                "zero_gate_ratio",
                "all_on_gate_ratio",
                "learned_mean_active_paths",
                "learned_zero_gate_ratio",
                "learned_all_on_gate_ratio",
                "learned_path_rate_min",
                "learned_path_rate_max",
                "learned_path_rate_std",
                "cached_mean_active_paths",
                "cached_zero_gate_ratio",
                "cached_all_on_gate_ratio",
                "gate_rate_loss",
                "gate_balance_loss",
                "gate_entropy",
                "gate_commitment_loss",
                "gate_teacher_bce_loss",
            ):
                if key in last_metrics:
                    summary[key] = last_metrics[key]
        elif args.model in LRP_MODELS and gates is not None:
            summary.update(gate_diagnostics(gates))
        write_json(run_dir / "summary.json", summary)
        return summary
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return failed_summary(run_dir, args, "failed_oom", repr(exc))
        return failed_summary(run_dir, args, "failed", repr(exc))
    except Exception as exc:
        return failed_summary(run_dir, args, "failed", repr(exc))


def main():
    args = parse_args()
    summary = train_model(args)
    if summary.get("status") not in ("pass",):
        print(f"LEVEL_TEST_TRAIN_ONE_PASS model={args.model} status={summary.get('status')}")
    else:
        print(f"LEVEL_TEST_TRAIN_ONE_PASS model={args.model}")


if __name__ == "__main__":
    main()

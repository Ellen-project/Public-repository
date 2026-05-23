from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import time
from pathlib import Path
from typing import Iterator, Optional, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

from Low_Rank_Path_SSM import load_gate_cache
from c4_data import build_tokenizer, make_c4_batch_iterator
from c4_lrp_lm import (
    C4LRPSSMLanguageModel,
    gate_statistics,
    generate_distilled_gates_for_input_ids,
    generate_snn_gates_for_input_ids,
    load_c4_lightweight_router,
    load_full_snn_router_for_dimensions,
    load_gate_feature_encoder,
    safe_perplexity,
    save_c4_lm_checkpoint,
    torch_load,
)


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_optional_int(value: Optional[str]) -> Optional[int]:
    if value is None or str(value).lower() == "none":
        return None
    return int(value)


def parse_args():
    parser = argparse.ArgumentParser(description="Train C4 LowRankPathSSM language model.")
    parser.add_argument("--mode", choices=["cached_gate", "online_snn", "distilled_router"], default="cached_gate")
    parser.add_argument("--tokenizer-name", type=str, default="gpt2")
    parser.add_argument("--subset", type=str, default="en")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="validation")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-train-steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--max-eval-batches", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--state-dim", type=int, default=128)
    parser.add_argument("--num-paths", type=int, default=8)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--tie-weights", action="store_true")
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-lrp-layers", type=int, default=None)
    parser.add_argument("--gamma-init", type=float, default=0.1)
    parser.add_argument("--path-residual-scale", type=float, default=1.0)
    parser.add_argument("--path-dropout", type=float, default=0.0)
    parser.add_argument("--force-min-active-paths", type=int, default=0)
    parser.add_argument("--topk-fallback", type=int, default=0)
    parser.add_argument("--gate-mode", choices=["cached_snn", "learned_sigmoid", "learned_topk_st", "hybrid_cached_plus_learned"], default="cached_snn")
    parser.add_argument("--target-active-paths", type=float, default=1.5)
    parser.add_argument("--gate-loss-weight", type=float, default=0.01)
    parser.add_argument("--gate-balance-weight", type=float, default=0.001)
    parser.add_argument("--router-hidden-dim", type=int, default=128)
    parser.add_argument("--router-temperature", type=float, default=1.0)
    parser.add_argument("--router-topk", type=int, default=2)
    parser.add_argument("--token-cache", type=str, default="c4_token_cache.pt")
    parser.add_argument("--gate-cache", type=str, default="c4_gate_cache.pt")
    parser.add_argument("--router-preset", type=str, default="best_router_preset.npz")
    parser.add_argument("--gate-feature-encoder", type=str, default="c4_gate_feature_encoder.pt")
    parser.add_argument("--lightweight-router", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="runs/c4_lrp_ssm")
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    parser.add_argument("--max-train-samples", type=parse_optional_int, default=None)
    parser.add_argument("--max-train-tokens", type=parse_optional_int, default=None)
    return parser.parse_args()


def cuda_sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def gpu_memory_mb(device: torch.device) -> Tuple[float, float]:
    if device.type != "cuda":
        return 0.0, 0.0
    return (
        float(torch.cuda.memory_allocated(device) / (1024 * 1024)),
        float(torch.cuda.memory_reserved(device) / (1024 * 1024)),
    )


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


def load_token_cache(path: str) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    payload = torch_load(path, map_location="cpu")
    return payload["input_ids"].long(), payload["labels"].long(), dict(payload.get("metadata", {}))


def load_gate_tensor(path: str) -> Tuple[torch.Tensor, dict]:
    gates_np, metadata = load_gate_cache(path)
    return torch.from_numpy(gates_np).float(), metadata


def resolve_vocab_info(args, token_metadata: Optional[dict] = None) -> Tuple[int, Optional[int], str]:
    token_metadata = token_metadata or {}
    tokenizer_name = token_metadata.get("tokenizer_name") or args.tokenizer_name
    vocab_size = token_metadata.get("vocab_size")
    pad_token_id = token_metadata.get("pad_token_id")
    if vocab_size is not None:
        return int(vocab_size), None if pad_token_id is None else int(pad_token_id), tokenizer_name
    tokenizer = build_tokenizer(tokenizer_name)
    return len(tokenizer), tokenizer.pad_token_id, tokenizer_name


def split_cache_dataset(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    gates: Optional[torch.Tensor],
    batch_size: int,
    max_eval_batches: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader]:
    num_samples = input_ids.shape[0]
    eval_count = 1 if num_samples <= 2 else max(1, min(num_samples // 10, batch_size * max_eval_batches, num_samples - 1))
    generator = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(num_samples, generator=generator)
    eval_idx = perm[:eval_count]
    train_idx = perm[eval_count:] if eval_count < num_samples else perm

    if gates is None:
        train_ds = TensorDataset(input_ids[train_idx], labels[train_idx])
        eval_ds = TensorDataset(input_ids[eval_idx], labels[eval_idx])
    else:
        train_ds = TensorDataset(input_ids[train_idx], labels[train_idx], gates[train_idx])
        eval_ds = TensorDataset(input_ids[eval_idx], labels[eval_idx], gates[eval_idx])
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False),
        DataLoader(eval_ds, batch_size=batch_size, shuffle=False, drop_last=False),
    )


def endless(loader: DataLoader) -> Iterator[tuple]:
    while True:
        for batch in loader:
            yield batch


def trainable_parameters(model: C4LRPSSMLanguageModel):
    return [param for param in model.parameters() if param.requires_grad]


@torch.no_grad()
def evaluate_loader(
    model: C4LRPSSMLanguageModel,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    max_batches: int,
    mode: str,
    gate_feature_encoder=None,
    lightweight_router=None,
    lightweight_threshold: float = 0.5,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    gate_stats = []
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        input_ids = batch[0].to(device)
        labels = batch[1].to(device)
        if mode == "cached_gate":
            gates = batch[2].to(device).float().detach() if len(batch) > 2 else None
        elif mode == "distilled_router":
            gates = generate_distilled_gates_for_input_ids(
                input_ids.cpu(),
                gate_feature_encoder,
                lightweight_router,
                threshold=lightweight_threshold,
                device=device,
            ).to(device)
        else:
            raise ValueError(f"Unsupported eval mode {mode}")
        with autocast_context(device, amp):
            output = model(input_ids, labels=labels, gates=gates, return_diagnostics=True)
        tokens = int(labels.numel())
        total_loss += float(output["loss"].detach().cpu()) * tokens
        total_tokens += tokens
        gate_stats.append(output["diagnostics"])
    if total_tokens == 0:
        return {"val_loss": float("inf"), "val_ppl": float("inf")}
    mean_loss = total_loss / total_tokens
    stats = {
        "val_loss": mean_loss,
        "val_ppl": safe_perplexity(mean_loss),
    }
    if gate_stats:
        for key in ("mean_active_paths", "zero_gate_ratio", "all_on_gate_ratio"):
            stats[key] = float(sum(item[key] for item in gate_stats) / len(gate_stats))
    model.train()
    return stats


def save_training_checkpoint(model, optimizer, args, tokenizer_name, model_config, step, metrics, output_dir: Path, name: str):
    return save_c4_lm_checkpoint(
        model=getattr(model, "_orig_mod", model),
        path=output_dir / name,
        tokenizer_name=tokenizer_name,
        model_config=model_config,
        optimizer=optimizer,
        step=step,
        metrics=metrics,
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "online_snn" and args.batch_size != 1:
        raise ValueError("online_snn mode is smoke/debug only and requires --batch-size 1")
    if args.mode == "online_snn":
        print("WARNING: online_snn mode runs FullSNNPathRouter inside the train loop and is intentionally slow.")

    token_metadata = {}
    gates = None
    train_loader = None
    eval_loader = None
    gate_feature_encoder = None
    lightweight_router = None
    lightweight_threshold = 0.5

    needs_cached_gates = args.gate_mode in ("cached_snn", "hybrid_cached_plus_learned")
    if args.mode in ("cached_gate", "distilled_router"):
        input_ids, labels, token_metadata = load_token_cache(args.token_cache)
        if input_ids.shape != labels.shape:
            raise ValueError("token cache input_ids and labels shape mismatch")
        if args.mode == "cached_gate" and needs_cached_gates:
            gates, gate_metadata = load_gate_tensor(args.gate_cache)
            if gates.shape[:2] != input_ids.shape:
                raise ValueError("gate cache shape does not match token cache")
            if gates.shape[-1] != args.num_paths:
                raise ValueError(f"gate cache num_paths {gates.shape[-1]} != --num-paths {args.num_paths}")
        elif args.mode == "distilled_router":
            gate_feature_encoder, _ = load_gate_feature_encoder(args.gate_feature_encoder, map_location="cpu")
            router_path = args.lightweight_router or ("c4_lightweight_router.pt" if Path("c4_lightweight_router.pt").exists() else "lightweight_router.pt")
            lightweight_router, lightweight_payload = load_c4_lightweight_router(router_path, map_location=device)
            lightweight_threshold = float(lightweight_payload.get("threshold", 0.5))
        train_loader, eval_loader = split_cache_dataset(
            input_ids,
            labels,
            gates,
            batch_size=args.batch_size,
            max_eval_batches=args.max_eval_batches,
            seed=args.seed,
        )

    vocab_size, pad_token_id, tokenizer_name = resolve_vocab_info(args, token_metadata)
    model_config = {
        "vocab_size": int(vocab_size),
        "model_dim": int(args.model_dim),
        "state_dim": int(args.state_dim),
        "num_paths": int(args.num_paths),
        "rank": int(args.rank),
        "dropout": float(args.dropout),
        "tie_weights": bool(args.tie_weights),
        "pad_token_id": pad_token_id,
        "num_layers": int(args.num_lrp_layers or args.num_layers),
        "gamma_init": float(args.gamma_init),
        "path_residual_scale": float(args.path_residual_scale),
        "path_dropout": float(args.path_dropout),
        "force_min_active_paths": int(args.force_min_active_paths),
        "topk_fallback": int(args.topk_fallback),
        "gate_mode": args.gate_mode,
        "target_active_paths": float(args.target_active_paths),
        "router_hidden_dim": int(args.router_hidden_dim),
        "router_temperature": float(args.router_temperature),
        "router_topk": int(args.router_topk),
    }
    model = C4LRPSSMLanguageModel(**model_config).to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(trainable_parameters(getattr(model, "_orig_mod", model)), lr=args.lr, weight_decay=args.weight_decay)
    scaler = make_grad_scaler(args.amp and device.type == "cuda")

    (output_dir / "train_config.json").write_text(
        json.dumps({**vars(args), "model_config": model_config}, indent=2),
        encoding="utf-8",
    )
    metrics_path = output_dir / "train_metrics.jsonl"
    best_val_loss = float("inf")
    best_val_ppl = float("inf")
    tokens_seen = 0

    if args.mode == "online_snn":
        tokenizer = build_tokenizer(tokenizer_name)
        gate_feature_encoder, _ = load_gate_feature_encoder(args.gate_feature_encoder, map_location="cpu")
        router, _ = load_full_snn_router_for_dimensions(
            args.router_preset,
            input_dim=args.model_dim,
            num_paths=args.num_paths,
            seed=args.seed,
        )
        stream_iter = make_c4_batch_iterator(
            tokenizer_name=tokenizer.name_or_path,
            subset=args.subset,
            split=args.train_split,
            block_size=args.block_size,
            batch_size=1,
            streaming=True if not args.streaming else args.streaming,
            shuffle_buffer=args.shuffle_buffer,
            max_samples=args.max_train_samples,
            max_tokens=args.max_train_tokens,
            seed=args.seed,
        )
        train_iter = iter(stream_iter)
    else:
        train_iter = endless(train_loader)

    for step in range(1, args.max_train_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if args.mode == "cached_gate":
            batch = next(train_iter)
            input_ids_b = batch[0].to(device)
            labels_b = batch[1].to(device)
            gates_b = batch[2].to(device).float().detach() if len(batch) > 2 else None
        elif args.mode == "distilled_router":
            batch = next(train_iter)
            input_ids_b = batch[0].to(device)
            labels_b = batch[1].to(device)
            gates_b = generate_distilled_gates_for_input_ids(
                input_ids_b.cpu(),
                gate_feature_encoder,
                lightweight_router,
                threshold=lightweight_threshold,
                device=device,
            ).to(device)
        else:
            try:
                batch = next(train_iter)
            except StopIteration:
                break
            input_ids_b = batch["input_ids"].to(device)
            labels_b = batch["labels"].to(device)
            gates_b = generate_snn_gates_for_input_ids(
                input_ids_b.cpu(),
                gate_feature_encoder,
                router,
                use_ema=False,
                use_gpu=False,
            ).to(device)

        if gates_b is not None:
            gates_b.requires_grad_(False)
        cuda_sync(device)
        start = time.perf_counter()
        with autocast_context(device, args.amp):
            output = model(input_ids_b, labels=labels_b, gates=gates_b, return_diagnostics=True)
            loss = output["loss"]
            if "gate_losses" in output:
                loss = loss + args.gate_loss_weight * output["gate_losses"]["gate_rate_loss"]
                loss = loss + args.gate_balance_weight * output["gate_losses"]["gate_balance_loss"]
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(getattr(model, "_orig_mod", model)), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        cuda_sync(device)
        elapsed = max(time.perf_counter() - start, 1e-9)

        batch_tokens = int(labels_b.numel())
        tokens_seen += batch_tokens
        alloc_mb, reserved_mb = gpu_memory_mb(device)
        train_loss = float(loss.detach().cpu())
        diag_metrics = output.get("diagnostics")
        if diag_metrics is None:
            diag_metrics = gate_statistics(gates_b) if gates_b is not None else {}
        metrics = {
            "step": int(step),
            "train_loss": train_loss,
            "train_ppl": safe_perplexity(train_loss),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "tokens_seen": int(tokens_seen),
            "tokens_per_sec": batch_tokens / elapsed,
            "samples_per_sec": input_ids_b.shape[0] / elapsed,
            "ms_per_step": elapsed * 1000.0,
            "grad_norm": float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm),
            "gpu_memory_allocated_mb": alloc_mb,
            "gpu_memory_reserved_mb": reserved_mb,
            **diag_metrics,
        }

        if args.eval_every > 0 and step % args.eval_every == 0 and eval_loader is not None:
            eval_metrics = evaluate_loader(
                getattr(model, "_orig_mod", model),
                eval_loader,
                device=device,
                amp=args.amp,
                max_batches=args.max_eval_batches,
                mode=args.mode,
                gate_feature_encoder=gate_feature_encoder,
                lightweight_router=lightweight_router,
                lightweight_threshold=lightweight_threshold,
            )
            metrics.update(eval_metrics)
            if eval_metrics["val_loss"] < best_val_loss:
                best_val_loss = float(eval_metrics["val_loss"])
                best_val_ppl = float(eval_metrics["val_ppl"])
                metrics["best_val_loss"] = best_val_loss
                metrics["best_val_ppl"] = best_val_ppl
                save_training_checkpoint(model, optimizer, args, tokenizer_name, model_config, step, metrics, output_dir, "best_model.pt")

        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics) + "\n")

        if args.save_every > 0 and step % args.save_every == 0:
            save_training_checkpoint(model, optimizer, args, tokenizer_name, model_config, step, metrics, output_dir, "latest_model.pt")

    final_metrics = {
        "step": int(min(args.max_train_steps, step)),
        "best_val_loss": best_val_loss,
        "best_val_ppl": best_val_ppl,
        "tokens_seen": int(tokens_seen),
    }
    save_training_checkpoint(model, optimizer, args, tokenizer_name, model_config, final_metrics["step"], final_metrics, output_dir, "latest_model.pt")
    if not (output_dir / "best_model.pt").exists():
        save_training_checkpoint(model, optimizer, args, tokenizer_name, model_config, final_metrics["step"], final_metrics, output_dir, "best_model.pt")
    print("C4_TRAIN_PASS")


if __name__ == "__main__":
    main()

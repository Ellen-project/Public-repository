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
    gate_statistics,
    generate_distilled_gates_for_input_ids,
    generate_snn_gates_for_input_ids,
    load_c4_lightweight_router,
    load_c4_lm_checkpoint,
    load_full_snn_router_for_dimensions,
    load_gate_feature_encoder,
    safe_perplexity,
    torch_load,
)


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a C4 LowRankPathSSM language model.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--mode",
        choices=["cached_gate", "online_snn", "distilled_router", "zero_gate", "all_on_gate", "random_gate"],
        default="cached_gate",
    )
    parser.add_argument("--tokenizer-name", type=str, default=None)
    parser.add_argument("--subset", type=str, default="en")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-eval-batches", type=int, default=100)
    parser.add_argument("--token-cache", type=str, default=None)
    parser.add_argument("--gate-cache", type=str, default=None)
    parser.add_argument("--router-preset", type=str, default="best_router_preset.npz")
    parser.add_argument("--gate-feature-encoder", type=str, default="c4_gate_feature_encoder.pt")
    parser.add_argument("--lightweight-router", type=str, default=None)
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-report", type=str, default="c4_eval_report.json")
    return parser.parse_args()


def cuda_sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autocast_context(device: torch.device, enabled: bool):
    active = bool(enabled and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=active)
    return torch.cuda.amp.autocast(enabled=active)


def load_token_cache(path: str) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    payload = torch_load(path, map_location="cpu")
    return payload["input_ids"].long(), payload["labels"].long(), dict(payload.get("metadata", {}))


def load_gate_tensor(path: str) -> Tuple[torch.Tensor, dict]:
    gates_np, metadata = load_gate_cache(path)
    return torch.from_numpy(gates_np).float(), metadata


def make_cache_loader(token_cache: str, gate_cache: Optional[str], batch_size: int):
    input_ids, labels, token_metadata = load_token_cache(token_cache)
    if gate_cache is not None:
        gates, gate_metadata = load_gate_tensor(gate_cache)
        if gates.shape[:2] != input_ids.shape:
            raise ValueError("gate cache shape does not match token cache")
        dataset = TensorDataset(input_ids, labels, gates)
    else:
        gate_metadata = {}
        dataset = TensorDataset(input_ids, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False), token_metadata, gate_metadata


def make_stream_loader(args, tokenizer_name: str) -> Iterator[dict]:
    return make_c4_batch_iterator(
        tokenizer_name=tokenizer_name,
        subset=args.subset,
        split=args.split,
        block_size=args.block_size,
        batch_size=args.batch_size,
        streaming=True,
        shuffle_buffer=0,
        max_samples=None,
        max_tokens=args.max_eval_batches * args.batch_size * args.block_size,
        seed=args.seed,
    )


def random_gate_probability(num_paths: int, gate_cache_metadata: dict, cached_gates: Optional[torch.Tensor]) -> float:
    if cached_gates is not None:
        return float(cached_gates.sum(dim=-1).float().mean().item() / max(1, num_paths))
    if "mean_active_paths" in gate_cache_metadata:
        return float(gate_cache_metadata["mean_active_paths"]) / max(1, num_paths)
    target_active = 1.5 if num_paths <= 8 else max(1.0, num_paths * 0.1875)
    return min(1.0, max(0.0, target_active / max(1, num_paths)))


def make_gates(
    mode: str,
    input_ids: torch.Tensor,
    num_paths: int,
    device: torch.device,
    cached_batch: Optional[torch.Tensor] = None,
    gate_probability: float = 0.25,
    gate_feature_encoder=None,
    full_router=None,
    lightweight_router=None,
    lightweight_threshold: float = 0.5,
):
    if mode == "cached_gate":
        if cached_batch is None:
            raise ValueError("cached_gate mode requires --gate-cache")
        gates = cached_batch.float()
    elif mode == "zero_gate":
        gates = torch.zeros(input_ids.shape[0], input_ids.shape[1], num_paths)
    elif mode == "all_on_gate":
        gates = torch.ones(input_ids.shape[0], input_ids.shape[1], num_paths)
    elif mode == "random_gate":
        gates = torch.bernoulli(torch.full((input_ids.shape[0], input_ids.shape[1], num_paths), gate_probability))
    elif mode == "online_snn":
        gates = generate_snn_gates_for_input_ids(input_ids.cpu(), gate_feature_encoder, full_router)
    elif mode == "distilled_router":
        gates = generate_distilled_gates_for_input_ids(
            input_ids.cpu(),
            gate_feature_encoder,
            lightweight_router,
            threshold=lightweight_threshold,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported mode {mode}")
    gates = gates.to(device).float().detach()
    gates.requires_grad_(False)
    return gates


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    checkpoint = load_c4_lm_checkpoint(args.checkpoint, map_location=device)
    model = checkpoint["model"].to(device)
    model.eval()
    model_config = checkpoint["model_config"]
    num_paths = int(model_config["num_paths"])
    model_dim = int(model_config["model_dim"])
    tokenizer_name = args.tokenizer_name or checkpoint.get("tokenizer_name") or "gpt2"

    gate_feature_encoder = None
    full_router = None
    lightweight_router = None
    lightweight_threshold = 0.5
    cached_gates_for_prob = None
    gate_metadata = {}

    if args.token_cache is not None:
        loader, token_metadata, gate_metadata = make_cache_loader(args.token_cache, args.gate_cache, args.batch_size)
        tokenizer_name = args.tokenizer_name or token_metadata.get("tokenizer_name") or tokenizer_name
        if args.gate_cache is not None:
            cached_gates_for_prob, _ = load_gate_tensor(args.gate_cache)
    else:
        if args.mode == "cached_gate":
            raise ValueError("cached_gate mode requires --token-cache and --gate-cache")
        build_tokenizer(tokenizer_name)
        loader = make_stream_loader(args, tokenizer_name)

    if args.mode in ("online_snn", "distilled_router"):
        gate_feature_encoder, _ = load_gate_feature_encoder(args.gate_feature_encoder, map_location="cpu")
    if args.mode == "online_snn":
        full_router, _ = load_full_snn_router_for_dimensions(
            args.router_preset,
            input_dim=model_dim,
            num_paths=num_paths,
            seed=args.seed,
        )
    if args.mode == "distilled_router":
        router_path = args.lightweight_router or ("c4_lightweight_router.pt" if Path("c4_lightweight_router.pt").exists() else "lightweight_router.pt")
        lightweight_router, lightweight_payload = load_c4_lightweight_router(router_path, map_location=device)
        lightweight_threshold = float(lightweight_payload.get("threshold", 0.5))

    gate_probability = random_gate_probability(num_paths, gate_metadata, cached_gates_for_prob)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    total_loss = 0.0
    total_tokens = 0
    total_samples = 0
    forward_ms = []
    gate_stats_accum = []
    start_all = time.perf_counter()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= args.max_eval_batches:
                break
            if isinstance(batch, dict):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                cached_batch = None
            else:
                input_ids = batch[0].to(device)
                labels = batch[1].to(device)
                cached_batch = batch[2] if len(batch) > 2 else None
            gates = make_gates(
                args.mode,
                input_ids,
                num_paths=num_paths,
                device=device,
                cached_batch=cached_batch,
                gate_probability=gate_probability,
                gate_feature_encoder=gate_feature_encoder,
                full_router=full_router,
                lightweight_router=lightweight_router,
                lightweight_threshold=lightweight_threshold,
            )
            cuda_sync(device)
            start = time.perf_counter()
            with autocast_context(device, args.amp):
                output = model(input_ids, labels=labels, gates=gates, return_diagnostics=True)
            cuda_sync(device)
            forward_ms.append((time.perf_counter() - start) * 1000.0)
            tokens = int(labels.numel())
            total_loss += float(output["loss"].detach().cpu()) * tokens
            total_tokens += tokens
            total_samples += int(input_ids.shape[0])
            gate_stats_accum.append(output["diagnostics"])

    elapsed = max(time.perf_counter() - start_all, 1e-9)
    eval_loss = total_loss / max(1, total_tokens)
    peak_gpu_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024)) if device.type == "cuda" else 0.0
    report = {
        "mode": args.mode,
        "checkpoint": args.checkpoint,
        "eval_loss": eval_loss,
        "eval_ppl": safe_perplexity(eval_loss),
        "total_tokens": int(total_tokens),
        "total_samples": int(total_samples),
        "elapsed_sec": elapsed,
        "tokens_per_sec": total_tokens / elapsed,
        "samples_per_sec": total_samples / elapsed,
        "avg_forward_ms": float(sum(forward_ms) / max(1, len(forward_ms))),
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "gate_probability": gate_probability if args.mode == "random_gate" else None,
    }
    if gate_stats_accum:
        for key in ("mean_active_paths", "zero_gate_ratio", "all_on_gate_ratio"):
            report[key] = float(sum(item[key] for item in gate_stats_accum) / len(gate_stats_accum))
    else:
        report.update(gate_statistics(torch.zeros(1, 1, num_paths)))

    Path(args.output_report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("C4_EVAL_PASS")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Optional, Tuple

import torch

from Low_Rank_Path_SSM import load_gate_cache
from c4_data import build_tokenizer, load_c4_stream
from c4_lrp_lm import (
    C4LRPSSMLanguageModel,
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


def parse_csv_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark C4 LowRankPathSSM pipeline.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--token-cache", type=str, default="c4_token_cache.pt")
    parser.add_argument("--gate-cache", type=str, default="c4_gate_cache.pt")
    parser.add_argument("--router-preset", type=str, default="best_router_preset.npz")
    parser.add_argument("--gate-feature-encoder", type=str, default="c4_gate_feature_encoder.pt")
    parser.add_argument("--lightweight-router", type=str, default=None)
    parser.add_argument("--tokenizer-name", type=str, default="gpt2")
    parser.add_argument("--subset", type=str, default="en")
    parser.add_argument("--block-sizes", type=str, default="64,128,256")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8,16")
    parser.add_argument("--num-paths", type=int, default=8)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--state-dim", type=int, default=128)
    parser.add_argument("--num-warmup", type=int, default=5)
    parser.add_argument("--num-iters", type=int, default=20)
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output-report", type=str, default="c4_benchmark_report.json")
    return parser.parse_args()


def cuda_sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))


def autocast_context(device: torch.device, enabled: bool):
    active = bool(enabled and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=active)
    return torch.cuda.amp.autocast(enabled=active)


def load_token_cache(path: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], dict]:
    if not Path(path).exists():
        return None, None, {}
    payload = torch_load(path, map_location="cpu")
    return payload["input_ids"].long(), payload["labels"].long(), dict(payload.get("metadata", {}))


def load_gate_tensor(path: str) -> Tuple[Optional[torch.Tensor], dict]:
    if not Path(path).exists():
        return None, {}
    gates_np, metadata = load_gate_cache(path)
    return torch.from_numpy(gates_np).float(), metadata


def repeat_to_shape(tensor: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
    rows = tensor
    if rows.shape[0] < batch_size:
        row_reps = (batch_size + rows.shape[0] - 1) // rows.shape[0]
        rows = rows.repeat((row_reps, 1, *([1] * (rows.ndim - 2))))
    rows = rows[:batch_size]
    if rows.shape[1] < seq_len:
        col_reps = (seq_len + rows.shape[1] - 1) // rows.shape[1]
        rows = rows.repeat((1, col_reps, *([1] * (rows.ndim - 2))))
    return rows[:, :seq_len].clone()


def make_tokens(input_ids, labels, batch_size: int, seq_len: int, vocab_size: int):
    if input_ids is not None and labels is not None:
        return repeat_to_shape(input_ids, batch_size, seq_len), repeat_to_shape(labels, batch_size, seq_len)
    x = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    return x, y


def make_gates(cached_gates, batch_size: int, seq_len: int, num_paths: int, kind: str = "cached"):
    if kind == "cached" and cached_gates is not None and cached_gates.shape[-1] == num_paths:
        return repeat_to_shape(cached_gates, batch_size, seq_len)
    if kind == "zero":
        return torch.zeros(batch_size, seq_len, num_paths)
    if kind == "all_on":
        return torch.ones(batch_size, seq_len, num_paths)
    p = 1.5 / max(1, num_paths)
    return torch.bernoulli(torch.full((batch_size, seq_len, num_paths), min(1.0, p)))


def environment_report(device: torch.device):
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "device": str(device),
    }


def benchmark_c4_streaming(args):
    try:
        tokenizer = build_tokenizer(args.tokenizer_name)
        dataset = load_c4_stream(args.subset, split="train", streaming=True, shuffle_buffer=0, seed=1)
        count = 0
        chars = 0
        tokens = 0
        start = time.perf_counter()
        for sample in dataset:
            text = sample.get("text", "")
            if not isinstance(text, str) or not text:
                continue
            count += 1
            chars += len(text)
            max_length = getattr(tokenizer, "model_max_length", 1024)
            if not isinstance(max_length, int) or max_length > 100000:
                max_length = 1024
            tokens += len(
                tokenizer(
                    text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_length,
                ).input_ids
            )
            if count >= 16:
                break
        elapsed = max(time.perf_counter() - start, 1e-9)
        return {
            "samples": count,
            "chars": chars,
            "tokens": tokens,
            "text_samples_per_sec": count / elapsed,
            "raw_char_per_sec": chars / elapsed,
            "tokenization_tokens_per_sec": tokens / elapsed,
            "elapsed_sec": elapsed,
        }
    except Exception as exc:
        return {"status": "SKIP", "reason": repr(exc)}


def fresh_model(model_config, checkpoint_path: Optional[str], device: torch.device):
    if checkpoint_path and Path(checkpoint_path).exists():
        return load_c4_lm_checkpoint(checkpoint_path, map_location=device)["model"].to(device)
    return C4LRPSSMLanguageModel(**model_config).to(device)


def benchmark_gate_generation(args, input_ids, model_config, device: torch.device):
    if input_ids is None or not Path(args.gate_feature_encoder).exists():
        return {"status": "SKIP", "reason": "token cache or gate feature encoder missing"}
    try:
        encoder, _ = load_gate_feature_encoder(args.gate_feature_encoder, map_location="cpu")
        router, _ = load_full_snn_router_for_dimensions(
            args.router_preset,
            input_dim=int(model_config["model_dim"]),
            num_paths=int(model_config["num_paths"]),
            seed=1,
        )
        sample_ids = repeat_to_shape(input_ids, min(4, input_ids.shape[0]), min(input_ids.shape[1], 64))
        start = time.perf_counter()
        gates = generate_snn_gates_for_input_ids(sample_ids, encoder, router)
        elapsed = max(time.perf_counter() - start, 1e-9)
        tokens = int(sample_ids.numel())
        return {
            "samples": int(sample_ids.shape[0]),
            "tokens": tokens,
            "elapsed_sec": elapsed,
            "snn_steps_per_sec": tokens / elapsed,
            "tokens_per_sec": tokens / elapsed,
            **gate_statistics(gates),
        }
    except Exception as exc:
        return {"status": "ERROR", "reason": repr(exc)}


def benchmark_forward(args, model_config, input_ids, labels, cached_gates, device):
    results = []
    for block_size in parse_csv_ints(args.block_sizes):
        for batch_size in parse_csv_ints(args.batch_sizes):
            model = fresh_model(model_config, args.checkpoint, device).eval()
            x, _ = make_tokens(input_ids, labels, batch_size, block_size, int(model_config["vocab_size"]))
            gates = make_gates(cached_gates, batch_size, block_size, int(model_config["num_paths"])).float()
            x = x.to(device)
            gates = gates.to(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            with torch.no_grad():
                for _ in range(args.num_warmup):
                    with autocast_context(device, args.amp):
                        model(x, gates=gates)
                cuda_sync(device)
                start = time.perf_counter()
                for _ in range(args.num_iters):
                    with autocast_context(device, args.amp):
                        model(x, gates=gates)
                cuda_sync(device)
            elapsed = max(time.perf_counter() - start, 1e-9)
            tokens = batch_size * block_size * args.num_iters
            results.append(
                {
                    "batch_size": batch_size,
                    "block_size": block_size,
                    "tokens_per_sec": tokens / elapsed,
                    "avg_forward_latency_ms": elapsed * 1000.0 / args.num_iters,
                    "avg_forward_ms": elapsed * 1000.0 / args.num_iters,
                    "peak_gpu_memory_mb": peak_memory_mb(device),
                }
            )
    return results


def benchmark_training(args, model_config, input_ids, labels, cached_gates, device):
    results = []
    for block_size in parse_csv_ints(args.block_sizes):
        for batch_size in parse_csv_ints(args.batch_sizes):
            model = fresh_model(model_config, args.checkpoint, device).train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            x, y = make_tokens(input_ids, labels, batch_size, block_size, int(model_config["vocab_size"]))
            gates = make_gates(cached_gates, batch_size, block_size, int(model_config["num_paths"])).float()
            x = x.to(device)
            y = y.to(device)
            gates = gates.to(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            cuda_sync(device)
            start = time.perf_counter()
            for _ in range(args.num_iters):
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(device, args.amp):
                    loss = model(x, labels=y, gates=gates)["loss"]
                loss.backward()
                optimizer.step()
            cuda_sync(device)
            elapsed = max(time.perf_counter() - start, 1e-9)
            tokens = batch_size * block_size * args.num_iters
            results.append(
                {
                    "batch_size": batch_size,
                    "block_size": block_size,
                    "tokens_per_sec": tokens / elapsed,
                    "steps_per_sec": args.num_iters / elapsed,
                    "peak_gpu_memory_mb": peak_memory_mb(device),
                }
            )
    return results


@torch.no_grad()
def eval_loss_for_gates(model, x, y, gates, device, amp):
    with autocast_context(device, amp):
        output = model(x.to(device), labels=y.to(device), gates=gates.to(device).float())
    loss = float(output["loss"].detach().cpu())
    return {"loss": loss, "ppl": safe_perplexity(loss), **gate_statistics(gates)}


def benchmark_evaluation_and_ablation(args, model_config, input_ids, labels, cached_gates, device):
    model = fresh_model(model_config, args.checkpoint, device).eval()
    block_size = parse_csv_ints(args.block_sizes)[0]
    batch_size = parse_csv_ints(args.batch_sizes)[0]
    x, y = make_tokens(input_ids, labels, batch_size, block_size, int(model_config["vocab_size"]))
    gates = make_gates(cached_gates, batch_size, block_size, int(model_config["num_paths"])).float()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(args.num_iters):
        eval_loss_for_gates(model, x, y, gates, device, args.amp)
    cuda_sync(device)
    elapsed = max(time.perf_counter() - start, 1e-9)
    evaluation = {
        "tokens_per_sec": batch_size * block_size * args.num_iters / elapsed,
        "avg_forward_ms": elapsed * 1000.0 / args.num_iters,
        "peak_gpu_memory_mb": peak_memory_mb(device),
    }
    ablation = {
        "cached_snn_gate": eval_loss_for_gates(model, x, y, gates, device, args.amp),
        "zero_gate": eval_loss_for_gates(model, x, y, make_gates(None, batch_size, block_size, int(model_config["num_paths"]), "zero"), device, args.amp),
        "all_on_gate": eval_loss_for_gates(model, x, y, make_gates(None, batch_size, block_size, int(model_config["num_paths"]), "all_on"), device, args.amp),
        "random_gate": eval_loss_for_gates(model, x, y, make_gates(None, batch_size, block_size, int(model_config["num_paths"]), "random"), device, args.amp),
    }
    router_path = args.lightweight_router or ("c4_lightweight_router.pt" if Path("c4_lightweight_router.pt").exists() else None)
    if router_path and Path(router_path).exists() and Path(args.gate_feature_encoder).exists():
        encoder, _ = load_gate_feature_encoder(args.gate_feature_encoder, map_location="cpu")
        light_router, payload = load_c4_lightweight_router(router_path, map_location=device)
        distilled = generate_distilled_gates_for_input_ids(x, encoder, light_router, payload.get("threshold", 0.5), device)
        ablation["distilled_router_gate"] = eval_loss_for_gates(model, x, y, distilled, device, args.amp)
    return evaluation, ablation


def main():
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    input_ids, labels, token_metadata = load_token_cache(args.token_cache)
    cached_gates, gate_metadata = load_gate_tensor(args.gate_cache)
    metadata_encoder_path = gate_metadata.get("gate_feature_encoder_path")
    if metadata_encoder_path and not Path(args.gate_feature_encoder).exists() and Path(metadata_encoder_path).exists():
        args.gate_feature_encoder = metadata_encoder_path

    if args.checkpoint and Path(args.checkpoint).exists():
        checkpoint = load_c4_lm_checkpoint(args.checkpoint, map_location=device)
        model_config = checkpoint["model_config"]
    else:
        vocab_size = int(token_metadata.get("vocab_size", 50257))
        pad_token_id = token_metadata.get("pad_token_id")
        model_config = {
            "vocab_size": vocab_size,
            "model_dim": int(gate_metadata.get("model_dim", args.model_dim)),
            "state_dim": int(gate_metadata.get("state_dim", args.state_dim)),
            "num_paths": int(gate_metadata.get("num_paths", args.num_paths)),
            "rank": int(gate_metadata.get("rank", args.rank)),
            "dropout": 0.0,
            "tie_weights": False,
            "pad_token_id": pad_token_id,
        }

    report = {
        "environment": environment_report(device),
        "c4_streaming": benchmark_c4_streaming(args),
        "gate_cache_generation": benchmark_gate_generation(args, input_ids, model_config, device),
    }
    report["ssm_forward"] = benchmark_forward(args, model_config, input_ids, labels, cached_gates, device)
    report["training"] = benchmark_training(args, model_config, input_ids, labels, cached_gates, device)
    evaluation, ablation = benchmark_evaluation_and_ablation(args, model_config, input_ids, labels, cached_gates, device)
    report["evaluation"] = evaluation
    report["ablation"] = ablation
    first_forward = report["ssm_forward"][0] if report["ssm_forward"] else {}
    report["summary"] = {
        "tokens_per_sec": first_forward.get("tokens_per_sec", evaluation.get("tokens_per_sec")),
        "avg_forward_ms": first_forward.get("avg_forward_ms", evaluation.get("avg_forward_ms")),
        "peak_gpu_memory_mb": first_forward.get("peak_gpu_memory_mb", evaluation.get("peak_gpu_memory_mb")),
    }
    Path(args.output_report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("C4_BENCHMARK_PASS")


if __name__ == "__main__":
    main()

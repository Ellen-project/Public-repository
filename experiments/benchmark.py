from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch


LEVEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LEVEL_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import load_gate_cache, load_token_cache, resolve_project_path, torch_load
from models import model_factory
from train_one import autocast_context, cuda_sync, level_path, peak_memory_mb


LRP_MODELS = {
    "lrp_ssm",
    "lrp_ssm_fixed_calibrated",
    "lrp_ssm_learned_router",
    "lrp_ssm_hybrid",
    "lrp_ssm_strong_path_bias_decay",
}


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_csv_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark one experiment run directory.")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--token-cache", type=str, default="../c4_token_cache.pt")
    parser.add_argument("--gate-cache", type=str, default="../c4_gate_cache.pt")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8")
    parser.add_argument("--block-sizes", type=str, default="64,128,256")
    parser.add_argument("--num-warmup", type=int, default=5)
    parser.add_argument("--num-iters", type=int, default=20)
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output-report", type=str, default=None)
    return parser.parse_args()


def repeat_batch(tensor: torch.Tensor, batch_size: int, block_size: int) -> torch.Tensor:
    rows = tensor
    if rows.shape[0] < batch_size:
        reps = (batch_size + rows.shape[0] - 1) // rows.shape[0]
        rows = rows.repeat((reps, 1, *([1] * (rows.ndim - 2))))
    return rows[:batch_size, :block_size].contiguous()


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return float(ordered[idx])


def load_run(run_dir: Path):
    config_payload = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    model_name = config_payload["model"]
    config = dict(config_payload["config"])
    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.exists():
        checkpoint_path = run_dir / "latest.pt"
    return model_name, config, checkpoint_path


def condition_result(status: str, batch_size: int, block_size: int, error: Optional[str] = None) -> dict:
    return {
        "batch_size": batch_size,
        "block_size": block_size,
        "forward_avg_ms": None,
        "forward_p50_ms": None,
        "forward_p95_ms": None,
        "tokens_per_sec": None,
        "train_step_ms": None,
        "train_tokens_per_sec": None,
        "peak_gpu_memory_mb": None,
        "status": status,
        "error": error,
    }


def run_condition(
    model_name: str,
    config: dict,
    checkpoint_path: Path,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    gates: Optional[torch.Tensor],
    batch_size: int,
    block_size: int,
    args,
    device: torch.device,
) -> dict:
    if block_size > input_ids.shape[1]:
        return condition_result("skipped", batch_size, block_size, "block_size exceeds cache sequence length")
    if block_size > int(config.get("max_seq_len", block_size)):
        return condition_result("skipped", batch_size, block_size, "block_size exceeds checkpoint max_seq_len")
    try:
        model = model_factory(model_name, config, vocab_size=int(config["vocab_size"])).to(device)
        checkpoint = torch_load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        x = repeat_batch(input_ids, batch_size, block_size).to(device)
        y = repeat_batch(labels, batch_size, block_size).to(device)
        kwargs = {}
        if model_name in LRP_MODELS and model_name != "lrp_ssm_learned_router":
            if gates is None:
                return condition_result("error", batch_size, block_size, "LRP-SSM benchmark requires gate cache")
            kwargs["gates"] = repeat_batch(gates, batch_size, block_size).to(device).float().detach()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.eval()
        with torch.no_grad():
            for _ in range(args.num_warmup):
                with autocast_context(device, args.amp):
                    model(x, **kwargs)
            cuda_sync(device)
            forward_times = []
            for _ in range(args.num_iters):
                start = time.perf_counter()
                with autocast_context(device, args.amp):
                    model(x, **kwargs)
                cuda_sync(device)
                forward_times.append((time.perf_counter() - start) * 1000.0)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        cuda_sync(device)
        train_start = time.perf_counter()
        for _ in range(args.num_iters):
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.amp):
                output = model(x, labels=y, **kwargs)
                loss = output["loss"]
            loss.backward()
            optimizer.step()
            cuda_sync(device)
        train_elapsed = max(time.perf_counter() - train_start, 1e-9)
        tokens = batch_size * block_size
        return {
            "batch_size": batch_size,
            "block_size": block_size,
            "forward_avg_ms": float(statistics.mean(forward_times)),
            "forward_p50_ms": percentile(forward_times, 50.0),
            "forward_p95_ms": percentile(forward_times, 95.0),
            "tokens_per_sec": tokens * args.num_iters / max(sum(forward_times) / 1000.0, 1e-9),
            "train_step_ms": train_elapsed * 1000.0 / args.num_iters,
            "train_tokens_per_sec": tokens * args.num_iters / train_elapsed,
            "peak_gpu_memory_mb": peak_memory_mb(device),
            "status": "pass",
            "error": None,
        }
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return condition_result("oom", batch_size, block_size, repr(exc))
        return condition_result("error", batch_size, block_size, repr(exc))
    except Exception as exc:
        return condition_result("error", batch_size, block_size, repr(exc))


def benchmark(args) -> dict:
    run_dir = level_path(args.run_dir)
    model_name, config, checkpoint_path = load_run(run_dir)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    input_ids, labels, token_metadata = load_token_cache(args.token_cache)
    config["vocab_size"] = int(token_metadata.get("vocab_size", config.get("vocab_size", 50257)))
    gates = None
    if model_name in LRP_MODELS and model_name != "lrp_ssm_learned_router":
        gates, _ = load_gate_cache(args.gate_cache)
    results = []
    for block_size in parse_csv_ints(args.block_sizes):
        for batch_size in parse_csv_ints(args.batch_sizes):
            results.append(
                run_condition(
                    model_name,
                    config,
                    checkpoint_path,
                    input_ids,
                    labels,
                    gates,
                    batch_size,
                    block_size,
                    args,
                    device,
                )
            )
    report = {
        "model": model_name,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "results": results,
        "report_generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    output_report = level_path(args.output_report) if args.output_report else run_dir / "benchmark.json"
    output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    args = parse_args()
    benchmark(args)
    print("LEVEL_TEST_BENCHMARK_PASS")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
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

from data import load_gate_cache, load_token_cache, make_dataloader, resolve_project_path, split_cache, torch_load
from models import count_parameters, model_factory, perplexity
from train_one import autocast_context, cuda_sync, level_path, peak_memory_mb


LRP_MODELS = {"lrp_ssm", "lrp_ssm_fixed_calibrated", "lrp_ssm_learned_router", "lrp_ssm_hybrid"}


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate one level_test checkpoint.")
    parser.add_argument("--model", choices=["lrp_ssm", "lrp_ssm_fixed_calibrated", "lrp_ssm_learned_router", "lrp_ssm_hybrid", "transformer", "linear_attention", "local_attention", "gru"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--token-cache", type=str, default="../c4_token_cache.pt")
    parser.add_argument("--gate-cache", type=str, default="../c4_gate_cache.pt")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-eval-batches", type=int, default=100)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output-report", type=str, default=None)
    return parser.parse_args()


def load_config(path: str) -> dict:
    payload = json.loads(resolve_project_path(path).read_text(encoding="utf-8"))
    return dict(payload.get("config", payload))


@torch.no_grad()
def evaluate(args) -> dict:
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    config = load_config(args.config)
    input_ids, labels, token_metadata = load_token_cache(args.token_cache)
    gates = None
    if args.model in LRP_MODELS and args.model != "lrp_ssm_learned_router":
        gates, _ = load_gate_cache(args.gate_cache)
    max_seq_len = int(config.get("max_seq_len", input_ids.shape[1]))
    if max_seq_len < input_ids.shape[1]:
        input_ids = input_ids[:, :max_seq_len].contiguous()
        labels = labels[:, :max_seq_len].contiguous()
        gates = None if gates is None else gates[:, :max_seq_len].contiguous()
    _, eval_split = split_cache(input_ids, labels, gates, val_ratio=args.val_ratio, seed=args.seed)
    loader = make_dataloader(*eval_split, batch_size=args.batch_size, shuffle=False)
    config["vocab_size"] = int(token_metadata.get("vocab_size", config.get("vocab_size", 50257)))
    model = model_factory(args.model, config, vocab_size=int(config["vocab_size"])).to(device)
    checkpoint = torch_load(resolve_project_path(args.checkpoint), map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    total_samples = 0
    forward_ms = []
    gate_stats = []
    start_all = time.perf_counter()
    for idx, batch in enumerate(loader):
        if idx >= args.max_eval_batches:
            break
        input_ids_b = batch["input_ids"].to(device)
        labels_b = batch["labels"].to(device)
        kwargs = {}
        if args.model in LRP_MODELS and "gates" in batch:
            kwargs["gates"] = batch["gates"].to(device).float().detach()
        cuda_sync(device)
        start = time.perf_counter()
        with autocast_context(device, args.amp):
            output = model(input_ids_b, labels=labels_b, **kwargs)
        cuda_sync(device)
        forward_ms.append((time.perf_counter() - start) * 1000.0)
        tokens = int(labels_b.numel())
        total_loss += float(output["loss"].detach().cpu()) * tokens
        total_tokens += tokens
        total_samples += int(input_ids_b.shape[0])
        if output.get("diagnostics"):
            gate_stats.append(output["diagnostics"])
    elapsed = max(time.perf_counter() - start_all, 1e-9)
    eval_loss = total_loss / max(1, total_tokens)
    report = {
        "model": args.model,
        "checkpoint": str(resolve_project_path(args.checkpoint)),
        "eval_loss": eval_loss,
        "eval_ppl": perplexity(eval_loss),
        "total_tokens": total_tokens,
        "total_samples": total_samples,
        "elapsed_sec": elapsed,
        "eval_tokens_per_sec": total_tokens / elapsed,
        "avg_forward_ms": float(sum(forward_ms) / max(1, len(forward_ms))),
        "peak_gpu_memory_mb": peak_memory_mb(device),
        "parameter_count": count_parameters(model),
        "report_generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if gate_stats:
        for key in ("mean_active_paths", "zero_gate_ratio", "all_on_gate_ratio"):
            report[key] = float(sum(float(item.get(key, 0.0)) for item in gate_stats) / len(gate_stats))
    output_report = level_path(args.output_report) if args.output_report else resolve_project_path(args.checkpoint).parent / "eval_report.json"
    output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    args = parse_args()
    evaluate(args)
    print(f"LEVEL_TEST_EVAL_PASS model={args.model}")


if __name__ == "__main__":
    main()

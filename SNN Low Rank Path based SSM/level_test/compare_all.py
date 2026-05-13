from __future__ import annotations

import argparse
import json
import platform
import sys
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch


LEVEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LEVEL_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark import benchmark as run_benchmark
from data import load_token_cache
from report import generate_report
from train_one import level_path, train_model


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Run all level_test comparisons.")
    parser.add_argument("--token-cache", type=str, default="../c4_token_cache.pt")
    parser.add_argument("--gate-cache", type=str, default="../c4_gate_cache.pt")
    parser.add_argument("--output-root", type=str, default="level_test/runs")
    parser.add_argument("--results-dir", type=str, default="level_test/results")
    parser.add_argument("--models", type=str, default="lrp_ssm,transformer,linear_attention,local_attention,gru")
    parser.add_argument("--max-train-steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--max-eval-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--state-dim", type=int, default=128)
    parser.add_argument("--num-paths", type=int, default=8)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--local-window-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--run-benchmark", action="store_true")
    parser.add_argument("--benchmark-batch-sizes", type=str, default="1,2,4,8")
    parser.add_argument("--benchmark-block-sizes", type=str, default="64,128,256")
    return parser.parse_args()


def model_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def make_configs(args, token_metadata: dict) -> dict[str, Path]:
    block_size = int(token_metadata.get("block_size", 128))
    vocab_size = int(token_metadata.get("vocab_size", 50257))
    pad_token_id = token_metadata.get("pad_token_id")
    configs_dir = level_path("level_test/configs")
    configs_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "vocab_size": vocab_size,
        "model_dim": int(args.model_dim),
        "num_layers": int(args.num_layers),
        "num_heads": int(args.num_heads),
        "mlp_ratio": 4.0,
        "dropout": float(args.dropout),
        "max_seq_len": block_size,
        "pad_token_id": pad_token_id,
    }
    configs: dict[str, dict[str, Any]] = {
        "lrp_ssm": {
            **common,
            "state_dim": int(args.state_dim),
            "num_paths": int(args.num_paths),
            "rank": int(args.rank),
            "tie_weights": False,
            "num_layers": int(args.num_layers),
            "gamma_init": 0.1,
        },
        "lrp_ssm_fixed_calibrated": {
            **common,
            "state_dim": int(args.state_dim),
            "num_paths": int(args.num_paths),
            "rank": int(args.rank),
            "tie_weights": False,
            "num_layers": int(args.num_layers),
            "gamma_init": 0.1,
            "force_min_active_paths": 1,
            "topk_fallback": 1,
            "gate_mode": "cached_snn"
        },
        "lrp_ssm_learned_router": {
            **common,
            "state_dim": int(args.state_dim),
            "num_paths": int(args.num_paths),
            "rank": int(args.rank),
            "tie_weights": False,
            "num_layers": int(args.num_layers),
            "gamma_init": 0.1,
            "gate_mode": "learned_topk_st",
            "target_active_paths": 1.5,
            "router_topk": 2,
            "router_hidden_dim": max(64, int(args.model_dim) * 2)
        },
        "lrp_ssm_hybrid": {
            **common,
            "state_dim": int(args.state_dim),
            "num_paths": int(args.num_paths),
            "rank": int(args.rank),
            "tie_weights": False,
            "num_layers": int(args.num_layers),
            "gamma_init": 0.1,
            "force_min_active_paths": 1,
            "gate_mode": "hybrid_cached_plus_learned",
            "target_active_paths": 1.5,
            "router_topk": 2,
            "router_hidden_dim": max(64, int(args.model_dim) * 2)
        },
        "transformer": dict(common),
        "linear_attention": {**common, "notes": "simplified ELU+1 causal prefix linear attention"},
        "local_attention": {**common, "window_size": int(args.local_window_size)},
        "gru": {
            "vocab_size": vocab_size,
            "model_dim": int(args.model_dim),
            "num_layers": max(1, int(args.num_layers)),
            "dropout": float(args.dropout),
            "max_seq_len": block_size,
            "pad_token_id": pad_token_id,
        },
    }
    paths = {}
    for name, cfg in configs.items():
        path = configs_dir / f"{name}.json"
        write_json(path, cfg)
        paths[name] = path
    return paths


def rank_models(models: dict, key: str, reverse: bool = False):
    entries = []
    for name, summary in models.items():
        value = summary.get(key)
        if isinstance(value, (int, float)):
            entries.append([name, value])
    return sorted(entries, key=lambda item: item[1], reverse=reverse)


def environment(device: str) -> dict:
    dev = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
        "device": str(dev),
    }


def run_all(args) -> dict:
    _, _, token_metadata = load_token_cache(args.token_cache)
    config_paths = make_configs(args, token_metadata)
    results_dir = level_path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    models: dict[str, dict] = {}

    for model_name in model_list(args.models):
        train_args = Namespace(
            model=model_name,
            token_cache=args.token_cache,
            gate_cache=args.gate_cache,
            output_dir=args.output_root,
            config=str(config_paths[model_name]),
            block_size=None,
            batch_size=args.batch_size,
            max_train_steps=args.max_train_steps,
            eval_every=args.eval_every,
            max_eval_batches=args.max_eval_batches,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            val_ratio=0.1,
            seed=args.seed,
            device=args.device,
            amp=args.amp,
            compile=args.compile,
        )
        try:
            summary = train_model(train_args)
        except Exception as exc:
            summary = {
                "model": model_name,
                "status": "failed",
                "error": repr(exc),
                "parameter_count": None,
                "best_eval_loss": None,
                "best_eval_ppl": None,
                "avg_train_tokens_per_sec": 0.0,
                "avg_eval_tokens_per_sec": 0.0,
                "peak_gpu_memory_mb": 0.0,
                "run_dir": None,
            }
        models[model_name] = summary
        if args.run_benchmark and summary.get("status") == "pass":
            try:
                bench_args = Namespace(
                    run_dir=summary["run_dir"],
                    token_cache=args.token_cache,
                    gate_cache=args.gate_cache,
                    batch_sizes=args.benchmark_batch_sizes,
                    block_sizes=args.benchmark_block_sizes,
                    num_warmup=2,
                    num_iters=5,
                    device=args.device,
                    amp=args.amp,
                    output_report=None,
                )
                run_benchmark(bench_args)
            except Exception as exc:
                models[model_name]["benchmark_error"] = repr(exc)

    comparison = {
        "experiment": {
            "token_cache": args.token_cache,
            "gate_cache": args.gate_cache,
            "models": model_list(args.models),
            "max_train_steps": int(args.max_train_steps),
            "eval_every": int(args.eval_every),
            "max_eval_batches": int(args.max_eval_batches),
            "batch_size": int(args.batch_size),
            "model_dim": int(args.model_dim),
            "state_dim": int(args.state_dim),
            "num_paths": int(args.num_paths),
            "rank": int(args.rank),
            "num_layers": int(args.num_layers),
            "num_heads": int(args.num_heads),
            "dropout": float(args.dropout),
            "local_window_size": int(args.local_window_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "grad_clip": float(args.grad_clip),
            "seed": int(args.seed),
            "device": args.device,
            "amp": bool(args.amp),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
        "environment": environment(args.device),
        "models": models,
        "ranking": {
            "best_eval_ppl": rank_models(models, "best_eval_ppl", reverse=False),
            "best_tokens_per_sec": rank_models(models, "avg_train_tokens_per_sec", reverse=True),
            "lowest_memory": rank_models(models, "peak_gpu_memory_mb", reverse=False),
        },
    }
    write_json(results_dir / "comparison_summary.json", comparison)
    has_fix_models = any(name.startswith("lrp_ssm_") for name in model_list(args.models))
    report_output = "level_test/LRP_FIX_REPORT.md" if has_fix_models else "level_test/REPORT.md"
    if report_output == "level_test/REPORT.md" and level_path(report_output).exists():
        report_output = "level_test/REPORT_latest.md"
    generate_report(
        Namespace(
            results_dir=args.results_dir,
            runs_dir=args.output_root,
            output=report_output,
            include_raw_tables=False,
            include_lrp_fix_analysis=has_fix_models,
            ablation_report=None,
            old_report="level_test/REPORT.md",
        )
    )
    return comparison


def main():
    args = parse_args()
    run_all(args)
    print("LEVEL_TEST_COMPARE_ALL_PASS")


if __name__ == "__main__":
    main()

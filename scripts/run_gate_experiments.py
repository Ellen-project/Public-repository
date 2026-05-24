from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_float_csv(text: str) -> list[float]:
    return [float(item) for item in parse_csv(text)]


def parse_args():
    parser = argparse.ArgumentParser(description="Run managed C4 gate-learning experiments.")
    parser.add_argument("--experiments", type=str, default="cached_snn,learned_topk_st,hybrid_cached_plus_learned")
    parser.add_argument("--max-train-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--state-dim", type=int, default=128)
    parser.add_argument("--num-paths", type=int, default=8)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--router-topk", type=int, default=2)
    parser.add_argument("--target-active-paths", type=float, default=1.5)
    parser.add_argument("--teacher-weights", type=str, default="0.01")
    parser.add_argument("--gate-entropy-weight", type=float, default=0.001)
    parser.add_argument("--gate-commitment-weight", type=float, default=0.001)
    parser.add_argument("--gate-loss-weight", type=float, default=0.01)
    parser.add_argument("--gate-balance-weight", type=float, default=0.001)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--token-cache", type=str, default="runs/cache/c4_token_cache.pt")
    parser.add_argument("--gate-cache", type=str, default="runs/cache/c4_gate_cache.pt")
    parser.add_argument("--gate-feature-encoder", type=str, default="runs/cache/c4_gate_feature_encoder.pt")
    parser.add_argument("--router-preset", type=str, default="runs/cache/best_router_preset.npz")
    parser.add_argument("--build-cache", action="store_true")
    parser.add_argument("--build-cache-if-missing", action="store_true", default=True)
    parser.add_argument("--no-build-cache-if-missing", dest="build_cache_if_missing", action="store_false")
    parser.add_argument("--cache-num-samples", type=int, default=4)
    parser.add_argument("--cache-max-raw-samples", type=int, default=20)
    parser.add_argument("--cache-max-tokens", type=int, default=4096)
    parser.add_argument("--allow-adapt-preset", action="store_true", default=True)
    parser.add_argument("--no-allow-adapt-preset", dest="allow_adapt_preset", action="store_false")

    parser.add_argument("--output-root", type=str, default="runs/gate_experiments")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", type=str, default=sys.executable)
    return parser.parse_args()


def run_command(command: list[str], dry_run: bool) -> dict[str, Any]:
    print(" ".join(command), flush=True)
    if dry_run:
        return {"returncode": 0, "dry_run": True}
    completed = subprocess.run(command, check=False)
    return {"returncode": int(completed.returncode), "dry_run": False}


def latest_jsonl_record(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    last = ""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last = line
    return json.loads(last) if last else {}


def make_cache_command(args) -> list[str]:
    command = [
        args.python,
        "scripts/c4_build_cache.py",
        "--num-samples",
        str(args.cache_num_samples),
        "--max-raw-samples",
        str(args.cache_max_raw_samples),
        "--max-tokens",
        str(args.cache_max_tokens),
        "--block-size",
        str(args.block_size),
        "--model-dim",
        str(args.model_dim),
        "--num-paths",
        str(args.num_paths),
        "--rank",
        str(args.rank),
        "--state-dim",
        str(args.state_dim),
        "--output-token-cache",
        args.token_cache,
        "--output-gate-cache",
        args.gate_cache,
        "--output-gate-feature-encoder",
        args.gate_feature_encoder,
        "--router-preset",
        args.router_preset,
    ]
    if args.allow_adapt_preset:
        command.append("--allow-adapt-preset")
    return command


def make_train_command(args, gate_mode: str, teacher_weight: float | None, output_dir: Path) -> list[str]:
    command = [
        args.python,
        "scripts/c4_train.py",
        "--gate-mode",
        gate_mode,
        "--max-train-steps",
        str(args.max_train_steps),
        "--batch-size",
        str(args.batch_size),
        "--block-size",
        str(args.block_size),
        "--model-dim",
        str(args.model_dim),
        "--state-dim",
        str(args.state_dim),
        "--num-paths",
        str(args.num_paths),
        "--rank",
        str(args.rank),
        "--num-layers",
        str(args.num_layers),
        "--router-topk",
        str(args.router_topk),
        "--target-active-paths",
        str(args.target_active_paths),
        "--gate-loss-weight",
        str(args.gate_loss_weight),
        "--gate-balance-weight",
        str(args.gate_balance_weight),
        "--gate-entropy-weight",
        str(args.gate_entropy_weight),
        "--gate-commitment-weight",
        str(args.gate_commitment_weight),
        "--lr",
        str(args.lr),
        "--seed",
        str(args.seed),
        "--token-cache",
        args.token_cache,
        "--gate-cache",
        args.gate_cache,
        "--gate-feature-encoder",
        args.gate_feature_encoder,
        "--router-preset",
        args.router_preset,
        "--output-dir",
        str(output_dir),
    ]
    if teacher_weight is not None:
        command.extend(["--gate-teacher-weight", str(teacher_weight)])
    if args.device:
        command.extend(["--device", args.device])
    if args.amp:
        command.append("--amp")
    return command


def main():
    args = parse_args()
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.output_root) / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    token_cache = Path(args.token_cache)
    gate_cache = Path(args.gate_cache)
    needs_gate_cache = any(mode in {"cached_snn", "hybrid_cached_plus_learned"} for mode in parse_csv(args.experiments))
    should_build_cache = args.build_cache or (
        args.build_cache_if_missing
        and (not token_cache.exists() or (needs_gate_cache and not gate_cache.exists()))
    )

    results: list[dict[str, Any]] = []
    if should_build_cache:
        result = run_command(make_cache_command(args), args.dry_run)
        results.append({"name": "build_cache", **result})
        if result["returncode"] != 0:
            raise SystemExit(result["returncode"])

    for gate_mode in parse_csv(args.experiments):
        teacher_weights: list[float | None]
        if gate_mode == "hybrid_cached_plus_learned":
            teacher_weights = parse_float_csv(args.teacher_weights)
        else:
            teacher_weights = [None]

        for teacher_weight in teacher_weights:
            suffix = gate_mode
            if teacher_weight is not None:
                suffix = f"{suffix}_teacher_{teacher_weight:g}"
            output_dir = run_root / suffix
            command = make_train_command(args, gate_mode, teacher_weight, output_dir)
            result = run_command(command, args.dry_run)
            metrics = latest_jsonl_record(output_dir / "train_metrics.jsonl") if not args.dry_run else {}
            results.append(
                {
                    "name": suffix,
                    "gate_mode": gate_mode,
                    "teacher_weight": teacher_weight,
                    "output_dir": str(output_dir),
                    "metrics": metrics,
                    **result,
                }
            )
            if result["returncode"] != 0:
                raise SystemExit(result["returncode"])

    summary = {
        "run_name": run_name,
        "output_root": str(run_root),
        "args": vars(args),
        "results": results,
    }
    summary_path = run_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")
    print("GATE_EXPERIMENTS_PASS")


if __name__ == "__main__":
    main()

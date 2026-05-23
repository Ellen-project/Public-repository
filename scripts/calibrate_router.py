from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np

from lrp_ssm.low_rank_path_ssm import FullSNNPathRouter, build_gate_cache


def parse_args():
    parser = argparse.ArgumentParser(description="Grid-search current_scale for FullSNNPathRouter.")
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--num-paths", type=int, default=8)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--state-dim", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--router-preset", type=str, default="best_router_preset.npz")
    parser.add_argument("--gate-cache", type=str, default="gate_cache.npy")
    parser.add_argument("--ssm-checkpoint", type=str, default="ssm_model.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--window-ms", type=float, default=0.1)
    parser.add_argument("--dt", type=float, default=0.025)
    parser.add_argument("--current-clip", type=float, default=1e-6)
    parser.add_argument("--scale-min", type=float, default=1e-9)
    parser.add_argument("--scale-max", type=float, default=1e-5)
    parser.add_argument("--scale-steps", type=int, default=5)
    parser.add_argument("--target-min", type=float, default=None)
    parser.add_argument("--target-max", type=float, default=None)
    parser.add_argument("--report", type=str, default="calibration_report.json")
    parser.add_argument("--best-preset", type=str, default="best_router_preset.npz")
    return parser.parse_args()


def target_range(num_paths: int, target_min: float | None, target_max: float | None):
    if target_min is not None and target_max is not None:
        return float(target_min), float(target_max)
    if num_paths <= 8:
        return 1.0, 2.0
    if num_paths <= 16:
        return 2.0, 4.0
    return max(1.0, num_paths * 0.125), max(2.0, num_paths * 0.25)


def metrics_for_gates(gates: np.ndarray):
    active = gates.sum(axis=-1)
    return {
        "mean_active_paths": float(active.mean()),
        "zero_gate_ratio": float((active == 0).mean()),
        "all_on_ratio": float((active == gates.shape[-1]).mean()),
        "path_firing_rate": gates.mean(axis=(0, 1)).astype(float).tolist(),
    }


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    x_dataset = rng.standard_normal((args.num_samples, args.seq_len, args.input_dim)).astype(np.float32)
    low, high = target_range(args.num_paths, args.target_min, args.target_max)
    scales = np.logspace(np.log10(args.scale_min), np.log10(args.scale_max), args.scale_steps)

    entries = []
    best_entry = None
    best_router = None
    best_score = float("inf")

    for idx, scale in enumerate(scales):
        router = FullSNNPathRouter(
            num_paths=args.num_paths,
            input_dim=args.input_dim,
            window_ms=args.window_ms,
            dt=args.dt,
            seed=args.seed + idx,
            current_scale=float(scale),
            current_clip=args.current_clip,
            n_basal=1,
            n_apical=1,
            n_tuft=1,
            record_traces=False,
            spike_times_max_len=0,
        )
        gates = build_gate_cache(router, x_dataset, use_ema=False, use_gpu=False, reset_each_sample=True)
        entry = {"current_scale": float(scale), **metrics_for_gates(gates)}
        entry["meets_target"] = low <= entry["mean_active_paths"] <= high
        entries.append(entry)

        if entry["meets_target"]:
            score = abs(entry["mean_active_paths"] - 0.5 * (low + high))
        else:
            score = min(abs(entry["mean_active_paths"] - low), abs(entry["mean_active_paths"] - high))
        if score < best_score:
            best_score = score
            best_entry = entry
            best_router = router

    assert best_router is not None and best_entry is not None
    best_router.save_preset(args.best_preset)

    report = {
        "target_active_paths": [low, high],
        "best": best_entry,
        "candidates": entries,
        "best_router_preset": str(Path(args.best_preset)),
    }
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Calibration complete. Best current_scale={best_entry['current_scale']:.6g}")
    print(f"Wrote {args.report} and {args.best_preset}")


if __name__ == "__main__":
    main()

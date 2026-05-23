from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import math
import time
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

from Low_Rank_Path_SSM import FullSNNPathRouter
from c4_lrp_lm import create_gate_feature_encoder, gate_statistics, load_gate_feature_encoder, torch_load


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate FullSNNPathRouter for a C4 gate feature model_dim.")
    parser.add_argument("--token-cache", type=str, default="c4_token_cache_medium.pt")
    parser.add_argument("--gate-feature-encoder", type=str, default=None)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--num-paths", type=int, default=4)
    parser.add_argument("--target-min-active", type=float, default=1.0)
    parser.add_argument("--target-max-active", type=float, default=2.0)
    parser.add_argument("--window-ms", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.025)
    parser.add_argument("--current-clip", type=float, default=1e-4)
    parser.add_argument("--scale-min", type=float, default=1e-4)
    parser.add_argument("--scale-max", type=float, default=1e-1)
    parser.add_argument("--scale-steps", type=int, default=9)
    parser.add_argument("--n-basal", type=int, default=1)
    parser.add_argument("--n-apical", type=int, default=1)
    parser.add_argument("--n-tuft", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-preset", type=str, default="best_router_preset_dim64.npz")
    parser.add_argument("--output-report", type=str, default="router_calibration_dim64_report.json")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def load_token_cache(path: str | Path):
    payload = torch_load(path, map_location="cpu")
    return payload["input_ids"].long(), dict(payload.get("metadata", {}))


def infer_gate_feature_path(token_cache: str | Path, model_dim: int) -> Optional[Path]:
    token_path = Path(token_cache)
    candidates = [
        token_path.with_name(token_path.name.replace("token_cache", "gate_feature_encoder")),
        token_path.with_name(f"c4_gate_feature_encoder_dim{model_dim}.pt"),
        token_path.with_name("c4_gate_feature_encoder_medium.pt"),
        Path(f"c4_gate_feature_encoder_dim{model_dim}.pt"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def make_features(input_ids: torch.Tensor, token_metadata: dict, args) -> tuple[torch.Tensor, dict]:
    feature_path = Path(args.gate_feature_encoder) if args.gate_feature_encoder else infer_gate_feature_path(args.token_cache, args.model_dim)
    if feature_path and feature_path.exists():
        encoder, payload = load_gate_feature_encoder(feature_path, map_location=args.device)
        if int(payload["model_dim"]) != int(args.model_dim):
            raise ValueError(f"gate_feature_encoder model_dim {payload['model_dim']} != requested {args.model_dim}")
        encoder_meta = {"gate_feature_encoder_path": str(feature_path), "created_new": False, "seed": payload.get("seed")}
    else:
        vocab_size = int(token_metadata.get("vocab_size", int(input_ids.max().item()) + 1))
        encoder = create_gate_feature_encoder(vocab_size, args.model_dim, seed=args.seed, device=args.device)
        encoder_meta = {"gate_feature_encoder_path": None, "created_new": True, "seed": int(args.seed)}
    with torch.no_grad():
        features = encoder(input_ids.to(args.device)).detach().cpu().float()
    return features, encoder_meta


def metrics_for_gates(gates: torch.Tensor) -> dict:
    stats = gate_statistics(gates)
    path_firing_rate = gates.float().mean(dim=(0, 1)).tolist()
    return {**stats, "path_firing_rate": [float(x) for x in path_firing_rate]}


def build_gates_for_scale(features: torch.Tensor, args, current_scale: float) -> tuple[torch.Tensor, FullSNNPathRouter]:
    router = FullSNNPathRouter(
        num_paths=args.num_paths,
        input_dim=args.model_dim,
        window_ms=args.window_ms,
        dt=args.dt,
        seed=args.seed,
        current_scale=float(current_scale),
        current_clip=args.current_clip,
        n_basal=args.n_basal,
        n_apical=args.n_apical,
        n_tuft=args.n_tuft,
        record_traces=False,
        spike_times_max_len=0,
    )
    gates = torch.zeros(features.shape[0], features.shape[1], args.num_paths, dtype=torch.float32)
    for n in range(features.shape[0]):
        router.reset_state()
        for t in range(features.shape[1]):
            gates[n, t] = torch.from_numpy(router.step(features[n, t].numpy()))
    return gates, router


def calibrate_router_from_token_cache(args) -> dict:
    input_ids, token_metadata = load_token_cache(args.token_cache)
    features, encoder_meta = make_features(input_ids, token_metadata, args)
    scales = torch.logspace(math.log10(args.scale_min), math.log10(args.scale_max), args.scale_steps).tolist()
    candidates = []
    best = None
    best_router = None
    best_score = float("inf")
    start_all = time.perf_counter()
    for scale in tqdm(scales, desc="router_calibration"):
        gates, router = build_gates_for_scale(features, args, float(scale))
        metrics = metrics_for_gates(gates)
        meets_target = args.target_min_active <= metrics["mean_active_paths"] <= args.target_max_active
        if meets_target:
            score = metrics["zero_gate_ratio"] + 0.1 * abs(metrics["mean_active_paths"] - 0.5 * (args.target_min_active + args.target_max_active))
        else:
            score = 10.0 + min(
                abs(metrics["mean_active_paths"] - args.target_min_active),
                abs(metrics["mean_active_paths"] - args.target_max_active),
            ) + metrics["zero_gate_ratio"]
        entry = {
            "current_scale": float(scale),
            "input_dim": int(args.model_dim),
            "model_dim": int(args.model_dim),
            "num_paths": int(args.num_paths),
            "meets_target": bool(meets_target),
            **metrics,
        }
        candidates.append(entry)
        if score < best_score:
            best_score = score
            best = entry
            best_router = router
    if best_router is None or best is None:
        raise RuntimeError("router calibration failed to produce a candidate")
    best_router.save_preset(args.output_preset)
    report = {
        "status": "PASS" if best["meets_target"] else "WARN_TARGET_NOT_MET",
        "token_cache": str(args.token_cache),
        "output_preset": str(args.output_preset),
        "target_active_paths": [float(args.target_min_active), float(args.target_max_active)],
        "best": best,
        "candidates": candidates,
        "encoder": encoder_meta,
        "elapsed_sec": max(time.perf_counter() - start_all, 1e-9),
    }
    Path(args.output_report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    args = parse_args()
    report = calibrate_router_from_token_cache(args)
    print(json.dumps(report["best"], indent=2))
    print("LRP_ROUTER_CALIBRATION_PASS")


if __name__ == "__main__":
    main()

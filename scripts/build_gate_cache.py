from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse

import numpy as np

from lrp_ssm.low_rank_path_ssm import (
    FullSNNPathRouter,
    build_gate_cache,
    gate_cache_metadata,
    save_gate_cache,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build a gate cache from FullSNNPathRouter.")
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
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--reset-each-sample", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    x_dataset = rng.standard_normal((args.num_samples, args.seq_len, args.input_dim)).astype(np.float32)

    if args.router_preset:
        try:
            router = FullSNNPathRouter.load_preset(args.router_preset)
            router_preset_path = args.router_preset
        except FileNotFoundError:
            router = FullSNNPathRouter(
                num_paths=args.num_paths,
                input_dim=args.input_dim,
                window_ms=args.window_ms,
                dt=args.dt,
                seed=args.seed,
                n_basal=1,
                n_apical=1,
                n_tuft=1,
                record_traces=False,
                spike_times_max_len=0,
            )
            router_preset_path = None
    else:
        router = FullSNNPathRouter(
            num_paths=args.num_paths,
            input_dim=args.input_dim,
            window_ms=args.window_ms,
            dt=args.dt,
            seed=args.seed,
            n_basal=1,
            n_apical=1,
            n_tuft=1,
            record_traces=False,
            spike_times_max_len=0,
        )
        router_preset_path = None

    gates = build_gate_cache(
        router,
        x_dataset,
        use_ema=args.use_ema,
        use_gpu=args.use_gpu,
        reset_each_sample=args.reset_each_sample,
    )
    metadata = gate_cache_metadata(
        gates,
        input_dim=x_dataset.shape[-1],
        router=router,
        use_ema=args.use_ema,
        use_gpu=args.use_gpu,
        router_preset_path=router_preset_path,
    )
    save_gate_cache(gates, args.gate_cache, metadata)
    print(f"Wrote {args.gate_cache} with shape {gates.shape}")


if __name__ == "__main__":
    main()

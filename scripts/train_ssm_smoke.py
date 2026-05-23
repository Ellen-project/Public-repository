from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from Low_Rank_Path_SSM import (
    CachedGateLowRankPathSSM,
    FullSNNPathRouter,
    LowRankPathSSMModel,
    build_gate_cache,
    gate_cache_metadata,
    load_gate_cache,
    load_ssm_checkpoint,
    save_gate_cache,
    save_ssm_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description="SSM training smoke test with cached gates.")
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--num-paths", type=int, default=4)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--router-preset", type=str, default="best_router_preset.npz")
    parser.add_argument("--gate-cache", type=str, default="gate_cache.npy")
    parser.add_argument("--ssm-checkpoint", type=str, default="ssm_model.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    x_np = rng.standard_normal((args.num_samples, args.seq_len, args.input_dim)).astype(np.float32)
    x = torch.from_numpy(x_np).to(device)
    target = torch.tanh(x[..., :1].repeat(1, 1, args.input_dim)).to(device)

    try:
        gates_np, _ = load_gate_cache(args.gate_cache)
        if gates_np.shape != (args.num_samples, args.seq_len, args.num_paths):
            raise ValueError("gate cache shape mismatch")
    except (FileNotFoundError, ValueError):
        try:
            router = FullSNNPathRouter.load_preset(args.router_preset)
        except FileNotFoundError:
            router = FullSNNPathRouter(
                num_paths=args.num_paths,
                input_dim=args.input_dim,
                window_ms=0.05,
                dt=0.025,
                seed=args.seed,
                n_basal=1,
                n_apical=1,
                n_tuft=1,
                record_traces=False,
                spike_times_max_len=0,
            )
        gates_np = build_gate_cache(router, x_np, reset_each_sample=True)
        metadata = gate_cache_metadata(
            gates_np,
            input_dim=args.input_dim,
            router=router,
            router_preset_path=args.router_preset,
        )
        save_gate_cache(gates_np, args.gate_cache, metadata)

    gates = torch.from_numpy(gates_np).to(device=device, dtype=x.dtype)
    model = LowRankPathSSMModel(
        input_dim=args.input_dim,
        state_dim=args.state_dim,
        num_paths=args.num_paths,
        rank=args.rank,
        output_dim=args.input_dim,
        router_kwargs={
            "window_ms": 0.05,
            "dt": 0.025,
            "seed": args.seed,
            "n_basal": 1,
            "n_apical": 1,
            "n_tuft": 1,
            "record_traces": False,
            "spike_times_max_len": 0,
        },
    ).to(device)
    cached_model = CachedGateLowRankPathSSM(model.ssm_core)
    optimizer = torch.optim.Adam(model.ssm_core.parameters(), lr=args.lr)

    last_loss = None
    for _ in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        y = cached_model(x, gates)
        loss = F.mse_loss(y, target)
        assert torch.isfinite(loss), loss
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())

    save_ssm_checkpoint(model, args.ssm_checkpoint)
    loaded = load_ssm_checkpoint(
        args.ssm_checkpoint,
        router_kwargs={
            "window_ms": 0.05,
            "dt": 0.025,
            "seed": args.seed,
            "n_basal": 1,
            "n_apical": 1,
            "n_tuft": 1,
            "record_traces": False,
            "spike_times_max_len": 0,
        },
        map_location=args.device,
    ).to(device)
    loaded_cached = CachedGateLowRankPathSSM(loaded.ssm_core)
    y_loaded = loaded_cached(x, gates)
    assert y_loaded.shape == y.shape
    assert torch.isfinite(y_loaded).all()

    print(f"Training smoke PASS. final_loss={last_loss:.6g}; wrote {args.ssm_checkpoint}")


if __name__ == "__main__":
    main()

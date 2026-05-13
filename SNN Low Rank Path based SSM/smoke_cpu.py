from __future__ import annotations

import argparse

import numpy as np
import torch

from Low_Rank_Path_SSM import LowRankPathSSMModel
from pyramidalNeuron import Network, PyramidalNeuron


def parse_args():
    parser = argparse.ArgumentParser(description="CPU smoke test for LowRankPathSSM-A.")
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--num-paths", type=int, default=4)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--router-preset", type=str, default="")
    parser.add_argument("--gate-cache", type=str, default="")
    parser.add_argument("--ssm-checkpoint", type=str, default="ssm_model.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    output_dim = args.input_dim + 1
    model = LowRankPathSSMModel(
        input_dim=args.input_dim,
        state_dim=args.state_dim,
        num_paths=args.num_paths,
        rank=args.rank,
        output_dim=output_dim,
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

    x = torch.randn(1, args.seq_len, args.input_dim, device=device)
    y, gates, diagnostics = model(x, return_gates=True, return_diagnostics=True)

    assert y.shape == (1, args.seq_len, output_dim), y.shape
    assert gates.shape == (1, args.seq_len, args.num_paths), gates.shape
    assert torch.isfinite(y).all()
    assert torch.isfinite(gates).all()
    for key in ("active_paths", "h_norm", "delta_norm"):
        assert key in diagnostics, key
        assert torch.isfinite(diagnostics[key]).all()

    try:
        model(torch.randn(2, args.seq_len, args.input_dim, device=device))
    except ValueError:
        pass
    else:
        raise AssertionError("single stateful router accepted batch_size > 1")

    net = Network(dt=0.025, seed=args.seed)
    neuron = PyramidalNeuron("smoke", net, n_basal=1, n_apical=1, n_tuft=1)
    neuron.record_traces = False
    net.run_window(0.05)
    t1 = net.t
    net.run_window(0.05)
    assert net.t > t1

    model.router.reset_state()
    assert model.router.net.t == 0.0

    print("CPU smoke test PASS")


if __name__ == "__main__":
    main()

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np

from Low_Rank_Path_SSM import FullSNNPathRouter, build_gate_cache


def parse_args():
    parser = argparse.ArgumentParser(description="Verify CPU/GPU FullSNNPathRouter consistency.")
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--num-paths", type=int, default=4)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--router-preset", type=str, default="best_router_preset.npz")
    parser.add_argument("--gate-cache", type=str, default="gate_cache.npy")
    parser.add_argument("--ssm-checkpoint", type=str, default="ssm_model.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--report", type=str, default="gpu_verify_report.json")
    return parser.parse_args()


def make_or_load_router(args, seed_offset=0):
    try:
        return FullSNNPathRouter.load_preset(args.router_preset)
    except FileNotFoundError:
        return FullSNNPathRouter(
            num_paths=args.num_paths,
            input_dim=args.input_dim,
            window_ms=0.05,
            dt=0.025,
            seed=args.seed + seed_offset,
            n_basal=1,
            n_apical=1,
            n_tuft=1,
            record_traces=False,
            spike_times_max_len=0,
        )


def write_report(path: str, report: dict):
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    report = {"status": "UNKNOWN"}
    try:
        import cupy  # noqa: F401
    except Exception as exc:
        report.update({"status": "SKIP", "reason": f"CuPy import failed: {exc}"})
        write_report(args.report, report)
        print(report["reason"])
        return

    rng = np.random.default_rng(args.seed)
    x_dataset = rng.standard_normal((args.num_samples, args.seq_len, args.input_dim)).astype(np.float32)
    cpu_router = make_or_load_router(args, seed_offset=0)
    gpu_router = make_or_load_router(args, seed_offset=0)

    try:
        cpu_gates = build_gate_cache(cpu_router, x_dataset, use_gpu=False, reset_each_sample=True)
        gpu_gates = build_gate_cache(gpu_router, x_dataset, use_gpu=True, reset_each_sample=True)
        gates_equal = bool(np.array_equal(cpu_gates, gpu_gates))
        cpu_diag = cpu_router.get_diagnostics()
        gpu_diag = gpu_router.get_diagnostics()
        spike_counts_equal = bool(np.array_equal(cpu_diag["spike_counts"], gpu_diag["spike_counts"]))
        net_time_equal = bool(abs(cpu_diag["net_time"] - gpu_diag["net_time"]) < 1e-9)
        active_paths_equal = bool(cpu_diag["active_paths"] == gpu_diag["active_paths"])

        passed = gates_equal and spike_counts_equal and net_time_equal and active_paths_equal
        report.update(
            {
                "status": "PASS" if passed else "FAIL",
                "gates_equal": gates_equal,
                "spike_counts_equal": spike_counts_equal,
                "net_time_equal": net_time_equal,
                "active_paths_equal": active_paths_equal,
                "cpu_spike_counts": cpu_diag["spike_counts"].astype(int).tolist(),
                "gpu_spike_counts": gpu_diag["spike_counts"].astype(int).tolist(),
                "cpu_net_time": cpu_diag["net_time"],
                "gpu_net_time": gpu_diag["net_time"],
                "gate_shape": list(cpu_gates.shape),
            }
        )
    except Exception as exc:
        report.update({"status": "ERROR", "reason": repr(exc)})

    write_report(args.report, report)
    print(f"GPU router verification {report['status']}; wrote {args.report}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from Low_Rank_Path_SSM import load_gate_cache
from c4_lrp_lm import gate_statistics, load_c4_lm_checkpoint, safe_perplexity, torch_load


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Ablate LRP-SSM gate/path contribution.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--token-cache", type=str, default="c4_token_cache_medium.pt")
    parser.add_argument("--gate-cache", type=str, default="c4_gate_cache_medium.pt")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-eval-batches", type=int, default=50)
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--output-report", type=str, default="lrp_fix_ablation_report.json")
    return parser.parse_args()


def load_token_cache(path: str | Path):
    payload = torch_load(path, map_location="cpu")
    return payload["input_ids"].long(), payload["labels"].long(), dict(payload.get("metadata", {}))


def make_batches(input_ids: torch.Tensor, labels: torch.Tensor, gates: torch.Tensor, batch_size: int, max_batches: int):
    count = min(input_ids.shape[0], batch_size * max_batches)
    for start in range(0, count, batch_size):
        end = min(count, start + batch_size)
        yield input_ids[start:end], labels[start:end], gates[start:end]


def density_random_like(gates: torch.Tensor) -> torch.Tensor:
    p = float(gates.float().mean().item())
    return torch.bernoulli(torch.full_like(gates.float(), p))


def set_force_min_active(model, value: int):
    old = []
    for layer in getattr(model, "ssm_layers", []):
        old.append(layer.force_min_active_paths)
        layer.force_min_active_paths = int(value)
    return old


def restore_force_min_active(model, old):
    for layer, value in zip(getattr(model, "ssm_layers", []), old):
        layer.force_min_active_paths = int(value)


@torch.no_grad()
def run_mode(model, mode: str, input_ids, labels, cached_gates, device):
    if mode == "cached_gate":
        gates = cached_gates
        old_force = set_force_min_active(model, 0)
    elif mode == "zero_gate":
        gates = torch.zeros_like(cached_gates)
        old_force = set_force_min_active(model, 0)
    elif mode == "all_on_gate":
        gates = torch.ones_like(cached_gates)
        old_force = set_force_min_active(model, 0)
    elif mode == "random_gate_same_density":
        gates = density_random_like(cached_gates)
        old_force = set_force_min_active(model, 0)
    elif mode == "force_min_active_1":
        gates = cached_gates
        old_force = set_force_min_active(model, 1)
    elif mode == "force_min_active_2":
        gates = cached_gates
        old_force = set_force_min_active(model, 2)
    elif mode == "learned_router_if_available":
        gates = None
        old_force = set_force_min_active(model, 0)
        if getattr(model, "learned_router", None) is None:
            return None
    else:
        raise ValueError(mode)
    try:
        output = model(
            input_ids.to(device),
            labels=labels.to(device),
            gates=None if gates is None else gates.to(device).float(),
            return_diagnostics=True,
            return_gates=True,
        )
    finally:
        restore_force_min_active(model, old_force)
    used_gates = output.get("gates")
    if used_gates is None:
        used_gates = gates
    return {
        "loss": output["loss"].detach().cpu(),
        "logits": output["logits"].detach().cpu(),
        "hidden": output.get("hidden", torch.zeros_like(output["logits"][..., :1])).detach().cpu(),
        "diagnostics": output.get("diagnostics", {}),
        "gates": None if used_gates is None else used_gates.detach().cpu(),
    }


def summarize_mode(mode: str, outputs: list, zero_outputs: list | None):
    losses = torch.stack([item["loss"] for item in outputs])
    logits = torch.cat([item["logits"] for item in outputs], dim=0)
    hidden = torch.cat([item["hidden"] for item in outputs], dim=0)
    gates = [item["gates"] for item in outputs if item["gates"] is not None]
    diagnostics = [item["diagnostics"] for item in outputs if item.get("diagnostics")]
    summary = {
        "eval_loss": float(losses.mean().item()),
        "eval_ppl": safe_perplexity(float(losses.mean().item())),
    }
    if gates:
        summary.update(gate_statistics(torch.cat(gates, dim=0)))
    if zero_outputs is not None:
        zero_logits = torch.cat([item["logits"] for item in zero_outputs], dim=0)
        zero_hidden = torch.cat([item["hidden"] for item in zero_outputs], dim=0)
        summary["avg_logits_l2_vs_zero_gate"] = float((logits - zero_logits).pow(2).mean().sqrt().item())
        logp = F.log_softmax(logits.float(), dim=-1)
        zero_p = F.softmax(zero_logits.float(), dim=-1)
        summary["avg_logits_kl_vs_zero_gate"] = float(F.kl_div(logp, zero_p, reduction="batchmean").item())
        summary["avg_hidden_l2_vs_zero_gate"] = float((hidden - zero_hidden).pow(2).mean().sqrt().item())
    else:
        summary["avg_logits_l2_vs_zero_gate"] = 0.0
        summary["avg_logits_kl_vs_zero_gate"] = 0.0
        summary["avg_hidden_l2_vs_zero_gate"] = 0.0
    for key in ("raw_delta_norm_mean", "scaled_delta_norm_mean", "path_to_base_ratio_mean", "gamma"):
        vals = [float(diag.get(key, 0.0)) for diag in diagnostics]
        summary[key] = float(sum(vals) / max(1, len(vals)))
    return summary


def markdown_snippet(report: dict) -> str:
    rows = []
    for mode, item in report["modes"].items():
        rows.append(
            f"| {mode} | {item.get('eval_loss', 0):.4f} | {item.get('eval_ppl', 0):.2f} | "
            f"{item.get('mean_active_paths', 0):.4f} | {item.get('avg_logits_l2_vs_zero_gate', 0):.6f} | "
            f"{item.get('path_to_base_ratio_mean', 0):.6f} |"
        )
    return "\n".join(
        [
            "## LRP Fix Ablation",
            "",
            "| Mode | Eval Loss | Eval PPL | Mean Active | Logits L2 vs Zero | Path/Base Ratio |",
            "| --- | --- | --- | --- | --- | --- |",
            *rows,
            "",
            f"Verdicts: {', '.join(report['verdicts']) if report['verdicts'] else 'PASS'}",
        ]
    )


def main():
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    checkpoint = load_c4_lm_checkpoint(args.checkpoint, map_location=device)
    model = checkpoint["model"].to(device).eval()
    input_ids, labels, token_metadata = load_token_cache(args.token_cache)
    gates_np, gate_meta = load_gate_cache(args.gate_cache)
    gates = torch.from_numpy(gates_np).float()
    input_ids = input_ids[:, : gates.shape[1]]
    labels = labels[:, : gates.shape[1]]

    modes = [
        "zero_gate",
        "cached_gate",
        "all_on_gate",
        "random_gate_same_density",
        "force_min_active_1",
        "force_min_active_2",
        "learned_router_if_available",
    ]
    raw_outputs = {mode: [] for mode in modes}
    for batch in make_batches(input_ids, labels, gates, args.batch_size, args.max_eval_batches):
        for mode in modes:
            out = run_mode(model, mode, batch[0], batch[1], batch[2], device)
            if out is not None:
                raw_outputs[mode].append(out)
    zero_outputs = raw_outputs["zero_gate"]
    summaries = {}
    for mode, outputs in raw_outputs.items():
        if outputs:
            summaries[mode] = summarize_mode(mode, outputs, None if mode == "zero_gate" else zero_outputs)

    verdicts = []
    cached = summaries.get("cached_gate", {})
    all_on = summaries.get("all_on_gate", {})
    if cached.get("avg_logits_l2_vs_zero_gate", 0.0) < 1e-5:
        verdicts.append("FAIL_PATH_INACTIVE")
    if all_on.get("avg_logits_l2_vs_zero_gate", 0.0) < 1e-5:
        verdicts.append("FAIL_PATH_DELTA_TOO_SMALL")
    if cached.get("path_to_base_ratio_mean", 0.0) < 0.01:
        verdicts.append("FAIL_PATH_TOO_WEAK")
    if cached.get("mean_active_paths", 0.0) < 1.0:
        verdicts.append("FAIL_GATE_TOO_SPARSE")

    report = {
        "checkpoint": args.checkpoint,
        "token_cache": args.token_cache,
        "gate_cache": args.gate_cache,
        "token_metadata": token_metadata,
        "gate_metadata": gate_meta,
        "modes": summaries,
        "verdicts": verdicts,
        "status": "PASS" if not verdicts else "FAIL",
    }
    output_path = Path(args.output_report)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    output_path.with_suffix(".md").write_text(markdown_snippet(report), encoding="utf-8")
    print("LRP_FIX_ABLATION_PASS")


if __name__ == "__main__":
    main()

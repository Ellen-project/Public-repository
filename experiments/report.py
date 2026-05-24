from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import math
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch


LEVEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LEVEL_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_one import level_path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate experiment Markdown report.")
    parser.add_argument("--results-dir", type=str, default="experiments/results")
    parser.add_argument("--runs-dir", type=str, default="experiments/runs")
    parser.add_argument("--output", type=str, default="experiments/TEST_RESULTS/REPORT.md")
    parser.add_argument("--include-raw-tables", action="store_true")
    parser.add_argument("--include-lrp-fix-analysis", action="store_true")
    parser.add_argument("--ablation-report", type=str, default=None)
    parser.add_argument("--old-report", type=str, default=None)
    return parser.parse_args()


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if value == float("inf"):
            return "inf"
        return f"{value:.{digits}f}"
    return str(value)


def table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(fmt(cell) for cell in row) + " |")
    return "\n".join(out)


def model_kind(model: str) -> str:
    return {
        "lrp_ssm": "Low-rank path SSM",
        "lrp_ssm_fixed_calibrated": "Low-rank path SSM fixed calibrated",
        "lrp_ssm_learned_router": "Low-rank path SSM learned router",
        "lrp_ssm_hybrid": "Low-rank path SSM hybrid",
        "transformer": "Full causal attention",
        "linear_attention": "Kernelized linear attention",
        "local_attention": "Sliding-window attention",
        "gru": "Recurrent",
    }.get(model, model)


def complexity(model: str, cfg: dict) -> str:
    if model == "transformer":
        return "O(T^2)"
    if model == "local_attention":
        return f"O(T*w), w={cfg.get('window_size', 'n/a')}"
    if model == "linear_attention":
        return "O(T), simplified ELU+1 prefix"
    if model == "gru":
        return "O(T) recurrent"
    if model.startswith("lrp_ssm"):
        return "O(T*paths*rank)"
    return "n/a"


def load_ablation_reports() -> list[tuple[str, dict]]:
    candidates = [
        ("cached_gate", PROJECT_ROOT / "c4_eval_smoke_report.json"),
        ("zero_gate", PROJECT_ROOT / "c4_eval_zero_gate_smoke.json"),
        ("all_on_gate", PROJECT_ROOT / "c4_eval_all_on_gate_smoke.json"),
        ("random_gate", PROJECT_ROOT / "c4_eval_random_gate_smoke.json"),
        ("distilled_router", PROJECT_ROOT / "c4_eval_distilled_router_smoke.json"),
    ]
    reports = []
    for name, path in candidates:
        payload = read_json(path)
        if payload:
            reports.append((name, payload))
    return reports


def read_torch_metadata(path_text: str | None):
    if not path_text:
        return {}
    candidates = [PROJECT_ROOT / path_text, LEVEL_DIR / path_text, level_path(path_text)]
    for path in candidates:
        try:
            if path.exists():
                try:
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                except TypeError:
                    payload = torch.load(path, map_location="cpu")
                return dict(payload.get("metadata", {}))
        except Exception:
            continue
    return {}


def latest_summary_for_prefix(runs_dir: Path, prefix: str):
    candidates = sorted(runs_dir.glob(f"{prefix}_*/summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        payload = read_json(path)
        if payload:
            return payload
    return {}


def generate_report(args) -> Path:
    results_dir = level_path(args.results_dir)
    runs_dir = level_path(args.runs_dir)
    output_path = level_path(args.output)
    comparison = read_json(results_dir / "comparison_summary.json", default={})
    models = comparison.get("models", {})
    experiment = comparison.get("experiment", {})
    environment = comparison.get("environment", {})
    generated_at = datetime.now().isoformat(timespec="seconds")

    lines = ["# Low Rank Path based SSM vs Attention Baselines", ""]
    lines += [
        "## 1. 실험 목적",
        "",
        "- Low Rank Path based SSM과 attention 계열 baseline의 language modeling 성능 및 효율 비교.",
        "- C4 cache 기반 동일 데이터 조건에서 비교.",
        "",
        "## 2. 실험 환경",
        "",
        table(
            ["항목", "값"],
            [
                ["Python", environment.get("python", platform.python_version())],
                ["PyTorch", environment.get("torch", torch.__version__)],
                ["CUDA available", environment.get("cuda_available", torch.cuda.is_available())],
                ["GPU", environment.get("gpu_name")],
                ["device", experiment.get("device", environment.get("device"))],
                ["AMP", experiment.get("amp")],
                ["seed", experiment.get("seed")],
                ["token cache", experiment.get("token_cache")],
                ["gate cache", experiment.get("gate_cache")],
            ],
        ),
        "",
        "## 3. 비교 모델",
        "",
    ]
    model_rows = []
    for name, summary in models.items():
        cfg = summary.get("config", {})
        model_rows.append(
            [
                name,
                model_kind(name),
                summary.get("parameter_count"),
                "yes" if name.startswith("lrp_ssm") else "no",
                complexity(name, cfg),
                "cached/learned gates" if name.startswith("lrp_ssm") else ("simplified baseline" if name == "linear_attention" else ""),
            ]
        )
    lines += [table(["Model", "Type", "Params", "Uses Gate", "Complexity", "Notes"], model_rows), ""]

    cfg0 = next((summary.get("config", {}) for summary in models.values() if isinstance(summary, dict)), {})
    lines += [
        "## 4. 학습 설정",
        "",
        table(
            ["항목", "값"],
            [
                ["block_size", cfg0.get("max_seq_len", experiment.get("block_size"))],
                ["batch_size", experiment.get("batch_size")],
                ["max_train_steps", experiment.get("max_train_steps")],
                ["optimizer", "AdamW"],
                ["lr", experiment.get("lr")],
                ["weight_decay", experiment.get("weight_decay")],
                ["grad_clip", experiment.get("grad_clip")],
                ["model_dim", experiment.get("model_dim", cfg0.get("model_dim"))],
                ["num_layers", experiment.get("num_layers", cfg0.get("num_layers"))],
                ["num_heads", experiment.get("num_heads", cfg0.get("num_heads"))],
                ["state_dim", experiment.get("state_dim", cfg0.get("state_dim"))],
                ["rank", experiment.get("rank", cfg0.get("rank"))],
                ["num_paths", experiment.get("num_paths", cfg0.get("num_paths"))],
            ],
        ),
        "",
        "## 5. 결과 요약",
        "",
    ]
    result_rows = []
    for name, summary in models.items():
        result_rows.append(
            [
                name,
                summary.get("parameter_count"),
                summary.get("best_eval_loss"),
                summary.get("best_eval_ppl"),
                summary.get("final_train_loss"),
                summary.get("avg_train_tokens_per_sec"),
                summary.get("avg_eval_tokens_per_sec"),
                summary.get("peak_gpu_memory_mb"),
                summary.get("status"),
            ]
        )
    lines += [
        table(
            [
                "Model",
                "Params",
                "Best Eval Loss",
                "Best Eval PPL",
                "Final Train Loss",
                "Avg Train Tokens/s",
                "Avg Eval Tokens/s",
                "Peak GPU Memory MB",
                "Status",
            ],
            result_rows,
        ),
        "",
        "## 6. 성능 벤치마크",
        "",
    ]
    bench_rows = []
    for name, summary in models.items():
        run_dir = Path(summary.get("run_dir", ""))
        benchmark = read_json(run_dir / "benchmark.json", default={})
        for item in benchmark.get("results", []):
            bench_rows.append(
                [
                    name,
                    item.get("batch_size"),
                    item.get("block_size"),
                    item.get("forward_avg_ms"),
                    item.get("tokens_per_sec"),
                    item.get("train_tokens_per_sec"),
                    item.get("peak_gpu_memory_mb"),
                    item.get("status"),
                ]
            )
    lines += [
        table(
            ["Model", "Batch Size", "Block Size", "Forward ms", "Tokens/s", "Train Tokens/s", "Peak Memory MB", "Status"],
            bench_rows or [["not available", None, None, None, None, None, None, "not available"]],
        ),
        "",
        "## 7. Gate 분석, LRP-SSM 전용",
        "",
    ]
    lrp = models.get("lrp_ssm_fixed_calibrated") or models.get("lrp_ssm", {})
    calibration = read_json(PROJECT_ROOT / "calibration_report.json", default={})
    gate_rows = [
        ["mean_active_paths", lrp.get("mean_active_paths")],
        ["zero_gate_ratio", lrp.get("zero_gate_ratio")],
        ["all_on_gate_ratio", lrp.get("all_on_gate_ratio")],
        ["calibration_current_scale", calibration.get("best", {}).get("current_scale")],
        ["target_active_paths", calibration.get("target_active_paths")],
    ]
    lines += [
        table(["Metric", "Value"], gate_rows),
        "",
        "cached gate가 sparse할수록 SSM update의 path 사용량이 줄어든다. 이 비교는 gate cache를 고정한 조건이므로 online FullSNN router 비용과는 분리해서 해석해야 한다.",
        "",
        "## 8. Ablation 권장",
        "",
    ]
    ablation_reports = load_ablation_reports()
    if ablation_reports:
        lines.append(
            table(
                ["Mode", "Eval Loss", "Eval PPL", "Mean Active Paths", "Zero Gate Ratio", "All On Gate Ratio"],
                [
                    [
                        name,
                        payload.get("eval_loss"),
                        payload.get("eval_ppl"),
                        payload.get("mean_active_paths"),
                        payload.get("zero_gate_ratio"),
                        payload.get("all_on_gate_ratio"),
                    ]
                    for name, payload in ablation_reports
                ],
            )
        )
    else:
        lines.append("Ablation reports are not available.")
    lines += [
        "",
        "## 9. 해석",
        "",
    ]
    ranking = comparison.get("ranking", {})
    best_ppl = ranking.get("best_eval_ppl", [])
    best_speed = ranking.get("best_tokens_per_sec", [])
    low_mem = ranking.get("lowest_memory", [])
    lines += [
        f"- Eval perplexity 기준 최상위 모델: {best_ppl[0][0] if best_ppl else 'not available'}.",
        f"- Train throughput 기준 최상위 모델: {best_speed[0][0] if best_speed else 'not available'}.",
        f"- Peak memory 기준 최상위 모델: {low_mem[0][0] if low_mem else 'not available'}.",
        "- LRP-SSM의 gate sparsity 기여는 cached/zero/all_on/random gate ablation을 함께 보아야 한다.",
        "- Local attention은 제한된 receptive field로 memory를 줄일 수 있고, linear attention은 단순화된 prefix baseline이라 full attention과 품질 차이가 날 수 있다.",
        "- 기본 보고서는 동일 hidden-size 조건이며 parameter count가 완전히 같지는 않다.",
        "",
        "## 10. 한계",
        "",
        "- C4 전체가 아니라 cache subset 기준 결과다.",
        "- cached gate는 고정 gate이므로 end-to-end learned routing이 아니다.",
        "- LRP-SSM과 Transformer의 parameter count가 완전히 동일하지 않을 수 있다.",
        "- linear attention 구현은 simplified baseline이다.",
        "- online FullSNN router runtime은 별도 비용이며 cached gate 결과와 구분해야 한다.",
        "",
        "## 11. 결론",
        "",
        f"- 가장 좋은 eval perplexity 모델: {best_ppl[0][0] if best_ppl else 'not available'}.",
        f"- 가장 빠른 모델: {best_speed[0][0] if best_speed else 'not available'}.",
        f"- memory 효율 모델: {low_mem[0][0] if low_mem else 'not available'}.",
        "- LRP-SSM은 cached SNN gate 기반 recurrent state update 계열로, attention baseline과 다른 효율/품질 tradeoff를 보인다.",
        "- 다음 개선점은 parameter-matched 설정, 더 큰 C4 cache, online router 비용 별도 측정, learned/distilled router 비교다.",
        "",
        f"Generated at: {generated_at}",
        f"Raw results: `{results_dir}`",
        f"Runs: `{runs_dir}`",
    ]
    if args.include_lrp_fix_analysis:
        ablation = read_json(level_path(args.ablation_report)) if args.ablation_report else None
        old_text_available = bool(args.old_report and level_path(args.old_report).exists())
        old_lrp = models.get("lrp_ssm", {})
        fixed_lrp = models.get("lrp_ssm_fixed_calibrated", {})
        learned_lrp = models.get("lrp_ssm_learned_router", {})
        hybrid_lrp = models.get("lrp_ssm_hybrid", {})
        old_gate_meta = read_torch_metadata("experiments/c4_gate_cache_medium.pt")
        new_gate_meta = read_torch_metadata(experiment.get("gate_cache"))
        fix_rows = []
        for name in ("lrp_ssm", "lrp_ssm_fixed_calibrated", "lrp_ssm_learned_router", "lrp_ssm_hybrid"):
            if name in models:
                item = models[name]
                fix_rows.append([
                    name,
                    item.get("mean_active_paths"),
                    item.get("zero_gate_ratio"),
                    item.get("all_on_gate_ratio"),
                    item.get("best_eval_loss"),
                    item.get("best_eval_ppl"),
                    item.get("avg_train_tokens_per_sec"),
                    item.get("peak_gpu_memory_mb"),
                    item.get("status"),
                ])
        path_rows = []
        if ablation:
            for mode, item in ablation.get("modes", {}).items():
                path_rows.append([
                    mode,
                    item.get("gamma"),
                    item.get("raw_delta_norm_mean"),
                    item.get("scaled_delta_norm_mean"),
                    item.get("path_to_base_ratio_mean"),
                    item.get("avg_logits_l2_vs_zero_gate"),
                ])
        lines += [
            "",
            "## LRP Fix Analysis",
            "",
            "기존 LRP 문제는 gate sparsity, router input_dim mismatch, 작은 path contribution, 단일 layer capacity, fixed random gate였다.",
            f"Old report available: {old_text_available}",
            "",
            "### Before vs After Gate Stats",
            "",
            table(
                ["Case", "Mean Active", "Zero Ratio", "All On Ratio", "Router Input Dim", "Model Dim", "Current Scale"],
                [
                    [
                        "before_lrp_ssm_old",
                        old_gate_meta.get("mean_active_paths", old_lrp.get("mean_active_paths")),
                        old_gate_meta.get("zero_gate_ratio", old_lrp.get("zero_gate_ratio")),
                        old_gate_meta.get("all_on_gate_ratio", old_lrp.get("all_on_gate_ratio")),
                        old_gate_meta.get("router_input_dim", old_gate_meta.get("model_dim")),
                        old_gate_meta.get("model_dim"),
                        old_gate_meta.get("current_scale"),
                    ],
                    [
                        "after_fixed_calibrated",
                        fixed_lrp.get("mean_active_paths", new_gate_meta.get("mean_active_paths")),
                        fixed_lrp.get("zero_gate_ratio", new_gate_meta.get("zero_gate_ratio")),
                        fixed_lrp.get("all_on_gate_ratio", new_gate_meta.get("all_on_gate_ratio")),
                        new_gate_meta.get("router_input_dim"),
                        new_gate_meta.get("model_dim"),
                        new_gate_meta.get("current_scale"),
                    ],
                    [
                        "after_learned_router",
                        learned_lrp.get("mean_active_paths"),
                        learned_lrp.get("zero_gate_ratio"),
                        learned_lrp.get("all_on_gate_ratio"),
                        "learned",
                        learned_lrp.get("config", {}).get("model_dim"),
                        "n/a",
                    ],
                    [
                        "after_hybrid",
                        hybrid_lrp.get("mean_active_paths"),
                        hybrid_lrp.get("zero_gate_ratio"),
                        hybrid_lrp.get("all_on_gate_ratio"),
                        "cached+learned",
                        hybrid_lrp.get("config", {}).get("model_dim"),
                        "n/a",
                    ],
                ],
            ),
            "",
            "### Before vs After Path Contribution",
            "",
            table(
                ["Mode", "Gamma", "Raw Delta", "Scaled Delta", "Path/Base", "Logits L2 vs Zero"],
                path_rows or [["not available", None, None, None, None, None]],
            ),
            "",
            "### Before vs After LM Quality",
            "",
            table(
                ["Case", "Train Loss", "Eval Loss", "Eval PPL", "Uniform Loss Gap"],
                [
                    [
                        "before_lrp_ssm_old",
                        old_lrp.get("final_train_loss"),
                        old_lrp.get("best_eval_loss"),
                        old_lrp.get("best_eval_ppl"),
                        (old_lrp.get("best_eval_loss") - math.log(old_lrp.get("config", {}).get("vocab_size", 50257))) if old_lrp.get("best_eval_loss") else None,
                    ],
                    [
                        "after_fixed_calibrated",
                        fixed_lrp.get("final_train_loss"),
                        fixed_lrp.get("best_eval_loss"),
                        fixed_lrp.get("best_eval_ppl"),
                        (fixed_lrp.get("best_eval_loss") - math.log(fixed_lrp.get("config", {}).get("vocab_size", 50257))) if fixed_lrp.get("best_eval_loss") else None,
                    ],
                    [
                        "after_learned_router",
                        learned_lrp.get("final_train_loss"),
                        learned_lrp.get("best_eval_loss"),
                        learned_lrp.get("best_eval_ppl"),
                        (learned_lrp.get("best_eval_loss") - math.log(learned_lrp.get("config", {}).get("vocab_size", 50257))) if learned_lrp.get("best_eval_loss") else None,
                    ],
                ],
            ),
            "",
            "### Updated Model Comparison",
            "",
            table(
                ["Model", "Params", "Eval Loss", "Eval PPL", "Uniform Gap", "Status"],
                [
                    [
                        name,
                        item.get("parameter_count"),
                        item.get("best_eval_loss"),
                        item.get("best_eval_ppl"),
                        (item.get("best_eval_loss") - math.log(item.get("config", {}).get("vocab_size", 50257))) if item.get("best_eval_loss") else None,
                        item.get("status"),
                    ]
                    for name, item in models.items()
                ],
            ),
            "",
            "### Throughput/Memory",
            "",
            table(
                ["Model", "Train Tokens/s", "Eval Tokens/s", "Forward ms", "Peak GPU Memory MB"],
                [
                    [
                        name,
                        item.get("avg_train_tokens_per_sec"),
                        item.get("avg_eval_tokens_per_sec"),
                        (read_json(Path(item.get("run_dir", "")) / "benchmark.json", default={}).get("results", [{}])[0].get("forward_avg_ms") if item.get("run_dir") else None),
                        item.get("peak_gpu_memory_mb"),
                    ]
                    for name, item in models.items()
                ],
            ),
        ]
    if args.include_raw_tables:
        lines += ["", "## Raw JSON Summary", "", "```json", json.dumps(comparison, indent=2), "```"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def main():
    args = parse_args()
    output = generate_report(args)
    print(f"LEVEL_TEST_REPORT_PASS output={output}")


if __name__ == "__main__":
    main()

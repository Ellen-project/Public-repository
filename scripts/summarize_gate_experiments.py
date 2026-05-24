from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_COLUMNS = [
    "name",
    "train_loss",
    "val_loss",
    "mean_active_paths",
    "zero_gate_ratio",
    "learned_mean_active_paths",
    "learned_path_rate_std",
    "gate_entropy",
    "gate_commitment_loss",
    "gate_teacher_bce_loss",
    "tokens_per_sec",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize managed gate experiment results.")
    parser.add_argument("run_dir", type=str, nargs="?", default=None)
    parser.add_argument("--columns", type=str, default=",".join(DEFAULT_COLUMNS))
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args()


def latest_run_dir(root: Path = Path("runs/gate_experiments")) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist")
    candidates = [path for path in root.iterdir() if path.is_dir() and (path / "summary.json").exists()]
    if not candidates:
        raise FileNotFoundError(f"No gate experiment summaries found under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def latest_jsonl_record(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    last = ""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last = line
    return json.loads(last) if last else {}


def load_summary(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"{summary_path} does not exist")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def normalize_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for result in summary.get("results", []):
        if result.get("name") == "build_cache":
            continue
        metrics = dict(result.get("metrics") or {})
        if not metrics and result.get("output_dir"):
            metrics = latest_jsonl_record(Path(result["output_dir"]) / "train_metrics.jsonl")
        row = {
            "name": result.get("name"),
            "gate_mode": result.get("gate_mode"),
            "teacher_weight": result.get("teacher_weight"),
            "returncode": result.get("returncode"),
            **metrics,
        }
        rows.append(row)
    return rows


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if abs(value) >= 1000.0:
            return f"{value:.2f}"
        return f"{value:.4g}"
    return str(value)


def print_table(rows: list[dict[str, Any]], columns: list[str]):
    widths = {
        column: max(
            len(column),
            *(len(fmt(row.get(column))) for row in rows),
        )
        for column in columns
    }
    header = " | ".join(column.ljust(widths[column]) for column in columns)
    rule = "-+-".join("-" * widths[column] for column in columns)
    print(header)
    print(rule)
    for row in rows:
        print(" | ".join(fmt(row.get(column)).ljust(widths[column]) for column in columns))


def main():
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir()
    summary = load_summary(run_dir)
    rows = normalize_rows(summary)
    columns = [column.strip() for column in args.columns.split(",") if column.strip()]

    if args.as_json:
        print(json.dumps({"run_dir": str(run_dir), "rows": rows}, indent=2))
        return

    print(f"run_dir: {run_dir}")
    print_table(rows, columns)


if __name__ == "__main__":
    main()

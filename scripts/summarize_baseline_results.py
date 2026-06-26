#!/usr/bin/env python
"""Summarize raw baseline JSON files by method/dataset/seed."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--status", default="development")
    parser.add_argument("--result-root", default="results/raw")
    parser.add_argument("--summary-root", default="results/summary")
    parser.add_argument("--output-prefix", default=None)
    return parser.parse_args()


def read_results(result_root: Path, method: str, status: str) -> list[dict[str, Any]]:
    results = []
    for path in sorted(result_root.glob(f"*/{method}/*.json")):
        payload = json.loads(path.read_text())
        if payload.get("method") != method or payload.get("status") != status:
            continue
        payload["_mtime"] = path.stat().st_mtime
        payload["_source_path"] = str(path.resolve())
        results.append(payload)
    return results


def latest_by_dataset_seed(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    for result in results:
        key = (str(result["dataset"]), int(result["model_seed"]))
        old = latest.get(key)
        if old is None or result["_mtime"] > old["_mtime"]:
            latest[key] = result
    return sorted(latest.values(), key=lambda r: (str(r["dataset"]), int(r["model_seed"])))


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_dataset[str(result["dataset"])].append(result)

    summary: dict[str, Any] = {}
    for dataset, items in sorted(by_dataset.items()):
        values = np.array([float(item["test_at_best"]) for item in items], dtype=float)
        target_mean = items[-1].get("target_mean")
        target_std = items[-1].get("target_std")
        summary[dataset] = {
            "num_seeds": int(values.shape[0]),
            "seeds": [int(item["model_seed"]) for item in items],
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "target_mean": target_mean,
            "target_std": target_std,
            "gap_from_target_mean": (
                float(values.mean()) - float(target_mean) if target_mean is not None else None
            ),
            "values": [float(value) for value in values.tolist()],
            "source_paths": [item["_source_path"] for item in items],
        }
    return summary


def render_markdown(method: str, status: str, summary: dict[str, Any]) -> str:
    lines = [
        f"# {method} Baseline Summary",
        "",
        f"- Status filter: `{status}`",
        "- Selection: latest raw JSON per dataset/seed",
        "- Metric: `test_at_best` accuracy / %",
        "",
        "| Dataset | Seeds | Local mean±std | Target mean±std | Gap |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for dataset, item in summary.items():
        target = (
            f"{float(item['target_mean']):.2f}±{float(item['target_std']):.2f}"
            if item["target_mean"] is not None
            else "NA"
        )
        gap = item["gap_from_target_mean"]
        lines.append(
            f"| {dataset} | {item['num_seeds']} | "
            f"{item['mean']:.2f}±{item['std']:.2f} | {target} | "
            f"{gap:.2f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    result_root = PROJECT_ROOT / args.result_root
    summary_root = PROJECT_ROOT / args.summary_root
    summary_root.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix or f"{args.method.lower()}_{args.status}"

    raw = read_results(result_root, args.method, args.status)
    selected = latest_by_dataset_seed(raw)
    summary = summarize(selected)

    payload = {
        "method": args.method,
        "status": args.status,
        "selection_rule": "latest_raw_json_per_dataset_seed",
        "summary": summary,
    }
    json_path = summary_root / f"{prefix}_summary.json"
    md_path = summary_root / f"{prefix}_summary.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    md_path.write_text(render_markdown(args.method, args.status, summary))
    print(md_path.read_text(), end="")
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()

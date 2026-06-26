#!/usr/bin/env python
"""Run small seed0/seed1 searches for Amazon GRACE candidate configs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/grace_1_1_8.yaml")
    parser.add_argument("--candidate-config", default="configs/grace_amazon_candidate_search.yaml")
    parser.add_argument("--candidates", default=None, help="Comma-separated candidate names. Default: all.")
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--status", default="candidate")
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_yaml(path: str) -> dict:
    return yaml.safe_load((PROJECT_ROOT / path).read_text())


def write_candidate_config(base: dict, candidate: dict, out_dir: Path) -> Path:
    config = json.loads(json.dumps(base))
    dataset = candidate["dataset"]
    config["pretrain"][dataset].update(candidate["overrides"])
    config["meta"]["candidate_name"] = candidate["name"]
    config["meta"]["candidate_source"] = candidate.get("source")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{candidate['name']}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def summarize_candidate(candidate_name: str, dataset: str) -> None:
    root = PROJECT_ROOT / "results" / "raw" / dataset / "GRACE"
    items = []
    if root.exists():
        for path in root.glob(f"grace_1_1_8_{candidate_name}_{dataset}_seed*.json"):
            payload = json.loads(path.read_text())
            if payload.get("run_tag") == candidate_name:
                items.append(payload)
    if not items:
        print(f"[{candidate_name}] no result files found yet")
        return
    items.sort(key=lambda x: x["model_seed"])
    values = np.array([item["test_at_best"] for item in items], dtype=float)
    seeds = [item["model_seed"] for item in items]
    print(
        f"[{candidate_name}] {dataset} seeds={seeds} "
        f"mean={values.mean():.2f} std={values.std(ddof=0):.2f} "
        f"values={[round(v, 2) for v in values]}"
    )


def main() -> None:
    args = parse_args()
    base = load_yaml(args.base_config)
    candidates_doc = load_yaml(args.candidate_config)
    wanted = None
    if args.candidates:
        wanted = {name.strip() for name in args.candidates.split(",") if name.strip()}
    candidates = [
        candidate for candidate in candidates_doc["candidates"]
        if wanted is None or candidate["name"] in wanted
    ]
    missing = wanted - {candidate["name"] for candidate in candidates} if wanted else set()
    if missing:
        raise ValueError(f"Unknown candidates: {sorted(missing)}")

    temp_dir = PROJECT_ROOT / "outputs" / "candidate_configs"
    for candidate in candidates:
        config_path = write_candidate_config(base, candidate, temp_dir)
        cmd = [
            sys.executable,
            "scripts/run_grace_1_1_8.py",
            "--config",
            str(config_path.relative_to(PROJECT_ROOT)),
            "--datasets",
            candidate["dataset"],
            "--seeds",
            args.seeds,
            "--gpu-id",
            str(args.gpu_id),
            "--status",
            args.status,
            "--evaluator",
            "logreg_val",
            "--run-tag",
            candidate["name"],
            "--log-every",
            str(args.log_every),
        ]
        print("\n=== Candidate:", candidate["name"], "===")
        print(" ".join(cmd), flush=True)
        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        summarize_candidate(candidate["name"], candidate["dataset"])


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Run supervised GCN under the project's node-classification protocol."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import yaml
from sklearn.model_selection import train_test_split
from torch import nn
from torch_geometric.datasets import Amazon, Planetoid, WikiCS
from torch_geometric.nn import GCNConv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GCN_DIR = PROJECT_ROOT / "baselines" / "GCN"
DATASET_NAMES = ("Cora", "CiteSeer", "PubMed", "WikiCS", "Computers", "Photo")


@dataclass
class Split:
    train: torch.Tensor
    val: torch.Tensor
    test: torch.Tensor
    path: Path
    split_type: str
    validation_role: str = "val"


class GCN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float,
        cached: bool,
        add_self_loops: bool,
        normalize: bool,
    ) -> None:
        super().__init__()
        self.conv1 = GCNConv(
            in_dim,
            hidden_dim,
            cached=cached,
            add_self_loops=add_self_loops,
            normalize=normalize,
        )
        self.conv2 = GCNConv(
            hidden_dim,
            out_dim,
            cached=cached,
            add_self_loops=add_self_loops,
            normalize=normalize,
        )
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gcn_1_1_8.yaml")
    parser.add_argument("--datasets", default="Cora")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--status", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--early-stopping-metric", choices=("val_accuracy", "val_loss"), default=None)
    parser.add_argument("--data-root", default="data/pyg")
    parser.add_argument("--split-root", default="splits")
    parser.add_argument("--result-root", default="results/raw")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--skip-if-result-exists", action="store_true")
    return parser.parse_args()


def git_rev(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def git_dirty(path: Path) -> bool:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(path), "status", "--short"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except Exception:
        return False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset(name: str, data_root: Path, dataset_cfg: dict[str, Any] | None = None):
    dataset_cfg = dataset_cfg or {}
    transform = T.NormalizeFeatures() if dataset_cfg.get("normalize_features", True) else None
    if name in {"Cora", "CiteSeer", "PubMed"}:
        return Planetoid(str(data_root / "Planetoid"), name, transform=transform)
    if name == "WikiCS":
        return WikiCS(
            str(data_root / "WikiCS"),
            transform=transform,
            is_undirected=dataset_cfg.get("is_undirected"),
        )
    if name in {"Computers", "Photo"}:
        return Amazon(str(data_root / "Amazon"), name, transform=transform)
    raise ValueError(f"Unsupported dataset: {name}")


def split_counts(labels: np.ndarray, indices: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(labels[indices], return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(values, counts)}


def create_or_load_random_split(
    dataset_name: str,
    labels: torch.Tensor,
    seed: int,
    split_root: Path,
) -> Split:
    split_dir = split_root / dataset_name
    split_dir.mkdir(parents=True, exist_ok=True)
    split_path = split_dir / f"split_seed_{seed}.json"
    y = labels.detach().cpu().numpy()
    all_idx = np.arange(y.shape[0])

    if split_path.exists():
        payload = json.loads(split_path.read_text())
        return Split(
            train=torch.tensor(payload["train"], dtype=torch.long),
            val=torch.tensor(payload["val"], dtype=torch.long),
            test=torch.tensor(payload["test"], dtype=torch.long),
            path=split_path,
            split_type=str(payload.get("split_type", "stratified_random_1_1_8")),
        )

    train_idx, tmp_idx = train_test_split(
        all_idx,
        train_size=0.1,
        random_state=seed,
        shuffle=True,
        stratify=y,
    )
    tmp_y = y[tmp_idx]
    val_idx, test_idx = train_test_split(
        tmp_idx,
        train_size=1.0 / 9.0,
        random_state=seed,
        shuffle=True,
        stratify=tmp_y,
    )

    payload = {
        "dataset_name": dataset_name,
        "split_type": "stratified_random_1_1_8",
        "split_seed": seed,
        "ratios": {"train": 0.1, "val": 0.1, "test": 0.8},
        "num_nodes": int(y.shape[0]),
        "num_classes": int(np.unique(y).shape[0]),
        "counts": {
            "train": int(train_idx.shape[0]),
            "val": int(val_idx.shape[0]),
            "test": int(test_idx.shape[0]),
        },
        "class_counts": {
            "train": split_counts(y, train_idx),
            "val": split_counts(y, val_idx),
            "test": split_counts(y, test_idx),
        },
        "train": sorted(map(int, train_idx.tolist())),
        "val": sorted(map(int, val_idx.tolist())),
        "test": sorted(map(int, test_idx.tolist())),
    }
    split_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return Split(
        train=torch.tensor(payload["train"], dtype=torch.long),
        val=torch.tensor(payload["val"], dtype=torch.long),
        test=torch.tensor(payload["test"], dtype=torch.long),
        path=split_path,
        split_type="stratified_random_1_1_8",
    )


def create_or_load_wikics_split(
    data: Any,
    seed: int,
    split_root: Path,
    validation_role: str,
) -> Split:
    split_dir = split_root / "WikiCS"
    split_dir.mkdir(parents=True, exist_ok=True)
    split_path = split_dir / f"official_split_{seed}.json"
    split_count = int(data.train_mask.size(1))
    split_idx = seed % split_count
    train_idx = data.train_mask[:, split_idx].nonzero(as_tuple=False).view(-1).cpu().numpy()
    val_idx = data.val_mask[:, split_idx].nonzero(as_tuple=False).view(-1).cpu().numpy()
    stopping_idx = data.stopping_mask[:, split_idx].nonzero(as_tuple=False).view(-1).cpu().numpy()
    test_idx = data.test_mask.nonzero(as_tuple=False).view(-1).cpu().numpy()
    y = data.y.detach().cpu().numpy()

    if split_path.exists():
        payload = json.loads(split_path.read_text())
        if "stopping" not in payload:
            payload["stopping"] = sorted(map(int, stopping_idx.tolist()))
            payload.setdefault("counts", {})["stopping"] = int(stopping_idx.shape[0])
            payload.setdefault("class_counts", {})["stopping"] = split_counts(y, stopping_idx)
            split_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        selected_val = payload["stopping"] if validation_role == "stopping" else payload["val"]
        return Split(
            train=torch.tensor(payload["train"], dtype=torch.long),
            val=torch.tensor(selected_val, dtype=torch.long),
            test=torch.tensor(payload["test"], dtype=torch.long),
            path=split_path,
            split_type="wikics_official",
            validation_role=validation_role,
        )

    payload = {
        "dataset_name": "WikiCS",
        "split_type": "wikics_official",
        "split_seed": seed,
        "official_split_index": split_idx,
        "num_nodes": int(y.shape[0]),
        "num_classes": int(np.unique(y).shape[0]),
        "counts": {
            "train": int(train_idx.shape[0]),
            "val": int(val_idx.shape[0]),
            "stopping": int(stopping_idx.shape[0]),
            "test": int(test_idx.shape[0]),
        },
        "class_counts": {
            "train": split_counts(y, train_idx),
            "val": split_counts(y, val_idx),
            "stopping": split_counts(y, stopping_idx),
            "test": split_counts(y, test_idx),
        },
        "train": sorted(map(int, train_idx.tolist())),
        "val": sorted(map(int, val_idx.tolist())),
        "stopping": sorted(map(int, stopping_idx.tolist())),
        "test": sorted(map(int, test_idx.tolist())),
    }
    split_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    selected_val = payload["stopping"] if validation_role == "stopping" else payload["val"]
    return Split(
        train=torch.tensor(payload["train"], dtype=torch.long),
        val=torch.tensor(selected_val, dtype=torch.long),
        test=torch.tensor(payload["test"], dtype=torch.long),
        path=split_path,
        split_type="wikics_official",
        validation_role=validation_role,
    )


def create_or_load_split(
    dataset_name: str,
    data: Any,
    seed: int,
    split_root: Path,
    dataset_cfg: dict[str, Any],
) -> Split:
    if dataset_name == "WikiCS":
        validation_role = str(dataset_cfg.get("early_stopping_mask", "stopping"))
        if validation_role not in {"val", "stopping"}:
            raise ValueError("WikiCS early_stopping_mask must be 'val' or 'stopping'")
        return create_or_load_wikics_split(data, seed, split_root, validation_role)
    return create_or_load_random_split(dataset_name, data.y.detach().cpu(), seed, split_root)


@torch.no_grad()
def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == labels).float().mean().item())


def count_parameters(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


def merged_dataset_config(config: dict[str, Any], dataset_name: str, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    model_cfg = dict(config["model"])
    train_cfg = dict(config["training"])
    overrides = dict(config.get("dataset_overrides", {}).get(dataset_name, {}))
    for key in ("hidden_dim", "dropout", "cached", "add_self_loops", "normalize"):
        if key in overrides:
            model_cfg[key] = overrides[key]
    for key in ("learning_rate", "weight_decay", "max_epochs", "patience"):
        if key in overrides:
            train_cfg[key] = overrides[key]
    if args.epochs is not None:
        train_cfg["max_epochs"] = args.epochs
    if args.patience is not None:
        train_cfg["patience"] = args.patience
    if args.hidden_dim is not None:
        model_cfg["hidden_dim"] = args.hidden_dim
    if args.dropout is not None:
        model_cfg["dropout"] = args.dropout
    if args.learning_rate is not None:
        train_cfg["learning_rate"] = args.learning_rate
    if args.weight_decay is not None:
        train_cfg["weight_decay"] = args.weight_decay
    if args.early_stopping_metric is not None:
        train_cfg["early_stopping_metric"] = args.early_stopping_metric
    return model_cfg, train_cfg


def train_one(
    dataset_name: str,
    seed: int,
    config: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    status = args.status or config["meta"].get("status_default", "development")
    model_cfg, train_cfg = merged_dataset_config(config, dataset_name, args)
    dataset_cfg = dict(config.get("dataset_overrides", {}).get(dataset_name, {}))
    dataset = load_dataset(dataset_name, PROJECT_ROOT / args.data_root, dataset_cfg)
    data = dataset[0].to(device)
    split = create_or_load_split(dataset_name, data, seed, PROJECT_ROOT / args.split_root, dataset_cfg)

    result_dir = PROJECT_ROOT / args.result_root / dataset_name / "GCN"
    result_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(result_dir.glob(f"gcn_1_1_8_{dataset_name}_seed{seed}_*.json"))
    if args.skip_if_result_exists and existing:
        return json.loads(existing[-1].read_text())

    set_seed(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = GCN(
        in_dim=int(dataset.num_features),
        hidden_dim=int(model_cfg["hidden_dim"]),
        out_dim=int(dataset.num_classes),
        dropout=float(model_cfg["dropout"]),
        cached=bool(model_cfg["cached"]),
        add_self_loops=bool(model_cfg["add_self_loops"]),
        normalize=bool(model_cfg["normalize"]),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    train_idx = split.train.to(device)
    val_idx = split.val.to(device)
    test_idx = split.test.to(device)
    y = data.y
    max_epochs = int(train_cfg["max_epochs"])
    patience = int(train_cfg["patience"])
    best = {
        "best_epoch": 0,
        "valid_at_best": -1.0,
        "valid_loss_at_best": float("inf"),
        "test_at_best": -1.0,
        "train_at_best": -1.0,
        "state": None,
    }
    wait = 0
    final_test = 0.0
    last_loss = 0.0
    last_val_loss = 0.0
    start_time = time.time()
    early_stopping_metric = str(train_cfg.get("early_stopping_metric", "val_accuracy"))
    if early_stopping_metric not in {"val_accuracy", "val_loss"}:
        raise ValueError("early_stopping_metric must be 'val_accuracy' or 'val_loss'")

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            train_acc = accuracy(logits[train_idx], y[train_idx])
            val_acc = accuracy(logits[val_idx], y[val_idx])
            test_acc = accuracy(logits[test_idx], y[test_idx])
            val_loss = float(F.cross_entropy(logits[val_idx], y[val_idx]).item())
        final_test = test_acc
        last_val_loss = val_loss

        improved = (
            val_acc > best["valid_at_best"]
            if early_stopping_metric == "val_accuracy"
            else val_loss < best["valid_loss_at_best"]
        )
        if improved:
            best.update(
                {
                    "best_epoch": epoch,
                    "valid_at_best": val_acc,
                    "valid_loss_at_best": val_loss,
                    "test_at_best": test_acc,
                    "train_at_best": train_acc,
                    "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                }
            )
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

        if epoch == 1 or epoch % args.log_every == 0:
            print(
                f"[{dataset_name} seed={seed}] epoch={epoch:04d}/{max_epochs} "
                f"loss={last_loss:.4f} val={val_acc * 100:.2f} "
                f"test={test_acc * 100:.2f}",
                flush=True,
            )

    elapsed = time.time() - start_time
    target = config.get("targets", {}).get(dataset_name, {})
    peak_memory_mb = None
    if device.type == "cuda":
        peak_memory_mb = float(torch.cuda.max_memory_allocated(device) / 1024 / 1024)

    result = {
        "run_id": f"gcn_1_1_8_{dataset_name}_seed{seed}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}",
        "method": "GCN",
        "dataset": dataset_name,
        "split_type": split.split_type,
        "split_seed": seed,
        "model_seed": seed,
        "status": status,
        "metric": "accuracy_percent",
        "config_path": str((PROJECT_ROOT / args.config).resolve()),
        "split_path": str(split.path.resolve()),
        "validation_role": split.validation_role,
        "code_source": config["meta"]["code_source"],
        "implementation": config["meta"].get("implementation"),
        "project_commit_hash": git_rev(PROJECT_ROOT),
        "project_dirty": git_dirty(PROJECT_ROOT),
        "baseline_commit_hash": git_rev(GCN_DIR),
        "baseline_dirty": git_dirty(GCN_DIR),
        "dataset_num_nodes": int(data.num_nodes),
        "dataset_num_edges": int(data.edge_index.size(1)),
        "dataset_num_features": int(dataset.num_features),
        "dataset_num_classes": int(dataset.num_classes),
        "model_config": model_cfg,
        "training_config": train_cfg,
        "parameter_count": count_parameters(model),
        "best_epoch": int(best["best_epoch"]),
        "epochs_ran": int(epoch),
        "train_at_best": best["train_at_best"] * 100.0,
        "valid_at_best": best["valid_at_best"] * 100.0,
        "test_at_best": best["test_at_best"] * 100.0,
        "final_test": final_test * 100.0,
        "last_train_loss": last_loss,
        "last_val_loss": last_val_loss,
        "elapsed_sec": elapsed,
        "peak_cuda_memory_mb": peak_memory_mb,
        "target_mean": target.get("mean"),
        "target_std": target.get("std"),
        "gap_from_target_mean": (
            best["test_at_best"] * 100.0 - float(target["mean"]) if "mean" in target else None
        ),
    }
    result_path = result_dir / f"{result['run_id']}.json"
    result["result_path"] = str(result_path.resolve())
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        f"[{dataset_name} seed={seed}] test@best-val={result['test_at_best']:.2f} "
        f"target={target.get('mean', 'NA')} result={result_path}",
        flush=True,
    )
    return result


def summarize(results: list[dict[str, Any]], config: dict[str, Any]) -> None:
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        by_dataset.setdefault(result["dataset"], []).append(result)
    print("\n=== GCN supervised summary ===")
    for dataset_name, items in by_dataset.items():
        values = np.array([r["test_at_best"] for r in items], dtype=float)
        target = config.get("targets", {}).get(dataset_name, {})
        print(
            f"{dataset_name}: {values.mean():.2f}+-{values.std(ddof=0):.2f} "
            f"over {len(values)} seed(s); target={target.get('mean', 'NA')}+-{target.get('std', 'NA')}"
        )


def main() -> None:
    args = parse_args()
    config = yaml.safe_load((PROJECT_ROOT / args.config).read_text())
    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    unknown = sorted(set(datasets) - set(DATASET_NAMES))
    if unknown:
        raise ValueError(f"Unsupported dataset(s): {unknown}. Supported: {DATASET_NAMES}")

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")
    print(f"device={device} datasets={datasets} seeds={seeds}")

    results = []
    for dataset_name in datasets:
        for seed in seeds:
            results.append(train_one(dataset_name, seed, config, args, device))
    summarize(results, config)


if __name__ == "__main__":
    main()

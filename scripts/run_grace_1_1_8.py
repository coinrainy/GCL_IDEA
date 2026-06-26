#!/usr/bin/env python
"""Run GRACE under the project's fixed 1:1:8 node-classification protocol."""

from __future__ import annotations

import argparse
import json
import os
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
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from torch import nn
from torch_geometric.datasets import Amazon, CitationFull, Planetoid
import torch_geometric.transforms as T

try:
    from torch_geometric.utils import dropout_edge
except ImportError:  # pragma: no cover - kept for older PyG
    dropout_edge = None
    from torch_geometric.utils import dropout_adj


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GRACE_DIR = PROJECT_ROOT / "baselines" / "GRACE"
sys.path.insert(0, str(GRACE_DIR))

from model import Encoder, Model, drop_feature  # noqa: E402
from torch_geometric.nn import GCNConv  # noqa: E402


DATASET_NAMES = ("Cora", "CiteSeer", "PubMed", "DBLP", "Computers", "Photo")


@dataclass
class Split:
    train: torch.Tensor
    val: torch.Tensor
    test: torch.Tensor
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/grace_1_1_8.yaml")
    parser.add_argument("--datasets", default="Cora")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--status", default=None)
    parser.add_argument("--evaluator", choices=("logreg_val", "torch_linear"), default=None)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--eval-epochs", type=int, default=None)
    parser.add_argument("--eval-patience", type=int, default=None)
    parser.add_argument("--eval-learning-rate", type=float, default=None)
    parser.add_argument("--eval-weight-decay", type=float, default=None)
    parser.add_argument("--no-normalize-embeddings", action="store_true")
    parser.add_argument("--loss-batch-size", type=int, default=None)
    parser.add_argument("--data-root", default="data/pyg")
    parser.add_argument("--split-root", default="splits")
    parser.add_argument("--result-root", default="results/raw")
    parser.add_argument("--log-every", type=int, default=25)
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


def load_dataset(name: str, data_root: Path):
    transform = T.NormalizeFeatures()
    if name in {"Cora", "CiteSeer", "PubMed"}:
        return Planetoid(str(data_root / "Planetoid"), name, transform=transform)
    if name == "DBLP":
        return CitationFull(str(data_root / "CitationFull"), "dblp", transform=transform)
    if name in {"Computers", "Photo"}:
        return Amazon(str(data_root / "Amazon"), name, transform=transform)
    raise ValueError(f"Unsupported dataset: {name}")


def split_counts(labels: np.ndarray, indices: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(labels[indices], return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(values, counts)}


def create_or_load_split(
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
    )


def activation_from_name(name: str):
    if name == "relu":
        return F.relu
    if name == "rrelu":
        return nn.RReLU()
    if name == "prelu":
        return nn.PReLU()
    raise ValueError(f"Unsupported activation: {name}")


def drop_edges(edge_index: torch.Tensor, p: float) -> torch.Tensor:
    if dropout_edge is not None:
        return dropout_edge(edge_index, p=p)[0]
    return dropout_adj(edge_index, p=p)[0]


def train_grace_epoch(
    model: Model,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    cfg: dict[str, Any],
    loss_batch_size: int,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    edge_index_1 = drop_edges(edge_index, cfg["drop_edge_rate_1"])
    edge_index_2 = drop_edges(edge_index, cfg["drop_edge_rate_2"])
    x_1 = drop_feature(x, cfg["drop_feature_rate_1"])
    x_2 = drop_feature(x, cfg["drop_feature_rate_2"])
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)
    loss = model.loss(z1, z2, batch_size=loss_batch_size)
    loss.backward()
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def encode(model: Model, x: torch.Tensor, edge_index: torch.Tensor, normalize: bool) -> torch.Tensor:
    model.eval()
    z = model(x, edge_index)
    return F.normalize(z, dim=1) if normalize else z


@torch.no_grad()
def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == labels).float().mean().item())


def linear_eval(
    z: torch.Tensor,
    y: torch.Tensor,
    split: Split,
    num_classes: int,
    cfg: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    set_seed(seed + 10_000)
    clf = nn.Linear(z.size(1), num_classes).to(z.device)
    optimizer = torch.optim.Adam(
        clf.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    max_epochs = int(cfg["max_epochs"])
    patience = int(cfg["patience"])

    train_idx = split.train.to(z.device)
    val_idx = split.val.to(z.device)
    test_idx = split.test.to(z.device)

    best = {
        "best_epoch": 0,
        "valid_at_best": -1.0,
        "test_at_best": -1.0,
        "state": None,
    }
    wait = 0
    final_test = 0.0

    for epoch in range(1, max_epochs + 1):
        clf.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(clf(z[train_idx]), y[train_idx])
        loss.backward()
        optimizer.step()

        clf.eval()
        with torch.no_grad():
            val_acc = accuracy(clf(z[val_idx]), y[val_idx])
            test_acc = accuracy(clf(z[test_idx]), y[test_idx])
        final_test = test_acc

        if val_acc > best["valid_at_best"]:
            best.update(
                {
                    "best_epoch": epoch,
                    "valid_at_best": val_acc,
                    "test_at_best": test_acc,
                    "state": {k: v.detach().cpu() for k, v in clf.state_dict().items()},
                }
            )
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    return {
        "evaluator_type": "torch_linear",
        "best_epoch": int(best["best_epoch"]),
        "valid_at_best": best["valid_at_best"] * 100.0,
        "test_at_best": best["test_at_best"] * 100.0,
        "final_test": final_test * 100.0,
        "eval_epochs_ran": epoch,
    }


def logreg_val_eval(
    z: torch.Tensor,
    y: torch.Tensor,
    split: Split,
    cfg: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    X = z.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    train_idx = split.train.cpu().numpy()
    val_idx = split.val.cpu().numpy()
    test_idx = split.test.cpu().numpy()

    best = {
        "valid_at_best": -1.0,
        "test_at_best": -1.0,
        "final_test": -1.0,
        "best_c": None,
        "best_clf": None,
    }
    c_values = [float(c) for c in cfg["c_values"]]
    for c in c_values:
        base_clf = LogisticRegression(
            C=c,
            penalty=cfg.get("penalty", "l2"),
            solver=cfg.get("solver", "liblinear"),
            tol=float(cfg.get("tol", 1e-4)),
            max_iter=int(cfg.get("max_iter", 5000)),
            fit_intercept=bool(cfg.get("fit_intercept", True)),
            class_weight=cfg.get("class_weight"),
            random_state=seed,
        )
        clf = OneVsRestClassifier(base_clf)
        clf.fit(X[train_idx], Y[train_idx])
        val_acc = accuracy_score(Y[val_idx], clf.predict(X[val_idx]))
        if val_acc > best["valid_at_best"]:
            best.update(
                {
                    "valid_at_best": float(val_acc),
                    "best_c": c,
                    "best_clf": clf,
                }
            )
    if best["best_clf"] is None:
        raise RuntimeError("Logistic Regression evaluator failed to fit any C value.")
    test_acc = accuracy_score(Y[test_idx], best["best_clf"].predict(X[test_idx]))
    best["test_at_best"] = float(test_acc)
    best["final_test"] = float(test_acc)

    return {
        "evaluator_type": "logreg_val",
        "best_epoch": None,
        "valid_at_best": best["valid_at_best"] * 100.0,
        "test_at_best": best["test_at_best"] * 100.0,
        "final_test": best["final_test"] * 100.0,
        "eval_epochs_ran": None,
        "best_c": best["best_c"],
        "c_values": c_values,
        "c_selection_rule": "fit_on_train_select_by_val_accuracy_report_test",
    }


def run_one(
    dataset_name: str,
    seed: int,
    config: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    pretrain_cfg = dict(config["pretrain"][dataset_name])
    evaluator_type = args.evaluator or config["meta"].get("default_evaluator", "logreg_val")
    if evaluator_type == "logreg_val":
        eval_cfg = dict(config["logreg_eval"])
    elif evaluator_type == "torch_linear":
        eval_cfg = dict(config["linear_eval"])
    else:
        raise ValueError(f"Unsupported evaluator: {evaluator_type}")
    if args.epochs is not None:
        pretrain_cfg["num_epochs"] = args.epochs
    if evaluator_type == "torch_linear" and args.eval_epochs is not None:
        eval_cfg["max_epochs"] = args.eval_epochs
    if evaluator_type == "torch_linear" and args.eval_patience is not None:
        eval_cfg["patience"] = args.eval_patience
    if evaluator_type == "torch_linear" and args.eval_learning_rate is not None:
        eval_cfg["learning_rate"] = args.eval_learning_rate
    if evaluator_type == "torch_linear" and args.eval_weight_decay is not None:
        eval_cfg["weight_decay"] = args.eval_weight_decay
    if args.no_normalize_embeddings:
        eval_cfg["normalize_embeddings"] = False
    if args.loss_batch_size is not None:
        pretrain_cfg["loss_batch_size"] = args.loss_batch_size

    status = args.status or config["meta"].get("status_default", "development")
    dataset = load_dataset(dataset_name, PROJECT_ROOT / args.data_root)
    data = dataset[0].to(device)
    num_classes = int(dataset.num_classes)
    split = create_or_load_split(
        dataset_name,
        data.y.detach().cpu(),
        seed,
        PROJECT_ROOT / args.split_root,
    )

    set_seed(seed)
    encoder = Encoder(
        dataset.num_features,
        int(pretrain_cfg["num_hidden"]),
        activation_from_name(pretrain_cfg["activation"]),
        base_model=GCNConv,
        k=int(pretrain_cfg["num_layers"]),
    ).to(device)
    model = Model(
        encoder,
        int(pretrain_cfg["num_hidden"]),
        int(pretrain_cfg["num_proj_hidden"]),
        float(pretrain_cfg["tau"]),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(pretrain_cfg["learning_rate"]),
        weight_decay=float(pretrain_cfg["weight_decay"]),
    )

    start_time = time.time()
    last_loss = None
    num_epochs = int(pretrain_cfg["num_epochs"])
    loss_batch_size = int(pretrain_cfg.get("loss_batch_size", 0))
    for epoch in range(1, num_epochs + 1):
        last_loss = train_grace_epoch(
            model,
            optimizer,
            data.x,
            data.edge_index,
            pretrain_cfg,
            loss_batch_size,
        )
        if epoch == 1 or epoch == num_epochs or epoch % args.log_every == 0:
            elapsed = time.time() - start_time
            print(
                f"[{dataset_name} seed={seed}] epoch={epoch:04d}/{num_epochs} "
                f"loss={last_loss:.4f} elapsed={elapsed:.1f}s",
                flush=True,
            )

    z = encode(
        model,
        data.x,
        data.edge_index,
        normalize=bool(eval_cfg["normalize_embeddings"]),
    ).detach()
    if evaluator_type == "logreg_val":
        eval_result = logreg_val_eval(z, data.y, split, eval_cfg, seed)
    else:
        eval_result = linear_eval(z, data.y, split, num_classes, eval_cfg, seed)
    elapsed = time.time() - start_time

    target = config.get("targets", {}).get(dataset_name, {})
    tag = f"_{args.run_tag}" if args.run_tag else ""
    result = {
        "run_id": f"grace_1_1_8{tag}_{dataset_name}_seed{seed}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}",
        "method": "GRACE",
        "run_tag": args.run_tag,
        "dataset": dataset_name,
        "split_type": "stratified_random_1_1_8",
        "split_seed": seed,
        "model_seed": seed,
        "status": status,
        "metric": "accuracy_percent",
        "config_path": str((PROJECT_ROOT / args.config).resolve()),
        "split_path": str(split.path.resolve()),
        "code_source": config["meta"]["code_source"],
        "project_commit_hash": git_rev(PROJECT_ROOT),
        "project_dirty": git_dirty(PROJECT_ROOT),
        "baseline_commit_hash": git_rev(GRACE_DIR),
        "baseline_dirty": git_dirty(GRACE_DIR),
        "dataset_num_nodes": int(data.num_nodes),
        "dataset_num_edges": int(data.edge_index.size(1)),
        "dataset_num_features": int(dataset.num_features),
        "dataset_num_classes": num_classes,
        "pretrain_config": pretrain_cfg,
        "evaluator_type": evaluator_type,
        "evaluator_config": eval_cfg,
        "last_train_loss": last_loss,
        "elapsed_sec": elapsed,
        "target_mean": target.get("mean"),
        "target_std": target.get("std"),
        "gap_from_target_mean": (
            eval_result["test_at_best"] - float(target["mean"]) if "mean" in target else None
        ),
        **eval_result,
    }

    result_dir = PROJECT_ROOT / args.result_root / dataset_name / "GRACE"
    result_dir.mkdir(parents=True, exist_ok=True)
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
    print("\n=== GRACE 1:1:8 summary ===")
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
            results.append(run_one(dataset_name, seed, config, args, device))
    summarize(results, config)


if __name__ == "__main__":
    main()

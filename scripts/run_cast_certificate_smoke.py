#!/usr/bin/env python
"""Run a minimal CAST latent target-prediction certificate smoke."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.linear_model import LinearRegression
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_grace_1_1_8 import (  # noqa: E402
    create_or_load_split,
    git_dirty,
    git_rev,
    load_dataset,
    logreg_val_eval,
    set_seed,
)
from run_iris_smoke import (  # noqa: E402
    cosine_matrix,
    false_negative_mass,
    label_agreement,
    node_degree,
    relation_overlap,
    relation_smooth_embeddings,
    sparse_row_norm,
    standardize_score,
    structural_signature,
    topk_candidate_mask,
    topk_relations,
    train_grace,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cast_certificate_smoke.yaml")
    parser.add_argument("--base-config", default="configs/grace_1_1_8.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--cert-epochs", type=int, default=None)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--run-tag", default="cast_cert_smoke")
    parser.add_argument("--status", default=None)
    parser.add_argument("--data-root", default="data/pyg")
    parser.add_argument("--split-root", default="splits")
    parser.add_argument("--result-root", default="results/raw")
    parser.add_argument("--summary-root", default="results/summary")
    parser.add_argument("--log-root", default="logs/cast_certificate")
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def as_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


class EgoPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def ego_summary(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    adj = sparse_row_norm(edge_index, x.size(0), x.device)
    neigh = torch.sparse.mm(adj, x)
    degree = node_degree(edge_index, x.size(0)).to(x.device).view(-1, 1)
    degree = degree / degree.max().clamp_min(1.0)
    return torch.cat([x, neigh, (x - neigh).abs(), degree], dim=1)


def train_certificate(
    summary: torch.Tensor,
    edge_index: torch.Tensor,
    target_z: torch.Tensor,
    cfg: dict[str, Any],
    seed: int,
    log_path: Path,
) -> tuple[EgoPredictor, float]:
    set_seed(seed + 7000)
    model = EgoPredictor(
        summary.size(1),
        int(cfg["hidden_dim"]),
        target_z.size(1),
    ).to(summary.device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    edges = edge_index.detach().to(summary.device)
    num_edges = edges.size(1)
    batch_edges = min(int(cfg["batch_edges"]), num_edges)
    epochs = int(cfg["train_epochs"])
    last_loss = 0.0
    with log_path.open("w") as f:
        for epoch in range(1, epochs + 1):
            model.train()
            perm = torch.randint(0, num_edges, (batch_edges,), device=summary.device)
            src = edges[0, perm]
            dst = edges[1, perm]
            pred = model(summary[src])
            if bool(cfg.get("output_normalize", True)):
                pred = F.normalize(pred, dim=1)
            loss = F.mse_loss(pred, target_z[dst])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())
            if epoch == 1 or epoch == epochs or epoch % 25 == 0:
                record = {"epoch": epoch, "loss": last_loss, "time": time.time()}
                f.write(json.dumps(record, sort_keys=True) + "\n")
                f.flush()
                print(f"[certificate] epoch={epoch:04d} loss={last_loss:.6f}", flush=True)
    return model, last_loss


def local_graph_diffusion_score(
    edge_index: torch.Tensor,
    num_nodes: int,
    steps: int,
    decay: float,
) -> np.ndarray:
    edges = edge_index.detach().cpu().numpy()
    neighbors: list[set[int]] = [set() for _ in range(num_nodes)]
    for src, dst in zip(edges[0].tolist(), edges[1].tolist()):
        if src != dst:
            neighbors[src].add(dst)
            neighbors[dst].add(src)

    score = np.full((num_nodes, num_nodes), -np.inf, dtype=np.float32)
    for root in range(num_nodes):
        visited = {root}
        frontier = {root}
        for hop in range(1, steps + 1):
            nxt: set[int] = set()
            for node in frontier:
                nxt.update(neighbors[node])
            nxt.difference_update(visited)
            if not nxt:
                break
            value = decay ** (hop - 1)
            for node in nxt:
                score[root, node] = max(score[root, node], value)
            visited.update(nxt)
            frontier = nxt
    np.fill_diagonal(score, -np.inf)
    return score


def sampled_partial_corr(
    score: np.ndarray,
    labels: np.ndarray,
    controls: list[np.ndarray],
    degree: np.ndarray,
    max_pairs: int,
    seed: int,
) -> float:
    """Estimate label-score partial correlation from finite certificate pairs only."""
    rng = np.random.default_rng(seed + 9100)
    rows = np.flatnonzero(np.isfinite(score).any(axis=1))
    if rows.size == 0:
        return float("nan")

    src: list[int] = []
    dst: list[int] = []
    per_anchor = max(1, max_pairs // max(1, rows.size))
    shuffled_rows = rng.permutation(rows)
    for row in shuffled_rows:
        finite = np.flatnonzero(np.isfinite(score[row]))
        finite = finite[finite != row]
        if finite.size == 0:
            continue
        take = min(per_anchor, finite.size, max_pairs - len(src))
        picked = rng.choice(finite, size=take, replace=False)
        src.extend([int(row)] * take)
        dst.extend([int(x) for x in picked])
        if len(src) >= max_pairs:
            break

    if len(src) < 3:
        return float("nan")
    src_arr = np.asarray(src, dtype=np.int64)
    dst_arr = np.asarray(dst, dtype=np.int64)
    y = (labels[src_arr] == labels[dst_arr]).astype(np.float32)
    x = score[src_arr, dst_arr].astype(np.float32)
    finite_mask = np.isfinite(x) & np.isfinite(y)
    if finite_mask.sum() < 3:
        return float("nan")

    design_parts = []
    for control in controls:
        design_parts.append(control[src_arr, dst_arr].astype(np.float32))
    design_parts.append(np.abs(degree[src_arr] - degree[dst_arr]).astype(np.float32))
    design = np.stack(design_parts, axis=1)
    finite_mask &= np.isfinite(design).all(axis=1)
    if finite_mask.sum() < design.shape[1] + 3:
        return float("nan")

    x_fit = x[finite_mask]
    y_fit = y[finite_mask]
    design_fit = design[finite_mask]
    x_res = x_fit - LinearRegression().fit(design_fit, x_fit).predict(design_fit)
    y_res = y_fit - LinearRegression().fit(design_fit, y_fit).predict(design_fit)
    if np.std(x_res) < 1e-8 or np.std(y_res) < 1e-8:
        return float("nan")
    return float(np.corrcoef(x_res, y_res)[0, 1])


@torch.no_grad()
def certificate_energy(
    model: EgoPredictor,
    summary: torch.Tensor,
    target_z: torch.Tensor,
    candidate_mask: np.ndarray,
    alpha_values: list[float],
    normalize_output: bool,
) -> np.ndarray:
    model.eval()
    n = summary.size(0)
    energy = np.full((n, n), np.inf, dtype=np.float32)
    for i in range(n):
        cand = np.where(candidate_mask[i])[0]
        if cand.size == 0:
            continue
        cand_t = torch.tensor(cand, dtype=torch.long, device=summary.device)
        best = None
        for alpha in alpha_values:
            mixed = (1.0 - alpha) * summary[i].view(1, -1) + alpha * summary[cand_t]
            pred = model(mixed)
            if normalize_output:
                pred = F.normalize(pred, dim=1)
            err = (pred - target_z[cand_t]).pow(2).mean(dim=1)
            best = err if best is None else torch.minimum(best, err)
        energy[i, cand] = as_numpy(best)
    np.fill_diagonal(energy, np.inf)
    return energy


def summarize_markdown(results: list[dict[str, Any]], path: Path, dataset_name: str, seed: int) -> None:
    lines = [
        "# CAST Certificate Smoke Summary",
        "",
        f"本文件为 {dataset_name} seed={seed} smoke/diagnostic 汇总，不支持 formal、SOTA 或 robust claim。",
        "",
        "| ID | Variant | Test@best-val | Label agreement | FN mass after | kNN overlap | PPR overlap | CAST overlap | Cert pcorr |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        diag = result["diagnostics"]
        eval_result = result["evaluation"]
        lines.append(
            f"| {result['variant_id']} | {result['variant_name']} | "
            f"{eval_result['test_at_best']:.2f} | "
            f"{diag['pair_label_agreement']:.4f} | "
            f"{diag['false_negative_mass']['same_label_negative_mass_after_closure']:.4f} | "
            f"{diag['overlap_with_knn_C1']:.4f} | "
            f"{diag['overlap_with_ppr_C6']:.4f} | "
            f"{diag['overlap_with_cast_C2']:.4f} | "
            f"{diag['certificate_partial_corr_label_after_controls']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    config = load_yaml(PROJECT_ROOT / args.config)
    base_config = load_yaml(PROJECT_ROOT / args.base_config)
    dataset_name = args.dataset or config["dataset"]["name"]
    seed = int(args.seed if args.seed is not None else config["dataset"]["seed"])
    allowed_datasets = {"Cora", "CiteSeer", "PubMed"}
    allowed_seeds = {0, 1, 2}
    if dataset_name not in allowed_datasets or seed not in allowed_seeds:
        raise ValueError(
            "CAST certificate Pilot-A gate currently allows only "
            "Cora/CiteSeer/PubMed seeds 0-2."
        )

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset(dataset_name, PROJECT_ROOT / args.data_root)
    data = dataset[0].to(device)
    split = create_or_load_split(dataset_name, data.y.detach().cpu(), seed, PROJECT_ROOT / args.split_root)
    pretrain_cfg = dict(base_config["pretrain"][dataset_name])
    pretrain_cfg.update(config.get("pretrain", {}).get(dataset_name, {}))
    if args.epochs is not None:
        pretrain_cfg["num_epochs"] = int(args.epochs)
    cert_cfg = dict(config["certificate"])
    diag_cfg = dict(config.get("diagnostics", {}))
    if args.cert_epochs is not None:
        cert_cfg["train_epochs"] = int(args.cert_epochs)
    eval_cfg = dict(base_config["logreg_eval"])
    status = args.status or config["meta"].get("status_default", "smoke")

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_name = f"{args.run_tag}_{dataset_name}_seed{seed}_{timestamp}"
    result_dir = PROJECT_ROOT / args.result_root / dataset_name / "CAST_CERTIFICATE_SMOKE" / run_name
    summary_dir = PROJECT_ROOT / args.summary_root
    log_dir = PROJECT_ROOT / args.log_root / run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    start_all = time.time()
    _, z, last_grace_loss = train_grace(
        dataset, data, pretrain_cfg, device, seed, log_dir / "grace_warmup.jsonl", args.log_every
    )
    target_z = torch.tensor(z, device=device, dtype=torch.float32)
    summary = ego_summary(data.x, data.edge_index).detach()
    cert_model, last_cert_loss = train_certificate(
        summary,
        data.edge_index,
        target_z,
        cert_cfg,
        seed,
        log_dir / "certificate_train.jsonl",
    )

    labels = as_numpy(data.y).astype(int)
    features = as_numpy(data.x).astype(np.float32)
    graph_sim = local_graph_diffusion_score(
        data.edge_index,
        int(data.num_nodes),
        int(diag_cfg.get("graph_diffusion_steps", 3)),
        float(diag_cfg.get("graph_diffusion_decay", 0.6)),
    )
    degree = as_numpy(node_degree(data.edge_index, int(data.num_nodes))).astype(np.float32)
    struct = structural_signature(features, degree, graph_sim)
    emb_sim = cosine_matrix(z)
    feat_sim = cosine_matrix(features)
    struct_sim = cosine_matrix(struct)
    cast_score = standardize_score(0.4 * feat_sim + 0.35 * emb_sim + 0.25 * struct_sim)
    pool_mask = topk_candidate_mask(cast_score, int(cert_cfg["candidate_pool_size"]))
    energy = certificate_energy(
        cert_model,
        summary,
        target_z,
        pool_mask,
        [float(a) for a in cert_cfg["alpha_values"]],
        bool(cert_cfg.get("output_normalize", True)),
    )
    cert_score = standardize_score(-energy)
    cert_plus_cast = standardize_score(0.7 * cast_score + 0.3 * cert_score)

    budget = int(cert_cfg["positive_budget"])
    relations = {
        "C0": np.zeros_like(emb_sim, dtype=bool),
        "C1": topk_relations(emb_sim, budget),
        "C2": topk_relations(cast_score, budget),
        "C3": topk_relations(cast_score, budget, pool_mask),
        "C6": topk_relations(graph_sim, budget),
        "C4": topk_relations(cert_score, budget, pool_mask),
        "C5": topk_relations(cert_plus_cast, budget, pool_mask),
    }
    certificate_partial_corr = sampled_partial_corr(
        cert_score,
        labels,
        [feat_sim, emb_sim, graph_sim],
        degree,
        int(diag_cfg.get("partial_corr_max_pairs", 250000)),
        seed,
    )

    results: list[dict[str, Any]] = []
    for variant in config["variants"]:
        rel = relations[variant["id"]]
        z_eval = z if variant["family"] == "grace_base" else relation_smooth_embeddings(
            z, rel, float(cert_cfg["relation_smoothing_weight"])
        )
        eval_result = logreg_val_eval(torch.tensor(z_eval), data.y.cpu(), split, eval_cfg, seed)
        diagnostics = {
            **label_agreement(rel, labels),
            "false_negative_mass": false_negative_mass(z, labels, rel, tau=0.4),
            "overlap_with_knn_C1": relation_overlap(rel, relations["C1"]),
            "overlap_with_ppr_C6": relation_overlap(rel, relations["C6"]),
            "overlap_with_cast_C2": relation_overlap(rel, relations["C2"]),
            "certificate_partial_corr_label_after_controls": certificate_partial_corr,
            "label_diagnostics_are_offline_only": True,
        }
        if variant["id"] in {"C4", "C5"}:
            selected_energy = energy[rel]
            diagnostics["certificate_energy_mean"] = float(np.mean(selected_energy)) if selected_energy.size else None
        result = {
            "run_id": f"{run_name}_{variant['id']}",
            "method": "CAST-Certificate-smoke",
            "variant_id": variant["id"],
            "variant_name": variant["name"],
            "variant_family": variant["family"],
            "dataset": dataset_name,
            "split_type": "stratified_random_1_1_8",
            "split_seed": seed,
            "model_seed": seed,
            "status": status,
            "metric": "accuracy_percent",
            "config_path": str((PROJECT_ROOT / args.config).resolve()),
            "base_config_path": str((PROJECT_ROOT / args.base_config).resolve()),
            "split_path": str(split.path.resolve()),
            "project_commit_hash": git_rev(PROJECT_ROOT),
            "project_dirty": git_dirty(PROJECT_ROOT),
            "dataset_num_nodes": int(data.num_nodes),
            "dataset_num_edges": int(data.edge_index.size(1)),
            "dataset_num_features": int(dataset.num_features),
            "dataset_num_classes": int(dataset.num_classes),
            "pretrain_config": pretrain_cfg,
            "certificate_config": cert_cfg,
            "diagnostics_config": diag_cfg,
            "last_grace_loss": last_grace_loss,
            "last_certificate_loss": last_cert_loss,
            "evaluator_type": "logreg_val",
            "evaluator_config": eval_cfg,
            "diagnostics": diagnostics,
            "evaluation": eval_result,
            "elapsed_sec": time.time() - start_all,
        }
        result_path = result_dir / f"{variant['id']}_{variant['name']}.json"
        result["result_path"] = str(result_path.resolve())
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        results.append(result)
        print(
            f"[{variant['id']} {variant['name']}] "
            f"test={eval_result['test_at_best']:.2f} "
            f"agree={diagnostics['pair_label_agreement']:.4f}",
            flush=True,
        )

    payload = {
        "run_name": run_name,
        "status": status,
        "dataset": dataset_name,
        "seed": seed,
        "config_path": str((PROJECT_ROOT / args.config).resolve()),
        "base_config_path": str((PROJECT_ROOT / args.base_config).resolve()),
        "result_dir": str(result_dir.resolve()),
        "log_dir": str(log_dir.resolve()),
        "results": results,
        "bridge_decision_inputs": {
            "smoke_only": True,
            "certificate_partial_corr_label_after_controls": certificate_partial_corr,
            "elapsed_sec": time.time() - start_all,
        },
    }
    summary_json = summary_dir / f"{run_name}_summary.json"
    summary_md = summary_dir / f"{run_name}_summary.md"
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    summarize_markdown(results, summary_md, dataset_name, seed)
    print(f"summary_json={summary_json}")
    print(f"summary_md={summary_md}")


if __name__ == "__main__":
    main()

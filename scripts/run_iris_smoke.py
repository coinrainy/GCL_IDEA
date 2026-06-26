#!/usr/bin/env python
"""Run the IRIS-GCL Cora seed=0 relation-diagnostic smoke."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_grace_1_1_8 import (  # noqa: E402
    Encoder,
    Model,
    activation_from_name,
    create_or_load_split,
    drop_edges,
    drop_feature,
    encode,
    git_dirty,
    git_rev,
    load_dataset,
    logreg_val_eval,
    set_seed,
    train_grace_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/iris_smoke.yaml")
    parser.add_argument("--base-config", default="configs/grace_1_1_8.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--status", default=None)
    parser.add_argument("--run-tag", default="iris_smoke")
    parser.add_argument("--data-root", default="data/pyg")
    parser.add_argument("--split-root", default="splits")
    parser.add_argument("--result-root", default="results/raw")
    parser.add_argument("--summary-root", default="results/summary")
    parser.add_argument("--log-root", default="logs/iris_smoke")
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def as_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def cosine_matrix(x: np.ndarray) -> np.ndarray:
    z = normalize_rows(x.astype(np.float32, copy=False))
    sim = z @ z.T
    np.fill_diagonal(sim, -np.inf)
    return sim


def dense_diffusion(edge_index: torch.Tensor, num_nodes: int, alpha: float, steps: int) -> np.ndarray:
    edges = to_undirected(edge_index.detach().cpu(), num_nodes=num_nodes)
    row = edges[0].numpy()
    col = edges[1].numpy()
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    adj[row, col] = 1.0
    np.fill_diagonal(adj, 1.0)
    deg = adj.sum(axis=1, keepdims=True)
    trans = adj / np.maximum(deg, 1.0)
    power = np.eye(num_nodes, dtype=np.float32)
    diffusion = alpha * power
    coeff = 1.0
    for _ in range(1, steps + 1):
        power = power @ trans
        coeff *= 1.0 - alpha
        diffusion += alpha * coeff * power
    np.fill_diagonal(diffusion, -np.inf)
    return diffusion


def sparse_row_norm(edge_index: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
    edges = to_undirected(edge_index, num_nodes=num_nodes)
    loops = torch.arange(num_nodes, device=device)
    loop_index = torch.stack([loops, loops], dim=0)
    edges = torch.cat([edges.to(device), loop_index], dim=1)
    values = torch.ones(edges.size(1), device=device)
    adj = torch.sparse_coo_tensor(edges, values, (num_nodes, num_nodes), device=device).coalesce()
    row = adj.indices()[0]
    deg = torch.sparse.sum(adj, dim=1).to_dense().clamp_min(1.0)
    values = adj.values() / deg[row]
    return torch.sparse_coo_tensor(adj.indices(), values, adj.shape, device=device).coalesce()


def local_smoothness(z: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    adj = sparse_row_norm(edge_index, num_nodes, z.device)
    neigh = torch.sparse.mm(adj, z)
    return (z - neigh).pow(2).mean(dim=1, keepdim=True)


@torch.no_grad()
def build_response_fingerprint(
    model: Model,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    base_z: torch.Tensor,
    cfg: dict[str, Any],
    seed: int,
) -> np.ndarray:
    model.eval()
    num_nodes = x.size(0)
    adj = sparse_row_norm(edge_index, num_nodes, x.device)
    x_low = torch.sparse.mm(adj, x)
    x_high = (x + float(cfg["high_frequency_scale"]) * (x - x_low)).clamp_min(0.0)

    set_seed(seed + 17)
    x_mask = drop_feature(x, float(cfg["feature_mask_rate"]))
    edge_drop = drop_edges(edge_index, float(cfg["edge_drop_rate"]))

    degree = node_degree(edge_index, num_nodes).to(x.device)
    perm = torch.arange(num_nodes, device=x.device)
    for lo, hi in [(0, 2), (2, 5), (5, 10), (10, 10**9)]:
        idx = torch.where((degree >= lo) & (degree < hi))[0]
        if idx.numel() > 1:
            perm[idx] = idx[torch.roll(torch.arange(idx.numel(), device=x.device), shifts=1)]
    x_role_swap = x_low[perm]

    set_seed(seed + 23)
    rewired_edge = drop_edges(edge_index, min(float(cfg["edge_drop_rate"]) * 0.5, 0.8))

    interventions = [
        ("high_frequency", x_high, edge_index),
        ("low_pass", x_low, edge_index),
        ("ego_edge_drop", x, edge_drop),
        ("feature_group_mask", x_mask, edge_index),
        ("neighbor_role_swap", x_role_swap, edge_index),
        ("degree_bin_rewire_proxy", x, rewired_edge),
    ]

    base_smooth = local_smoothness(base_z, edge_index, num_nodes)
    base_consistency = F.cosine_similarity(base_z, torch.sparse.mm(adj, base_z), dim=1).view(-1, 1)
    blocks = []
    for _, x_i, e_i in interventions:
        z_i = F.normalize(model(x_i, e_i), dim=1)
        adj_i = sparse_row_norm(e_i, num_nodes, x.device)
        smooth_i = local_smoothness(z_i, e_i, num_nodes)
        consistency_i = F.cosine_similarity(z_i, torch.sparse.mm(adj_i, z_i), dim=1).view(-1, 1)
        block = torch.cat(
            [
                z_i - base_z,
                float(cfg["local_smoothness_weight"]) * (smooth_i - base_smooth),
                float(cfg["ego_consistency_weight"]) * (consistency_i - base_consistency),
            ],
            dim=1,
        )
        blocks.append(as_numpy(block))

    response = np.concatenate(blocks, axis=1).astype(np.float32)
    response = (response - response.mean(axis=0, keepdims=True)) / np.maximum(
        response.std(axis=0, keepdims=True), 1e-6
    )
    return response


def node_degree(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    edges = to_undirected(edge_index.detach().cpu(), num_nodes=num_nodes)
    deg = torch.bincount(edges[0], minlength=num_nodes).float()
    return deg


def structural_signature(x: np.ndarray, degree: np.ndarray, diffusion: np.ndarray) -> np.ndarray:
    finite_diffusion = np.where(np.isfinite(diffusion), diffusion, 0.0)
    degree_col = degree.reshape(-1, 1).astype(np.float32)
    graph_energy = (finite_diffusion > 0).sum(axis=1, keepdims=True).astype(np.float32)
    feat_norm = np.linalg.norm(x, axis=1, keepdims=True)
    feat_sum = x.sum(axis=1, keepdims=True)
    return np.concatenate([degree_col, graph_energy, feat_norm, feat_sum], axis=1).astype(np.float32)


def quantile_threshold(values: np.ndarray, q: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.quantile(finite, q)) if finite.size else 0.0


def anti_proximity_mask(emb_sim: np.ndarray, graph_sim: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, dict[str, float]]:
    emb_thr = quantile_threshold(emb_sim, float(cfg["embedding_near_quantile"]))
    graph_thr = quantile_threshold(graph_sim, float(cfg["graph_near_quantile"]))
    mask = (emb_sim <= emb_thr) & (graph_sim <= graph_thr)
    np.fill_diagonal(mask, False)
    return mask, {
        "embedding_near_threshold": emb_thr,
        "graph_near_threshold": graph_thr,
        "retained_pair_fraction": float(mask.mean()),
    }


def topk_relations(score: np.ndarray, budget: int, mask: np.ndarray | None = None) -> np.ndarray:
    n = score.shape[0]
    rel = np.zeros((n, n), dtype=bool)
    effective = score.copy()
    if mask is not None:
        effective = np.where(mask, effective, -np.inf)
    np.fill_diagonal(effective, -np.inf)
    for i in range(n):
        row = effective[i]
        finite = np.isfinite(row)
        if not finite.any():
            continue
        k = min(budget, int(finite.sum()))
        idx = np.argpartition(row, -k)[-k:]
        idx = idx[np.argsort(row[idx])[::-1]]
        rel[i, idx] = True
    return rel


def topk_candidate_mask(score: np.ndarray, pool_size: int) -> np.ndarray:
    n = score.shape[0]
    mask = np.zeros((n, n), dtype=bool)
    effective = score.copy()
    np.fill_diagonal(effective, -np.inf)
    for i in range(n):
        row = effective[i]
        finite = np.isfinite(row)
        if not finite.any():
            continue
        k = min(pool_size, int(finite.sum()))
        idx = np.argpartition(row, -k)[-k:]
        mask[i, idx] = True
    return mask


def probabilistic_score(
    emb_sim: np.ndarray,
    feat_sim: np.ndarray,
    graph_sim: np.ndarray,
    max_pairs: int,
    seed: int,
) -> np.ndarray:
    combined = (np.nan_to_num(emb_sim, neginf=-1.0) + np.nan_to_num(feat_sim, neginf=-1.0) + np.nan_to_num(graph_sim, neginf=0.0)) / 3.0
    n = combined.shape[0]
    tri = np.triu_indices(n, k=1)
    values = combined[tri].reshape(-1, 1)
    rng = np.random.default_rng(seed)
    if values.shape[0] > max_pairs:
        values = values[rng.choice(values.shape[0], max_pairs, replace=False)]
    gm = GaussianMixture(n_components=2, random_state=seed, covariance_type="full")
    gm.fit(values)
    high = int(np.argmax(gm.means_.reshape(-1)))
    flat = combined.reshape(-1, 1)
    prob = gm.predict_proba(flat)[:, high].reshape(n, n)
    np.fill_diagonal(prob, -np.inf)
    return prob


def standardize_score(score: np.ndarray) -> np.ndarray:
    out = score.astype(np.float32, copy=True)
    finite = np.isfinite(out)
    if finite.any():
        out[finite] = (out[finite] - out[finite].mean()) / max(out[finite].std(), 1e-6)
    np.fill_diagonal(out, -np.inf)
    return out


def residualize_pair_score(
    target: np.ndarray,
    controls: list[np.ndarray],
    degree: np.ndarray,
    max_fit_pairs: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    n = target.shape[0]
    tri = np.triu_indices(n, k=1)
    valid = np.isfinite(target[tri])
    idx = np.where(valid)[0]
    rng = np.random.default_rng(seed + 313)
    if idx.size > max_fit_pairs:
        idx = rng.choice(idx, max_fit_pairs, replace=False)
    r = tri[0][idx]
    c = tri[1][idx]
    y = target[r, c].astype(np.float32)
    degree_gap = np.abs(degree[r] - degree[c]).astype(np.float32)
    design = np.stack([ctrl[r, c].astype(np.float32) for ctrl in controls] + [degree_gap], axis=1)
    design = np.nan_to_num(design, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    reg = LinearRegression().fit(design, y)

    residual = np.empty_like(target, dtype=np.float32)
    for i in range(n):
        row_degree_gap = np.abs(degree[i] - degree).astype(np.float32)
        row_design = np.stack([ctrl[i].astype(np.float32) for ctrl in controls] + [row_degree_gap], axis=1)
        row_design = np.nan_to_num(row_design, nan=0.0, posinf=0.0, neginf=0.0)
        row_target = np.nan_to_num(target[i].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        residual[i] = row_target - reg.predict(row_design).astype(np.float32)
    residual = 0.5 * (residual + residual.T)
    np.fill_diagonal(residual, -np.inf)
    return standardize_score(residual), {
        "fit_pairs": int(idx.size),
        "controls": ["feature_similarity", "embedding_similarity", "graph_proximity", "degree_gap"],
        "intercept": float(reg.intercept_),
        "coefficients": [float(v) for v in reg.coef_.reshape(-1)],
    }


def relation_smooth_embeddings(z: np.ndarray, rel: np.ndarray, weight: float) -> np.ndarray:
    counts = rel.sum(axis=1, keepdims=True).astype(np.float32)
    neighbor_sum = rel.astype(np.float32) @ z
    neighbor_mean = np.divide(neighbor_sum, np.maximum(counts, 1.0), where=counts >= 0)
    out = z + weight * np.where(counts > 0, neighbor_mean, 0.0)
    out = out / (1.0 + weight * (counts > 0).astype(np.float32))
    return normalize_rows(out.astype(np.float32))


def label_agreement(rel: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    counts = rel.sum(axis=1)
    selected = rel.sum()
    if selected == 0:
        return {
            "mean_sibling_count": 0.0,
            "anchor_coverage": 0.0,
            "pair_label_agreement": 0.0,
            "per_anchor_label_agreement": 0.0,
        }
    same = labels[:, None] == labels[None, :]
    pair_agree = float((rel & same).sum() / selected)
    per_anchor = np.divide((rel & same).sum(axis=1), np.maximum(counts, 1), where=counts >= 0)
    return {
        "mean_sibling_count": float(counts.mean()),
        "anchor_coverage": float((counts > 0).mean()),
        "pair_label_agreement": pair_agree,
        "per_anchor_label_agreement": float(per_anchor[counts > 0].mean()) if (counts > 0).any() else 0.0,
    }


def partial_corr(
    score: np.ndarray,
    labels: np.ndarray,
    controls: list[np.ndarray],
    degree: np.ndarray,
    max_pairs: int,
    seed: int,
) -> float:
    n = score.shape[0]
    tri = np.triu_indices(n, k=1)
    valid = np.isfinite(score[tri])
    idx = np.where(valid)[0]
    rng = np.random.default_rng(seed)
    if idx.size > max_pairs:
        idx = rng.choice(idx, max_pairs, replace=False)
    r = tri[0][idx]
    c = tri[1][idx]
    y = (labels[r] == labels[c]).astype(np.float32)
    x = score[r, c].astype(np.float32)
    control_cols = [ctrl[r, c].astype(np.float32) for ctrl in controls]
    degree_gap = np.abs(degree[r] - degree[c]).astype(np.float32)
    design = np.stack(control_cols + [degree_gap], axis=1)
    design = np.nan_to_num(design, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0
    lr_x = LinearRegression().fit(design, x)
    lr_y = LinearRegression().fit(design, y)
    rx = x - lr_x.predict(design)
    ry = y - lr_y.predict(design)
    denom = np.std(rx) * np.std(ry)
    return float(np.corrcoef(rx, ry)[0, 1]) if denom > 1e-12 else 0.0


def false_negative_mass(z: np.ndarray, labels: np.ndarray, rel: np.ndarray, tau: float) -> dict[str, float]:
    sim = normalize_rows(z) @ normalize_rows(z).T
    logits = np.exp(np.clip(sim / tau, -20.0, 20.0)).astype(np.float64)
    np.fill_diagonal(logits, 0.0)
    same = labels[:, None] == labels[None, :]
    denom_before = logits.sum(axis=1) + 1e-12
    same_before = (logits * same).sum(axis=1) / denom_before
    keep_after = ~rel
    np.fill_diagonal(keep_after, False)
    logits_after = logits * keep_after
    denom_after = logits_after.sum(axis=1) + 1e-12
    same_after = (logits_after * same * keep_after).sum(axis=1) / denom_after
    selected_mass = (logits * rel * same).sum(axis=1) / denom_before
    return {
        "same_label_negative_mass_before": float(same_before.mean()),
        "same_label_negative_mass_after_closure": float(same_after.mean()),
        "same_label_mass_removed_by_closure": float(selected_mass.mean()),
    }


def train_grace(
    dataset,
    data,
    pretrain_cfg: dict[str, Any],
    device: torch.device,
    seed: int,
    log_path: Path,
    log_every: int,
) -> tuple[Model, np.ndarray, float]:
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
    loss_batch_size = int(pretrain_cfg.get("loss_batch_size", 0))
    last_loss = math.nan
    with log_path.open("w") as f:
        for epoch in range(1, int(pretrain_cfg["num_epochs"]) + 1):
            last_loss = train_grace_epoch(
                model, optimizer, data.x, data.edge_index, pretrain_cfg, loss_batch_size
            )
            if epoch == 1 or epoch == int(pretrain_cfg["num_epochs"]) or epoch % log_every == 0:
                record = {"epoch": epoch, "loss": last_loss, "time": time.time()}
                f.write(json.dumps(record, sort_keys=True) + "\n")
                f.flush()
                print(f"[GRACE warmup] epoch={epoch:04d} loss={last_loss:.4f}", flush=True)
    z = as_numpy(encode(model, data.x, data.edge_index, normalize=True))
    return model, z, float(last_loss)


def build_variant_relations(
    variants: list[dict[str, Any]],
    matrices: dict[str, np.ndarray],
    labels: np.ndarray,
    cfg: dict[str, Any],
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    budget = int(cfg["positive_budget"])
    anti_mask, anti_diag = anti_proximity_mask(
        matrices["emb_sim"], matrices["graph_sim"], cfg["anti_proximity"]
    )
    rng = np.random.default_rng(seed + 101)
    relations: dict[str, np.ndarray] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    shuffled = rng.permutation(matrices["response"].shape[0])
    shuffled_response_sim = cosine_matrix(matrices["response"][shuffled])
    prob_score = probabilistic_score(
        matrices["emb_sim"],
        matrices["feat_sim"],
        matrices["graph_sim"],
        int(cfg["diagnostics"]["gmm_sample_pairs"]),
        seed,
    )
    cast_score = 0.4 * matrices["feat_sim"] + 0.35 * matrices["emb_sim"] + 0.25 * matrices["struct_sim"]
    cast_score = standardize_score(cast_score)
    residual_score = matrices["residual_response_sim"]
    soft_penalty = float(cfg["residualization"].get("soft_proximity_penalty", 0.15))
    hybrid_cast_weight = float(cfg["residualization"].get("hybrid_cast_weight", 0.35))
    certification_weight = float(cfg["residualization"].get("certification_weight", 0.3))
    pool_size = int(cfg.get("candidate_pool_size", 128))
    emb_control_score = np.nan_to_num(standardize_score(matrices["emb_sim"]), nan=0.0, posinf=0.0, neginf=0.0)
    graph_control_score = np.nan_to_num(standardize_score(matrices["graph_sim"]), nan=0.0, posinf=0.0, neginf=0.0)
    proximity_score = 0.5 * emb_control_score + 0.5 * graph_control_score
    np.fill_diagonal(proximity_score, 0.0)
    residual_soft_score = standardize_score(residual_score - soft_penalty * proximity_score)
    response_cast_hybrid = standardize_score(
        (1.0 - hybrid_cast_weight) * residual_score + hybrid_cast_weight * cast_score
    )
    certified_knn_score = standardize_score(
        (1.0 - certification_weight) * standardize_score(matrices["emb_sim"])
        + certification_weight * residual_score
    )
    certified_cast_score = standardize_score(
        (1.0 - certification_weight) * cast_score + certification_weight * residual_score
    )
    certified_cast_score_w015 = standardize_score(0.85 * cast_score + 0.15 * residual_score)
    certified_cast_score_w045 = standardize_score(0.55 * cast_score + 0.45 * residual_score)
    certified_knn_score_w015 = standardize_score(0.85 * standardize_score(matrices["emb_sim"]) + 0.15 * residual_score)
    certified_knn_score_w045 = standardize_score(0.55 * standardize_score(matrices["emb_sim"]) + 0.45 * residual_score)
    knn_pool_mask = topk_candidate_mask(matrices["emb_sim"], pool_size)
    cast_pool_mask = topk_candidate_mask(cast_score, pool_size)
    score_by_family = {
        "grace_base": None,
        "embedding_knn": matrices["emb_sim"],
        "graph_diffusion": matrices["graph_sim"],
        "probabilistic_mining": prob_score,
        "cast_proxy": cast_score,
        "iris_full": matrices["response_sim"],
        "iris_response_shuffled": shuffled_response_sim,
        "iris_no_anti_proximity": matrices["response_sim"],
        "iris_structural_only": matrices["struct_sim"],
        "iris_no_gradient_proxy": matrices["response_sim"],
        "residualized_response": residual_score,
        "raw_response_no_residual": matrices["response_sim"],
        "residualized_soft_penalty": residual_soft_score,
        "response_cast_hybrid": response_cast_hybrid,
        "response_reranked_knn_pool": residual_score,
        "response_reranked_cast_pool": residual_score,
        "response_certified_knn_score": certified_knn_score,
        "response_certified_cast_score": certified_cast_score,
        "response_certified_cast_w015": certified_cast_score_w015,
        "response_certified_cast_w045": certified_cast_score_w045,
        "response_certified_knn_w015": certified_knn_score_w015,
        "response_certified_knn_w045": certified_knn_score_w045,
    }
    mask_by_family = {
        "grace_base": None,
        "embedding_knn": None,
        "graph_diffusion": None,
        "probabilistic_mining": None,
        "cast_proxy": None,
        "iris_full": anti_mask,
        "iris_response_shuffled": anti_mask,
        "iris_no_anti_proximity": None,
        "iris_structural_only": anti_mask,
        "iris_no_gradient_proxy": anti_mask,
        "residualized_response": None,
        "raw_response_no_residual": None,
        "residualized_soft_penalty": None,
        "response_cast_hybrid": None,
        "response_reranked_knn_pool": knn_pool_mask,
        "response_reranked_cast_pool": cast_pool_mask,
        "response_certified_knn_score": None,
        "response_certified_cast_score": None,
        "response_certified_cast_w015": None,
        "response_certified_cast_w045": None,
        "response_certified_knn_w015": None,
        "response_certified_knn_w045": None,
    }
    for variant in variants:
        family = variant["family"]
        if family == "grace_base":
            rel = np.zeros_like(matrices["emb_sim"], dtype=bool)
        else:
            rel = topk_relations(score_by_family[family], budget, mask_by_family[family])
        relations[variant["id"]] = rel
        diagnostics[variant["id"]] = {
            "variant_family": family,
            "positive_budget": budget,
            "label_diagnostics_are_offline_only": True,
            "anti_proximity": anti_diag if mask_by_family[family] is not None else None,
            "residualization": matrices.get("residualization_diagnostics") if family.startswith("residualized") or family == "response_cast_hybrid" else None,
            **label_agreement(rel, labels),
        }
    return relations, diagnostics


def markdown_summary(results: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# IRIS-GCL Smoke Summary",
        "",
        "本文件为 Cora seed=0 smoke/diagnostic 汇总，不支持 formal、SOTA 或 robust claim。",
        "",
        "| ID | Variant | Test@best-val | Label agreement | FN mass after | Response pcorr | Residual pcorr | Notes |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        diag = result["diagnostics"]
        eval_res = result["evaluation"]
        note = result.get("implementation_note") or ""
        lines.append(
            f"| {result['variant_id']} | {result['variant_name']} | "
            f"{eval_res['test_at_best']:.2f} | "
            f"{diag['pair_label_agreement']:.4f} | "
            f"{diag['false_negative_mass']['same_label_negative_mass_after_closure']:.4f} | "
            f"{diag['partial_corr_label_response_after_controls']:.4f} | "
            f"{diag['partial_corr_label_residual_response_after_controls']:.4f} | {note} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    config = load_yaml(PROJECT_ROOT / args.config)
    base_config = load_yaml(PROJECT_ROOT / args.base_config)
    dataset_name = args.dataset or config["dataset"]["name"]
    seed = int(args.seed if args.seed is not None else config["dataset"]["seed"])
    if dataset_name != "Cora" or seed != 0:
        raise ValueError("IRIS smoke gate currently allows only Cora seed=0.")

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset(dataset_name, PROJECT_ROOT / args.data_root)
    data = dataset[0].to(device)
    split = create_or_load_split(dataset_name, data.y.detach().cpu(), seed, PROJECT_ROOT / args.split_root)
    pretrain_cfg = dict(base_config["pretrain"][dataset_name])
    pretrain_cfg.update(config.get("pretrain", {}).get(dataset_name, {}))
    if args.epochs is not None:
        pretrain_cfg["num_epochs"] = int(args.epochs)
    eval_cfg = dict(base_config["logreg_eval"])
    status = args.status or config["meta"].get("status_default", "smoke")

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_name = f"{args.run_tag}_{dataset_name}_seed{seed}_{timestamp}"
    result_dir = PROJECT_ROOT / args.result_root / dataset_name / "IRIS_GCL_SMOKE" / run_name
    summary_dir = PROJECT_ROOT / args.summary_root
    log_dir = PROJECT_ROOT / args.log_root / run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    start_all = time.time()
    model, z, last_loss = train_grace(
        dataset, data, pretrain_cfg, device, seed, log_dir / "grace_warmup.jsonl", args.log_every
    )
    labels = as_numpy(data.y).astype(int)
    features = as_numpy(data.x).astype(np.float32)
    iris_cfg = config["iris"]
    diffusion = dense_diffusion(
        data.edge_index,
        int(data.num_nodes),
        float(iris_cfg["diffusion"]["alpha"]),
        int(iris_cfg["diffusion"]["steps"]),
    )
    response = build_response_fingerprint(
        model, data.x, data.edge_index, torch.tensor(z, device=device), iris_cfg["response"], seed
    )
    degree = as_numpy(node_degree(data.edge_index, int(data.num_nodes))).astype(np.float32)
    struct = structural_signature(features, degree, diffusion)
    emb_sim = cosine_matrix(z)
    feat_sim = cosine_matrix(features)
    graph_sim = diffusion
    response_sim = cosine_matrix(response)
    struct_sim = cosine_matrix(struct)
    residual_response_sim, residualization_diagnostics = residualize_pair_score(
        response_sim,
        [feat_sim, emb_sim, graph_sim],
        degree,
        int(iris_cfg["residualization"]["max_fit_pairs"]),
        seed,
    )
    matrices = {
        "emb_sim": emb_sim,
        "feat_sim": feat_sim,
        "graph_sim": graph_sim,
        "response": response,
        "response_sim": response_sim,
        "residual_response_sim": residual_response_sim,
        "residualization_diagnostics": residualization_diagnostics,
        "struct_sim": struct_sim,
    }
    relations, relation_diagnostics = build_variant_relations(
        config["variants"], matrices, labels, iris_cfg, seed
    )

    controls = [matrices["feat_sim"], matrices["emb_sim"], matrices["graph_sim"]]
    response_partial = partial_corr(
        matrices["response_sim"],
        labels,
        controls,
        degree,
        int(iris_cfg["diagnostics"]["partial_corr_max_pairs"]),
        seed,
    )
    residual_response_partial = partial_corr(
        matrices["residual_response_sim"],
        labels,
        controls,
        degree,
        int(iris_cfg["diagnostics"]["partial_corr_max_pairs"]),
        seed,
    )

    results: list[dict[str, Any]] = []
    for variant in config["variants"]:
        rel = relations[variant["id"]]
        z_eval = z if variant["family"] == "grace_base" else relation_smooth_embeddings(
            z, rel, float(iris_cfg["relation_smoothing_weight"])
        )
        eval_result = logreg_val_eval(torch.tensor(z_eval), data.y.cpu(), split, eval_cfg, seed)
        diag = relation_diagnostics[variant["id"]]
        diag["partial_corr_label_response_after_controls"] = response_partial
        diag["partial_corr_label_residual_response_after_controls"] = residual_response_partial
        diag["false_negative_mass"] = false_negative_mass(
            z,
            labels,
            rel,
            float(iris_cfg["diagnostics"]["tau_for_repulsion_mass"]),
        )
        result = {
            "run_id": f"{run_name}_{variant['id']}",
            "method": "IRIS-GCL-smoke",
            "variant_id": variant["id"],
            "variant_name": variant["name"],
            "variant_family": variant["family"],
            "implementation_note": variant.get("implementation_note"),
            "embedding_mode": "relation_smoothed_frozen_grace_diagnostic",
            "dataset": dataset_name,
            "split_type": "stratified_random_1_1_8",
            "split_seed": seed,
            "model_seed": seed,
            "status": status,
            "metric": "accuracy_percent",
            "config_path": str((PROJECT_ROOT / args.config).resolve()),
            "base_config_path": str((PROJECT_ROOT / args.base_config).resolve()),
            "split_path": str(split.path.resolve()),
            "train_log_path": str((log_dir / "grace_warmup.jsonl").resolve()),
            "project_commit_hash": git_rev(PROJECT_ROOT),
            "project_dirty": git_dirty(PROJECT_ROOT),
            "dataset_num_nodes": int(data.num_nodes),
            "dataset_num_edges": int(data.edge_index.size(1)),
            "dataset_num_features": int(dataset.num_features),
            "dataset_num_classes": int(dataset.num_classes),
            "pretrain_config": pretrain_cfg,
            "last_grace_loss": last_loss,
            "evaluator_type": "logreg_val",
            "evaluator_config": eval_cfg,
            "diagnostics": diag,
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
            f"agree={diag['pair_label_agreement']:.4f} "
            f"partial={response_partial:.4f}",
            flush=True,
        )

    summary_payload = {
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
            "response_partial_corr_label_after_controls": response_partial,
            "residual_response_partial_corr_label_after_controls": residual_response_partial,
            "residualization_diagnostics": residualization_diagnostics,
            "elapsed_sec": time.time() - start_all,
        },
    }
    summary_json = summary_dir / f"{run_name}_summary.json"
    summary_md = summary_dir / f"{run_name}_summary.md"
    summary_json.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n")
    markdown_summary(results, summary_md)
    print(f"summary_json={summary_json}")
    print(f"summary_md={summary_md}")


if __name__ == "__main__":
    main()

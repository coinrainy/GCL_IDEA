#!/usr/bin/env python
"""Run ORBIT-GCL OR-M0-001 kill-smoke.

ORBIT replaces node-node negative targets and masked reconstruction with a
node-level operator-response basis target. This runner is intentionally scoped
to Cora seed0 smoke and strong controls from refine-logs/EXPERIMENT_PLAN.md.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
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
    git_dirty,
    git_rev,
    load_dataset,
    logreg_val_eval,
    set_seed,
    train_grace_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/orbit_smoke.yaml")
    parser.add_argument("--base-config", default="configs/grace_1_1_8.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--status", default=None)
    parser.add_argument("--run-tag", default="or_m0_001")
    parser.add_argument("--data-root", default="data/pyg")
    parser.add_argument("--split-root", default="splits")
    parser.add_argument("--result-root", default="results/raw")
    parser.add_argument("--summary-root", default="results/summary")
    parser.add_argument("--log-root", default="logs/orbit_smoke")
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def exact_command() -> str:
    return " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])


def as_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def standardize_np(x: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return ((x - mean) / np.maximum(std, eps)).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def row_normalized_adj(edge_index: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
    edges = to_undirected(edge_index, num_nodes=num_nodes).to(device)
    loops = torch.arange(num_nodes, device=device)
    edges = torch.cat([edges, torch.stack([loops, loops], dim=0)], dim=1)
    values = torch.ones(edges.size(1), device=device)
    adj = torch.sparse_coo_tensor(edges, values, (num_nodes, num_nodes), device=device).coalesce()
    row = adj.indices()[0]
    deg = torch.sparse.sum(adj, dim=1).to_dense().clamp_min(1.0)
    values = adj.values() / deg[row]
    return torch.sparse_coo_tensor(adj.indices(), values, adj.shape, device=device).coalesce()


def smooth_features(x: torch.Tensor, edge_index: torch.Tensor, steps: int = 1) -> torch.Tensor:
    out = x
    adj = row_normalized_adj(edge_index, x.size(0), x.device)
    for _ in range(max(1, int(steps))):
        out = torch.sparse.mm(adj, out)
    return out


def make_model(dataset, cfg: dict[str, Any], device: torch.device) -> Model:
    encoder = Encoder(
        dataset.num_features,
        int(cfg["num_hidden"]),
        activation_from_name(cfg["activation"]),
        base_model=GCNConv,
        k=int(cfg["num_layers"]),
    ).to(device)
    return Model(
        encoder,
        int(cfg["num_hidden"]),
        int(cfg["num_proj_hidden"]),
        float(cfg["tau"]),
    ).to(device)


class OrbitModel(nn.Module):
    def __init__(self, dataset, pretrain_cfg: dict[str, Any], orbit_cfg: dict[str, Any], target_dim: int, num_ops: int) -> None:
        super().__init__()
        hidden = int(pretrain_cfg["num_hidden"])
        proj_hidden = int(orbit_cfg["projector_hidden_dim"])
        dec_hidden = int(orbit_cfg["decoder_hidden_dim"])
        self.tau = float(pretrain_cfg["tau"])
        self.encoder = Encoder(
            dataset.num_features,
            hidden,
            activation_from_name(pretrain_cfg["activation"]),
            base_model=GCNConv,
            k=int(pretrain_cfg["num_layers"]),
        )
        self.predictor = nn.Sequential(nn.Linear(hidden, proj_hidden), nn.ELU(), nn.Linear(proj_hidden, target_dim))
        self.decoder = nn.Sequential(nn.Linear(hidden, dec_hidden), nn.ELU(), nn.Linear(dec_hidden, dataset.num_features))
        self.projector = nn.Sequential(nn.Linear(hidden, proj_hidden), nn.ELU(), nn.Linear(proj_hidden, hidden))
        self.op_head = nn.Linear(hidden, num_ops)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        return F.normalize(z1, dim=1) @ F.normalize(z2, dim=1).T

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        refl = torch.exp(self.sim(z1, z1) / self.tau)
        between = torch.exp(self.sim(z1, z2) / self.tau)
        return -torch.log(between.diag() / (refl.sum(1) + between.sum(1) - refl.diag()).clamp_min(1e-12))

    def contrastive_loss(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        return 0.5 * (self.semi_loss(z1, z2) + self.semi_loss(z2, z1)).mean()


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def feature_frequency_mask(x: torch.Tensor, rate: float) -> torch.Tensor:
    x_new = x.clone()
    k = max(1, int(x.size(1) * float(rate)))
    freq = x.abs().sum(dim=0)
    cols = torch.topk(freq, k=min(k, x.size(1))).indices
    x_new[:, cols] = 0.0
    return x_new


def rewire_edges_same_degree(edge_index: torch.Tensor, num_nodes: int, rate: float, seed: int) -> torch.Tensor:
    edge = to_undirected(edge_index, num_nodes=num_nodes).detach().cpu()
    src, dst = edge[0].numpy().copy(), edge[1].numpy().copy()
    degree = np.bincount(src, minlength=num_nodes)
    quantiles = np.quantile(degree, [0.25, 0.5, 0.75])
    buckets = np.digitize(degree, quantiles, right=True)
    rng = np.random.default_rng(seed + 12_017)
    mask = rng.random(dst.shape[0]) < float(rate)
    bucket_to_nodes = {b: np.where(buckets == b)[0] for b in range(4)}
    for idx in np.where(mask)[0]:
        candidates = bucket_to_nodes[int(buckets[dst[idx]])]
        if candidates.size > 0:
            dst[idx] = int(rng.choice(candidates))
    out = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long, device=edge_index.device)
    return to_undirected(out, num_nodes=num_nodes)


def ego_crop_edges(edge_index: torch.Tensor, x: torch.Tensor, drop_rate: float, seed: int) -> torch.Tensor:
    edge = to_undirected(edge_index, num_nodes=x.size(0))
    src, dst = edge[0], edge[1]
    feat_norm = x.norm(dim=1)
    diff = (feat_norm[src] - feat_norm[dst]).abs()
    if diff.numel() == 0:
        return edge
    threshold = torch.quantile(diff.detach(), 1.0 - float(drop_rate))
    keep = diff <= threshold
    if int(keep.sum().item()) == 0:
        return edge
    return edge[:, keep]


def positional_perturb(x: torch.Tensor, edge_index: torch.Tensor, scale: float) -> torch.Tensor:
    edge = to_undirected(edge_index, num_nodes=x.size(0))
    degree = torch.bincount(edge[0], minlength=x.size(0)).float().to(x.device)
    pos = (degree - degree.mean()) / degree.std().clamp_min(1e-6)
    smooth_pos = smooth_features(pos[:, None], edge_index, steps=2).squeeze(1)
    signal = (pos + smooth_pos)[:, None]
    direction = torch.linspace(-1.0, 1.0, x.size(1), device=x.device)[None, :]
    return x + float(scale) * signal * direction


def build_operator_views(data, cfg: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    op_cfg = cfg["operator_bank"]
    smooth = smooth_features(data.x, data.edge_index, steps=1)
    return [
        {
            "name": "feature_frequency_mask",
            "x": feature_frequency_mask(data.x, float(op_cfg["feature_frequency_mask_rate"])),
            "edge_index": data.edge_index,
            "heldout": False,
        },
        {
            "name": "degree_preserving_edge_rewire",
            "x": data.x,
            "edge_index": rewire_edges_same_degree(data.edge_index, data.num_nodes, float(op_cfg["edge_rewire_rate"]), seed),
            "heldout": False,
        },
        {
            "name": "diffusion_temperature_shift",
            "x": (1.0 - float(op_cfg["diffusion_alpha"])) * data.x + float(op_cfg["diffusion_alpha"]) * smooth,
            "edge_index": data.edge_index,
            "heldout": False,
        },
        {
            "name": "ego_radius_crop",
            "x": data.x,
            "edge_index": ego_crop_edges(data.edge_index, data.x, float(op_cfg["ego_crop_drop_rate"]), seed),
            "heldout": False,
        },
        {
            "name": "positional_perturbation",
            "x": positional_perturb(data.x, data.edge_index, float(op_cfg["positional_scale"])),
            "edge_index": data.edge_index,
            "heldout": True,
        },
    ]


@torch.no_grad()
def encode_any(model: OrbitModel | Model, x: torch.Tensor, edge_index: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    model.eval()
    h = model(x, edge_index)
    return F.normalize(h, dim=1) if normalize else h


@torch.no_grad()
def compute_teacher_responses(
    teacher: Model,
    data,
    operators: list[dict[str, Any]],
    basis_dim: int,
) -> dict[str, Any]:
    teacher.eval()
    h0 = teacher(data.x, data.edge_index)
    train_responses = []
    heldout_response = None
    for op in operators:
        hm = teacher(op["x"], op["edge_index"])
        resp = hm - h0
        if op["heldout"]:
            heldout_response = resp
        else:
            train_responses.append(resp)
    raw = torch.cat(train_responses, dim=1)
    raw_np = as_numpy(raw).astype(np.float64)
    raw_std, raw_mean, raw_scale = standardize_np(raw_np)
    centered = raw_std - raw_std.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    dim = min(int(basis_dim), vt.shape[0])
    basis = centered @ vt[:dim].T
    basis, basis_mean, basis_scale = standardize_np(basis)
    total = float(np.square(s).sum())
    explained = float(np.square(s[:dim]).sum() / max(total, 1e-12))
    if heldout_response is None:
        heldout_response = train_responses[-1]
    heldout_np, _, _ = standardize_np(as_numpy(heldout_response).astype(np.float64))
    magnitude = torch.stack([resp.norm(dim=1) for resp in train_responses], dim=1)
    magnitude_np, _, _ = standardize_np(as_numpy(magnitude))
    mean_response = torch.stack(train_responses, dim=0).mean(dim=0)
    mean_np, _, _ = standardize_np(as_numpy(mean_response))
    return {
        "raw": torch.tensor(raw_std, dtype=torch.float32, device=data.x.device),
        "basis": torch.tensor(basis, dtype=torch.float32, device=data.x.device),
        "magnitude": torch.tensor(magnitude_np, dtype=torch.float32, device=data.x.device),
        "mean_response": torch.tensor(mean_np, dtype=torch.float32, device=data.x.device),
        "heldout": torch.tensor(heldout_np, dtype=torch.float32, device=data.x.device),
        "basis_explained_variance": explained,
        "basis_effective_rank": effective_rank(basis),
        "basis_mean": basis_mean.tolist(),
        "basis_scale": basis_scale.tolist(),
        "raw_mean": raw_mean.tolist(),
        "raw_scale": raw_scale.tolist(),
        "train_operator_names": [op["name"] for op in operators if not op["heldout"]],
        "heldout_operator_names": [op["name"] for op in operators if op["heldout"]],
    }


def build_shortcut_targets(data, seed: int) -> dict[str, torch.Tensor]:
    x = data.x
    edge = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    degree = torch.bincount(edge[0], minlength=data.num_nodes).float().to(x.device)
    feat_norm = x.norm(dim=1)
    smooth_norm = smooth_features(x, data.edge_index, steps=1).norm(dim=1)
    residual_norm = (x - smooth_features(x, data.edge_index, steps=1)).norm(dim=1)
    shortcut = torch.stack([degree, feat_norm, smooth_norm, residual_norm], dim=1)
    shortcut_np, _, _ = standardize_np(as_numpy(shortcut))
    rng = np.random.default_rng(seed + 88_777)
    return {
        "degree_feature": torch.tensor(shortcut_np, dtype=torch.float32, device=x.device),
        "random_operator": torch.tensor(rng.normal(size=(x.size(0), 4)).astype(np.float32), device=x.device),
    }


def make_weak_view(data, cfg: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    x = drop_feature(data.x, float(cfg["feature_drop_rate"]))
    edge = drop_edges(data.edge_index, float(cfg["edge_drop_rate"]))
    return x, edge


def normalize_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape[1] == target.shape[1]:
        return F.mse_loss(F.normalize(pred, dim=1), F.normalize(target.detach(), dim=1))
    return F.mse_loss(pred, target.detach())


def effective_rank(z: np.ndarray) -> float:
    x = z.astype(np.float64) - z.astype(np.float64).mean(axis=0, keepdims=True)
    s = np.linalg.svd(x, full_matrices=False, compute_uv=False)
    p = s / max(float(s.sum()), 1e-12)
    return float(np.exp(-np.sum(p * np.log(np.maximum(p, 1e-12)))))


def covariance_redundancy(z: np.ndarray) -> float:
    x = z.astype(np.float64)
    x = (x - x.mean(axis=0, keepdims=True)) / np.maximum(x.std(axis=0, keepdims=True), 1e-9)
    c = (x.T @ x) / max(1, x.shape[0] - 1)
    mask = ~np.eye(c.shape[0], dtype=bool)
    return float(np.abs(c[mask]).mean())


def uniformity_score(z: np.ndarray, max_pairs: int, seed: int) -> float:
    x = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    n = x.shape[0]
    rng = np.random.default_rng(seed + 71)
    row = rng.integers(0, n, size=min(max_pairs, n * 80))
    col = rng.integers(0, n, size=row.shape[0])
    keep = row != col
    dist2 = ((x[row[keep]] - x[col[keep]]) ** 2).sum(axis=1)
    return float(np.log(np.exp(-2.0 * dist2).mean() + 1e-12))


@torch.no_grad()
def alignment_score(model: OrbitModel | Model, data, cfg: dict[str, Any], seed: int) -> float:
    model.eval()
    set_seed(seed + 99)
    x1 = drop_feature(data.x, 0.2)
    x2 = drop_feature(data.x, 0.3)
    e1 = drop_edges(data.edge_index, 0.2)
    e2 = drop_edges(data.edge_index, 0.4)
    h1 = F.normalize(model(x1, e1), dim=1)
    h2 = F.normalize(model(x2, e2), dim=1)
    return float((h1 - h2).pow(2).sum(dim=1).mean().item())


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def partial_corr(a: np.ndarray, b: np.ndarray, controls: np.ndarray) -> float:
    controls = np.asarray(controls, dtype=np.float64)
    controls = np.column_stack([np.ones(controls.shape[0]), controls])
    beta_a = np.linalg.lstsq(controls, a, rcond=None)[0]
    beta_b = np.linalg.lstsq(controls, b, rcond=None)[0]
    return corr(a - controls @ beta_a, b - controls @ beta_b)


def shortcut_diagnostics(score: np.ndarray, data) -> dict[str, float]:
    x = as_numpy(data.x)
    edge = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    degree = as_numpy(torch.bincount(edge[0], minlength=data.num_nodes).float())
    feature_norm = np.linalg.norm(x, axis=1)
    smooth_norm = as_numpy(smooth_features(data.x, data.edge_index, steps=1).norm(dim=1))
    controls = np.column_stack([degree, feature_norm, smooth_norm])
    return {
        "corr_degree": corr(score, degree),
        "corr_feature_norm": corr(score, feature_norm),
        "corr_diffusion_norm": corr(score, smooth_norm),
        "partial_corr_degree_given_feature_diffusion": partial_corr(score, degree, controls[:, 1:]),
    }


def heldout_response_error(pred: np.ndarray, heldout: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray) -> float | None:
    if pred.ndim != 2 or heldout.ndim != 2 or pred.shape[0] != heldout.shape[0]:
        return None
    x_train = np.column_stack([np.ones(train_idx.shape[0]), pred[train_idx]])
    x_val = np.column_stack([np.ones(val_idx.shape[0]), pred[val_idx]])
    y_train = heldout[train_idx]
    y_val = heldout[val_idx]
    try:
        w = np.linalg.lstsq(x_train, y_train, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    err = np.square(x_val @ w - y_val).mean()
    return float(err)


def train_grace_variant(
    variant: dict[str, Any],
    dataset,
    data,
    cfg: dict[str, Any],
    orbit_cfg: dict[str, Any],
    device: torch.device,
    seed: int,
    log_path: Path,
    log_every: int,
) -> tuple[Model, dict[str, Any]]:
    set_seed(seed)
    train_cfg = dict(cfg)
    if variant["type"] == "grace_matched":
        train_cfg["num_proj_hidden"] = int(orbit_cfg["matched_grace_proj_hidden"])
    model = make_model(dataset, train_cfg, device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    last = math.nan
    start = time.time()
    with log_path.open("w") as f:
        for epoch in range(1, int(cfg["num_epochs"]) + 1):
            last = train_grace_epoch(model, opt, data.x, data.edge_index, train_cfg, int(train_cfg.get("loss_batch_size", 0)))
            if epoch == 1 or epoch == int(cfg["num_epochs"]) or epoch % log_every == 0:
                f.write(json.dumps({"epoch": epoch, "loss": float(last), "time": time.time()}, sort_keys=True) + "\n")
                f.flush()
                print(f"[{variant['id']} {variant['name']}] epoch={epoch:04d}/{int(cfg['num_epochs'])} loss={last:.4f} elapsed={time.time()-start:.1f}s", flush=True)
    return model, {
        "last_train_loss": float(last),
        "epochs": int(cfg["num_epochs"]),
        "elapsed_sec": time.time() - start,
        "model_seed": int(seed),
        "parameter_count": count_parameters(model),
        "train_config": train_cfg,
    }


def target_for_variant(
    variant_type: str,
    targets: dict[str, Any],
    shortcuts: dict[str, torch.Tensor],
    data,
    seed: int,
) -> torch.Tensor | None:
    if variant_type in {"orbit_full"}:
        return targets["basis"]
    if variant_type == "shuffled_response_basis":
        gen = torch.Generator(device=data.x.device)
        gen.manual_seed(seed + 44_123)
        return targets["basis"][torch.randperm(data.num_nodes, generator=gen, device=data.x.device)]
    if variant_type == "random_teacher_basis":
        return targets["random_basis"]
    if variant_type == "random_operator_labels":
        return shortcuts["random_operator"]
    if variant_type == "response_magnitude" or variant_type == "discrepancy_regression":
        return targets["magnitude"]
    if variant_type == "degree_feature_shortcut":
        return shortcuts["degree_feature"]
    if variant_type == "same_node_response" or variant_type == "graph_jepa":
        return targets["mean_response"]
    if variant_type == "raw_response":
        return targets["raw"]
    if variant_type == "graphmae2_lite":
        return targets["teacher_clean"]
    return None


def train_orbit_variant(
    variant: dict[str, Any],
    dataset,
    data,
    pretrain_cfg: dict[str, Any],
    orbit_cfg: dict[str, Any],
    targets: dict[str, Any],
    shortcuts: dict[str, torch.Tensor],
    operators: list[dict[str, Any]],
    device: torch.device,
    seed: int,
    log_path: Path,
    log_every: int,
) -> tuple[OrbitModel, dict[str, Any]]:
    variant_type = variant["type"]
    target = target_for_variant(variant_type, targets, shortcuts, data, seed)
    if variant_type == "operator_id":
        target_dim = int(pretrain_cfg["num_hidden"])
    elif variant_type == "gcmae_hybrid":
        target_dim = int(pretrain_cfg["num_hidden"])
    elif target is None:
        target_dim = int(pretrain_cfg["num_hidden"])
    else:
        target_dim = int(target.size(1))
    set_seed(seed)
    model = OrbitModel(dataset, pretrain_cfg, orbit_cfg, target_dim, len(operators)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(orbit_cfg["learning_rate"]), weight_decay=float(orbit_cfg["weight_decay"]))
    last: dict[str, float] = {}
    start = time.time()
    with log_path.open("w") as f:
        for epoch in range(1, int(pretrain_cfg["num_epochs"]) + 1):
            model.train()
            opt.zero_grad(set_to_none=True)
            xw, ew = make_weak_view(data, orbit_cfg)
            h = model(xw, ew)
            loss = h.new_tensor(0.0)
            diag: dict[str, float] = {}
            if variant_type == "gcmae_hybrid":
                x2, e2 = make_weak_view(data, orbit_cfg)
                h2 = model(x2, e2)
                recon = F.mse_loss(model.decoder(h), data.x)
                contrast = model.contrastive_loss(h, h2)
                loss = float(orbit_cfg["lambda_recon"]) * recon + float(orbit_cfg["lambda_contrast"]) * contrast
                diag = {"recon_loss": float(recon.detach().item()), "contrast_loss": float(contrast.detach().item())}
            elif variant_type == "operator_id":
                losses = []
                accs = []
                for op_id, op in enumerate(operators):
                    hop = model(op["x"], op["edge_index"])
                    logits = model.op_head(hop)
                    labels = torch.full((data.num_nodes,), op_id, dtype=torch.long, device=device)
                    losses.append(F.cross_entropy(logits, labels))
                    accs.append((logits.argmax(dim=1) == labels).float().mean())
                loss = torch.stack(losses).mean()
                diag = {"operator_id_acc": float(torch.stack(accs).mean().detach().item() * 100.0)}
            elif target is not None:
                pred = model.predictor(h)
                loss = float(orbit_cfg["lambda_target"]) * normalize_loss(pred, target)
                diag = {"target_mse": float(F.mse_loss(pred, target.detach()).detach().item())}
            else:
                raise ValueError(f"Unsupported variant type: {variant_type}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(orbit_cfg["max_grad_norm"]))
            opt.step()
            last = {"loss": float(loss.detach().item()), **diag}
            if epoch == 1 or epoch == int(pretrain_cfg["num_epochs"]) or epoch % log_every == 0:
                f.write(json.dumps({"epoch": epoch, **last, "time": time.time()}, sort_keys=True) + "\n")
                f.flush()
                print(f"[{variant['id']} {variant['name']}] epoch={epoch:04d}/{int(pretrain_cfg['num_epochs'])} loss={last['loss']:.4f} elapsed={time.time()-start:.1f}s", flush=True)
    return model, {
        "last_train_loss": last.get("loss"),
        "last_train_diagnostics": last,
        "epochs": int(pretrain_cfg["num_epochs"]),
        "elapsed_sec": time.time() - start,
        "model_seed": int(seed),
        "parameter_count": count_parameters(model),
    }


def evaluate(
    model: OrbitModel | Model,
    variant_type: str,
    data,
    split,
    eval_cfg: dict[str, Any],
    targets: dict[str, Any],
    shortcuts: dict[str, torch.Tensor],
    seed: int,
    max_pairs: int,
) -> dict[str, Any]:
    h = encode_any(model, data.x, data.edge_index, normalize=bool(eval_cfg["normalize_embeddings"]))
    z = as_numpy(h)
    eval_result = logreg_val_eval(h, data.y, split, eval_cfg, seed)
    pred_np = None
    response_target_error = None
    if isinstance(model, OrbitModel):
        with torch.no_grad():
            pred = model.predictor(model(data.x, data.edge_index))
            pred_np = as_numpy(pred)
            target = target_for_variant(variant_type, targets, shortcuts, data, seed)
            if target is not None and pred.shape == target.shape:
                response_target_error = float(F.mse_loss(pred, target).detach().item())
    train_idx = split.train.detach().cpu().numpy()
    val_idx = split.val.detach().cpu().numpy()
    heldout_err = heldout_response_error(
        pred_np if pred_np is not None else z,
        as_numpy(targets["heldout"]),
        train_idx,
        val_idx,
    )
    score = np.linalg.norm(pred_np, axis=1) if pred_np is not None else np.linalg.norm(z, axis=1)
    return {
        **eval_result,
        "accuracy": eval_result["test_at_best"],
        "response_target_mse": response_target_error,
        "heldout_operator_response_val_mse": heldout_err,
        "response_basis_explained_variance": targets["basis_explained_variance"],
        "response_basis_effective_rank": targets["basis_effective_rank"],
        "effective_rank": effective_rank(z),
        "alignment": alignment_score(model, data, {"unused": 0}, seed),
        "uniformity": uniformity_score(z, max_pairs, seed),
        "covariance_redundancy": covariance_redundancy(z),
        "shortcut_correlations": shortcut_diagnostics(score, data),
    }


def control_gaps(results: dict[str, dict[str, Any]]) -> dict[str, float]:
    acc = {k: float(v["test_at_best"]) for k, v in results.items()}
    o14 = acc.get("O14", math.nan)
    strongest = max(acc.get(k, -math.inf) for k in ["O0", "O1", "O2", "O3", "O4", "O5", "O6", "O7", "O13"])
    shortcut_best = max(acc.get(k, -math.inf) for k in ["O9", "O10", "O11", "O12"])
    return {
        "o14_minus_o0_grace": o14 - acc.get("O0", math.nan),
        "o14_minus_o1_matched_grace": o14 - acc.get("O1", math.nan),
        "o14_minus_o2_graphmae2": o14 - acc.get("O2", math.nan),
        "o14_minus_o3_gcmae": o14 - acc.get("O3", math.nan),
        "o14_minus_o4_graph_jepa": o14 - acc.get("O4", math.nan),
        "o14_minus_o5_tter": o14 - acc.get("O5", math.nan),
        "o14_minus_o6_dsla": o14 - acc.get("O6", math.nan),
        "o14_minus_o7_response_control": o14 - acc.get("O7", math.nan),
        "o14_minus_o13_raw_response": o14 - acc.get("O13", math.nan),
        "o14_minus_strongest_matched_control": o14 - strongest,
        "o14_minus_shortcut_best": o14 - shortcut_best,
    }


def decide(results: dict[str, dict[str, Any]], gaps: dict[str, float], dirty: bool) -> dict[str, Any]:
    triggered = []
    if dirty:
        triggered.append("BLOCK_PROVENANCE_BEFORE_PILOT")
    if gaps["o14_minus_strongest_matched_control"] <= 0:
        triggered.append("KILL_ORBIT_AS_WEAKER_THAN_MATCHED_CONTROL")
    if gaps["o14_minus_shortcut_best"] <= 0.3:
        triggered.append("KILL_ORBIT_AS_SHORTCUT_OR_ARTIFACT")
    o14_heldout = results["O14"].get("heldout_operator_response_val_mse")
    random_heldout = min(
        results.get(k, {}).get("heldout_operator_response_val_mse") or math.inf
        for k in ["O10", "O11", "O12"]
    )
    if o14_heldout is None or o14_heldout >= random_heldout:
        triggered.append("KILL_HELDOUT_OPERATOR_RESPONSE_STORY")
    if results["O14"]["test_at_best"] <= max(results["O0"]["test_at_best"], results["O1"]["test_at_best"]):
        triggered.append("KILL_ACCURACY_NOT_CONVERTED")
    return {
        "decision": "GO_TO_OR_M1_WITH_CAUTION" if not triggered else "KILL_OR_PIVOT_REQUIRED",
        "triggered_rules": triggered,
    }


def main() -> None:
    args = parse_args()
    config_path = (PROJECT_ROOT / args.config).resolve()
    base_config_path = (PROJECT_ROOT / args.base_config).resolve()
    config = load_yaml(config_path)
    base_config = load_yaml(base_config_path)
    command = exact_command()
    dirty_at_start = git_dirty(PROJECT_ROOT)
    commit_hash = git_rev(PROJECT_ROOT)
    dataset_name = args.dataset or str(config["dataset"])
    seed = int(args.seed if args.seed is not None else config["seed"])
    if dataset_name != "Cora" or seed != 0:
        raise ValueError("OR-M0-001 gate allows only Cora seed0.")
    status = args.status or config["meta"].get("status_default", "smoke")
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    pretrain_cfg = dict(base_config["pretrain"][dataset_name])
    pretrain_cfg.update(config.get("pretrain_overrides", {}).get(dataset_name, {}))
    if args.epochs is not None:
        pretrain_cfg["num_epochs"] = int(args.epochs)
    orbit_cfg = dict(config["orbit"])
    eval_cfg = dict(config["logreg_eval"])

    dataset = load_dataset(dataset_name, PROJECT_ROOT / args.data_root)
    data = dataset[0].to(device)
    split = create_or_load_split(dataset_name, data.y.detach().cpu(), seed, PROJECT_ROOT / args.split_root)
    split.train = split.train.to(device)
    split.val = split.val.to(device)
    split.test = split.test.to(device)

    run_id = f"orbit_smoke_{dataset_name}_seed{seed}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    result_dir = PROJECT_ROOT / args.result_root / dataset_name / "ORBIT_GCL_SMOKE" / run_id
    summary_dir = PROJECT_ROOT / args.summary_root
    log_dir = PROJECT_ROOT / args.log_root / run_id
    result_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    operators = build_operator_views(data, orbit_cfg, seed)
    teacher_variant = {"id": "OT", "name": "GRACE teacher for response targets", "type": "grace"}
    teacher, teacher_info = train_grace_variant(
        teacher_variant,
        dataset,
        data,
        pretrain_cfg,
        orbit_cfg,
        device,
        seed + 77,
        log_dir / "OT_teacher_grace.jsonl",
        args.log_every,
    )
    targets = compute_teacher_responses(teacher, data, operators, int(orbit_cfg["basis_dim"]))
    with torch.no_grad():
        targets["teacher_clean"] = teacher(data.x, data.edge_index).detach()
    random_teacher = make_model(dataset, pretrain_cfg, device)
    random_targets = compute_teacher_responses(random_teacher, data, operators, int(orbit_cfg["basis_dim"]))
    targets["random_basis"] = random_targets["basis"]
    shortcuts = build_shortcut_targets(data, seed)

    results: dict[str, dict[str, Any]] = {}
    for variant in config["variants"]:
        variant_seed = seed + int(variant["id"][1:]) * 1000
        if variant["type"] in {"grace", "grace_matched"}:
            model, train_info = train_grace_variant(
                variant,
                dataset,
                data,
                pretrain_cfg,
                orbit_cfg,
                device,
                variant_seed,
                log_dir / f"{variant['id']}_{variant['type']}.jsonl",
                args.log_every,
            )
        else:
            model, train_info = train_orbit_variant(
                variant,
                dataset,
                data,
                pretrain_cfg,
                orbit_cfg,
                targets,
                shortcuts,
                operators,
                device,
                variant_seed,
                log_dir / f"{variant['id']}_{variant['type']}.jsonl",
                args.log_every,
            )
        metrics = evaluate(model, variant["type"], data, split, eval_cfg, targets, shortcuts, seed, int(orbit_cfg["diagnostics_max_pairs"]))
        result = {
            "variant_id": variant["id"],
            "variant_name": variant["name"],
            "variant_type": variant["type"],
            "model_seed": int(variant_seed),
            "parameter_count": int(train_info["parameter_count"]),
            "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "status": status,
            "training": train_info,
            **metrics,
        }
        results[variant["id"]] = result
        print(f"[{variant['id']}] acc={result['test_at_best']:.2f} val={result['valid_at_best']:.2f}", flush=True)

    gaps = control_gaps(results)
    decision = decide(results, gaps, dirty_at_start)
    payload = {
        "run_id": run_id,
        "smoke_id": config["meta"]["smoke_id"],
        "method": config["meta"]["method"],
        "method_status": config["meta"]["method_status"],
        "dataset": dataset_name,
        "split_type": config["meta"]["split_type"],
        "split_seed": seed,
        "model_seed": seed,
        "status": status,
        "metric": "accuracy_percent",
        "exact_run_command": command,
        "project_commit_hash": commit_hash,
        "project_dirty": dirty_at_start,
        "project_dirty_at_start": dirty_at_start,
        "project_dirty_at_write": git_dirty(PROJECT_ROOT),
        "config_path": str(config_path),
        "base_config_path": str(base_config_path),
        "split_path": str(split.path.resolve()),
        "result_dir": str(result_dir.resolve()),
        "summary_dir": str(summary_dir.resolve()),
        "log_dir": str(log_dir.resolve()),
        "dataset_num_nodes": int(data.num_nodes),
        "dataset_num_edges": int(data.edge_index.size(1)),
        "dataset_num_features": int(dataset.num_features),
        "dataset_num_classes": int(dataset.num_classes),
        "pretrain_config": pretrain_cfg,
        "orbit_config": orbit_cfg,
        "teacher_training": teacher_info,
        "operator_names": [op["name"] for op in operators],
        "operator_target_info": {k: v for k, v in targets.items() if k.endswith("_names") or k in {"basis_explained_variance", "basis_effective_rank"}},
        "evaluator_config": eval_cfg,
        "integrity": {
            "operator_targets_use_labels": False,
            "test_labels_used_for_operator_targets": False,
            "test_labels_used_for_thresholds": False,
            "validation_labels_used_for_operator_targets": False,
            "uses_knn_ppr_cast_relations": False,
            "uses_positive_mining": False,
            "uses_train_label_probe_routing": False,
        },
        "variants": results,
        "control_gaps": gaps,
        "decision": decision,
    }
    result_path = result_dir / f"{run_id}.json"
    payload["result_path"] = str(result_path.resolve())
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    summary_json = summary_dir / f"{run_id}_summary.json"
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    summary_md = summary_dir / f"{run_id}_summary.md"
    lines = [
        f"# ORBIT-GCL {config['meta']['smoke_id']} Kill-Smoke",
        "",
        f"- Run: `{run_id}`",
        f"- Dataset/seed: `{dataset_name}` / `{seed}`",
        f"- Command: `{command}`",
        f"- Commit: `{commit_hash}`",
        f"- Dirty at start: `{dirty_at_start}`",
        f"- Decision: `{decision['decision']}`",
        "",
        "| ID | System | Test@best-val | Val@best | C | Heldout response MSE | Target MSE | Eff rank | Align | Uniformity | Cov redundancy |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in config["variants"]:
        r = results[variant["id"]]
        heldout = r.get("heldout_operator_response_val_mse")
        target_mse = r.get("response_target_mse")
        lines.append(
            f"| {variant['id']} | {variant['name']} | {r['test_at_best']:.2f} | {r['valid_at_best']:.2f} | "
            f"{r['best_c']} | {heldout if heldout is not None else 'NA'} | {target_mse if target_mse is not None else 'NA'} | "
            f"{r['effective_rank']:.2f} | {r['alignment']:.4f} | {r['uniformity']:.4f} | {r['covariance_redundancy']:.4f} |"
        )
    lines.extend(["", "## Control Gaps", ""])
    for key, value in gaps.items():
        lines.append(f"- `{key}`: `{value:.4f}`")
    lines.extend(["", "## Triggered Rules", ""])
    if decision["triggered_rules"]:
        for rule in decision["triggered_rules"]:
            lines.append(f"- `{rule}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Integrity", ""])
    for key, value in payload["integrity"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    summary_md.write_text("\n".join(lines) + "\n")
    print(f"[done] result={result_path}", flush=True)
    print(f"[done] summary={summary_md}", flush=True)


if __name__ == "__main__":
    main()

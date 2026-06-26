#!/usr/bin/env python
"""Run SPECTRA-GCL SP-M0-001 smoke.

The runner intentionally avoids cross-node positive mining, kNN/PPR/CAST
relations, score fusion, and post-hoc relation smoothing. S1-S6 are trained
with same-node two-view negative-free objectives only.
"""

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
    parser.add_argument("--config", default="configs/spectra_smoke.yaml")
    parser.add_argument("--base-config", default="configs/grace_1_1_8.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--status", default=None)
    parser.add_argument("--run-tag", default="sp_m0_001")
    parser.add_argument("--data-root", default="data/pyg")
    parser.add_argument("--split-root", default="splits")
    parser.add_argument("--result-root", default="results/raw")
    parser.add_argument("--summary-root", default="results/summary")
    parser.add_argument("--log-root", default="logs/spectra_smoke")
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def as_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def normalize_rows_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(denom, eps)).astype(np.float32)


def sparse_row_norm(edge_index: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
    edges = to_undirected(edge_index, num_nodes=num_nodes).to(device)
    loops = torch.arange(num_nodes, device=device)
    loop_index = torch.stack([loops, loops], dim=0)
    edges = torch.cat([edges, loop_index], dim=1)
    values = torch.ones(edges.size(1), device=device)
    adj = torch.sparse_coo_tensor(edges, values, (num_nodes, num_nodes), device=device).coalesce()
    row = adj.indices()[0]
    deg = torch.sparse.sum(adj, dim=1).to_dense().clamp_min(1.0)
    values = adj.values() / deg[row]
    return torch.sparse_coo_tensor(adj.indices(), values, adj.shape, device=device).coalesce()


def smooth_tensor(x: torch.Tensor, edge_index: torch.Tensor, steps: int) -> torch.Tensor:
    out = x
    adj = sparse_row_norm(edge_index, x.size(0), x.device)
    for _ in range(max(1, int(steps))):
        out = torch.sparse.mm(adj, out)
    return out


def residual_energy(x: torch.Tensor, edge_index: torch.Tensor, steps: int, mode: str) -> torch.Tensor:
    smooth = smooth_tensor(x, edge_index, steps)
    if mode == "low":
        target = smooth
    else:
        target = x - smooth
    return target.pow(2).sum(dim=1).add(1e-12).sqrt()


def make_bucket_ids(reference_energy: torch.Tensor, num_buckets: int) -> torch.Tensor:
    energy_np = as_numpy(reference_energy).astype(np.float64)
    quantiles = np.quantile(energy_np, np.linspace(0.0, 1.0, num_buckets + 1)[1:-1])
    bucket_np = np.digitize(energy_np, quantiles, right=True)
    return torch.tensor(bucket_np, dtype=torch.long, device=reference_energy.device)


def bucket_profile(energy: torch.Tensor, bucket_ids: torch.Tensor, num_buckets: int, eps: float) -> torch.Tensor:
    values = []
    for bucket in range(num_buckets):
        mask = bucket_ids == bucket
        if bool(mask.any()):
            values.append(energy[mask].mean())
        else:
            values.append(energy.new_tensor(0.0))
    profile = torch.stack(values)
    return profile / profile.sum().clamp_min(eps)


def target_profile(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    steps: int,
    bucket_ids: torch.Tensor,
    num_buckets: int,
    eps: float,
    mode: str,
    seed: int,
) -> torch.Tensor:
    energy = residual_energy(x, edge_index, steps, "low" if mode == "low" else "high")
    profile = bucket_profile(energy, bucket_ids, num_buckets, eps).detach()
    if mode == "random":
        generator = torch.Generator(device=profile.device)
        generator.manual_seed(seed + 44_000)
        profile = profile[torch.randperm(profile.numel(), generator=generator, device=profile.device)]
    return profile


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def barlow_loss(z1: torch.Tensor, z2: torch.Tensor, lambd: float, eps: float = 1e-9) -> tuple[torch.Tensor, dict[str, float]]:
    n = z1.size(0)
    z1n = (z1 - z1.mean(dim=0)) / z1.std(dim=0).clamp_min(eps)
    z2n = (z2 - z2.mean(dim=0)) / z2.std(dim=0).clamp_min(eps)
    c = (z1n.T @ z2n) / n
    on_diag = torch.diagonal(c).add_(-1.0).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()
    loss = on_diag + float(lambd) * off_diag
    return loss, {
        "rr_on_diag_loss": float(on_diag.detach().item()),
        "rr_off_diag_loss": float(off_diag.detach().item()),
        "rr_total_loss": float(loss.detach().item()),
    }


def rank_guard_loss(h: torch.Tensor, gamma: float, eps: float = 1e-4) -> torch.Tensor:
    centered = h - h.mean(dim=0, keepdim=True)
    std = torch.sqrt(centered.var(dim=0, unbiased=False) + eps)
    return F.relu(float(gamma) - std).mean()


class SpectraModel(nn.Module):
    def __init__(self, in_dim: int, cfg: dict[str, Any], spectra_cfg: dict[str, Any]) -> None:
        super().__init__()
        hidden_dim = int(cfg["num_hidden"])
        self.encoder = Encoder(
            in_dim,
            hidden_dim,
            activation_from_name(cfg["activation"]),
            base_model=GCNConv,
            k=int(cfg["num_layers"]),
        )
        proj_hidden = int(spectra_cfg["projector_hidden_dim"])
        proj_out = int(spectra_cfg["projector_out_dim"])
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, proj_hidden),
            nn.ELU(),
            nn.Linear(proj_hidden, proj_out),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x, edge_index)
        z = self.projector(h)
        return h, z


@torch.no_grad()
def encode_spectra(model: SpectraModel, x: torch.Tensor, edge_index: torch.Tensor, normalize: bool) -> torch.Tensor:
    model.eval()
    h, _ = model(x, edge_index)
    return F.normalize(h, dim=1) if normalize else h


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
        float(pretrain_cfg.get("tau", 0.4)),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(pretrain_cfg["learning_rate"]),
        weight_decay=float(pretrain_cfg["weight_decay"]),
    )
    last_loss = math.nan
    start = time.time()
    with log_path.open("w") as f:
        for epoch in range(1, int(pretrain_cfg["num_epochs"]) + 1):
            last_loss = train_grace_epoch(
                model,
                optimizer,
                data.x,
                data.edge_index,
                pretrain_cfg,
                int(pretrain_cfg.get("loss_batch_size", 0)),
            )
            if epoch == 1 or epoch == int(pretrain_cfg["num_epochs"]) or epoch % log_every == 0:
                record = {"epoch": epoch, "loss": last_loss, "time": time.time()}
                f.write(json.dumps(record, sort_keys=True) + "\n")
                f.flush()
                print(
                    f"[S0 GRACE] epoch={epoch:04d}/{int(pretrain_cfg['num_epochs'])} "
                    f"loss={last_loss:.4f} elapsed={time.time() - start:.1f}s",
                    flush=True,
                )
    z = as_numpy(encode(model, data.x, data.edge_index, normalize=True))
    return model, z, float(last_loss)


def boundary_loss(
    h: torch.Tensor,
    edge_index: torch.Tensor,
    cfg: dict[str, Any],
    bucket_ids: torch.Tensor,
    target: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    energy_mode = "low" if mode == "low" else "high"
    energy = residual_energy(h, edge_index, int(cfg["smooth_steps"]), energy_mode)
    current = bucket_profile(
        energy,
        bucket_ids,
        int(cfg["boundary_num_buckets"]),
        float(cfg["boundary_eps"]),
    )
    return F.mse_loss(current, target)


def train_spectra_variant(
    dataset,
    data,
    pretrain_cfg: dict[str, Any],
    spectra_cfg: dict[str, Any],
    variant: dict[str, Any],
    device: torch.device,
    seed: int,
    log_path: Path,
    log_every: int,
) -> tuple[SpectraModel, np.ndarray, dict[str, Any]]:
    set_seed(seed + int(variant["id"][1:]) * 1000)
    model = SpectraModel(dataset.num_features, pretrain_cfg, spectra_cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(spectra_cfg.get("learning_rate", pretrain_cfg["learning_rate"])),
        weight_decay=float(pretrain_cfg["weight_decay"]),
    )
    input_high_energy = residual_energy(
        data.x,
        data.edge_index,
        int(spectra_cfg["smooth_steps"]),
        "high",
    ).detach()
    bucket_ids = make_bucket_ids(input_high_energy, int(spectra_cfg["boundary_num_buckets"]))
    mode = str(variant.get("boundary_mode", "none"))
    tgt = target_profile(
        data.x,
        data.edge_index,
        int(spectra_cfg["smooth_steps"]),
        bucket_ids,
        int(spectra_cfg["boundary_num_buckets"]),
        float(spectra_cfg["boundary_eps"]),
        "random" if mode == "random" else ("low" if mode == "low" else "high"),
        seed,
    )
    last: dict[str, Any] = {}
    start = time.time()
    epochs = int(pretrain_cfg["num_epochs"])
    with log_path.open("w") as f:
        for epoch in range(1, epochs + 1):
            model.train()
            edge_index_1 = drop_edges(data.edge_index, float(pretrain_cfg["drop_edge_rate_1"]))
            edge_index_2 = drop_edges(data.edge_index, float(pretrain_cfg["drop_edge_rate_2"]))
            x1 = drop_feature(data.x, float(pretrain_cfg["drop_feature_rate_1"]))
            x2 = drop_feature(data.x, float(pretrain_cfg["drop_feature_rate_2"]))
            h1, z1 = model(x1, edge_index_1)
            h2, z2 = model(x2, edge_index_2)
            rr, rr_diag = barlow_loss(z1, z2, float(spectra_cfg["barlow_lambda"]))
            loss = rr
            boundary_value = h1.new_tensor(0.0)
            rank_value = h1.new_tensor(0.0)
            if bool(variant.get("use_boundary", False)):
                boundary_value = 0.5 * (
                    boundary_loss(h1, data.edge_index, spectra_cfg, bucket_ids, tgt, mode)
                    + boundary_loss(h2, data.edge_index, spectra_cfg, bucket_ids, tgt, mode)
                )
                loss = loss + float(spectra_cfg["boundary_lambda"]) * boundary_value
            if bool(variant.get("use_rank_guard", False)):
                rank_value = 0.5 * (
                    rank_guard_loss(h1, float(spectra_cfg["rank_gamma"]))
                    + rank_guard_loss(h2, float(spectra_cfg["rank_gamma"]))
                )
                loss = loss + float(spectra_cfg["rank_lambda"]) * rank_value
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            max_grad_norm = float(spectra_cfg.get("max_grad_norm", 0.0))
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            last = {
                "epoch": epoch,
                "loss": float(loss.detach().item()),
                "boundary_loss": float(boundary_value.detach().item()),
                "rank_guard_loss": float(rank_value.detach().item()),
                **rr_diag,
                "time": time.time(),
            }
            if epoch == 1 or epoch == epochs or epoch % log_every == 0:
                f.write(json.dumps(last, sort_keys=True) + "\n")
                f.flush()
                print(
                    f"[{variant['id']} {variant['name']}] epoch={epoch:04d}/{epochs} "
                    f"loss={last['loss']:.4f} rr={last['rr_total_loss']:.4f} "
                    f"b={last['boundary_loss']:.6f} r={last['rank_guard_loss']:.4f} "
                    f"elapsed={time.time() - start:.1f}s",
                    flush=True,
                )
    h = encode_spectra(model, data.x, data.edge_index, normalize=True)
    return model, as_numpy(h), last


def effective_rank(x: np.ndarray, center: bool = True) -> float:
    values = x.astype(np.float64)
    if center:
        values = values - values.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(values, full_matrices=False, compute_uv=False)
    total = float(singular.sum())
    if total <= 1e-12:
        return 0.0
    prob = singular / total
    entropy = -float(np.sum(prob * np.log(np.maximum(prob, 1e-12))))
    return float(np.exp(entropy))


def participation_ratio(x: np.ndarray, center: bool = True) -> float:
    values = x.astype(np.float64)
    if center:
        values = values - values.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(values, full_matrices=False, compute_uv=False)
    eig = singular ** 2
    denom = float(np.sum(eig ** 2))
    return float((np.sum(eig) ** 2) / max(denom, 1e-12))


def covariance_offdiag_mean(x: np.ndarray) -> float:
    values = x.astype(np.float64)
    values = values - values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    values = values / np.maximum(std, 1e-12)
    cov = values.T @ values / max(values.shape[0] - 1, 1)
    mask = ~np.eye(cov.shape[0], dtype=bool)
    return float(np.mean(np.abs(cov[mask])))


def uniformity_score(x: np.ndarray, max_pairs: int, seed: int) -> float:
    values = normalize_rows_np(x)
    n = values.shape[0]
    rng = np.random.default_rng(seed + 17_001)
    total_pairs = n * (n - 1) // 2
    if total_pairs <= max_pairs:
        row, col = np.triu_indices(n, k=1)
    else:
        row = rng.integers(0, n, size=max_pairs)
        col = rng.integers(0, n, size=max_pairs)
        keep = row != col
        row = row[keep]
        col = col[keep]
    dist2 = np.sum((values[row] - values[col]) ** 2, axis=1)
    return float(np.log(np.mean(np.exp(-2.0 * dist2)) + 1e-12))


def alignment_score(model: Any, data, pretrain_cfg: dict[str, Any], seed: int) -> float:
    set_seed(seed + 91_000)
    model.eval()
    with torch.no_grad():
        edge_index_1 = drop_edges(data.edge_index, float(pretrain_cfg["drop_edge_rate_1"]))
        edge_index_2 = drop_edges(data.edge_index, float(pretrain_cfg["drop_edge_rate_2"]))
        x1 = drop_feature(data.x, float(pretrain_cfg["drop_feature_rate_1"]))
        x2 = drop_feature(data.x, float(pretrain_cfg["drop_feature_rate_2"]))
        if isinstance(model, SpectraModel):
            h1, _ = model(x1, edge_index_1)
            h2, _ = model(x2, edge_index_2)
        else:
            h1 = model(x1, edge_index_1)
            h2 = model(x2, edge_index_2)
        h1 = F.normalize(h1, dim=1)
        h2 = F.normalize(h2, dim=1)
        return float((h1 - h2).pow(2).sum(dim=1).mean().item())


def boundary_diagnostics(
    embedding: np.ndarray,
    data,
    spectra_cfg: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    device = data.x.device
    x = data.x
    h = torch.tensor(embedding, dtype=torch.float32, device=device)
    steps = int(spectra_cfg["smooth_steps"])
    input_high = residual_energy(x, data.edge_index, steps, "high").detach()
    input_low = residual_energy(x, data.edge_index, steps, "low").detach()
    h_high = residual_energy(h, data.edge_index, steps, "high").detach()
    bucket_ids = make_bucket_ids(input_high, int(spectra_cfg["boundary_num_buckets"]))
    input_profile = bucket_profile(
        input_high,
        bucket_ids,
        int(spectra_cfg["boundary_num_buckets"]),
        float(spectra_cfg["boundary_eps"]),
    )
    h_profile = bucket_profile(
        h_high,
        bucket_ids,
        int(spectra_cfg["boundary_num_buckets"]),
        float(spectra_cfg["boundary_eps"]),
    )
    retention = 1.0 - 0.5 * torch.abs(input_profile - h_profile).sum().item()
    degree = torch.bincount(to_undirected(data.edge_index, num_nodes=data.num_nodes)[0], minlength=data.num_nodes)
    degree_np = as_numpy(degree.float())
    h_high_np = as_numpy(h_high)
    input_high_np = as_numpy(input_high)
    corr = 0.0
    if np.std(h_high_np) > 1e-12 and np.std(input_high_np) > 1e-12:
        corr = float(np.corrcoef(h_high_np, input_high_np)[0, 1])
    bucket_values: dict[str, float] = {}
    quantiles = np.quantile(degree_np, [0.25, 0.5, 0.75])
    degree_bucket = np.digitize(degree_np, quantiles, right=True)
    for bucket in range(4):
        mask = degree_bucket == bucket
        bucket_values[f"degree_bucket_{bucket}_h_high_energy"] = float(h_high_np[mask].mean()) if mask.any() else 0.0
    return {
        "boundary_energy_input_high_mean": float(input_high.mean().item()),
        "boundary_energy_input_low_mean": float(input_low.mean().item()),
        "boundary_energy_h_high_mean": float(h_high.mean().item()),
        "boundary_energy_retention": float(retention),
        "boundary_energy_node_corr": corr,
        "boundary_energy_input_profile": [float(v) for v in as_numpy(input_profile)],
        "boundary_energy_h_profile": [float(v) for v in as_numpy(h_profile)],
        "high_pass_residual_variance": float(h_high.var(unbiased=False).item()),
        "boundary_energy_by_degree_bucket": bucket_values,
        "diagnostic_seed": int(seed),
        "uses_labels": False,
    }


def label_only_diagnostics(embedding: np.ndarray, labels: np.ndarray, split, eval_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "uses_labels": True,
        "allowed_after_training_only": True,
        "test_at_best_val": float(eval_result["test_at_best"]),
        "valid_at_best": float(eval_result["valid_at_best"]),
        "train_size": int(split.train.numel()),
        "val_size": int(split.val.numel()),
        "test_size": int(split.test.numel()),
        "num_classes": int(np.unique(labels).shape[0]),
    }


def representation_diagnostics(
    embedding: np.ndarray,
    model: Any,
    data,
    pretrain_cfg: dict[str, Any],
    spectra_cfg: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    return {
        "effective_rank": effective_rank(embedding),
        "participation_ratio": participation_ratio(embedding),
        "cov_offdiag_mean": covariance_offdiag_mean(embedding),
        "alignment": alignment_score(model, data, pretrain_cfg, seed),
        "uniformity": uniformity_score(embedding, int(spectra_cfg["diagnostics_max_pairs"]), seed),
        **boundary_diagnostics(embedding, data, spectra_cfg, seed),
    }


def markdown_summary(results: list[dict[str, Any]], path: Path, decision: str) -> None:
    lines = [
        "# SPECTRA-GCL SP-M0-001 Smoke Summary",
        "",
        "本文件为 Cora seed0 smoke 汇总，不支持 pilot/formal/SOTA/robust claim。",
        "",
        f"- Bridge decision: `{decision}`",
        "",
        "| ID | Variant | Test@best-val | Val | Eff rank | Boundary retention | Cov offdiag | Alignment | Uniformity |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        eval_res = result["evaluation"]
        diag = result["diagnostics"]
        lines.append(
            f"| {result['variant_id']} | {result['variant_name']} | "
            f"{eval_res['test_at_best']:.2f} | {eval_res['valid_at_best']:.2f} | "
            f"{diag['effective_rank']:.2f} | {diag['boundary_energy_retention']:.4f} | "
            f"{diag['cov_offdiag_mean']:.4f} | {diag['alignment']:.4f} | "
            f"{diag['uniformity']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- S0/S7 are GRACE controls and do not use SPECTRA losses.",
            "- S1-S6 are same-node two-view negative-free objectives only; no cross-node positive mining is used.",
            "- Boundary diagnostics are label-free. Label-only diagnostics are saved in JSON after training and are not used for thresholding or model selection.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def bridge_decision(results: list[dict[str, Any]]) -> str:
    by_id = {r["variant_id"]: r for r in results}
    s0 = by_id["S0"]["evaluation"]["test_at_best"]
    s1 = by_id["S1"]["evaluation"]["test_at_best"]
    s2 = by_id["S2"]["evaluation"]["test_at_best"]
    s3_ret = by_id["S3"]["diagnostics"]["boundary_energy_retention"]
    s2_ret = by_id["S2"]["diagnostics"]["boundary_energy_retention"]
    s5_ret = by_id["S5"]["diagnostics"]["boundary_energy_retention"]
    if s1 >= s2 and abs(s5_ret - s2_ret) < 0.01:
        return "KILL_BOUNDARY_ENERGY_STORY"
    if s0 - max(s1, s2) > 1.0:
        return "PIVOT_AWAY_FROM_NEGATIVE_FREE_BOUNDARY"
    if s2 >= s1 and s2_ret > s3_ret and abs(s5_ret - s2_ret) >= 0.01:
        return "GO_TO_SP_M1_WITH_CAUTION"
    if s2_ret > s3_ret and s2 < max(s0, s1):
        return "REVISE_OBJECTIVE_NOT_DIAGNOSTIC"
    return "INCONCLUSIVE_SMOKE"


def main() -> None:
    args = parse_args()
    config = load_yaml(PROJECT_ROOT / args.config)
    base_config = load_yaml(PROJECT_ROOT / args.base_config)
    dataset_name = args.dataset or config["dataset"]["name"]
    seed = int(args.seed if args.seed is not None else config["dataset"]["seed"])
    if dataset_name != "Cora" or seed != 0:
        raise ValueError("SPECTRA SP-M0-001 gate allows only Cora seed=0.")

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
    spectra_cfg = dict(config["spectra"])

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_name = f"{args.run_tag}_{dataset_name}_seed{seed}_{timestamp}"
    result_dir = PROJECT_ROOT / args.result_root / dataset_name / "SPECTRA_GCL_SMOKE" / run_name
    summary_dir = PROJECT_ROOT / args.summary_root
    log_dir = PROJECT_ROOT / args.log_root / run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    labels = as_numpy(data.y).astype(int)
    start_all = time.time()
    grace_model, grace_embedding, grace_loss = train_grace(
        dataset,
        data,
        pretrain_cfg,
        device,
        seed,
        log_dir / "S0_GRACE.jsonl",
        args.log_every,
    )

    results: list[dict[str, Any]] = []
    trained_embeddings: dict[str, np.ndarray] = {"S0": grace_embedding, "S7": grace_embedding}
    trained_models: dict[str, Any] = {"S0": grace_model, "S7": grace_model}
    train_logs: dict[str, Any] = {"S0": {"last_loss": grace_loss}, "S7": {"last_loss": grace_loss}}
    train_log_paths: dict[str, Path] = {
        "S0": log_dir / "S0_GRACE.jsonl",
        "S7": log_dir / "S0_GRACE.jsonl",
    }

    for variant in config["variants"]:
        if variant["family"] in {"grace", "grace_matched"}:
            continue
        model, embedding, last_log = train_spectra_variant(
            dataset,
            data,
            pretrain_cfg,
            spectra_cfg,
            variant,
            device,
            seed,
            log_dir / f"{variant['id']}_{variant['name']}.jsonl",
            args.log_every,
        )
        trained_embeddings[variant["id"]] = embedding
        trained_models[variant["id"]] = model
        train_logs[variant["id"]] = last_log
        train_log_paths[variant["id"]] = log_dir / f"{variant['id']}_{variant['name']}.jsonl"

    for variant in config["variants"]:
        embedding = trained_embeddings[variant["id"]]
        model = trained_models[variant["id"]]
        eval_result = logreg_val_eval(torch.tensor(embedding), data.y.cpu(), split, eval_cfg, seed)
        diag = representation_diagnostics(embedding, model, data, pretrain_cfg, spectra_cfg, seed)
        result = {
            "run_id": f"{run_name}_{variant['id']}",
            "method": "SPECTRA-GCL-smoke",
            "variant_id": variant["id"],
            "variant_name": variant["name"],
            "variant_family": variant["family"],
            "variant_description": variant.get("description"),
            "dataset": dataset_name,
            "split_type": "stratified_random_1_1_8",
            "split_seed": seed,
            "model_seed": seed,
            "status": status,
            "metric": "accuracy_percent",
            "config_path": str((PROJECT_ROOT / args.config).resolve()),
            "base_config_path": str((PROJECT_ROOT / args.base_config).resolve()),
            "split_path": str(split.path.resolve()),
            "train_log_path": str(train_log_paths[variant["id"]].resolve()),
            "project_commit_hash": git_rev(PROJECT_ROOT),
            "project_dirty": git_dirty(PROJECT_ROOT),
            "dataset_num_nodes": int(data.num_nodes),
            "dataset_num_edges": int(data.edge_index.size(1)),
            "dataset_num_features": int(dataset.num_features),
            "dataset_num_classes": int(dataset.num_classes),
            "pretrain_config": pretrain_cfg,
            "spectra_config": spectra_cfg,
            "evaluator_type": "logreg_val",
            "evaluator_config": eval_cfg,
            "train_last_log": train_logs[variant["id"]],
            "diagnostics": diag,
            "label_only_diagnostics": label_only_diagnostics(embedding, labels, split, eval_result),
            "evaluation": eval_result,
            "elapsed_sec": time.time() - start_all,
            "integrity": {
                "uses_cross_node_positive_mining": False,
                "uses_knn_ppr_cast_mining": False,
                "uses_test_labels_for_thresholds": False,
                "evaluator_c_selection": "train_fit_val_select_test_report",
            },
        }
        result_path = result_dir / f"{variant['id']}_{variant['name']}.json"
        result["result_path"] = str(result_path.resolve())
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        results.append(result)
        print(
            f"[{variant['id']} {variant['name']}] "
            f"test={eval_result['test_at_best']:.2f} "
            f"rank={diag['effective_rank']:.2f} "
            f"boundary={diag['boundary_energy_retention']:.4f} "
            f"uniformity={diag['uniformity']:.4f}",
            flush=True,
        )

    decision = bridge_decision(results)
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
        "bridge_decision": decision,
        "bridge_decision_inputs": {
            "smoke_only": True,
            "elapsed_sec": time.time() - start_all,
            "no_test_label_thresholding": True,
            "no_pair_mining": True,
        },
    }
    summary_json = summary_dir / f"{run_name}_summary.json"
    summary_md = summary_dir / f"{run_name}_summary.md"
    summary_json.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n")
    markdown_summary(results, summary_md, decision)
    print(f"bridge_decision={decision}")
    print(f"summary_json={summary_json}")
    print(f"summary_md={summary_md}")


if __name__ == "__main__":
    main()

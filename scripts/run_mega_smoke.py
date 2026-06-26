#!/usr/bin/env python
"""Run MEGA-GCL ME-M0-001 kill-smoke.

This runner compares masked evidence-group discrimination against strong
masked-autoencoder, codebook/prototype, token-corruption, and GRACE controls.
Evidence groups are generated without labels; labels are used only by the
standard frozen-encoder Logistic Regression evaluator and post-hoc diagnostics.
"""

from __future__ import annotations

import argparse
import copy
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
    git_dirty,
    git_rev,
    load_dataset,
    logreg_val_eval,
    set_seed,
    train_grace_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mega_smoke.yaml")
    parser.add_argument("--base-config", default="configs/grace_1_1_8.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--status", default=None)
    parser.add_argument("--run-tag", default="me_m0_001")
    parser.add_argument("--data-root", default="data/pyg")
    parser.add_argument("--split-root", default="splits")
    parser.add_argument("--result-root", default="results/raw")
    parser.add_argument("--summary-root", default="results/summary")
    parser.add_argument("--log-root", default="logs/mega_smoke")
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def exact_command() -> str:
    return " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])


def as_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def make_grace_model(dataset, cfg: dict[str, Any], device: torch.device) -> Model:
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


class MegaModel(nn.Module):
    def __init__(self, dataset, pretrain_cfg: dict[str, Any], mega_cfg: dict[str, Any], group_sizes: dict[str, int]) -> None:
        super().__init__()
        hidden = int(pretrain_cfg["num_hidden"])
        proj_hidden = int(mega_cfg["projector_hidden_dim"])
        proj_out = int(mega_cfg["projector_out_dim"])
        dec_hidden = int(mega_cfg["decoder_hidden_dim"])
        self.tau = float(pretrain_cfg["tau"])
        self.encoder = Encoder(
            dataset.num_features,
            hidden,
            activation_from_name(pretrain_cfg["activation"]),
            base_model=GCNConv,
            k=int(pretrain_cfg["num_layers"]),
        )
        self.projector = nn.Sequential(nn.Linear(hidden, proj_hidden), nn.ELU(), nn.Linear(proj_hidden, proj_out))
        self.predictor = nn.Sequential(nn.Linear(hidden, proj_hidden), nn.ELU(), nn.Linear(proj_hidden, hidden))
        self.decoder = nn.Sequential(nn.Linear(hidden, dec_hidden), nn.ELU(), nn.Linear(dec_hidden, dataset.num_features))
        self.heads = nn.ModuleDict({name: nn.Linear(hidden, size) for name, size in group_sizes.items()})
        self.codebook = nn.Parameter(torch.randn(int(mega_cfg["num_codebook_entries"]), hidden) * 0.02)
        self.prototypes = nn.Linear(hidden, int(mega_cfg["num_prototypes"]), bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, h: torch.Tensor) -> torch.Tensor:
        return self.projector(h)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        return F.normalize(z1, dim=1) @ F.normalize(z2, dim=1).T

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        refl = torch.exp(self.sim(z1, z1) / self.tau)
        between = torch.exp(self.sim(z1, z2) / self.tau)
        return -torch.log(between.diag() / (refl.sum(1) + between.sum(1) - refl.diag()).clamp_min(1e-12))

    def contrastive_loss(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        z1 = self.projection(h1)
        z2 = self.projection(h2)
        return 0.5 * (self.semi_loss(z1, z2) + self.semi_loss(z2, z1)).mean()


def quantile_bucket(values: np.ndarray, num_groups: int) -> np.ndarray:
    values = values.astype(np.float64)
    if np.allclose(values.max(), values.min()):
        return np.zeros_like(values, dtype=np.int64)
    qs = np.quantile(values, np.linspace(0.0, 1.0, num_groups + 1)[1:-1])
    return np.digitize(values, qs, right=True).astype(np.int64)


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


def build_evidence(data, cfg: dict[str, Any], seed: int, device: torch.device) -> dict[str, torch.Tensor]:
    x_np = as_numpy(data.x).astype(np.float64)
    centered = x_np - x_np.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    comps = centered @ vt[:4].T
    bits = comps > np.median(comps, axis=0, keepdims=True)
    feature = np.zeros(x_np.shape[0], dtype=np.int64)
    for i in range(bits.shape[1]):
        feature += bits[:, i].astype(np.int64) << i
    feature %= int(cfg["num_feature_groups"])

    src = data.edge_index[0].detach().cpu().numpy()
    dst = data.edge_index[1].detach().cpu().numpy()
    degree = np.bincount(src, minlength=data.num_nodes).astype(np.float64)
    degree_group = quantile_bucket(degree, int(cfg["num_degree_groups"]))
    reciprocal = np.bincount(dst, minlength=data.num_nodes).astype(np.float64)
    struct_score = np.log1p(degree) + 0.5 * np.log1p(reciprocal)
    structure = quantile_bucket(struct_score, int(cfg["num_structure_groups"]))

    adj = row_normalized_adj(data.edge_index, data.num_nodes, device)
    smooth = torch.sparse.mm(adj, data.x)
    residual_norm = as_numpy((data.x - smooth).pow(2).sum(dim=1).sqrt())
    residual = quantile_bucket(residual_norm, int(cfg["num_residual_groups"]))
    mint = (feature + 3 * structure + 5 * residual) % int(cfg["num_mint_groups"])

    rng = np.random.default_rng(seed + 41_009)
    random_feature = rng.permutation(feature)
    random_structure = rng.permutation(structure)
    random_residual = rng.integers(0, int(cfg["num_residual_groups"]), size=data.num_nodes)

    return {
        "feature": torch.tensor(feature, dtype=torch.long, device=device),
        "structure": torch.tensor(structure, dtype=torch.long, device=device),
        "residual": torch.tensor(residual, dtype=torch.long, device=device),
        "degree": torch.tensor(degree_group, dtype=torch.long, device=device),
        "mint": torch.tensor(mint, dtype=torch.long, device=device),
        "random_feature": torch.tensor(random_feature, dtype=torch.long, device=device),
        "random_structure": torch.tensor(random_structure, dtype=torch.long, device=device),
        "random_residual": torch.tensor(random_residual, dtype=torch.long, device=device),
    }


def mask_input(x: torch.Tensor, mask_rate: float, seed: int, epoch: int, disable: bool = False) -> torch.Tensor:
    if disable or mask_rate <= 0:
        return x
    gen = torch.Generator(device=x.device)
    gen.manual_seed(seed + epoch * 9973)
    keep = torch.rand(x.shape, generator=gen, device=x.device) > float(mask_rate)
    return x * keep.float()


def class_separation(z: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    x = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    rng = np.random.default_rng(17)
    n = x.shape[0]
    sample = min(120000, n * 40)
    row = rng.integers(0, n, size=sample)
    col = rng.integers(0, n, size=sample)
    keep = row != col
    row, col = row[keep], col[keep]
    sim = (x[row] * x[col]).sum(axis=1)
    same = labels[row] == labels[col]
    return {
        "same_class_cosine_mean": float(sim[same].mean()) if same.any() else 0.0,
        "diff_class_cosine_mean": float(sim[~same].mean()) if (~same).any() else 0.0,
        "class_separation_gap": float((sim[same].mean() if same.any() else 0.0) - (sim[~same].mean() if (~same).any() else 0.0)),
    }


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
def alignment_score(model: MegaModel | Model, data, cfg: dict[str, Any], seed: int) -> float:
    model.eval()
    x1 = mask_input(data.x, float(cfg["drop_feature_rate_1"]), seed + 1, 1)
    x2 = mask_input(data.x, float(cfg["drop_feature_rate_2"]), seed + 2, 1)
    e1 = drop_edges(data.edge_index, float(cfg["drop_edge_rate_1"]))
    e2 = drop_edges(data.edge_index, float(cfg["drop_edge_rate_2"]))
    h1 = F.normalize(model(x1, e1), dim=1)
    h2 = F.normalize(model(x2, e2), dim=1)
    return float((h1 - h2).pow(2).sum(dim=1).mean().item())


def evidence_loss(model: MegaModel, h: torch.Tensor, evidence: dict[str, torch.Tensor], keys: list[str]) -> tuple[torch.Tensor, dict[str, float]]:
    losses = []
    accs = {}
    for key in keys:
        logits = model.heads[key](h)
        target = evidence[key]
        losses.append(F.cross_entropy(logits, target))
        accs[f"{key}_evidence_acc"] = float((logits.argmax(dim=1) == target).float().mean().detach().item() * 100.0)
    return torch.stack(losses).mean(), accs


def usage_stats(assignments: torch.Tensor, size: int, prefix: str) -> dict[str, float]:
    hist = torch.bincount(assignments.detach().cpu(), minlength=size).float()
    prob = hist / hist.sum().clamp_min(1.0)
    entropy = float(-(prob * torch.log(prob.clamp_min(1e-12))).sum().item())
    return {
        f"{prefix}_active": int((hist > 0).sum().item()),
        f"{prefix}_entropy": entropy,
        f"{prefix}_perplexity": float(math.exp(entropy)),
        f"{prefix}_max_share": float((hist.max() / hist.sum().clamp_min(1.0)).item()),
    }


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def train_grace_variant(
    variant: dict[str, Any],
    dataset,
    data,
    cfg: dict[str, Any],
    mega_cfg: dict[str, Any],
    device: torch.device,
    seed: int,
    log_path: Path,
    log_every: int,
) -> tuple[Model, dict[str, Any]]:
    set_seed(seed)
    train_cfg = dict(cfg)
    if variant["type"] == "grace_matched":
        train_cfg["num_proj_hidden"] = int(mega_cfg["matched_grace_proj_hidden"])
    model = make_grace_model(dataset, train_cfg, device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    last = math.nan
    start = time.time()
    with log_path.open("w") as f:
        for epoch in range(1, int(cfg["num_epochs"]) + 1):
            last = train_grace_epoch(model, opt, data.x, data.edge_index, train_cfg, int(train_cfg.get("loss_batch_size", 0)))
            if epoch == 1 or epoch == int(cfg["num_epochs"]) or epoch % log_every == 0:
                rec = {"epoch": epoch, "loss": float(last), "time": time.time()}
                f.write(json.dumps(rec, sort_keys=True) + "\n")
                f.flush()
                print(f"[{variant['id']} {variant['name']}] epoch={epoch:04d}/{int(cfg['num_epochs'])} loss={last:.4f} elapsed={time.time()-start:.1f}s", flush=True)
    elapsed = time.time() - start
    return model, {
        "last_train_loss": float(last),
        "epochs": int(cfg["num_epochs"]),
        "elapsed_sec": elapsed,
        "model_seed": int(seed),
        "parameter_count": count_parameters(model),
        "train_config": train_cfg,
    }


def train_ssl_variant(
    variant: dict[str, Any],
    dataset,
    data,
    pretrain_cfg: dict[str, Any],
    mega_cfg: dict[str, Any],
    evidence: dict[str, torch.Tensor],
    device: torch.device,
    seed: int,
    log_path: Path,
    log_every: int,
) -> tuple[MegaModel, dict[str, Any]]:
    group_sizes = {
        "feature": int(mega_cfg["num_feature_groups"]),
        "structure": int(mega_cfg["num_structure_groups"]),
        "residual": int(mega_cfg["num_residual_groups"]),
        "degree": int(mega_cfg["num_degree_groups"]),
        "mint": int(mega_cfg["num_mint_groups"]),
        "random_feature": int(mega_cfg["num_feature_groups"]),
        "random_structure": int(mega_cfg["num_structure_groups"]),
        "random_residual": int(mega_cfg["num_residual_groups"]),
    }
    set_seed(seed)
    model = MegaModel(dataset, pretrain_cfg, mega_cfg, group_sizes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(mega_cfg["learning_rate"]), weight_decay=float(mega_cfg["weight_decay"]))
    variant_type = variant["type"]
    last: dict[str, float] = {}
    start = time.time()
    with log_path.open("w") as f:
        for epoch in range(1, int(pretrain_cfg["num_epochs"]) + 1):
            model.train()
            opt.zero_grad(set_to_none=True)
            masked = variant_type not in {"no_mask_evidence"}
            x1 = mask_input(data.x, float(mega_cfg["mask_rate"]), seed, epoch, disable=not masked)
            x2 = mask_input(data.x, float(mega_cfg["mask_rate"]), seed + 13, epoch, disable=not masked)
            e1 = drop_edges(data.edge_index, float(mega_cfg["edge_drop_rate"]))
            e2 = drop_edges(data.edge_index, float(mega_cfg["edge_drop_rate"]))
            h1 = model(x1, e1)
            h2 = model(x2, e2)
            h_full = model(data.x, data.edge_index)
            loss = h1.new_tensor(0.0)
            diag: dict[str, float] = {}

            if variant_type == "graphmae_recon":
                recon = model.decoder(h1)
                loss = float(mega_cfg["lambda_recon"]) * F.mse_loss(recon, data.x)
            elif variant_type == "latent_remask":
                pred = model.predictor(h1)
                loss = float(mega_cfg["lambda_latent"]) * F.mse_loss(F.normalize(pred, dim=1), F.normalize(h_full.detach(), dim=1))
            elif variant_type == "gcmae_hybrid":
                recon = F.mse_loss(model.decoder(h1), data.x)
                contrast = model.contrastive_loss(h1, h2)
                loss = float(mega_cfg["lambda_recon"]) * recon + float(mega_cfg["lambda_contrast"]) * contrast
            elif variant_type == "vq_codebook":
                dist = torch.cdist(h1, model.codebook)
                assign = dist.argmin(dim=1)
                quant = model.codebook[assign]
                recon = F.mse_loss(model.decoder(quant), data.x)
                commit = F.mse_loss(h1, quant.detach()) + F.mse_loss(quant, h1.detach())
                loss = recon + float(mega_cfg["lambda_vq"]) * commit
                diag.update(usage_stats(assign, int(mega_cfg["num_codebook_entries"]), "codebook"))
            elif variant_type == "swav_proto":
                logits1 = model.prototypes(F.normalize(h1, dim=1))
                logits2 = model.prototypes(F.normalize(h2, dim=1))
                target1 = logits2.detach().argmax(dim=1)
                target2 = logits1.detach().argmax(dim=1)
                assign = logits1.detach().argmax(dim=1)
                prob = F.softmax(logits1, dim=1).mean(dim=0)
                balance = (prob * torch.log(prob.clamp_min(1e-12))).sum()
                loss = 0.5 * (F.cross_entropy(logits1, target1) + F.cross_entropy(logits2, target2)) + float(mega_cfg["lambda_balance"]) * balance
                diag.update(usage_stats(assign, int(mega_cfg["num_prototypes"]), "prototype"))
            elif variant_type == "mint_token":
                ev_loss, accs = evidence_loss(model, h1, evidence, ["mint"])
                corrupt = evidence["mint"][torch.randperm(data.num_nodes, device=device)]
                logits = model.heads["mint"](h2)
                corrupt_margin = F.relu(logits.gather(1, corrupt[:, None]).squeeze(1) - logits.gather(1, evidence["mint"][:, None]).squeeze(1) + 0.2).mean()
                loss = ev_loss + 0.25 * corrupt_margin
                diag.update(accs)
            elif variant_type == "mega_feature":
                ev_loss, accs = evidence_loss(model, h1, evidence, ["feature"])
                loss = float(mega_cfg["lambda_evidence"]) * ev_loss + float(mega_cfg["lambda_view"]) * F.mse_loss(F.normalize(h1, dim=1), F.normalize(h2, dim=1))
                diag.update(accs)
            elif variant_type == "mega_structure":
                ev_loss, accs = evidence_loss(model, h1, evidence, ["structure"])
                loss = ev_loss + float(mega_cfg["lambda_view"]) * F.mse_loss(F.normalize(h1, dim=1), F.normalize(h2, dim=1))
                diag.update(accs)
            elif variant_type == "mega_degree":
                ev_loss, accs = evidence_loss(model, h1, evidence, ["degree"])
                loss = ev_loss + float(mega_cfg["lambda_view"]) * F.mse_loss(F.normalize(h1, dim=1), F.normalize(h2, dim=1))
                diag.update(accs)
            elif variant_type == "mega_full":
                ev_loss, accs = evidence_loss(model, h1, evidence, ["feature", "structure", "residual"])
                loss = ev_loss + float(mega_cfg["lambda_view"]) * F.mse_loss(F.normalize(h1, dim=1), F.normalize(h2, dim=1))
                diag.update(accs)
            elif variant_type == "random_evidence":
                ev_loss, accs = evidence_loss(model, h1, evidence, ["random_feature", "random_structure", "random_residual"])
                loss = ev_loss + float(mega_cfg["lambda_view"]) * F.mse_loss(F.normalize(h1, dim=1), F.normalize(h2, dim=1))
                diag.update(accs)
            elif variant_type == "no_mask_evidence":
                ev_loss, accs = evidence_loss(model, h1, evidence, ["feature", "structure", "residual"])
                loss = ev_loss
                diag.update(accs)
            else:
                raise ValueError(f"Unsupported SSL variant type: {variant_type}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(mega_cfg["max_grad_norm"]))
            opt.step()
            last = {"loss": float(loss.detach().item()), **diag}
            if epoch == 1 or epoch == int(pretrain_cfg["num_epochs"]) or epoch % log_every == 0:
                rec = {"epoch": epoch, **last, "time": time.time()}
                f.write(json.dumps(rec, sort_keys=True) + "\n")
                f.flush()
                print(f"[{variant['id']} {variant['name']}] epoch={epoch:04d}/{int(pretrain_cfg['num_epochs'])} loss={last['loss']:.4f} elapsed={time.time()-start:.1f}s", flush=True)
    elapsed = time.time() - start
    return model, {
        "last_train_loss": last.get("loss"),
        "epochs": int(pretrain_cfg["num_epochs"]),
        "elapsed_sec": elapsed,
        "model_seed": int(seed),
        "parameter_count": count_parameters(model),
        "last_train_diagnostics": last,
    }


@torch.no_grad()
def encode_model(model: MegaModel | Model, data, normalize: bool) -> torch.Tensor:
    model.eval()
    h = model(data.x, data.edge_index)
    return F.normalize(h, dim=1) if normalize else h


@torch.no_grad()
def evidence_eval(model: MegaModel | Model, evidence: dict[str, torch.Tensor], split) -> dict[str, Any]:
    if not isinstance(model, MegaModel):
        return {
            "evidence_prediction": {},
            "evidence_prediction_accuracy": None,
            "evidence_prediction_accuracy_train": None,
            "evidence_prediction_accuracy_val": None,
        }
    h = model._last_eval_h if hasattr(model, "_last_eval_h") else None
    if h is None:
        return {
            "evidence_prediction": {},
            "evidence_prediction_accuracy": None,
            "evidence_prediction_accuracy_train": None,
            "evidence_prediction_accuracy_val": None,
        }
    out: dict[str, Any] = {}
    train_values = []
    val_values = []
    train_idx = split.train.to(h.device)
    val_idx = split.val.to(h.device)
    for key, target in evidence.items():
        if key not in model.heads:
            continue
        logits = model.heads[key](h)
        pred = logits.argmax(dim=1)
        all_acc = float((pred == target).float().mean().item() * 100.0)
        train_acc = float((pred[train_idx] == target[train_idx]).float().mean().item() * 100.0)
        val_acc = float((pred[val_idx] == target[val_idx]).float().mean().item() * 100.0)
        out[key] = {"all": all_acc, "train": train_acc, "val": val_acc}
        train_values.append(train_acc)
        val_values.append(val_acc)
    mean_acc = None
    if out:
        mean_acc = float(np.mean([v["all"] for v in out.values()]))
    return {
        "evidence_prediction": out,
        "evidence_prediction_accuracy": mean_acc,
        "evidence_prediction_accuracy_train": float(np.mean(train_values)) if train_values else None,
        "evidence_prediction_accuracy_val": float(np.mean(val_values)) if val_values else None,
    }


@torch.no_grad()
def codebook_prototype_eval(model: MegaModel | Model) -> dict[str, Any]:
    if not isinstance(model, MegaModel) or not hasattr(model, "_last_eval_h"):
        return {"codebook_prototype_usage": {}}
    h = model._last_eval_h
    dist = torch.cdist(h, model.codebook)
    code_assign = dist.argmin(dim=1)
    proto_assign = model.prototypes(F.normalize(h, dim=1)).argmax(dim=1)
    return {
        "codebook_prototype_usage": {
            **usage_stats(code_assign, model.codebook.size(0), "final_codebook"),
            **usage_stats(proto_assign, model.prototypes.out_features, "final_prototype"),
        }
    }


def evaluate(
    model: MegaModel | Model,
    data,
    split,
    eval_cfg: dict[str, Any],
    evidence: dict[str, torch.Tensor],
    seed: int,
    max_pairs: int,
) -> dict[str, Any]:
    h = encode_model(model, data, normalize=bool(eval_cfg["normalize_embeddings"]))
    if isinstance(model, MegaModel):
        model._last_eval_h = h
    z = as_numpy(h)
    labels = as_numpy(data.y).astype(np.int64)
    eval_result = logreg_val_eval(h, data.y, split, eval_cfg, seed)
    return {
        **eval_result,
        "accuracy": eval_result["test_at_best"],
        **evidence_eval(model, evidence, split),
        **codebook_prototype_eval(model),
        "effective_rank": effective_rank(z),
        "alignment": alignment_score(model, data, {"drop_feature_rate_1": 0.2, "drop_feature_rate_2": 0.3, "drop_edge_rate_1": 0.2, "drop_edge_rate_2": 0.4}, seed),
        "uniformity": uniformity_score(z, max_pairs, seed),
        "covariance_redundancy": covariance_redundancy(z),
        "class_evidence_separation": class_separation(z, labels),
    }


def control_gaps(results: dict[str, dict[str, Any]]) -> dict[str, float]:
    acc = {k: float(v["test_at_best"]) for k, v in results.items()}
    m10 = acc.get("M10", math.nan)
    return {
        "m10_minus_m0_grace": m10 - acc.get("M0", math.nan),
        "m10_minus_m1_graphmae": m10 - acc.get("M1", math.nan),
        "m10_minus_m2_graphmae2": m10 - acc.get("M2", math.nan),
        "m10_minus_m3_gcmae": m10 - acc.get("M3", math.nan),
        "m10_minus_m4_vq": m10 - acc.get("M4", math.nan),
        "m10_minus_m5_swav": m10 - acc.get("M5", math.nan),
        "m10_minus_m6_mint": m10 - acc.get("M6", math.nan),
        "m10_minus_m11_random": m10 - acc.get("M11", math.nan),
        "m10_minus_m12_no_mask": m10 - acc.get("M12", math.nan),
        "m10_minus_m13_matched_grace": m10 - acc.get("M13", math.nan),
    }


def decide(gaps: dict[str, float], dirty: bool) -> dict[str, Any]:
    triggered = []
    if dirty:
        triggered.append("BLOCK_PROVENANCE_BEFORE_PILOT")
    if gaps["m10_minus_m0_grace"] <= 0 or gaps["m10_minus_m13_matched_grace"] <= 0:
        triggered.append("KILL_MEGA_AS_WEAKER_THAN_GRACE")
    if min(gaps["m10_minus_m1_graphmae"], gaps["m10_minus_m2_graphmae2"], gaps["m10_minus_m3_gcmae"]) <= 0:
        triggered.append("KILL_MEGA_AS_MASKED_AUTOENCODER_VARIANT")
    if min(gaps["m10_minus_m4_vq"], gaps["m10_minus_m5_swav"]) <= 0:
        triggered.append("KILL_MEGA_AS_CODEBOOK_OR_PROTOTYPE_VARIANT")
    if gaps["m10_minus_m6_mint"] <= 0:
        triggered.append("KILL_MEGA_AS_TOKEN_CORRUPTION_VARIANT")
    if gaps["m10_minus_m11_random"] <= 0:
        triggered.append("KILL_EVIDENCE_GROUP_STORY")
    if gaps["m10_minus_m12_no_mask"] <= 0:
        triggered.append("KILL_MASKING_STORY")
    return {
        "decision": "GO_TO_ME_M1_WITH_CAUTION" if not triggered else "KILL_OR_PIVOT_REQUIRED",
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
    status = args.status or config["meta"].get("status_default", "smoke")
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    pretrain_cfg = dict(base_config["pretrain"][dataset_name])
    pretrain_cfg.update(config.get("pretrain_overrides", {}).get(dataset_name, {}))
    if args.epochs is not None:
        pretrain_cfg["num_epochs"] = int(args.epochs)
    mega_cfg = dict(config["mega"])
    eval_cfg = dict(config["logreg_eval"])

    dataset = load_dataset(dataset_name, PROJECT_ROOT / args.data_root)
    data = dataset[0].to(device)
    split = create_or_load_split(dataset_name, data.y.detach().cpu(), seed, PROJECT_ROOT / args.split_root)
    split.train = split.train.to(device)
    split.val = split.val.to(device)
    split.test = split.test.to(device)
    evidence = build_evidence(data, mega_cfg, seed, device)

    run_id = f"mega_smoke_{dataset_name}_seed{seed}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    result_dir = PROJECT_ROOT / args.result_root / dataset_name / "MEGA_GCL_SMOKE" / run_id
    summary_dir = PROJECT_ROOT / args.summary_root
    log_dir = PROJECT_ROOT / args.log_root / run_id
    result_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, Any]] = {}
    for variant in config["variants"]:
        variant_seed = seed + int(variant["id"][1:]) * 1000
        if variant["type"] in {"grace", "grace_matched"}:
            model, train_info = train_grace_variant(
                variant, dataset, data, pretrain_cfg, mega_cfg, device, variant_seed, log_dir / f"{variant['id']}_{variant['type']}.jsonl", args.log_every
            )
        else:
            model, train_info = train_ssl_variant(
                variant, dataset, data, pretrain_cfg, mega_cfg, evidence, device, variant_seed, log_dir / f"{variant['id']}_{variant['type']}.jsonl", args.log_every
            )
        metrics = evaluate(model, data, split, eval_cfg, evidence, seed, int(mega_cfg["diagnostics_max_pairs"]))
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
    decision = decide(gaps, dirty_at_start)
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
        "mega_config": mega_cfg,
        "evaluator_config": eval_cfg,
        "integrity": {
            "evidence_groups_use_labels": False,
            "test_labels_used_for_evidence_groups": False,
            "test_labels_used_for_thresholds": False,
            "validation_labels_used_for_evidence_groups": False,
            "uses_knn_ppr_cast_relations": False,
            "uses_positive_mining": False,
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
        f"# MEGA-GCL {config['meta']['smoke_id']} Kill-Smoke",
        "",
        f"- Run: `{run_id}`",
        f"- Dataset/seed: `{dataset_name}` / `{seed}`",
        f"- Command: `{command}`",
        f"- Commit: `{commit_hash}`",
        f"- Dirty at start: `{dirty_at_start}`",
        f"- Decision: `{decision['decision']}`",
        "",
        "| ID | System | Test@best-val | Val@best | C | Evidence acc | Evidence train | Evidence val | Codebook ppl | Proto ppl | Eff rank | Align | Uniformity | Cov redundancy |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in config["variants"]:
        r = results[variant["id"]]
        usage = r.get("codebook_prototype_usage", {})
        ev_acc = r.get("evidence_prediction_accuracy")
        ev_train = r.get("evidence_prediction_accuracy_train")
        ev_val = r.get("evidence_prediction_accuracy_val")
        lines.append(
            f"| {variant['id']} | {variant['name']} | {r['test_at_best']:.2f} | {r['valid_at_best']:.2f} | "
            f"{r['best_c']} | {ev_acc if ev_acc is not None else 'NA'} | "
            f"{ev_train if ev_train is not None else 'NA'} | {ev_val if ev_val is not None else 'NA'} | "
            f"{usage.get('final_codebook_perplexity', 'NA')} | {usage.get('final_prototype_perplexity', 'NA')} | "
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
    lines.append("")
    summary_md.write_text("\n".join(lines) + "\n")
    print(f"[done] result={result_path}", flush=True)
    print(f"[done] summary={summary_md}", flush=True)


if __name__ == "__main__":
    main()

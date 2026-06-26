#!/usr/bin/env python
"""Minimal DSR-GCL smoke runner.

This script intentionally reuses the project's GRACE loader, split, and
Logistic Regression evaluator. It is a mechanism diagnostic, not a formal
experiment runner.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_remaining_self_loops, degree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "baselines" / "GRACE"))

from model import Encoder, Model, drop_feature  # noqa: E402
from run_grace_1_1_8 import (  # noqa: E402
    activation_from_name,
    create_or_load_split,
    drop_edges,
    git_dirty,
    git_rev,
    load_dataset,
    logreg_val_eval,
    set_seed,
)


@dataclass
class BranchOutput:
    z_sem_1: torch.Tensor | None = None
    z_sem_2: torch.Tensor | None = None
    z_res_1: torch.Tensor | None = None
    z_res_2: torch.Tensor | None = None
    z_single_1: torch.Tensor | None = None
    z_single_2: torch.Tensor | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dsr_smoke.yaml")
    parser.add_argument("--base-config", default="configs/grace_1_1_8.yaml")
    parser.add_argument("--dataset", default="Cora")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--status", default=None)
    parser.add_argument("--run-tag", default="dsr_smoke")
    parser.add_argument("--data-root", default="data/pyg")
    parser.add_argument("--split-root", default="splits")
    parser.add_argument("--result-root", default="results/raw")
    parser.add_argument("--summary-root", default="results/summary")
    parser.add_argument("--log-root", default="logs/dsr_smoke")
    return parser.parse_args()


def count_parameters(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def grad_norm(params: list[torch.Tensor], loss: torch.Tensor) -> float:
    used = [p for p in params if p.requires_grad]
    grads = torch.autograd.grad(loss, used, retain_graph=True, allow_unused=True)
    total = loss.new_tensor(0.0)
    for grad in grads:
        if grad is not None:
            total = total + grad.detach().pow(2).sum()
    return float(torch.sqrt(total).item())


def low_pass_features(x: torch.Tensor, edge_index: torch.Tensor, k: int) -> torch.Tensor:
    edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.size(0))
    row, col = edge_index
    deg = degree(row, x.size(0), dtype=x.dtype).clamp_min(1.0)
    deg_inv_sqrt = deg.pow(-0.5)
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    out = x
    for _ in range(k):
        new_out = torch.zeros_like(out)
        new_out.index_add_(0, col, out[row] * norm.view(-1, 1))
        out = new_out
    return out


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def variance_loss(z: torch.Tensor, gamma: float, eps: float) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(F.relu(gamma - std))


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / max(z.size(0) - 1, 1)
    return off_diagonal(cov).pow(2).sum() / z.size(1)


def vicreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    lambda_var: float,
    lambda_cov: float,
    gamma: float,
    eps: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    align = F.mse_loss(z1, z2)
    var = variance_loss(z1, gamma, eps) + variance_loss(z2, gamma, eps)
    cov = covariance_loss(z1) + covariance_loss(z2)
    loss = align + lambda_var * var + lambda_cov * cov
    return loss, {
        "alignment": float(align.detach().item()),
        "variance": float(var.detach().item()),
        "covariance": float(cov.detach().item()),
    }


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, tau: float) -> torch.Tensor:
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    f = lambda x: torch.exp(x / tau)
    refl_1 = f(z1 @ z1.T)
    refl_2 = f(z2 @ z2.T)
    between_12 = f(z1 @ z2.T)
    between_21 = f(z2 @ z1.T)
    l1 = -torch.log(
        between_12.diag()
        / (refl_1.sum(1) + between_12.sum(1) - refl_1.diag()).clamp_min(1e-12)
    )
    l2 = -torch.log(
        between_21.diag()
        / (refl_2.sum(1) + between_21.sum(1) - refl_2.diag()).clamp_min(1e-12)
    )
    return 0.5 * (l1 + l2).mean()


def corr_matrix(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    z1 = z1 - z1.mean(dim=0)
    z2 = z2 - z2.mean(dim=0)
    z1 = z1 / z1.std(dim=0).clamp_min(1e-6)
    z2 = z2 / z2.std(dim=0).clamp_min(1e-6)
    return (z1.T @ z2) / max(z1.size(0) - 1, 1)


def corr_frobenius(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    return corr_matrix(z1, z2).pow(2).mean()


@torch.no_grad()
def effective_rank(z: torch.Tensor | None) -> float | None:
    if z is None:
        return None
    z = z.detach().float().cpu()
    z = z - z.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(z)
    singular_values = singular_values[singular_values > 1e-12]
    if singular_values.numel() == 0:
        return 0.0
    probs = singular_values / singular_values.sum()
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum()
    return float(torch.exp(entropy).item())


@torch.no_grad()
def uniformity(z: torch.Tensor | None) -> float | None:
    if z is None:
        return None
    z = F.normalize(z, dim=1)
    return float(torch.pdist(z, p=2).pow(2).mul(-2).exp().mean().log().item())


@torch.no_grad()
def branch_corr(z_sem: torch.Tensor | None, z_res: torch.Tensor | None) -> float | None:
    if z_sem is None or z_res is None:
        return None
    return float(corr_frobenius(z_sem, z_res).item())


class Projector(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.elu(self.fc1(z)))


class DSRModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        branch_dim: int,
        activation_name: str,
        num_layers: int,
        use_semantic: bool,
        use_residual: bool,
    ):
        super().__init__()
        self.use_semantic = use_semantic
        self.use_residual = use_residual
        if use_semantic:
            self.encoder_sem = Encoder(
                in_dim,
                branch_dim,
                activation_from_name(activation_name),
                base_model=GCNConv,
                k=num_layers,
            )
            self.projector_sem = Projector(branch_dim, branch_dim)
        if use_residual:
            self.encoder_res = Encoder(
                in_dim,
                branch_dim,
                activation_from_name(activation_name),
                base_model=GCNConv,
                k=num_layers,
            )
            self.projector_res = Projector(branch_dim, branch_dim)

    def sem_params(self) -> list[torch.Tensor]:
        if not self.use_semantic:
            return []
        return list(self.encoder_sem.parameters()) + list(self.projector_sem.parameters())

    def res_params(self) -> list[torch.Tensor]:
        if not self.use_residual:
            return []
        return list(self.encoder_res.parameters()) + list(self.projector_res.parameters())

    def encode_views(
        self,
        x1: torch.Tensor,
        edge1: torch.Tensor,
        x2: torch.Tensor,
        edge2: torch.Tensor,
        low_pass_k: int,
    ) -> BranchOutput:
        out = BranchOutput()
        if self.use_semantic:
            x1_sem = low_pass_features(x1, edge1, low_pass_k)
            x2_sem = low_pass_features(x2, edge2, low_pass_k)
            out.z_sem_1 = F.normalize(self.projector_sem(self.encoder_sem(x1_sem, edge1)), dim=1)
            out.z_sem_2 = F.normalize(self.projector_sem(self.encoder_sem(x2_sem, edge2)), dim=1)
        if self.use_residual:
            x1_low = low_pass_features(x1, edge1, low_pass_k).detach()
            x2_low = low_pass_features(x2, edge2, low_pass_k).detach()
            x1_res = x1 - x1_low
            x2_res = x2 - x2_low
            out.z_res_1 = F.normalize(self.projector_res(self.encoder_res(x1_res, edge1)), dim=1)
            out.z_res_2 = F.normalize(self.projector_res(self.encoder_res(x2_res, edge2)), dim=1)
        return out

    @torch.no_grad()
    def encode_eval(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        low_pass_k: int,
    ) -> dict[str, torch.Tensor]:
        self.eval()
        embeddings = {}
        if self.use_semantic:
            x_sem = low_pass_features(x, edge_index, low_pass_k)
            embeddings["z_sem"] = F.normalize(
                self.projector_sem(self.encoder_sem(x_sem, edge_index)),
                dim=1,
            )
        if self.use_residual:
            x_low = low_pass_features(x, edge_index, low_pass_k).detach()
            x_res = x - x_low
            embeddings["z_res"] = F.normalize(
                self.projector_res(self.encoder_res(x_res, edge_index)),
                dim=1,
            )
        if "z_sem" in embeddings and "z_res" in embeddings:
            embeddings["concat"] = torch.cat([embeddings["z_sem"], embeddings["z_res"].detach()], dim=1)
        return embeddings


class SingleHeadModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, activation_name: str, num_layers: int):
        super().__init__()
        self.encoder = Encoder(
            in_dim,
            hidden_dim,
            activation_from_name(activation_name),
            base_model=GCNConv,
            k=num_layers,
        )
        self.projector = Projector(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projector(self.encoder(x, edge_index)), dim=1)

    @torch.no_grad()
    def encode_eval(self, x: torch.Tensor, edge_index: torch.Tensor) -> dict[str, torch.Tensor]:
        self.eval()
        return {"single": self.forward(x, edge_index)}


def resolve_single_head_hidden_dim(
    in_dim: int,
    activation_name: str,
    num_layers: int,
    target_params: int,
    requested: int | str | None,
) -> tuple[int, dict[str, int | float | str]]:
    if requested is None:
        return target_params, {"match_mode": "default_base_hidden"}
    if isinstance(requested, int):
        return requested, {"match_mode": "explicit_hidden_dim"}
    if str(requested) != "param_match_dsr_full":
        raise ValueError(f"Unsupported hidden_dim request: {requested}")

    best_hidden = None
    best_params = None
    best_abs_diff = None
    for hidden_dim in range(1, 513):
        probe = SingleHeadModel(in_dim, hidden_dim, activation_name, num_layers)
        params = count_parameters(probe)
        abs_diff = abs(params - target_params)
        if best_abs_diff is None or abs_diff < best_abs_diff:
            best_hidden = hidden_dim
            best_params = params
            best_abs_diff = abs_diff
    assert best_hidden is not None and best_params is not None and best_abs_diff is not None
    return int(best_hidden), {
        "match_mode": "closest_single_head_hidden_dim_to_dsr_full",
        "target_parameter_count": int(target_params),
        "actual_parameter_count": int(best_params),
        "absolute_parameter_gap": int(best_abs_diff),
        "relative_parameter_gap": float(best_abs_diff / max(target_params, 1)),
    }


def train_grace_variant(
    dataset_name: str,
    seed: int,
    data,
    dataset,
    split,
    pretrain_cfg: dict[str, Any],
    eval_cfg: dict[str, Any],
    device: torch.device,
    log_path: Path,
) -> tuple[nn.Module, dict[str, Any], dict[str, torch.Tensor]]:
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
    logs = []
    start = time.time()
    num_epochs = int(pretrain_cfg["num_epochs"])
    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        edge1 = drop_edges(data.edge_index, float(pretrain_cfg["drop_edge_rate_1"]))
        edge2 = drop_edges(data.edge_index, float(pretrain_cfg["drop_edge_rate_2"]))
        x1 = drop_feature(data.x, float(pretrain_cfg["drop_feature_rate_1"]))
        x2 = drop_feature(data.x, float(pretrain_cfg["drop_feature_rate_2"]))
        z1 = model(x1, edge1)
        z2 = model(x2, edge2)
        loss = model.loss(z1, z2, batch_size=int(pretrain_cfg.get("loss_batch_size", 0)))
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch == num_epochs or epoch % int(pretrain_cfg["log_every"]) == 0:
            row = {"epoch": epoch, "loss": float(loss.detach().item()), "elapsed_sec": time.time() - start}
            logs.append(row)
            print(f"[{dataset_name} seed={seed} A0 grace] epoch={epoch:04d} loss={row['loss']:.4f}", flush=True)
    log_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in logs) + "\n")
    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)
        if bool(eval_cfg.get("normalize_embeddings", True)):
            z = F.normalize(z, dim=1)
    diagnostics = {
        "effective_rank_grace": effective_rank(z),
        "uniformity_grace": uniformity(z),
        "parameter_count": count_parameters(model),
        "negative_gradient_leakage": None,
        "firewall_pass": None,
    }
    return model, diagnostics, {"grace": z.detach()}


def train_dsr_variant(
    dataset_name: str,
    seed: int,
    data,
    dataset,
    variant_cfg: dict[str, Any],
    pretrain_cfg: dict[str, Any],
    device: torch.device,
    log_path: Path,
) -> tuple[nn.Module, dict[str, Any], dict[str, torch.Tensor]]:
    set_seed(seed)
    base_hidden = int(pretrain_cfg["num_hidden"])
    branch_dim = max(base_hidden // 2, 1)
    model = DSRModel(
        dataset.num_features,
        branch_dim,
        str(pretrain_cfg["activation"]),
        int(pretrain_cfg["num_layers"]),
        bool(variant_cfg.get("use_semantic", False)),
        bool(variant_cfg.get("use_residual", False)),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(pretrain_cfg["learning_rate"]),
        weight_decay=float(pretrain_cfg["weight_decay"]),
    )
    logs = []
    start = time.time()
    last_diag: dict[str, float | None] = {}
    num_epochs = int(pretrain_cfg["num_epochs"])
    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        edge1 = drop_edges(data.edge_index, float(pretrain_cfg["drop_edge_rate_1"]))
        edge2 = drop_edges(data.edge_index, float(pretrain_cfg["drop_edge_rate_2"]))
        x1 = drop_feature(data.x, float(pretrain_cfg["drop_feature_rate_1"]))
        x2 = drop_feature(data.x, float(pretrain_cfg["drop_feature_rate_2"]))
        out = model.encode_views(x1, edge1, x2, edge2, int(pretrain_cfg["low_pass_k"]))

        loss = data.x.new_tensor(0.0)
        a9_scale_reference = data.x.new_tensor(0.0)
        neg_loss = None
        diag: dict[str, float | None] = {}
        semantic_loss_value = None
        residual_nce_value = None
        semantic_nce_value = None
        orth_value = None
        if bool(variant_cfg.get("semantic_negative_free", False)) and out.z_sem_1 is not None:
            sem_loss, sem_diag = vicreg_loss(
                out.z_sem_1,
                out.z_sem_2,
                float(pretrain_cfg["lambda_var"]),
                float(pretrain_cfg["lambda_cov"]),
                float(pretrain_cfg["variance_gamma"]),
                float(pretrain_cfg["eps"]),
            )
            loss = loss + sem_loss
            a9_scale_reference = a9_scale_reference + sem_loss
            semantic_loss_value = sem_loss
            diag.update({f"semantic_{k}": v for k, v in sem_diag.items()})
        if bool(variant_cfg.get("residual_infonce", False)) and out.z_res_1 is not None:
            res_nce = info_nce_loss(out.z_res_1, out.z_res_2, float(pretrain_cfg["tau"]))
            residual_weight = float(variant_cfg.get("residual_infonce_weight", pretrain_cfg["alpha_res"]))
            loss = loss + residual_weight * res_nce
            a9_scale_reference = a9_scale_reference + float(pretrain_cfg["alpha_res"]) * res_nce
            neg_component = residual_weight * res_nce
            neg_loss = neg_component if neg_loss is None else neg_loss + neg_component
            residual_nce_value = res_nce
            diag["residual_infonce"] = float(res_nce.detach().item())
            diag["residual_infonce_weight"] = float(residual_weight)
        if bool(variant_cfg.get("semantic_infonce", False)) and out.z_sem_1 is not None:
            sem_nce = info_nce_loss(out.z_sem_1, out.z_sem_2, float(pretrain_cfg["tau"]))
            semantic_weight = float(variant_cfg.get("semantic_infonce_weight", 1.0))
            loss = loss + semantic_weight * sem_nce
            neg_component = semantic_weight * sem_nce
            neg_loss = neg_component if neg_loss is None else neg_loss + neg_component
            semantic_nce_value = sem_nce
            diag["semantic_infonce"] = float(sem_nce.detach().item())
            diag["semantic_infonce_weight"] = float(semantic_weight)
        if bool(variant_cfg.get("orthogonality", False)) and out.z_sem_1 is not None and out.z_res_1 is not None:
            orth = corr_frobenius(out.z_sem_1.detach(), out.z_res_1) + corr_frobenius(
                out.z_sem_2.detach(),
                out.z_res_2,
            )
            loss = loss + float(pretrain_cfg["beta_orth"]) * orth
            a9_scale_reference = a9_scale_reference + float(pretrain_cfg["beta_orth"]) * orth
            orth_value = orth
            diag["orthogonality"] = float(orth.detach().item())

        loss_scale = 1.0
        if variant_cfg.get("total_loss_scale_mode") == "match_a9_loss_scale":
            raw_loss_for_scale = float(loss.detach().abs().item())
            ref_loss_for_scale = float(a9_scale_reference.detach().abs().item())
            if raw_loss_for_scale > 0:
                loss_scale = ref_loss_for_scale / raw_loss_for_scale
                loss = loss * loss_scale
            diag["loss_scale_mode"] = "match_a9_loss_scale"
            diag["loss_scale"] = float(loss_scale)
            diag["a9_reference_loss_scale"] = float(ref_loss_for_scale)
            diag["raw_unscaled_loss"] = float(raw_loss_for_scale)
        else:
            diag["loss_scale_mode"] = "none"
            diag["loss_scale"] = float(loss_scale)

        if neg_loss is not None and model.sem_params() and model.res_params():
            sem_grad = grad_norm(model.sem_params(), neg_loss)
            res_grad = grad_norm(model.res_params(), neg_loss)
            leak = sem_grad / (res_grad + float(pretrain_cfg["eps"]))
            diag["negative_gradient_leakage"] = float(leak)
            diag["negative_grad_norm_sem"] = float(sem_grad)
            diag["negative_grad_norm_res"] = float(res_grad)
        else:
            diag["negative_gradient_leakage"] = None
            diag["negative_grad_norm_sem"] = None
            diag["negative_grad_norm_res"] = None

        loss.backward()
        optimizer.step()
        last_diag = diag
        if epoch == 1 or epoch == num_epochs or epoch % int(pretrain_cfg["log_every"]) == 0:
            row = {
                "epoch": epoch,
                "loss": float(loss.detach().item()),
                "elapsed_sec": time.time() - start,
                "semantic_loss": float(semantic_loss_value.detach().item()) if semantic_loss_value is not None else None,
                "residual_infonce_loss": float(residual_nce_value.detach().item()) if residual_nce_value is not None else None,
                "semantic_infonce_loss": float(semantic_nce_value.detach().item()) if semantic_nce_value is not None else None,
                "orthogonality_loss": float(orth_value.detach().item()) if orth_value is not None else None,
                **diag,
            }
            logs.append(row)
            print(
                f"[{dataset_name} seed={seed} {variant_cfg['id']} {variant_cfg['name']}] "
                f"epoch={epoch:04d} loss={row['loss']:.4f} leak={row['negative_gradient_leakage']}",
                flush=True,
            )
    log_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in logs) + "\n")
    embeddings = model.encode_eval(data.x, data.edge_index, int(pretrain_cfg["low_pass_k"]))
    diagnostics = {
        **last_diag,
        "effective_rank_sem": effective_rank(embeddings.get("z_sem")),
        "effective_rank_res": effective_rank(embeddings.get("z_res")),
        "effective_rank_concat": effective_rank(embeddings.get("concat")),
        "branch_correlation": branch_corr(embeddings.get("z_sem"), embeddings.get("z_res")),
        "uniformity_res": uniformity(embeddings.get("z_res")),
        "uniformity_concat": uniformity(embeddings.get("concat")),
        "parameter_count": count_parameters(model),
    }
    leak = diagnostics.get("negative_gradient_leakage")
    diagnostics["firewall_pass"] = (
        bool(leak < float(pretrain_cfg["firewall_threshold"])) if leak is not None else None
    )
    return model, diagnostics, {k: v.detach() for k, v in embeddings.items()}


def train_single_head_variant(
    dataset_name: str,
    seed: int,
    data,
    dataset,
    variant_cfg: dict[str, Any],
    pretrain_cfg: dict[str, Any],
    target_params: int,
    device: torch.device,
    log_path: Path,
) -> tuple[nn.Module, dict[str, Any], dict[str, torch.Tensor]]:
    set_seed(seed)
    hidden_dim, match_info = resolve_single_head_hidden_dim(
        int(dataset.num_features),
        str(pretrain_cfg["activation"]),
        int(pretrain_cfg["num_layers"]),
        target_params,
        variant_cfg.get("hidden_dim", int(pretrain_cfg["num_hidden"])),
    )
    model = SingleHeadModel(
        dataset.num_features,
        int(hidden_dim),
        str(pretrain_cfg["activation"]),
        int(pretrain_cfg["num_layers"]),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(pretrain_cfg["learning_rate"]),
        weight_decay=float(pretrain_cfg["weight_decay"]),
    )
    logs = []
    start = time.time()
    last_diag: dict[str, float | None] = {}
    num_epochs = int(pretrain_cfg["num_epochs"])
    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        edge1 = drop_edges(data.edge_index, float(pretrain_cfg["drop_edge_rate_1"]))
        edge2 = drop_edges(data.edge_index, float(pretrain_cfg["drop_edge_rate_2"]))
        x1 = drop_feature(data.x, float(pretrain_cfg["drop_feature_rate_1"]))
        x2 = drop_feature(data.x, float(pretrain_cfg["drop_feature_rate_2"]))
        z1 = model(x1, edge1)
        z2 = model(x2, edge2)
        sem_loss, sem_diag = vicreg_loss(
            z1,
            z2,
            float(pretrain_cfg["lambda_var"]),
            float(pretrain_cfg["lambda_cov"]),
            float(pretrain_cfg["variance_gamma"]),
            float(pretrain_cfg["eps"]),
        )
        neg = info_nce_loss(z1, z2, float(pretrain_cfg["tau"]))
        loss = sem_loss + neg
        loss.backward()
        optimizer.step()
        last_diag = {
            "single_head_infonce": float(neg.detach().item()),
            **{f"single_{k}": v for k, v in sem_diag.items()},
        }
        if epoch == 1 or epoch == num_epochs or epoch % int(pretrain_cfg["log_every"]) == 0:
            row = {"epoch": epoch, "loss": float(loss.detach().item()), "elapsed_sec": time.time() - start, **last_diag}
            logs.append(row)
            print(f"[{dataset_name} seed={seed} A4 single-head] epoch={epoch:04d} loss={row['loss']:.4f}", flush=True)
    log_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in logs) + "\n")
    embeddings = model.encode_eval(data.x, data.edge_index)
    z = embeddings["single"]
    diagnostics = {
        **last_diag,
        "effective_rank_single": effective_rank(z),
        "uniformity_single": uniformity(z),
        "parameter_count": count_parameters(model),
        "single_head_hidden_dim": int(hidden_dim),
        **match_info,
        "negative_gradient_leakage": None,
        "firewall_pass": None,
    }
    return model, diagnostics, {k: v.detach() for k, v in embeddings.items()}


def evaluate_embeddings(
    embeddings: dict[str, torch.Tensor],
    y: torch.Tensor,
    split,
    eval_cfg: dict[str, Any],
    seed: int,
    main_key: str,
) -> dict[str, Any]:
    evals = {}
    for key, z in embeddings.items():
        evals[key] = logreg_val_eval(z, y, split, eval_cfg, seed)
    main = evals[main_key]
    return {
        "main_eval_embedding": main_key,
        "valid_at_best": main["valid_at_best"],
        "test_at_best": main["test_at_best"],
        "final_test": main["final_test"],
        "best_c": main.get("best_c"),
        "embedding_evals": evals,
    }


def eval_metric(evals: dict[str, Any], key: str, metric: str = "test_at_best") -> float | None:
    if key not in evals:
        return None
    return float(evals[key][metric])


def result_markdown(results: list[dict[str, Any]], summary_path: Path) -> None:
    lines = [
        "# DSR-GCL Audit-Smoke Summary",
        "",
        "Scope: Cora seed=0 only, stratified random 1:1:8, frozen encoder + Logistic Regression.",
        "Status: audit-smoke diagnostic only; no formal result and no performance claim.",
        "",
        "## Audit Notes",
        "",
        "- Previous A9 DSR-full result was poor relative to A2/A5/A4/A0.",
        "- A3 residual-only was almost ineffective, so residual utility is under question.",
        "- Previous A5 was unfair because it added semantic InfoNCE on top of residual InfoNCE.",
        "- Previous A4 was not strictly parameter-matched to A9; this run adds A4b.",
        "- This summary explicitly reports embedding-level results, params, ranks, leakage, raw JSON, and logs.",
        "",
        "## Complete Result Table",
        "",
        "| ID | Variant | main z | valid@best | test@best | final test | params | raw JSON | log |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['variant_id']} | {r['variant_name']} | {r['evaluation']['main_eval_embedding']} | "
            f"{r['evaluation']['valid_at_best']:.2f} | {r['evaluation']['test_at_best']:.2f} | "
            f"{r['evaluation']['final_test']:.2f} | {r['diagnostics'].get('parameter_count')} | "
            f"`{r['result_path']}` | `{r['train_log_path']}` |"
        )

    def fmt(value: Any) -> str:
        if value is None:
            return "NA"
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, (float, int)):
            return f"{value:.4g}"
        return str(value)

    lines.extend(["", "## Embedding-Level Table", ""])
    lines.extend(
        [
            "| ID | z_sem test | z_res test | concat test | grace test | single test | rank_sem | rank_res | rank_concat | rank_single | branch_corr |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for r in results:
        d = r["diagnostics"]
        evals = r["evaluation"]["embedding_evals"]
        lines.append(
            f"| {r['variant_id']} | {fmt(eval_metric(evals, 'z_sem'))} | {fmt(eval_metric(evals, 'z_res'))} | "
            f"{fmt(eval_metric(evals, 'concat'))} | {fmt(eval_metric(evals, 'grace'))} | "
            f"{fmt(eval_metric(evals, 'single'))} | {fmt(d.get('effective_rank_sem'))} | "
            f"{fmt(d.get('effective_rank_res'))} | {fmt(d.get('effective_rank_concat'))} | "
            f"{fmt(d.get('effective_rank_single'))} | {fmt(d.get('branch_correlation'))} |"
        )

    lines.extend(["", "## Parameter Count Table", ""])
    lines.extend(["| ID | Variant | params | target params | abs gap | rel gap | hidden dim | note |", "|---|---|---:|---:|---:|---:|---:|---|"])
    for r in results:
        d = r["diagnostics"]
        lines.append(
            f"| {r['variant_id']} | {r['variant_name']} | {fmt(d.get('parameter_count'))} | "
            f"{fmt(d.get('target_parameter_count'))} | {fmt(d.get('absolute_parameter_gap'))} | "
            f"{fmt(d.get('relative_parameter_gap'))} | {fmt(d.get('single_head_hidden_dim'))} | "
            f"{r['variant_config'].get('audit_note', '')} |"
        )

    lines.extend(["", "## Leakage Table", ""])
    lines.extend(["| ID | leak | sem grad | res grad | firewall pass | res weight | sem weight | loss scale |", "|---|---:|---:|---:|---|---:|---:|---:|"])
    for r in results:
        d = r["diagnostics"]
        lines.append(
            f"| {r['variant_id']} | {fmt(d.get('negative_gradient_leakage'))} | "
            f"{fmt(d.get('negative_grad_norm_sem'))} | {fmt(d.get('negative_grad_norm_res'))} | "
            f"{fmt(d.get('firewall_pass'))} | {fmt(d.get('residual_infonce_weight'))} | "
            f"{fmt(d.get('semantic_infonce_weight'))} | {fmt(d.get('loss_scale'))} |"
        )
    full = next((r for r in results if r["variant_id"] == "A9"), None)
    sem_only = next((r for r in results if r["variant_id"] == "A2"), None)
    residual_only = next((r for r in results if r["variant_id"] == "A3"), None)
    same_head = next((r for r in results if r["variant_id"] == "A4"), None)
    matched_same_head = next((r for r in results if r["variant_id"] == "A4b"), None)
    no_firewall = next((r for r in results if r["variant_id"] == "A5"), None)
    budget_no_firewall = next((r for r in results if r["variant_id"] == "A5b"), None)
    scaled_no_firewall = next((r for r in results if r["variant_id"] == "A5c"), None)
    conclusions = []
    triggered = []
    if full:
        leak = full["diagnostics"].get("negative_gradient_leakage")
        conclusions.append(
            "C1 firewall diagnostic: "
            + ("PASS" if leak is not None and leak < 1e-8 else "REVISE")
            + f" (A9 leak={leak})."
        )
    if full and no_firewall:
        conclusions.append(
            "A5 no-firewall check: "
            f"A5 leak={no_firewall['diagnostics'].get('negative_gradient_leakage')}, "
            f"A9 leak={full['diagnostics'].get('negative_gradient_leakage')}."
        )
    if full and sem_only:
        gap = full["evaluation"]["test_at_best"] - sem_only["evaluation"]["test_at_best"]
        conclusions.append(f"A2 semantic-only vs A9 full smoke gap: {gap:.2f} points; single seed only.")
        if gap < 0:
            triggered.append("A9_below_A2_semantic_only")
    if full and residual_only:
        res_value = residual_only["evaluation"]["test_at_best"]
        conclusions.append(f"A3 residual-only test@best: {res_value:.2f}; single seed only.")
        if res_value < 40.0:
            triggered.append("A3_residual_only_near_random_or_extremely_low")
    if full and same_head:
        gap = full["evaluation"]["test_at_best"] - same_head["evaluation"]["test_at_best"]
        conclusions.append(f"A4 same-head vs A9 full smoke gap: {gap:.2f} points; single seed only.")
        if gap < 0:
            triggered.append("A9_below_A4_same_head")
    if full and matched_same_head:
        gap = full["evaluation"]["test_at_best"] - matched_same_head["evaluation"]["test_at_best"]
        conclusions.append(f"A4b param-matched same-head vs A9 full smoke gap: {gap:.2f} points; single seed only.")
        if gap < 0:
            triggered.append("A9_below_A4b_param_matched_single_head")
    for candidate, label in [(budget_no_firewall, "A5b"), (scaled_no_firewall, "A5c")]:
        if full and candidate:
            gap = full["evaluation"]["test_at_best"] - candidate["evaluation"]["test_at_best"]
            conclusions.append(f"{label} fairer no-firewall vs A9 full smoke gap: {gap:.2f} points; single seed only.")
            if gap < 0:
                triggered.append(f"{label}_above_A9_firewall_claim_not_supported")
    lines.extend(["", "## Kill Rule Readout", ""])
    lines.extend(f"- {item}" for item in conclusions)
    lines.append(f"- Triggered rules: `{', '.join(triggered) if triggered else 'none'}`.")
    if len(triggered) >= 2:
        decision = "`PIVOT_REQUIRED`"
        recommendation = "不建议继续推进 DSR-GCL formal；应先重构 residual branch 或放弃当前 firewall 叙事。"
    else:
        decision = "`REVISE`"
        recommendation = "只允许继续做更小范围机制修正，不进入 formal。"
    lines.append("")
    lines.append(f"Decision: {decision}. {recommendation}")
    summary_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    config = yaml.safe_load((PROJECT_ROOT / args.config).read_text())
    base_config = yaml.safe_load((PROJECT_ROOT / args.base_config).read_text())
    dataset_name = args.dataset
    seed = int(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")
    print(f"device={device} dataset={dataset_name} seed={seed}")

    dataset = load_dataset(dataset_name, PROJECT_ROOT / args.data_root)
    data = dataset[0].to(device)
    split = create_or_load_split(dataset_name, data.y.detach().cpu(), seed, PROJECT_ROOT / args.split_root)
    pretrain_cfg = dict(base_config["pretrain"][dataset_name])
    pretrain_cfg.update(config["pretrain"][dataset_name])
    if args.epochs is not None:
        pretrain_cfg["num_epochs"] = int(args.epochs)
    eval_cfg = dict(base_config["logreg_eval"])
    status = args.status or config["meta"].get("status_default", "smoke")

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_name = f"{args.run_tag}_{dataset_name}_seed{seed}_{timestamp}"
    result_dir = PROJECT_ROOT / args.result_root / dataset_name / "DSR_GCL_SMOKE" / run_name
    log_dir = PROJECT_ROOT / args.log_root / run_name
    summary_dir = PROJECT_ROOT / args.summary_root
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    results = []
    dsr_target_probe = DSRModel(
        dataset.num_features,
        max(int(pretrain_cfg["num_hidden"]) // 2, 1),
        str(pretrain_cfg["activation"]),
        int(pretrain_cfg["num_layers"]),
        True,
        True,
    )
    dsr_target_params = count_parameters(dsr_target_probe)
    for variant_cfg in config["variants"]:
        variant_id = variant_cfg["id"]
        variant_name = variant_cfg["name"]
        log_path = log_dir / f"{variant_id}_{variant_name}.jsonl"
        start = time.time()
        if variant_cfg["family"] == "grace":
            _, diagnostics, embeddings = train_grace_variant(
                dataset_name, seed, data, dataset, split, pretrain_cfg, eval_cfg, device, log_path
            )
        elif variant_cfg["family"] == "dsr":
            _, diagnostics, embeddings = train_dsr_variant(
                dataset_name, seed, data, dataset, variant_cfg, pretrain_cfg, device, log_path
            )
        elif variant_cfg["family"] == "single_head":
            _, diagnostics, embeddings = train_single_head_variant(
                dataset_name, seed, data, dataset, variant_cfg, pretrain_cfg, dsr_target_params, device, log_path
            )
        else:
            raise ValueError(f"Unknown variant family: {variant_cfg['family']}")

        main_key = str(variant_cfg["main_eval_embedding"])
        evaluation = evaluate_embeddings(embeddings, data.y, split, eval_cfg, seed, main_key)
        result = {
            "run_id": f"{run_name}_{variant_id}",
            "method": "DSR-GCL-smoke",
            "variant_id": variant_id,
            "variant_name": variant_name,
            "variant_config": variant_cfg,
            "dataset": dataset_name,
            "split_type": "stratified_random_1_1_8",
            "split_seed": seed,
            "model_seed": seed,
            "status": status,
            "metric": "accuracy_percent",
            "config_path": str((PROJECT_ROOT / args.config).resolve()),
            "base_config_path": str((PROJECT_ROOT / args.base_config).resolve()),
            "split_path": str(split.path.resolve()),
            "train_log_path": str(log_path.resolve()),
            "project_commit_hash": git_rev(PROJECT_ROOT),
            "project_dirty": git_dirty(PROJECT_ROOT),
            "dataset_num_nodes": int(data.num_nodes),
            "dataset_num_edges": int(data.edge_index.size(1)),
            "dataset_num_features": int(dataset.num_features),
            "dataset_num_classes": int(dataset.num_classes),
            "pretrain_config": pretrain_cfg,
            "evaluator_type": "logreg_val",
            "evaluator_config": eval_cfg,
            "diagnostics": diagnostics,
            "evaluation": evaluation,
            "elapsed_sec": time.time() - start,
        }
        result_path = result_dir / f"{variant_id}_{variant_name}.json"
        result["result_path"] = str(result_path.resolve())
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        results.append(result)
        print(
            f"[{variant_id} {variant_name}] main={evaluation['main_eval_embedding']} "
            f"valid={evaluation['valid_at_best']:.2f} test={evaluation['test_at_best']:.2f} "
            f"result={result_path}",
            flush=True,
        )

    summary_json = summary_dir / f"{run_name}_summary.json"
    summary_md = summary_dir / f"{run_name}_summary.md"
    summary_payload = {
        "run_name": run_name,
        "status": status,
        "dataset": dataset_name,
        "seed": seed,
        "config_path": str((PROJECT_ROOT / args.config).resolve()),
        "result_dir": str(result_dir.resolve()),
        "log_dir": str(log_dir.resolve()),
        "results": results,
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n")
    result_markdown(results, summary_md)
    print(f"summary_json={summary_json}")
    print(f"summary_md={summary_md}")


if __name__ == "__main__":
    main()

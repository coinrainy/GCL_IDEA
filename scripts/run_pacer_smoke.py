#!/usr/bin/env python
"""Run PACER-GCL PA-CER-M0-001 smoke.

PACER is a train-label-calibrated / semi-supervised GCL smoke. The runner
keeps the downstream evaluator identical to the project GRACE protocol:
frozen encoder embeddings plus Logistic Regression selected by validation
accuracy. Test labels are never used for probe fitting, routing, thresholds,
or hyperparameter selection.
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
    parser.add_argument("--config", default="configs/pacer_smoke.yaml")
    parser.add_argument("--base-config", default="configs/grace_1_1_8.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--probe-epochs", type=int, default=None)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--status", default=None)
    parser.add_argument("--run-tag", default="pa_cer_m0_001")
    parser.add_argument("--data-root", default="data/pyg")
    parser.add_argument("--split-root", default="splits")
    parser.add_argument("--result-root", default="results/raw")
    parser.add_argument("--summary-root", default="results/summary")
    parser.add_argument("--log-root", default="logs/pacer_smoke")
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def exact_command() -> str:
    return " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])


def as_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


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


class ProbeHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 0) -> None:
        super().__init__()
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.net = nn.Linear(in_dim, num_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


def margin_from_logits(logits: torch.Tensor) -> torch.Tensor:
    top2 = logits.topk(k=2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def train_probe(
    h: torch.Tensor,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    cfg: dict[str, Any],
    seed: int,
    num_classes: int,
    label_mode: str = "true",
) -> tuple[ProbeHead, torch.Tensor, dict[str, Any]]:
    set_seed(seed)
    labels = y.clone()
    if label_mode == "shuffled":
        generator = torch.Generator(device=y.device)
        generator.manual_seed(seed + 71_003)
        shuffled = y[train_idx][torch.randperm(train_idx.numel(), generator=generator, device=y.device)]
        labels[train_idx] = shuffled
    elif label_mode != "true":
        raise ValueError(f"Unsupported label_mode: {label_mode}")

    probe = ProbeHead(h.size(1), num_classes, int(cfg.get("probe_hidden_dim", 0))).to(h.device)
    optimizer = torch.optim.Adam(
        probe.parameters(),
        lr=float(cfg["probe_lr"]),
        weight_decay=float(cfg["probe_weight_decay"]),
    )
    best = {"loss": math.inf, "state": None, "epoch": 0}
    wait = 0
    patience = int(cfg["probe_patience"])
    max_epochs = int(cfg["probe_epochs"])
    for epoch in range(1, max_epochs + 1):
        probe.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(probe(h.detach())[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().item())
        if loss_value < best["loss"]:
            best = {
                "loss": loss_value,
                "state": {k: v.detach().clone() for k, v in probe.state_dict().items()},
                "epoch": epoch,
            }
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best["state"] is not None:
        probe.load_state_dict(best["state"])
    probe.eval()
    with torch.no_grad():
        train_acc = (probe(h)[train_idx].argmax(dim=1) == labels[train_idx]).float().mean()
    return probe, labels, {
        "probe_type": "torch_linear_train_labels_only",
        "label_mode": label_mode,
        "best_epoch_by_train_loss": int(best["epoch"]),
        "train_loss": float(best["loss"]),
        "train_probe_acc_percent": float(train_acc.item() * 100.0),
        "uses_train_labels": True,
        "uses_validation_labels": False,
        "uses_test_labels": False,
    }


@torch.no_grad()
def probe_margins(probe: ProbeHead, h: torch.Tensor) -> torch.Tensor:
    probe.eval()
    return margin_from_logits(probe(h))


def two_view_embeddings(model: Model, data, cfg: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    edge_1 = drop_edges(data.edge_index, float(cfg["drop_edge_rate_1"]))
    edge_2 = drop_edges(data.edge_index, float(cfg["drop_edge_rate_2"]))
    x_1 = drop_feature(data.x, float(cfg["drop_feature_rate_1"]))
    x_2 = drop_feature(data.x, float(cfg["drop_feature_rate_2"]))
    return model(x_1, edge_1), model(x_2, edge_2)


def nodewise_grace_loss(model: Model, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
    z1 = model.projection(h1)
    z2 = model.projection(h2)
    return 0.5 * (model.semi_loss(z1, z2) + model.semi_loss(z2, z1))


def representation_consistency_loss(h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(F.normalize(h1, dim=1), F.normalize(h2, dim=1))


def logit_consistency_loss(
    probe: ProbeHead,
    target_logits: torch.Tensor,
    h_view: torch.Tensor,
    mask: torch.Tensor,
    margin_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if int(mask.sum().item()) == 0:
        zero = h_view.new_tensor(0.0)
        return zero, zero, zero
    target_prob = F.softmax(target_logits[mask], dim=1).detach()
    view_logits = probe(h_view[mask])
    logit_loss = F.kl_div(F.log_softmax(view_logits, dim=1), target_prob, reduction="batchmean")
    margin = margin_from_logits(view_logits)
    margin_loss = F.relu(float(margin_floor) - margin).mean()
    return logit_loss, margin_loss, margin.detach().mean()


def supervised_contrastive_loss(
    h1: torch.Tensor,
    h2: torch.Tensor,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    if train_idx.numel() < 2:
        return h1.new_tensor(0.0)
    z = torch.cat([h1[train_idx], h2[train_idx]], dim=0)
    labels = torch.cat([y[train_idx], y[train_idx]], dim=0)
    z = F.normalize(z, dim=1)
    sim = (z @ z.T) / float(tau)
    self_mask = torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
    pos_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
    exp_sim = torch.exp(sim).masked_fill(self_mask, 0.0)
    denom = exp_sim.sum(dim=1).clamp_min(1e-12)
    numer = exp_sim.masked_fill(~pos_mask, 0.0).sum(dim=1)
    valid = numer > 0
    if int(valid.sum().item()) == 0:
        return h1.new_tensor(0.0)
    return -torch.log(numer[valid].clamp_min(1e-12) / denom[valid]).mean()


@torch.no_grad()
def augmentation_margin_drop(
    model: Model,
    probe: ProbeHead,
    data,
    cfg: dict[str, Any],
    base_margin: torch.Tensor,
    views: int,
) -> torch.Tensor:
    drops = []
    for _ in range(max(1, int(views))):
        edge = drop_edges(data.edge_index, float(cfg["drop_edge_rate_1"]))
        x_aug = drop_feature(data.x, float(cfg["drop_feature_rate_1"]))
        h_aug = model(x_aug, edge)
        drops.append(base_margin - probe_margins(probe, h_aug))
    return torch.stack(drops, dim=0).mean(dim=0)


def zscore_train(values: torch.Tensor, train_idx: torch.Tensor) -> torch.Tensor:
    ref = values[train_idx]
    return (values - ref.mean()) / ref.std(unbiased=False).clamp_min(1e-6)


def fragility_scores(
    warmup_model: Model,
    probe: ProbeHead,
    data,
    split,
    cfg: dict[str, Any],
    warmup_h: torch.Tensor,
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        base_margin = probe_margins(probe, warmup_h)
        drop = augmentation_margin_drop(
            warmup_model,
            probe,
            data,
            cfg,
            base_margin,
            int(cfg["augmentation_margin_drop_views"]),
        )
        q = -zscore_train(base_margin, split.train.to(warmup_h.device)) + zscore_train(
            drop, split.train.to(warmup_h.device)
        )
    return {"q": q, "base_margin": base_margin, "aug_margin_drop": drop}


def cv_threshold_q(
    warmup_h: torch.Tensor,
    y: torch.Tensor,
    split,
    cfg: dict[str, Any],
    seed: int,
    num_classes: int,
) -> tuple[torch.Tensor, ProbeHead, dict[str, Any]]:
    train_idx = split.train.to(warmup_h.device)
    generator = torch.Generator(device=warmup_h.device)
    generator.manual_seed(seed + 72_113)
    perm = train_idx[torch.randperm(train_idx.numel(), generator=generator, device=warmup_h.device)]
    folds = int(cfg["cv_folds"])
    oof_margin = torch.zeros(warmup_h.size(0), device=warmup_h.device)
    all_logits = torch.zeros(warmup_h.size(0), num_classes, device=warmup_h.device)
    states: list[dict[str, torch.Tensor]] = []
    used = torch.zeros(warmup_h.size(0), dtype=torch.bool, device=warmup_h.device)
    for fold in range(folds):
        val_fold = perm[fold::folds]
        train_fold = torch.tensor(
            [int(i) for i in train_idx.detach().cpu().tolist() if int(i) not in set(val_fold.detach().cpu().tolist())],
            dtype=torch.long,
            device=warmup_h.device,
        )
        probe, _, _ = train_probe(warmup_h, y, train_fold, cfg, seed + fold + 19, num_classes)
        with torch.no_grad():
            logits = probe(warmup_h)
            all_logits += logits / folds
            oof_margin[val_fold] = margin_from_logits(logits[val_fold])
        states.append({k: v.detach().clone() for k, v in probe.state_dict().items()})
        used[val_fold] = True
    avg_state = {}
    for key in states[0]:
        avg_state[key] = torch.stack([state[key] for state in states], dim=0).mean(dim=0)
    cv_probe = ProbeHead(warmup_h.size(1), num_classes, int(cfg.get("probe_hidden_dim", 0))).to(warmup_h.device)
    cv_probe.load_state_dict(avg_state)
    cv_probe.eval()
    ensemble_margin = margin_from_logits(all_logits)
    q_all = -zscore_train(ensemble_margin, train_idx)
    q_train_oof = -(
        (oof_margin[train_idx] - oof_margin[train_idx].mean())
        / oof_margin[train_idx].std(unbiased=False).clamp_min(1e-6)
    )
    threshold = torch.quantile(q_all[train_idx], float(cfg["fragile_quantile"]))
    return q_all, cv_probe, {
        "probe_type": "train_only_internal_cv_probe",
        "cv_folds": folds,
        "threshold_source": "train_fold_ensemble_margins_only",
        "cv_fragility_threshold": float(threshold.detach().item()),
        "train_oof_q_mean": float(q_train_oof.mean().detach().item()),
        "train_oof_q_std": float(q_train_oof.std(unbiased=False).detach().item()),
    }


def build_route(
    q: torch.Tensor,
    train_idx: torch.Tensor,
    quantile: float,
    mode: str,
    seed: int,
    matched_fraction: float | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if mode == "random":
        fraction = float(matched_fraction if matched_fraction is not None else 0.4)
        generator = torch.Generator(device=q.device)
        generator.manual_seed(seed + 81_177)
        score = torch.rand(q.numel(), generator=generator, device=q.device)
        threshold = torch.quantile(score, 1.0 - fraction)
        fragile = score >= threshold
        return fragile, {
            "route_mode": "random_matched_fraction",
            "fragile_threshold": float(threshold.detach().item()),
            "matched_fraction": fraction,
        }
    threshold = torch.quantile(q[train_idx], float(quantile))
    fragile = q >= threshold
    return fragile, {
        "route_mode": "probe_fragility_quantile",
        "fragile_threshold": float(threshold.detach().item()),
        "fragile_quantile_on_train": float(quantile),
    }


def degree_buckets(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    src = edge_index[0].detach().cpu().numpy()
    degree = np.bincount(src, minlength=num_nodes).astype(np.float64)
    q = np.quantile(degree, [0.25, 0.5, 0.75])
    return torch.tensor(np.digitize(degree, q, right=True), dtype=torch.long)


def route_distribution(route: torch.Tensor, buckets: torch.Tensor) -> dict[str, Any]:
    route_cpu = route.detach().cpu().bool()
    out: dict[str, Any] = {
        "fragile_anchor_fraction": float(route_cpu.float().mean().item()),
        "fragile_anchor_count": int(route_cpu.sum().item()),
    }
    for bucket in range(4):
        mask = buckets == bucket
        out[f"degree_bucket_{bucket}_fragile_fraction"] = (
            float(route_cpu[mask].float().mean().item()) if int(mask.sum().item()) else 0.0
        )
        out[f"degree_bucket_{bucket}_count"] = int(mask.sum().item())
    return out


def evaluate_variant(
    model: Model,
    probe: ProbeHead,
    warmup_margins: torch.Tensor,
    data,
    split,
    eval_cfg: dict[str, Any],
    seed: int,
    route: torch.Tensor,
    buckets: torch.Tensor,
) -> dict[str, Any]:
    h = encode(model, data.x, data.edge_index, normalize=bool(eval_cfg["normalize_embeddings"]))
    eval_result = logreg_val_eval(h, data.y, split, eval_cfg, seed)
    final_margin = probe_margins(probe, h)
    train_idx = split.train.to(h.device)
    val_idx = split.val.to(h.device)
    train_delta = float((final_margin[train_idx].mean() - warmup_margins[train_idx].mean()).item())
    val_delta = float((final_margin[val_idx].mean() - warmup_margins[val_idx].mean()).item())
    return {
        **eval_result,
        "accuracy": eval_result["test_at_best"],
        "train_probe_margin_delta": train_delta,
        "val_probe_margin_delta": val_delta,
        "train_val_probe_margin_delta_gap": float(train_delta - val_delta),
        "route_distribution": route_distribution(route, buckets),
    }


def train_warmup(
    dataset,
    data,
    pretrain_cfg: dict[str, Any],
    warmup_epochs: int,
    device: torch.device,
    seed: int,
    log_path: Path,
    log_every: int,
) -> tuple[Model, float]:
    set_seed(seed)
    model = make_model(dataset, pretrain_cfg, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(pretrain_cfg["learning_rate"]),
        weight_decay=float(pretrain_cfg["weight_decay"]),
    )
    last_loss = math.nan
    start = time.time()
    with log_path.open("w") as f:
        for epoch in range(1, int(warmup_epochs) + 1):
            last_loss = train_grace_epoch(
                model,
                optimizer,
                data.x,
                data.edge_index,
                pretrain_cfg,
                int(pretrain_cfg.get("loss_batch_size", 0)),
            )
            if epoch == 1 or epoch == int(warmup_epochs) or epoch % log_every == 0:
                record = {"epoch": epoch, "loss": float(last_loss), "time": time.time()}
                f.write(json.dumps(record, sort_keys=True) + "\n")
                f.flush()
                print(
                    f"[warmup] epoch={epoch:04d}/{int(warmup_epochs)} "
                    f"loss={last_loss:.4f} elapsed={time.time() - start:.1f}s",
                    flush=True,
                )
    return model, float(last_loss)


def train_variant(
    variant: dict[str, Any],
    base_state: dict[str, torch.Tensor],
    dataset,
    data,
    pretrain_cfg: dict[str, Any],
    pacer_cfg: dict[str, Any],
    device: torch.device,
    seed: int,
    adaptation_epochs: int,
    real_probe: ProbeHead,
    variant_probe: ProbeHead,
    target_logits: torch.Tensor,
    warmup_margins: torch.Tensor,
    q: torch.Tensor,
    route: torch.Tensor,
    split,
    log_path: Path,
    log_every: int,
) -> tuple[Model, dict[str, Any]]:
    model = make_model(dataset, pretrain_cfg, device)
    model.load_state_dict(base_state)
    if variant["type"] == "probe_only":
        return model, {"last_train_loss": None, "adaptation_epochs": 0}

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(pacer_cfg["learning_rate"]),
        weight_decay=float(pacer_cfg["weight_decay"]),
    )
    train_idx = split.train.to(device)
    stable = ~route
    fragile = route
    q_norm = (q - q[train_idx].min()) / (q[train_idx].max() - q[train_idx].min()).clamp_min(1e-6)
    weights = float(pacer_cfg["scalar_weight_min"]) + (
        float(pacer_cfg["scalar_weight_max"]) - float(pacer_cfg["scalar_weight_min"])
    ) * q_norm
    margin_floor = float(torch.quantile(warmup_margins[train_idx], float(pacer_cfg["margin_floor_quantile"])).item())
    last: dict[str, float] = {}
    start = time.time()
    with log_path.open("w") as f:
        for epoch in range(1, int(adaptation_epochs) + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            h1, h2 = two_view_embeddings(model, data, pretrain_cfg)
            node_loss = nodewise_grace_loss(model, h1, h2)
            if variant["type"] == "grace":
                loss = node_loss.mean()
            elif variant["type"] == "mask_only":
                loss = representation_consistency_loss(h1, h2)
            elif variant["type"] in {"pacer_full", "shuffled_probe", "random_route", "train_only_cv_probe"}:
                grace_loss = node_loss[stable].mean() if bool(stable.any()) else node_loss.mean() * 0.0
                logit_1, margin_1, _ = logit_consistency_loss(variant_probe, target_logits, h1, fragile, margin_floor)
                logit_2, margin_2, _ = logit_consistency_loss(variant_probe, target_logits, h2, fragile, margin_floor)
                logit_loss = 0.5 * (logit_1 + logit_2)
                margin_loss = 0.5 * (margin_1 + margin_2)
                loss = grace_loss + float(pacer_cfg["lambda_logit"]) * logit_loss + float(pacer_cfg["lambda_margin"]) * margin_loss
            elif variant["type"] == "scalar_reweight":
                loss = (node_loss * weights.detach()).mean()
            elif variant["type"] == "supcon_control":
                grace_loss = node_loss.mean()
                sup_loss = supervised_contrastive_loss(h1, h2, data.y, train_idx, float(pretrain_cfg["tau"]))
                loss = grace_loss + float(pacer_cfg["lambda_supcon"]) * sup_loss
            else:
                raise ValueError(f"Unsupported variant type: {variant['type']}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(pacer_cfg["max_grad_norm"]))
            optimizer.step()
            last = {"loss": float(loss.detach().item())}
            if epoch == 1 or epoch == int(adaptation_epochs) or epoch % log_every == 0:
                record = {"epoch": epoch, **last, "time": time.time()}
                f.write(json.dumps(record, sort_keys=True) + "\n")
                f.flush()
                print(
                    f"[{variant['id']} {variant['name']}] epoch={epoch:04d}/{int(adaptation_epochs)} "
                    f"loss={last['loss']:.4f} elapsed={time.time() - start:.1f}s",
                    flush=True,
                )
    return model, {"last_train_loss": last.get("loss"), "adaptation_epochs": int(adaptation_epochs)}


def control_gaps(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    acc = {k: float(v["test_at_best"]) for k, v in results.items()}
    p3 = acc.get("P3", math.nan)
    return {
        "p3_minus_p0_grace": p3 - acc.get("P0", math.nan),
        "p3_minus_p2_mask_only": p3 - acc.get("P2", math.nan),
        "shuffled_control_gap_p3_minus_p4": p3 - acc.get("P4", math.nan),
        "random_route_control_gap_p3_minus_p5": p3 - acc.get("P5", math.nan),
        "scalar_control_gap_p3_minus_p6": p3 - acc.get("P6", math.nan),
        "train_cv_control_gap_p3_minus_p7": p3 - acc.get("P7", math.nan),
        "supcon_control_gap_p3_minus_p8": p3 - acc.get("P8", math.nan),
    }


def decide(gaps: dict[str, Any], results: dict[str, dict[str, Any]], dirty_at_start: bool) -> dict[str, Any]:
    triggered = []
    if dirty_at_start:
        triggered.append("BLOCK_PROVENANCE_BEFORE_PILOT")
    if gaps["p3_minus_p0_grace"] <= 0 or gaps["p3_minus_p2_mask_only"] <= 0:
        triggered.append("KILL_PACER")
    if gaps["shuffled_control_gap_p3_minus_p4"] <= 0:
        triggered.append("KILL_LABEL_CALIBRATION_STORY")
    if gaps["random_route_control_gap_p3_minus_p5"] <= 0:
        triggered.append("KILL_ROUTING_STORY")
    if gaps["scalar_control_gap_p3_minus_p6"] <= 0:
        triggered.append("KILL_OBJECTIVE_ROUTING_STORY")
    if gaps["supcon_control_gap_p3_minus_p8"] <= 0:
        triggered.append("KILL_PACER_AS_UNNECESSARY")
    p3 = results.get("P3", {})
    if p3.get("train_probe_margin_delta", 0.0) > 0 and p3.get("val_probe_margin_delta", 0.0) <= 0:
        triggered.append("KILL_TRAIN_PROBE_OVERFIT")
    if not triggered:
        decision = "GO_TO_PA_CER_M1_WITH_CAUTION"
    elif triggered == ["BLOCK_PROVENANCE_BEFORE_PILOT"]:
        decision = "BLOCKED_BY_PROVENANCE"
    else:
        decision = "KILL_OR_PIVOT_REQUIRED"
    return {"decision": decision, "triggered_rules": triggered}


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
    total_epochs = int(pretrain_cfg["num_epochs"])
    pacer_cfg = dict(config["pacer"])
    if args.probe_epochs is not None:
        pacer_cfg["probe_epochs"] = int(args.probe_epochs)
    warmup_epochs = int(args.warmup_epochs if args.warmup_epochs is not None else pacer_cfg["warmup_epochs"])
    warmup_epochs = max(1, min(warmup_epochs, total_epochs))
    pacer_cfg["effective_warmup_epochs"] = warmup_epochs
    adaptation_epochs = max(0, total_epochs - warmup_epochs)
    pacer_cfg["effective_adaptation_epochs"] = adaptation_epochs
    eval_cfg = dict(config["logreg_eval"])

    set_seed(seed)
    dataset = load_dataset(dataset_name, PROJECT_ROOT / args.data_root)
    data = dataset[0].to(device)
    split = create_or_load_split(dataset_name, data.y.detach().cpu(), seed, PROJECT_ROOT / args.split_root)
    split.train = split.train.to(device)
    split.val = split.val.to(device)
    split.test = split.test.to(device)
    num_classes = int(dataset.num_classes)

    run_id = f"pacer_smoke_{dataset_name}_seed{seed}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    result_dir = PROJECT_ROOT / args.result_root / dataset_name / "PACER_GCL_SMOKE" / run_id
    summary_dir = PROJECT_ROOT / args.summary_root
    log_dir = PROJECT_ROOT / args.log_root / run_id
    result_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    warmup_model, warmup_last_loss = train_warmup(
        dataset,
        data,
        pretrain_cfg,
        warmup_epochs,
        device,
        seed,
        log_dir / "warmup.jsonl",
        args.log_every,
    )
    warmup_state = copy.deepcopy(warmup_model.state_dict())
    warmup_h = encode(warmup_model, data.x, data.edge_index, normalize=bool(eval_cfg["normalize_embeddings"])).detach()

    real_probe, real_probe_labels, real_probe_info = train_probe(
        warmup_h, data.y, split.train, pacer_cfg, seed + 100, num_classes, label_mode="true"
    )
    shuffled_probe, shuffled_labels, shuffled_probe_info = train_probe(
        warmup_h, data.y, split.train, pacer_cfg, seed + 200, num_classes, label_mode="shuffled"
    )

    real_fragility = fragility_scores(warmup_model, real_probe, data, split, {**pretrain_cfg, **pacer_cfg}, warmup_h)
    shuffled_fragility = fragility_scores(
        warmup_model, shuffled_probe, data, split, {**pretrain_cfg, **pacer_cfg}, warmup_h
    )
    train_idx = split.train.to(device)
    route_p3, route_p3_info = build_route(
        real_fragility["q"], train_idx, float(pacer_cfg["fragile_quantile"]), "probe", seed
    )
    route_p4, route_p4_info = build_route(
        shuffled_fragility["q"], train_idx, float(pacer_cfg["fragile_quantile"]), "probe", seed + 1
    )
    route_p5, route_p5_info = build_route(
        real_fragility["q"],
        train_idx,
        float(pacer_cfg["fragile_quantile"]),
        "random",
        seed + 2,
        matched_fraction=float(route_p3.float().mean().item()),
    )
    q_cv, cv_probe, cv_info = cv_threshold_q(warmup_h, data.y, split, pacer_cfg, seed + 300, num_classes)
    route_p7, route_p7_info = build_route(q_cv, train_idx, float(pacer_cfg["fragile_quantile"]), "probe", seed + 3)
    route_empty = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    route_all = torch.ones(data.num_nodes, dtype=torch.bool, device=device)
    buckets = degree_buckets(data.edge_index, data.num_nodes)
    with torch.no_grad():
        real_target_logits = real_probe(warmup_h).detach()
        shuffled_target_logits = shuffled_probe(warmup_h).detach()
        real_warmup_margins = probe_margins(real_probe, warmup_h)
        shuffled_warmup_margins = probe_margins(shuffled_probe, warmup_h)

    results: dict[str, dict[str, Any]] = {}
    variant_meta: dict[str, dict[str, Any]] = {}
    for variant in config["variants"]:
        variant_id = variant["id"]
        if variant_id == "P4":
            probe = shuffled_probe
            target_logits = shuffled_target_logits
            margins = shuffled_warmup_margins
            q = shuffled_fragility["q"]
            route = route_p4
            route_info = route_p4_info
            probe_info = shuffled_probe_info
            label_mode = "shuffled_train_labels"
        elif variant_id == "P5":
            probe = real_probe
            target_logits = real_target_logits
            margins = real_warmup_margins
            q = real_fragility["q"]
            route = route_p5
            route_info = route_p5_info
            probe_info = real_probe_info
            label_mode = "true_train_labels_random_route"
        elif variant_id == "P7":
            probe = cv_probe
            with torch.no_grad():
                cv_target_logits = cv_probe(warmup_h).detach()
                cv_warmup_margins = probe_margins(cv_probe, warmup_h)
            target_logits = cv_target_logits
            margins = cv_warmup_margins
            q = q_cv
            route = route_p7
            route_info = {**route_p7_info, **cv_info}
            probe_info = cv_info
            label_mode = "true_train_labels_internal_cv_route"
        else:
            probe = real_probe
            target_logits = real_target_logits
            margins = real_warmup_margins
            q = real_fragility["q"]
            route = route_empty if variant["type"] in {"grace", "probe_only", "mask_only", "supcon_control"} else route_p3
            route_info = route_p3_info if variant["type"] not in {"grace", "probe_only", "mask_only", "supcon_control"} else {"route_mode": "none"}
            probe_info = real_probe_info
            label_mode = "true_train_labels"

        if variant["type"] == "grace":
            train_route = route_all
        else:
            train_route = route
        model, train_info = train_variant(
            variant,
            warmup_state,
            dataset,
            data,
            pretrain_cfg,
            pacer_cfg,
            device,
            seed + int(variant_id[1:]) * 1000,
            adaptation_epochs,
            real_probe,
            probe,
            target_logits,
            margins,
            q,
            train_route if variant["type"] == "grace" else route,
            split,
            log_dir / f"{variant_id}_{variant['type']}.jsonl",
            args.log_every,
        )
        eval_result = evaluate_variant(
            model,
            real_probe,
            real_warmup_margins,
            data,
            split,
            eval_cfg,
            seed,
            route,
            buckets,
        )
        result = {
            "variant_id": variant_id,
            "variant_name": variant["name"],
            "variant_type": variant["type"],
            "method_status": config["meta"]["method_status"],
            "status": status,
            "uses_train_labels_for_pretrain": bool(variant.get("uses_train_labels_for_pretrain", False)),
            "uses_validation_labels_for_probe_or_threshold": False,
            "uses_test_labels_for_probe_or_threshold": False,
            "label_mode": label_mode,
            "probe_info": probe_info,
            "route_info": route_info,
            "training": train_info,
            "warmup_last_loss": warmup_last_loss,
            **eval_result,
        }
        results[variant_id] = result
        variant_meta[variant_id] = {
            "q_train_mean": float(q[train_idx].mean().detach().item()),
            "q_train_std": float(q[train_idx].std(unbiased=False).detach().item()),
            "base_train_margin_mean": float(margins[train_idx].mean().detach().item()),
            "base_val_margin_mean": float(margins[split.val].mean().detach().item()),
        }
        print(
            f"[{variant_id}] acc={result['test_at_best']:.2f} "
            f"val={result['valid_at_best']:.2f} fragile={result['route_distribution']['fragile_anchor_fraction']:.3f}",
            flush=True,
        )

    gaps = control_gaps(results)
    decision = decide(gaps, results, dirty_at_start)
    run_payload = {
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
        "dataset_num_classes": num_classes,
        "total_epochs": total_epochs,
        "warmup_epochs": warmup_epochs,
        "adaptation_epochs": adaptation_epochs,
        "pretrain_config": pretrain_cfg,
        "pacer_config": pacer_cfg,
        "evaluator_config": eval_cfg,
        "integrity": {
            "test_labels_used_for_thresholds": False,
            "test_labels_used_for_probe_training": False,
            "validation_labels_used_for_probe_training": False,
            "validation_labels_used_for_logreg_C_selection": True,
            "train_labels_used_for_probe_calibration": True,
            "pacer_is_unsupervised": False,
        },
        "probe_global": {
            "real_probe_info": real_probe_info,
            "shuffled_probe_info": shuffled_probe_info,
        },
        "variant_probe_diagnostics": variant_meta,
        "variants": results,
        "control_gaps": gaps,
        "decision": decision,
    }
    result_path = result_dir / f"{run_id}.json"
    run_payload["result_path"] = str(result_path.resolve())
    result_path.write_text(json.dumps(run_payload, indent=2, sort_keys=True) + "\n")

    summary_path = summary_dir / f"{run_id}_summary.json"
    summary_path.write_text(json.dumps(run_payload, indent=2, sort_keys=True) + "\n")
    md_path = summary_dir / f"{run_id}_summary.md"
    lines = [
        f"# PACER-GCL {config['meta']['smoke_id']} Smoke",
        "",
        f"- Run: `{run_id}`",
        f"- Dataset/seed: `{dataset_name}` / `{seed}`",
        f"- Method status: `{config['meta']['method_status']}`",
        f"- Split: `{split.path.resolve()}`",
        f"- Command: `{command}`",
        f"- Commit: `{commit_hash}`",
        f"- Dirty at start: `{dirty_at_start}`",
        f"- Decision: `{decision['decision']}`",
        "",
        "| ID | System | Test@best-val | Val@best | C | Train margin delta | Val margin delta | Fragile fraction |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in config["variants"]:
        r = results[variant["id"]]
        lines.append(
            f"| {variant['id']} | {variant['name']} | {r['test_at_best']:.2f} | "
            f"{r['valid_at_best']:.2f} | {r['best_c']} | {r['train_probe_margin_delta']:.4f} | "
            f"{r['val_probe_margin_delta']:.4f} | {r['route_distribution']['fragile_anchor_fraction']:.4f} |"
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
    md_path.write_text("\n".join(lines) + "\n")
    print(f"[done] result={result_path}", flush=True)
    print(f"[done] summary={md_path}", flush=True)


if __name__ == "__main__":
    main()

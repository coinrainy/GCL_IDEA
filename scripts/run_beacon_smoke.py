#!/usr/bin/env python
"""Run BEACON-GCL BE-M0 smoke with boundary/geometry diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_cast_certificate_smoke import (  # noqa: E402
    as_numpy,
    certificate_energy,
    ego_summary,
    local_graph_diffusion_score,
    train_certificate,
)
from run_grace_1_1_8 import (  # noqa: E402
    create_or_load_split,
    git_dirty,
    git_rev,
    load_dataset,
    logreg_val_eval,
)
from run_iris_smoke import (  # noqa: E402
    cosine_matrix,
    false_negative_mass,
    label_agreement,
    node_degree,
    relation_overlap,
    relation_smooth_embeddings,
    standardize_score,
    structural_signature,
    topk_candidate_mask,
    topk_relations,
    train_grace,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/beacon_smoke.yaml")
    parser.add_argument("--base-config", default="configs/grace_1_1_8.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--cert-epochs", type=int, default=None)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--run-tag", default="be_m0_001")
    parser.add_argument("--status", default=None)
    parser.add_argument("--data-root", default="data/pyg")
    parser.add_argument("--split-root", default="splits")
    parser.add_argument("--result-root", default="results/raw")
    parser.add_argument("--summary-root", default="results/summary")
    parser.add_argument("--log-root", default="logs/beacon_smoke")
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def finite_z(score: np.ndarray) -> np.ndarray:
    out = standardize_score(score)
    return np.where(np.isfinite(out), out, 0.0).astype(np.float32)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(denom, 1e-12)).astype(np.float32)


def train_probe(
    z: np.ndarray,
    y: np.ndarray,
    split,
    eval_cfg: dict[str, Any],
    seed: int,
) -> tuple[OneVsRestClassifier, dict[str, Any]]:
    train_idx = split.train.cpu().numpy()
    val_idx = split.val.cpu().numpy()
    best: dict[str, Any] = {
        "valid_at_best": -1.0,
        "best_c": None,
        "clf": None,
    }
    for c in [float(v) for v in eval_cfg["c_values"]]:
        base = LogisticRegression(
            C=c,
            penalty=eval_cfg.get("penalty", "l2"),
            solver=eval_cfg.get("solver", "liblinear"),
            tol=float(eval_cfg.get("tol", 1e-4)),
            max_iter=int(eval_cfg.get("max_iter", 5000)),
            fit_intercept=bool(eval_cfg.get("fit_intercept", True)),
            class_weight=eval_cfg.get("class_weight"),
            random_state=seed,
        )
        clf = OneVsRestClassifier(base)
        clf.fit(z[train_idx], y[train_idx])
        val_acc = accuracy_score(y[val_idx], clf.predict(z[val_idx]))
        if val_acc > best["valid_at_best"]:
            best.update({"valid_at_best": float(val_acc), "best_c": c, "clf": clf})
    return best["clf"], {
        "probe_type": "train_fit_val_select_logreg",
        "valid_at_best": best["valid_at_best"] * 100.0,
        "best_c": best["best_c"],
        "uses_test_labels": False,
    }


def probe_margin(clf: OneVsRestClassifier, z: np.ndarray) -> np.ndarray:
    scores = clf.decision_function(z)
    if scores.ndim == 1:
        scores = np.stack([-scores, scores], axis=1)
    part = np.partition(scores, -2, axis=1)
    return (part[:, -1] - part[:, -2]).astype(np.float32)


def boundary_risk_matrix(
    z: np.ndarray,
    candidate_mask: np.ndarray,
    clf: OneVsRestClassifier,
    eta: float,
) -> tuple[np.ndarray, np.ndarray]:
    z_norm = normalize_rows(z)
    base_margin = probe_margin(clf, z_norm)
    risk = np.full(candidate_mask.shape, np.inf, dtype=np.float32)
    delta = np.full(candidate_mask.shape, np.nan, dtype=np.float32)
    for i in range(candidate_mask.shape[0]):
        cand = np.where(candidate_mask[i])[0]
        if cand.size == 0:
            continue
        edited = z_norm[i][None, :] + eta * (z_norm[cand] - z_norm[i][None, :])
        edited = normalize_rows(edited)
        after = probe_margin(clf, edited)
        d = after - base_margin[i]
        delta[i, cand] = d.astype(np.float32)
        risk[i, cand] = np.maximum(0.0, -d).astype(np.float32)
    np.fill_diagonal(risk, np.inf)
    np.fill_diagonal(delta, np.nan)
    return risk, delta


def degree_bucket_concentration(rel: np.ndarray, degree: np.ndarray) -> dict[str, float]:
    selected_by_anchor = rel.sum(axis=1).astype(np.float32)
    if selected_by_anchor.sum() <= 0:
        return {"degree_bucket_max_pair_share": 0.0, "degree_bucket_entropy": 0.0}
    quantiles = np.quantile(degree, [0.25, 0.5, 0.75])
    buckets = np.digitize(degree, quantiles, right=True)
    shares = []
    for bucket in range(4):
        shares.append(float(selected_by_anchor[buckets == bucket].sum() / selected_by_anchor.sum()))
    entropy = -sum(s * np.log(max(s, 1e-12)) for s in shares)
    return {
        "degree_bucket_max_pair_share": float(max(shares)),
        "degree_bucket_entropy": float(entropy),
    }


def effective_rank(z: np.ndarray, center: bool) -> float:
    x = z.astype(np.float64)
    if center:
        x = x - x.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(x, full_matrices=False, compute_uv=False)
    prob = singular / max(float(singular.sum()), 1e-12)
    entropy = -float(np.sum(prob * np.log(np.maximum(prob, 1e-12))))
    return float(np.exp(entropy))


def uniformity_score(z: np.ndarray, max_pairs: int, seed: int) -> float:
    x = normalize_rows(z)
    n = x.shape[0]
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
    dist2 = np.sum((x[row] - x[col]) ** 2, axis=1)
    return float(np.log(np.mean(np.exp(-2.0 * dist2)) + 1e-12))


def gate_relations(
    utility: np.ndarray,
    candidate_mask: np.ndarray,
    boundary_risk: np.ndarray,
    cfg: dict[str, Any],
    use_boundary_threshold: bool,
) -> tuple[np.ndarray, dict[str, float]]:
    rel = np.zeros_like(candidate_mask, dtype=bool)
    cand_values = utility[candidate_mask & np.isfinite(utility)]
    if cand_values.size == 0:
        return rel, {"utility_threshold": float("nan"), "boundary_threshold": float("nan")}
    utility_threshold = float(np.quantile(cand_values, float(cfg["utility_quantile"])))
    boundary_values = boundary_risk[candidate_mask & np.isfinite(boundary_risk)]
    boundary_threshold = (
        float(np.quantile(boundary_values, float(cfg["boundary_risk_quantile"])))
        if boundary_values.size
        else float("inf")
    )
    budget = int(cfg["positive_budget"])
    for i in range(candidate_mask.shape[0]):
        cand = np.where(candidate_mask[i] & np.isfinite(utility[i]) & (utility[i] >= utility_threshold))[0]
        if use_boundary_threshold:
            cand = cand[boundary_risk[i, cand] <= boundary_threshold]
        if cand.size == 0:
            continue
        order = cand[np.argsort(utility[i, cand])[::-1]]
        rel[i, order[: min(budget, order.size)]] = True
    return rel, {
        "utility_threshold": utility_threshold,
        "boundary_threshold": boundary_threshold,
    }


def relation_stats(
    rel: np.ndarray,
    references: dict[str, np.ndarray],
    labels: np.ndarray,
    z_base: np.ndarray,
    z_eval: np.ndarray,
    degree: np.ndarray,
    utility: np.ndarray,
    candidate_mask: np.ndarray,
    margin_delta: np.ndarray,
    rank_base: float,
    uniformity_base: float,
    cfg: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    selected = rel & np.isfinite(utility)
    rejected = candidate_mask & ~rel & np.isfinite(utility)
    accepted_utility = utility[selected]
    rejected_utility = utility[rejected]
    selected_delta = margin_delta[rel & np.isfinite(margin_delta)]
    rank_eval = effective_rank(z_eval, bool(cfg.get("rank_center", True)))
    uniformity_eval = uniformity_score(z_eval, int(cfg["uniformity_sample_pairs"]), seed)
    diagnostics: dict[str, Any] = {
        **label_agreement(rel, labels),
        "false_negative_mass": false_negative_mass(z_base, labels, rel, tau=0.4),
        "accepted_pairs": int(rel.sum()),
        "accepted_pairs_per_anchor": float(rel.sum(axis=1).mean()),
        "overlap_with_knn_B1": relation_overlap(rel, references["B1"]),
        "overlap_with_ppr": relation_overlap(rel, references["PPR"]),
        "overlap_with_cast_B2": relation_overlap(rel, references["B2"]),
        "overlap_with_c4_B3": relation_overlap(rel, references["B3"]),
        "probe_margin_delta_mean": float(np.nanmean(selected_delta)) if selected_delta.size else 0.0,
        "probe_margin_delta_min": float(np.nanmin(selected_delta)) if selected_delta.size else 0.0,
        "effective_rank": rank_eval,
        "effective_rank_delta": rank_eval - rank_base,
        "uniformity": uniformity_eval,
        "uniformity_delta": uniformity_eval - uniformity_base,
        "utility_accepted_mean": float(np.mean(accepted_utility)) if accepted_utility.size else None,
        "utility_accepted_std": float(np.std(accepted_utility)) if accepted_utility.size else None,
        "utility_rejected_mean": float(np.mean(rejected_utility)) if rejected_utility.size else None,
        "utility_rejected_std": float(np.std(rejected_utility)) if rejected_utility.size else None,
        "label_diagnostics_are_offline_only": True,
        "gate_uses_test_labels": False,
    }
    diagnostics.update(degree_bucket_concentration(rel, degree))
    return diagnostics


def summarize_markdown(results: list[dict[str, Any]], path: Path, dataset_name: str, seed: int) -> None:
    lines = [
        "# BEACON-GCL BE-M0 Smoke Summary",
        "",
        f"本文件为 {dataset_name} seed={seed} smoke 汇总，不支持 formal、SOTA 或 robust claim。",
        "",
        "| ID | Variant | Test@best-val | Agreement | Accepted/anchor | kNN ov | PPR ov | CAST ov | C4 ov | Margin delta | Rank delta | Uniformity delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        diag = result["diagnostics"]
        eval_result = result["evaluation"]
        lines.append(
            f"| {result['variant_id']} | {result['variant_name']} | "
            f"{eval_result['test_at_best']:.2f} | "
            f"{diag['pair_label_agreement']:.4f} | "
            f"{diag['accepted_pairs_per_anchor']:.2f} | "
            f"{diag['overlap_with_knn_B1']:.4f} | "
            f"{diag['overlap_with_ppr']:.4f} | "
            f"{diag['overlap_with_cast_B2']:.4f} | "
            f"{diag['overlap_with_c4_B3']:.4f} | "
            f"{diag['probe_margin_delta_mean']:.4f} | "
            f"{diag['effective_rank_delta']:.4f} | "
            f"{diag['uniformity_delta']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    config = load_yaml(PROJECT_ROOT / args.config)
    base_config = load_yaml(PROJECT_ROOT / args.base_config)
    dataset_name = args.dataset or config["dataset"]["name"]
    seed = int(args.seed if args.seed is not None else config["dataset"]["seed"])
    if dataset_name != "Cora" or seed != 0:
        raise ValueError("BE-M0 gate currently allows only Cora seed 0.")

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset(dataset_name, PROJECT_ROOT / args.data_root)
    data = dataset[0].to(device)
    split = create_or_load_split(dataset_name, data.y.detach().cpu(), seed, PROJECT_ROOT / args.split_root)
    pretrain_cfg = dict(base_config["pretrain"][dataset_name])
    pretrain_cfg.update(config.get("pretrain", {}).get(dataset_name, {}))
    if args.epochs is not None:
        pretrain_cfg["num_epochs"] = int(args.epochs)
    cert_cfg = dict(config["certificate"])
    if args.cert_epochs is not None:
        cert_cfg["train_epochs"] = int(args.cert_epochs)
    beacon_cfg = dict(config["beacon"])
    diag_cfg = dict(config.get("diagnostics", {}))
    eval_cfg = dict(base_config["logreg_eval"])
    status = args.status or config["meta"].get("status_default", "smoke")

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_name = f"{args.run_tag}_{dataset_name}_seed{seed}_{timestamp}"
    result_dir = PROJECT_ROOT / args.result_root / dataset_name / "BEACON_GCL_SMOKE" / run_name
    summary_dir = PROJECT_ROOT / args.summary_root
    log_dir = PROJECT_ROOT / args.log_root / run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    start_all = time.time()
    _, z, last_grace_loss = train_grace(
        dataset, data, pretrain_cfg, device, seed, log_dir / "grace_warmup.jsonl", args.log_every
    )
    labels = as_numpy(data.y).astype(int)
    features = as_numpy(data.x).astype(np.float32)
    degree = as_numpy(node_degree(data.edge_index, int(data.num_nodes))).astype(np.float32)
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

    graph_sim = local_graph_diffusion_score(
        data.edge_index,
        int(data.num_nodes),
        int(diag_cfg.get("graph_diffusion_steps", 3)),
        float(diag_cfg.get("graph_diffusion_decay", 0.6)),
    )
    emb_sim = cosine_matrix(z)
    feat_sim = cosine_matrix(features)
    struct = structural_signature(features, degree, graph_sim)
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

    budget = int(cert_cfg["positive_budget"])
    rel_knn = topk_relations(emb_sim, budget)
    rel_cast = topk_relations(cast_score, budget)
    rel_ppr = topk_relations(graph_sim, budget)
    rel_c4 = topk_relations(cert_score, budget, pool_mask)
    candidate_mask = rel_knn | rel_cast | rel_ppr | rel_c4
    np.fill_diagonal(candidate_mask, False)

    probe, probe_diag = train_probe(normalize_rows(z), labels, split, eval_cfg, seed)
    boundary_risk, margin_delta = boundary_risk_matrix(
        z,
        candidate_mask,
        probe,
        float(beacon_cfg["edit_eta"]),
    )
    degree_gap = np.abs(degree[:, None] - degree[None, :]).astype(np.float32)
    benefit = finite_z(feat_sim) + finite_z(emb_sim) + finite_z(graph_sim) + finite_z(cert_score)
    benefit = standardize_score(np.where(candidate_mask, benefit / 4.0, -np.inf))
    geometry_raw = (1.0 - np.clip(emb_sim, -1.0, 1.0)) / 2.0
    geometry_raw += float(beacon_cfg.get("degree_gap_weight", 0.25)) * (
        degree_gap / max(float(degree_gap.max()), 1.0)
    )
    geometry_risk = standardize_score(np.where(candidate_mask, geometry_raw, -np.inf))
    boundary_score = standardize_score(np.where(candidate_mask, boundary_risk, -np.inf))

    full_utility = standardize_score(
        np.where(
            candidate_mask,
            benefit
            - float(beacon_cfg["boundary_weight"]) * np.where(np.isfinite(boundary_score), boundary_score, 0.0)
            - float(beacon_cfg["geometry_weight"]) * np.where(np.isfinite(geometry_risk), geometry_risk, 0.0),
            -np.inf,
        )
    )
    no_boundary_utility = standardize_score(
        np.where(
            candidate_mask,
            benefit - float(beacon_cfg["geometry_weight"]) * np.where(np.isfinite(geometry_risk), geometry_risk, 0.0),
            -np.inf,
        )
    )
    no_geometry_utility = standardize_score(
        np.where(
            candidate_mask,
            benefit
            - float(beacon_cfg["boundary_weight"]) * np.where(np.isfinite(boundary_score), boundary_score, 0.0),
            -np.inf,
        )
    )
    rng = np.random.default_rng(seed + 51_003)
    shuffled_utility = np.full_like(full_utility, -np.inf, dtype=np.float32)
    cand_pos = np.where(candidate_mask & np.isfinite(full_utility))
    shuffled_utility[cand_pos] = rng.permutation(full_utility[cand_pos]).astype(np.float32)

    rel_b5, gate_full_diag = gate_relations(full_utility, candidate_mask, boundary_risk, beacon_cfg, True)
    rel_b6, gate_shuffle_diag = gate_relations(shuffled_utility, candidate_mask, boundary_risk, beacon_cfg, True)
    rel_b7, gate_no_boundary_diag = gate_relations(no_boundary_utility, candidate_mask, boundary_risk, beacon_cfg, False)
    rel_b8, gate_no_geometry_diag = gate_relations(no_geometry_utility, candidate_mask, boundary_risk, beacon_cfg, True)

    relations = {
        "B0": np.zeros_like(rel_knn, dtype=bool),
        "B1": rel_knn,
        "B2": rel_cast,
        "B3": rel_c4,
        "B4": candidate_mask,
        "B5": rel_b5,
        "B6": rel_b6,
        "B7": rel_b7,
        "B8": rel_b8,
    }
    gate_diagnostics = {
        "B5": gate_full_diag,
        "B6": gate_shuffle_diag,
        "B7": gate_no_boundary_diag,
        "B8": gate_no_geometry_diag,
    }
    references = {"B1": rel_knn, "B2": rel_cast, "B3": rel_c4, "PPR": rel_ppr}
    utilities = {
        "B0": full_utility,
        "B1": finite_z(emb_sim),
        "B2": finite_z(cast_score),
        "B3": finite_z(cert_score),
        "B4": benefit,
        "B5": full_utility,
        "B6": shuffled_utility,
        "B7": no_boundary_utility,
        "B8": no_geometry_utility,
    }
    rank_base = effective_rank(z, bool(beacon_cfg.get("rank_center", True)))
    uniformity_base = uniformity_score(z, int(beacon_cfg["uniformity_sample_pairs"]), seed)

    results: list[dict[str, Any]] = []
    for variant in config["variants"]:
        rel = relations[variant["id"]]
        z_eval = z if variant["family"] == "grace_base" else relation_smooth_embeddings(
            z,
            rel,
            float(cert_cfg["relation_smoothing_weight"]),
        )
        eval_result = logreg_val_eval(torch.tensor(z_eval), data.y.cpu(), split, eval_cfg, seed)
        diagnostics = relation_stats(
            rel,
            references,
            labels,
            z,
            z_eval,
            degree,
            utilities[variant["id"]],
            candidate_mask,
            margin_delta,
            rank_base,
            uniformity_base,
            beacon_cfg,
            seed,
        )
        diagnostics["probe"] = probe_diag
        diagnostics["gate"] = gate_diagnostics.get(variant["id"], {})
        result = {
            "run_id": f"{run_name}_{variant['id']}",
            "method": "BEACON-GCL-smoke",
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
            "beacon_config": beacon_cfg,
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
            f"agree={diagnostics['pair_label_agreement']:.4f} "
            f"accepted={diagnostics['accepted_pairs_per_anchor']:.2f}",
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
            "gate_uses_test_labels": False,
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

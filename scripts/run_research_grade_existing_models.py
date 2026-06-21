#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from csi_er.classical import FEATURE_KIND, compute_feature
from csi_er.data import load_processed
from csi_er.models import build_model, count_parameters
from csi_er.utils import PROJECT_ROOT, now_ts
from csi_robustbench.statistics import (
    fit_ols,
    grouped_cv_r2,
    inv_probit,
    pair_clean_summary,
    probit,
    seed_level_mpc,
    summarize_seed_mpc,
)
from csi_robustbench.validity import assess_er_validity


DEEP_GROUPS = {
    "MLP": "MLP",
    "SimpleCNN": "CNN",
    "GRU": "RNN",
    "LSTM": "RNN",
    "CNNGRU": "Hybrid",
    "TinyViT": "Transformer",
}
CORRUPTION_SEEDS = [42, 123, 2026, 3407, 777]
CANONICAL = ["gaussian_snr", "subcarrier_mask", "time_dropout", "amplitude_scaling"]
STRUCTURED = ["contiguous_subcarrier_block", "burst_time_dropout", "smooth_gain_drift"]
CORRUPTIONS = CANONICAL + STRUCTURED
SEVERITIES = [1, 2, 3, 4, 5]
SNR_DB = [30, 25, 20, 15, 10]
RATIOS = [0.05, 0.10, 0.20, 0.30, 0.40]
GAIN_DB = [1, 2, 3, 4, 6]
ID_COLS = [
    "dataset",
    "model_name",
    "configuration_id",
    "run_id",
    "base_configuration_id",
    "feature_family",
    "classifier_family",
    "architecture_family",
    "model_group",
    "is_classical",
    "train_seed",
    "train_fraction",
    "model_size",
]


def model_metadata(name: str, *, is_classical: bool, train_seed: int = 42) -> dict:
    if is_classical:
        feature_map = {
            "TimeStats-SVM": ("time_stats", "linear_svm"),
            "TimeStats-RF": ("time_stats", "random_forest"),
            "FFT-SVM": ("fft_stats", "linear_svm"),
            "FFT-RF": ("fft_stats", "random_forest"),
            "PCA-kNN": ("pca_raw", "knn"),
            "PCA-LogReg": ("pca_raw", "logistic_regression"),
            "StatsFFT-SVM": ("time_fft_stats", "linear_svm"),
            "Fusion-Boosting": ("fusion_stats_fft", "hist_gradient_boosting"),
        }
        feature_family, classifier_family = feature_map.get(name, ("unknown_feature", "unknown_classifier"))
        base = f"{feature_family}__{classifier_family}__existing"
        return {
            "configuration_id": base,
            "run_id": f"{base}__seed{train_seed}",
            "base_configuration_id": base,
            "feature_family": feature_family,
            "classifier_family": classifier_family,
            "architecture_family": "classical",
            "model_group": classifier_family,
            "train_seed": train_seed,
            "train_fraction": 1.0,
            "model_size": "existing",
        }
    arch = DEEP_GROUPS[name]
    base = f"{arch}__{name}__existing"
    return {
        "configuration_id": base,
        "run_id": f"{base}__seed{train_seed}",
        "base_configuration_id": base,
        "feature_family": "deep",
        "classifier_family": "deep",
        "architecture_family": arch,
        "model_group": arch,
        "train_seed": train_seed,
        "train_fraction": 1.0,
        "model_size": "existing",
    }


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def apply_research_corruption(x: np.ndarray, corruption: str, severity: int, seed: int) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32)
    out = a.copy()
    r = rng(seed)
    sev_idx = int(severity) - 1
    if corruption == "gaussian_snr":
        snr = SNR_DB[sev_idx]
        signal_power = np.mean(a * a, axis=tuple(range(1, a.ndim)), keepdims=True)
        noise_power = signal_power / (10.0 ** (snr / 10.0))
        out = a + r.normal(0.0, np.sqrt(noise_power + 1e-12), size=a.shape).astype(np.float32)
    elif corruption == "subcarrier_mask":
        ratio = RATIOS[sev_idx]
        f = a.shape[-1]
        k = max(1, int(round(f * ratio)))
        for i in range(a.shape[0]):
            out[i, :, :, r.choice(f, size=min(k, f), replace=False)] = 0.0
    elif corruption == "time_dropout":
        ratio = RATIOS[sev_idx]
        t = a.shape[-2]
        k = max(1, int(round(t * ratio)))
        for i in range(a.shape[0]):
            out[i, :, r.choice(t, size=min(k, t), replace=False), :] = 0.0
    elif corruption == "amplitude_scaling":
        gain = GAIN_DB[sev_idx]
        factors = 10.0 ** (r.uniform(-gain, gain, size=(a.shape[0],) + (1,) * (a.ndim - 1)) / 20.0)
        out = a * factors.astype(np.float32)
    elif corruption == "contiguous_subcarrier_block":
        ratio = RATIOS[sev_idx]
        f = a.shape[-1]
        k = max(1, int(round(f * ratio)))
        for i in range(a.shape[0]):
            start = int(r.integers(0, max(1, f - k + 1)))
            out[i, :, :, start : start + k] = 0.0
    elif corruption == "burst_time_dropout":
        ratio = RATIOS[sev_idx]
        t = a.shape[-2]
        k = max(1, int(round(t * ratio)))
        for i in range(a.shape[0]):
            start = int(r.integers(0, max(1, t - k + 1)))
            out[i, :, start : start + k, :] = 0.0
    elif corruption == "smooth_gain_drift":
        gain = GAIN_DB[sev_idx]
        t = a.shape[-2]
        start = r.uniform(-gain, gain, size=(a.shape[0], 1, 1, 1))
        end = r.uniform(-gain, gain, size=(a.shape[0], 1, 1, 1))
        ramp = np.linspace(0, 1, t, dtype=np.float32).reshape(1, 1, t, 1)
        db = start * (1 - ramp) + end * ramp
        out = a * (10.0 ** (db / 20.0)).astype(np.float32)
    else:
        raise ValueError(corruption)
    return out.astype(np.float32, copy=False)


def predict_deep(model, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    loader = DataLoader(TensorDataset(torch.tensor(x, dtype=torch.float32)), batch_size=batch_size)
    with torch.no_grad():
        for (xb,) in loader:
            preds.extend(model(xb.to(device)).argmax(1).cpu().numpy())
    return np.asarray(preds)


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def summarize_corruptions(corr: pd.DataFrame) -> pd.DataFrame:
    seed_scores = seed_level_mpc(corr, group_cols=ID_COLS, metrics=["accuracy", "macro_f1", "balanced_accuracy"])
    return summarize_seed_mpc(seed_scores, group_cols=ID_COLS, metrics=["accuracy", "macro_f1", "balanced_accuracy"])


def compute_validity(clean: pd.DataFrame, summary: pd.DataFrame, metric: str, n_test: int, output_dir: Path) -> pd.DataFrame:
    df = pair_clean_summary(
        clean,
        summary,
        keys=ID_COLS,
        clean_metric=f"clean_{metric}",
        summary_metric=f"mPC_family_{metric}",
    )
    clean_col = f"clean_{metric}"
    corr_col = f"mPC_family_{metric}"
    df["z_clean"] = probit(df[clean_col].values, n_test=n_test)
    df["z_corr"] = probit(df[corr_col].values, n_test=n_test)
    rows = []
    rng_boot = np.random.default_rng(20260620)
    for _, target in df.iterrows():
        ref = df[df["architecture_family"] != target["architecture_family"]].copy()
        if len(ref) < 2:
            status = assess_er_validity(
                n_unique_configs=len(ref),
                cv_r2=float("nan"),
                slope=float("nan"),
                support_min=float("nan"),
                support_max=float("nan"),
                target_clean=float(target[clean_col]),
                bootstrap_ci_low=float("nan"),
                bootstrap_ci_high=float("nan"),
            )
            pred = np.nan
            resid_pp = np.nan
            fit = None
            cv = np.nan
            ci_low = np.nan
            ci_high = np.nan
        else:
            fit = fit_ols(ref["z_clean"].values, ref["z_corr"].values)
            cv_groups = np.where(ref["is_classical"], ref["base_configuration_id"], ref["architecture_family"])
            cv = grouped_cv_r2(ref["z_clean"].values, ref["z_corr"].values, cv_groups)
            pred_z = fit.slope * float(target["z_clean"]) + fit.intercept
            pred = float(inv_probit(pred_z))
            resid_pp = float(target[corr_col] - pred)
            boot = []
            groups = np.asarray(sorted(ref["base_configuration_id"].unique()))
            for _ in range(1000):
                sampled = rng_boot.choice(groups, size=len(groups), replace=True)
                b = pd.concat([ref[ref["base_configuration_id"] == g] for g in sampled], ignore_index=True)
                if len(b) < 2 or b["z_clean"].nunique() < 2:
                    continue
                bf = fit_ols(b["z_clean"].values, b["z_corr"].values)
                boot.append(float(target[corr_col] - inv_probit(bf.slope * float(target["z_clean"]) + bf.intercept)))
            if boot:
                ci_low, ci_high = np.percentile(boot, [2.5, 97.5]).astype(float)
                success = len(boot) / 1000
            else:
                ci_low, ci_high, success = np.nan, np.nan, 0.0
            status = assess_er_validity(
                n_unique_configs=int(ref["configuration_id"].nunique()),
                cv_r2=float(cv) if np.isfinite(cv) else -np.inf,
                slope=float(fit.slope),
                support_min=float(ref[clean_col].min()),
                support_max=float(ref[clean_col].max()),
                target_clean=float(target[clean_col]),
                bootstrap_ci_low=float(ci_low) if np.isfinite(ci_low) else 1.0,
                bootstrap_ci_high=float(ci_high) if np.isfinite(ci_high) else 0.0,
                bootstrap_success_rate=success,
            )
        rows.append(
            {
                "dataset": target["dataset"],
                "model_name": target["model_name"],
                "configuration_id": target["configuration_id"],
                "run_id": target["run_id"],
                "model_group": target["model_group"],
                "feature_family": target["feature_family"],
                "classifier_family": target["classifier_family"],
                "architecture_family": target["architecture_family"],
                "metric": metric,
                "clean_metric": float(target[clean_col]),
                "mPC_metric": float(target[corr_col]),
                "predicted_mPC": pred,
                "CSI_ER": resid_pp,
                "CSI_ER_ci_low": ci_low,
                "CSI_ER_ci_high": ci_high,
                **asdict(status),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "results" / f"validity_gated_csi_er_{metric}.csv", index=False)
    return out


def save_fig(fig, outbase: Path) -> None:
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(outbase.with_suffix(f".{ext}"), dpi=600 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def make_figures(clean: pd.DataFrame, corr: pd.DataFrame, summary: pd.DataFrame, er: pd.DataFrame, out: Path) -> None:
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.text(0.03, 0.85, "WiFi CSI RobustBench", fontsize=16, weight="bold")
    ax.text(0.03, 0.65, "UT-HAR existing trained models -> 7 corruptions x 5 severities x 5 seeds", fontsize=10)
    ax.text(0.03, 0.48, "Metrics: Macro-F1, Accuracy, Balanced Accuracy", fontsize=10)
    ax.text(0.03, 0.31, "CSI-ER: probit LOFO fit + support gate + cluster bootstrap CI", fontsize=10)
    save_fig(fig, figdir / "fig01_benchmark_overview")

    fig, ax = plt.subplots(figsize=(8, 4))
    for grp, g in clean.groupby("architecture_family"):
        ax.scatter(g["clean_macro_f1"], g["clean_accuracy"], label=grp)
    ax.set_xlabel("Clean Macro-F1")
    ax.set_ylabel("Clean Accuracy")
    ax.legend(fontsize=7)
    ax.set_title("Model bank clean metric coverage")
    save_fig(fig, figdir / "fig02_model_bank_coverage")

    paired = pair_clean_summary(
        clean,
        summary,
        keys=ID_COLS,
        clean_metric="clean_macro_f1",
        summary_metric="mPC_family_macro_f1",
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(paired["clean_macro_f1"], paired["mPC_family_macro_f1"])
    if len(clean) > 2:
        zfit = fit_ols(probit(paired["clean_macro_f1"].values, 996), probit(paired["mPC_family_macro_f1"].values, 996))
        ax.text(0.05, 0.9, f"probit in-sample R2={zfit.r2:.3f}", transform=ax.transAxes)
    ax.set_xlabel("Clean Macro-F1")
    ax.set_ylabel("Family-balanced mPC Macro-F1")
    ax.set_title("Macro-F1 on the line")
    save_fig(fig, figdir / "fig03_macro_f1_on_the_line")

    top = summary.sort_values("mPC_family_macro_f1")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["model_name"], top["mPC_family_macro_f1"], xerr=[top["mPC_family_macro_f1"] - top["mPC_family_macro_f1_ci_low"], top["mPC_family_macro_f1_ci_high"] - top["mPC_family_macro_f1"]])
    ax.set_xlabel("Family-balanced mPC Macro-F1")
    ax.set_title("Raw robustness ranking with seed-level 95% CI")
    save_fig(fig, figdir / "fig04_raw_robustness_ranking")

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, g in corr.groupby("model_name"):
        curve = g.groupby("severity")["macro_f1"].mean()
        ax.plot(curve.index, curve.values, marker="o", label=name)
    ax.set_xlabel("Severity")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Corruption curves averaged over corruption families and seeds")
    ax.legend(fontsize=6, ncol=2)
    save_fig(fig, figdir / "fig05_corruption_curves")

    valid = er[er["status"] == "VALID"].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axvline(0, color="black", linewidth=1)
    if valid.empty:
        counts = er["status"].value_counts()
        ax.barh(counts.index, counts.values)
        ax.set_xlabel("Model count")
        ax.set_title("No model satisfies the preregistered VALID CSI-ER gate")
    else:
        plot_er = valid.sort_values("CSI_ER")
        ax.barh(plot_er["model_name"], plot_er["CSI_ER"].fillna(0))
        ax.set_xlabel("CSI-ER Macro-F1 percentage-point residual")
        ax.set_title("VALID CSI-ER models only")
    save_fig(fig, figdir / "fig06_csi_er_ranking")

    heat = corr.groupby(["model_name", "corruption"])["macro_f1"].mean().unstack()
    retention = heat.div(clean.set_index("model_name")["clean_macro_f1"], axis=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(retention.values, aspect="auto", vmin=0, vmax=max(1, np.nanmax(retention.values)))
    ax.set_xticks(range(len(retention.columns)), retention.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(retention.index)), retention.index)
    fig.colorbar(im, ax=ax, label="mPC / clean Macro-F1")
    ax.set_title("Raw retention heatmap")
    save_fig(fig, figdir / "fig07_retention_heatmap")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    ax.text(0.05, 0.75, "Synthetic vs natural shift", fontsize=14, weight="bold")
    ax.text(0.05, 0.55, "Data now staged: NTU-Fi HAR drop-box and Widar.", fontsize=10)
    ax.text(0.05, 0.40, "Natural-shift training/evaluation is not part of this existing-model UT-HAR run.", fontsize=10)
    save_fig(fig, figdir / "fig08_synthetic_vs_natural_shift_blocker")

    fam = summary.groupby("architecture_family")["mPC_family_macro_f1"].agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(fam.index, fam["mean"], yerr=fam["std"].fillna(0))
    ax.set_ylabel("Mean mPC Macro-F1")
    ax.set_title("Family analysis")
    save_fig(fig, figdir / "fig09_family_analysis")

    fig, ax = plt.subplots(figsize=(6, 4))
    hardest_row = corr.groupby(["corruption", "severity"])["macro_f1"].mean().sort_values().reset_index().iloc[0]
    hardest = hardest_row["corruption"]
    hardest_sev = int(hardest_row["severity"])
    vals = corr[(corr["corruption"] == hardest) & (corr["severity"] == hardest_sev)].groupby("model_name")["macro_f1"].mean().sort_values()
    ax.barh(vals.index, vals.values)
    ax.set_title(f"Hardest condition: {hardest.replace('_', ' ')} severity {hardest_sev}")
    ax.set_xlabel("Macro-F1")
    save_fig(fig, figdir / "fig10_hardest_condition")

    dash = er.groupby("status")["model_name"].count()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(dash.index, dash.values)
    ax.set_ylabel("Model count")
    ax.set_title("Statistical validity dashboard")
    save_fig(fig, figdir / "fig11_validity_dashboard")

    eff = pair_clean_summary(clean, summary, keys=ID_COLS, clean_metric="clean_macro_f1", summary_metric="mPC_family_macro_f1")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(eff["param_count"].replace(0, np.nan), eff["mPC_family_macro_f1"])
    for _, row in eff.iterrows():
        ax.text(max(row["param_count"], 1), row["mPC_family_macro_f1"], row["model_name"], fontsize=6)
    ax.set_xscale("log")
    ax.set_xlabel("Parameter count (deep only; classical plotted at proxy=0)")
    ax.set_ylabel("Family-balanced mPC Macro-F1")
    ax.set_title("Parameter count vs robustness diagnostic")
    save_fig(fig, figdir / "fig12_robustness_efficiency")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-output", default="outputs_v2_20260619_143332")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()
    src = PROJECT_ROOT / args.source_output
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out = PROJECT_ROOT / (args.output_dir or f"outputs_research_grade_{timestamp}")
    for sub in ["results", "figures", "logs"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    d = load_processed(smoke=False)
    xte = d["X_test"].astype("float32")
    yte = d["y_test"].astype("int64")
    ncls = int(max(d["y_train"].max(), d["y_test"].max()) + 1)
    input_shape = tuple(xte.shape[1:])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    pca = joblib.load(src / "checkpoints" / "pca.joblib")

    clean_rows: list[dict] = []
    corr_rows: list[dict] = []

    classical_models = {}
    classical_meta = pd.read_csv(src / "results" / "classical_clean_metrics.csv")
    for _, meta in classical_meta.iterrows():
        name = meta["model_name"]
        classical_models[name] = joblib.load(src / "checkpoints" / "classical" / f"{name}.joblib")

    deep_models = {}
    for name in ["MLP", "SimpleCNN", "GRU", "LSTM", "CNNGRU", "TinyViT"]:
        ck = src / "checkpoints" / "deep" / f"{name}_best.pt"
        state = torch.load(ck, map_location=device)
        model = build_model(name, ncls, input_shape=state.get("input_shape", input_shape)).to(device)
        model.load_state_dict(state["state_dict"], strict=True)
        deep_models[name] = (model, int(state.get("seed", 42)))

    # Clean pass.
    for name, model in classical_models.items():
        pred = model.predict(compute_feature(FEATURE_KIND[name], xte, pca))
        meta = model_metadata(name, is_classical=True, train_seed=42)
        row = {
            "timestamp": now_ts(),
            "dataset": "UT_HAR",
            "model_name": name,
            "is_classical": True,
            "param_count": 0,
            **meta,
        }
        row.update({f"clean_{k}": v for k, v in metric_row(yte, pred).items()})
        clean_rows.append(row)

    for name, (model, train_seed) in deep_models.items():
        pred = predict_deep(model, xte, batch_size, device)
        meta = model_metadata(name, is_classical=False, train_seed=train_seed)
        row = {
            "timestamp": now_ts(),
            "dataset": "UT_HAR",
            "model_name": name,
            "is_classical": False,
            "param_count": count_parameters(model),
            **meta,
        }
        row.update({f"clean_{k}": v for k, v in metric_row(yte, pred).items()})
        clean_rows.append(row)

    # Corruption pass: generate each corrupted test set once, then evaluate all models.
    total = len(CORRUPTIONS) * len(SEVERITIES) * len(CORRUPTION_SEEDS)
    done = 0
    progress_path = out / "logs" / "progress.log"
    for corruption in CORRUPTIONS:
        for severity in SEVERITIES:
            for seed in CORRUPTION_SEEDS:
                done += 1
                xc = apply_research_corruption(xte, corruption, severity, seed)
                feature_cache = {}
                for name, model in classical_models.items():
                    kind = FEATURE_KIND[name]
                    if kind not in feature_cache:
                        feature_cache[kind] = compute_feature(kind, xc, pca)
                    pc = model.predict(feature_cache[kind])
                    meta = model_metadata(name, is_classical=True, train_seed=42)
                    rr = {
                        "dataset": "UT_HAR",
                        "model_name": name,
                        "is_classical": True,
                        **meta,
                        "corruption": corruption,
                        "severity": severity,
                        "corruption_seed": seed,
                    }
                    rr.update(metric_row(yte, pc))
                    corr_rows.append(rr)
                for name, (model, train_seed) in deep_models.items():
                    pc = predict_deep(model, xc, batch_size, device)
                    meta = model_metadata(name, is_classical=False, train_seed=train_seed)
                    rr = {
                        "dataset": "UT_HAR",
                        "model_name": name,
                        "is_classical": False,
                        **meta,
                        "corruption": corruption,
                        "severity": severity,
                        "corruption_seed": seed,
                    }
                    rr.update(metric_row(yte, pc))
                    corr_rows.append(rr)
                if done == 1 or done % 10 == 0 or done == total:
                    progress_path.write_text(f"{now_ts()} completed {done}/{total} corruption combinations\n", encoding="utf-8")

    clean = pd.DataFrame(clean_rows)
    corr = pd.DataFrame(corr_rows)
    summary = summarize_corruptions(corr)
    clean.to_csv(out / "results" / "clean_metrics_research.csv", index=False)
    corr.to_csv(out / "results" / "corruption_metrics_research_long.csv", index=False)
    summary.to_csv(out / "results" / "robustness_summary_research.csv", index=False)
    er_macro = compute_validity(clean, summary, "macro_f1", len(yte), out)
    er_acc = compute_validity(clean, summary, "accuracy", len(yte), out)

    merged = pair_clean_summary(clean, summary, keys=ID_COLS, clean_metric="clean_macro_f1", summary_metric="mPC_family_macro_f1")
    spear_mpc = float(spearmanr(merged["clean_macro_f1"], merged["mPC_family_macro_f1"]).statistic)
    make_figures(clean, corr, summary, er_macro, out)
    report = f"""# Research-Grade Existing-Model Completion Report

Generated: {now_ts()}

## Scope

This run evaluates the existing trained UT-HAR model bank from `{args.source_output}` with structured corruption statistics. It should be interpreted within the configured benchmark scope and the datasets actually evaluated.

## Evaluation scale

- Dataset: UT-HAR
- Models: {len(clean)}
- Corruptions: {len(CORRUPTIONS)} ({', '.join(CORRUPTIONS)})
- Severities per corruption: {len(SEVERITIES)}
- Corruption seeds: {len(CORRUPTION_SEEDS)} ({CORRUPTION_SEEDS})
- Total corruption rows: {len(corr)}

## Metrics

- Primary: Macro-F1
- Secondary: Accuracy, Balanced Accuracy
- Clean-vs-corrupted Spearman correlation for Macro-F1: {spear_mpc:.4f}

## CSI-ER validity

Because this run uses the existing 14-model bank rather than the full preregistered reference bank, the validity gate is expected to mark most or all targets as `EXPLORATORY`, `OUT_OF_SUPPORT`, or `INVALID_FIT`. This is intentional and prevents overclaiming.

Macro-F1 status counts:

{er_macro['status'].value_counts().to_string()}

Accuracy status counts:

{er_acc['status'].value_counts().to_string()}

## Multi-dataset and natural-shift status

NTU-Fi HAR and Widar are now staged under `data/raw`. NTU-Fi HAR uses the documented five-class drop-box manifest because the recovered official `box` test split contains only one sample. This existing-model run still evaluates UT-HAR only, so synthetic-vs-natural shift correlation is not reported as a completed result here.
"""
    (out / "results" / "research_grade_report.md").write_text(report, encoding="utf-8")
    print(out)
    print(report)


if __name__ == "__main__":
    main()

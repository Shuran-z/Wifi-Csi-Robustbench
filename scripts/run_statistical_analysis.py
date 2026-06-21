#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from csi_er.data import load_processed
from csi_robustbench.statistics import fit_ols, grouped_cv_r2, inv_probit, pair_clean_summary, probit, seed_level_mpc, summarize_seed_mpc
from csi_robustbench.validity import assess_er_validity


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
METRICS = ["accuracy", "macro_f1", "balanced_accuracy"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build final combined statistics for a completed WiFi CSI RobustBench run.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--bootstrap-reps", type=int, default=2000)
    return p.parse_args()


def read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def normalize_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def load_inputs(out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean_parts = []
    corr_parts = []
    for stem in ["classical_bank", "deep_bank"]:
        clean_parts.append(read_required(out / "results" / f"{stem}_clean.csv"))
        corr_parts.append(read_required(out / "results" / f"{stem}_corruptions_long.csv"))
    clean = pd.concat(clean_parts, ignore_index=True, sort=False)
    corr = pd.concat(corr_parts, ignore_index=True, sort=False)
    for frame in [clean, corr]:
        if "param_count" not in frame.columns:
            frame["param_count"] = 0
        frame["is_classical"] = normalize_bool(frame["is_classical"])
        frame["train_fraction"] = frame["train_fraction"].astype(float)
        frame["train_seed"] = frame["train_seed"].astype(int)
    for col in ID_COLS:
        if col not in clean.columns or col not in corr.columns:
            raise KeyError(f"missing ID column: {col}")
    return clean, corr


def compute_validity(clean: pd.DataFrame, summary: pd.DataFrame, metric: str, n_test: int, reps: int) -> pd.DataFrame:
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
    rng = np.random.default_rng(20260620)
    fit_cache = {}
    for arch in sorted(df["architecture_family"].unique()):
        ref = df[df["architecture_family"] != arch].copy()
        cache = {
            "ok": False,
            "ref": ref,
            "fit": None,
            "cv": float("nan"),
            "boot_slopes": np.asarray([], dtype=float),
            "boot_intercepts": np.asarray([], dtype=float),
            "success": 0.0,
        }
        if len(ref) >= 2 and ref["z_clean"].nunique() >= 2:
            fit = fit_ols(ref["z_clean"].values, ref["z_corr"].values)
            groups = np.where(ref["is_classical"].astype(bool), ref["base_configuration_id"], ref["architecture_family"])
            cv = grouped_cv_r2(ref["z_clean"].values, ref["z_corr"].values, groups)
            ref_x = ref["z_clean"].to_numpy(dtype=float)
            ref_y = ref["z_corr"].to_numpy(dtype=float)
            _, cluster_codes = np.unique(ref["base_configuration_id"].to_numpy(), return_inverse=True)
            n_clusters = int(cluster_codes.max() + 1)
            counts = rng.multinomial(n_clusters, np.full(n_clusters, 1.0 / n_clusters), size=int(reps)).astype(float)
            weights = counts[:, cluster_codes]
            sw = weights.sum(axis=1)
            sx = weights @ ref_x
            sy = weights @ ref_y
            sxx_raw = weights @ (ref_x * ref_x)
            sxy_raw = weights @ (ref_x * ref_y)
            mx = sx / np.maximum(sw, 1e-12)
            my = sy / np.maximum(sw, 1e-12)
            sxx = sxx_raw - sw * mx * mx
            sxy = sxy_raw - sw * mx * my
            valid = np.isfinite(sxx) & (sxx > 1e-12)
            slopes = np.full(int(reps), np.nan, dtype=float)
            intercepts = np.full(int(reps), np.nan, dtype=float)
            slopes[valid] = sxy[valid] / sxx[valid]
            intercepts[valid] = my[valid] - slopes[valid] * mx[valid]
            cache.update(
                {
                    "ok": True,
                    "fit": fit,
                    "cv": float(cv) if np.isfinite(cv) else float("nan"),
                    "boot_slopes": slopes[np.isfinite(slopes) & np.isfinite(intercepts)],
                    "boot_intercepts": intercepts[np.isfinite(slopes) & np.isfinite(intercepts)],
                    "success": float(valid.mean()),
                }
            )
        fit_cache[arch] = cache

    rows = []
    for _, target in df.iterrows():
        cache = fit_cache[target["architecture_family"]]
        ref = cache["ref"]
        pred = np.nan
        resid = np.nan
        ci_low = np.nan
        ci_high = np.nan
        success = float(cache["success"])
        if cache["ok"]:
            fit = cache["fit"]
            cv = cache["cv"]
            pred = float(inv_probit(fit.slope * float(target["z_clean"]) + fit.intercept))
            resid = float(target[corr_col] - pred)
            boot_slopes = cache["boot_slopes"]
            boot_intercepts = cache["boot_intercepts"]
            if len(boot_slopes):
                boot = target[corr_col] - inv_probit(boot_slopes * float(target["z_clean"]) + boot_intercepts)
                ci_low, ci_high = np.percentile(boot, [2.5, 97.5]).astype(float)
            status = assess_er_validity(
                n_unique_configs=int(ref["configuration_id"].nunique()),
                cv_r2=float(cv) if np.isfinite(cv) else float("nan"),
                slope=float(fit.slope),
                support_min=float(ref[clean_col].min()),
                support_max=float(ref[clean_col].max()),
                target_clean=float(target[clean_col]),
                bootstrap_ci_low=float(ci_low) if np.isfinite(ci_low) else float("nan"),
                bootstrap_ci_high=float(ci_high) if np.isfinite(ci_high) else float("nan"),
                bootstrap_success_rate=success,
            )
        else:
            status = assess_er_validity(
                n_unique_configs=int(ref["configuration_id"].nunique()),
                cv_r2=float("nan"),
                slope=float("nan"),
                support_min=float("nan"),
                support_max=float("nan"),
                target_clean=float(target[clean_col]),
                bootstrap_ci_low=float("nan"),
                bootstrap_ci_high=float("nan"),
                bootstrap_success_rate=0.0,
            )
        row = {k: target[k] for k in ID_COLS}
        row.update(
            {
                "metric": metric,
                "clean_metric": float(target[clean_col]),
                "mPC_metric": float(target[corr_col]),
                "predicted_mPC": pred,
                "CSI_ER": resid,
                "CSI_ER_ci_low": ci_low,
                "CSI_ER_ci_high": ci_high,
                "bootstrap_success_rate": success,
                **asdict(status),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def save_fig(fig, path: Path) -> None:
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def make_figures(clean: pd.DataFrame, summary: pd.DataFrame, validity: pd.DataFrame, out: Path) -> None:
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    paired = pair_clean_summary(clean, summary, keys=ID_COLS, clean_metric="clean_macro_f1", summary_metric="mPC_family_macro_f1")

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, g in paired.groupby("architecture_family"):
        ax.scatter(g["clean_macro_f1"], g["mPC_family_macro_f1"], s=12, alpha=0.55, label=name)
    ax.set_xlabel("Clean Macro-F1")
    ax.set_ylabel("Family-balanced mPC Macro-F1")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Final model bank: clean vs corrupted")
    save_fig(fig, figdir / "final_clean_vs_mpc_macro_f1")

    top = summary.sort_values("mPC_family_macro_f1", ascending=False).head(30)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(top["model_name"], top["mPC_family_macro_f1"])
    ax.invert_yaxis()
    ax.set_xlabel("Family-balanced mPC Macro-F1")
    ax.set_title("Top 30 raw robustness models")
    save_fig(fig, figdir / "final_top30_raw_robustness")

    fig, ax = plt.subplots(figsize=(7, 4))
    counts = validity["status"].value_counts()
    ax.bar(counts.index, counts.values)
    ax.set_ylabel("Model count")
    ax.set_title("CSI-ER validity gate status")
    save_fig(fig, figdir / "final_csi_er_validity_status")


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir)
    (out / "results").mkdir(parents=True, exist_ok=True)
    clean, corr = load_inputs(out)
    clean.to_csv(out / "results" / "combined_clean_metrics.csv", index=False)
    corr.to_csv(out / "results" / "combined_corruption_metrics_long.csv", index=False)

    seed_scores = seed_level_mpc(corr, group_cols=ID_COLS, metrics=METRICS)
    summary = summarize_seed_mpc(seed_scores, group_cols=ID_COLS, metrics=METRICS)
    summary.to_csv(out / "results" / "combined_robustness_summary.csv", index=False)

    n_test = int(load_processed(smoke=False)["y_test"].shape[0])
    validity_frames = []
    for metric in ["macro_f1", "accuracy"]:
        er = compute_validity(clean, summary, metric, n_test=n_test, reps=args.bootstrap_reps)
        er.to_csv(out / "results" / f"combined_validity_gated_csi_er_{metric}.csv", index=False)
        validity_frames.append(er)
    validity = pd.concat(validity_frames, ignore_index=True)
    make_figures(clean, summary, validity[validity["metric"] == "macro_f1"], out)

    report = out / "reports" / "final_statistical_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    valid_counts = validity.groupby(["metric", "status"]).size().unstack(fill_value=0)
    payload = {
        "n_models": int(len(clean)),
        "n_corruption_rows": int(len(corr)),
        "n_test": n_test,
        "metrics": METRICS,
        "validity_counts": valid_counts.to_dict(),
    }
    (out / "artifacts").mkdir(parents=True, exist_ok=True)
    (out / "artifacts" / "final_statistical_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    report.write_text(
        "# Final Statistical Report\n\n"
        f"- Models: {len(clean)}\n"
        f"- Corruption rows: {len(corr)}\n"
        f"- UT-HAR test samples: {n_test}\n"
        "- Primary metric: family-balanced mPC Macro-F1\n"
        "- Secondary metric: flat-7 mPC plus Accuracy/Balanced Accuracy\n\n"
        "## CSI-ER Validity Counts\n\n"
        "```text\n"
        f"{valid_counts.to_string()}\n"
        "```\n\n"
        "CSI-ER rankings must be interpreted only for rows marked `VALID`; otherwise the raw mPC and retention metrics are the supported result.\n",
        encoding="utf-8",
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

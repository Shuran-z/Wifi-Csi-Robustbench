#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from csi_robustbench.features import FeatureTransformer
from scripts.run_research_grade_existing_models import CORRUPTION_SEEDS, CORRUPTIONS, SEVERITIES, apply_research_corruption


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a real Widar train/test natural-shift minimum bank.")
    p.add_argument("--root", default="data/raw/SenseFi_extracted/Widardata")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-per-class", type=int, default=None)
    p.add_argument("--max-test-per-class", type=int, default=None)
    p.add_argument("--corruption-seed-limit", type=int, default=3)
    return p.parse_args()


def class_dirs(root: Path, split: str) -> list[Path]:
    return sorted([p for p in (root / split).iterdir() if p.is_dir()], key=lambda p: int(p.name.split("-")[0]))


def load_split(root: Path, split: str, *, max_per_class: int | None, seed: int):
    xs, ys, paths = [], [], []
    rng = np.random.default_rng(seed)
    for label, d in enumerate(class_dirs(root, split)):
        files = sorted(d.glob("*.csv"))
        if max_per_class is not None and len(files) > max_per_class:
            files = [files[i] for i in sorted(rng.choice(len(files), size=max_per_class, replace=False).tolist())]
        for f in files:
            arr = np.loadtxt(f, delimiter=",", dtype=np.float32)
            if arr.shape != (22, 400):
                raise ValueError(f"{f} has shape {arr.shape}, expected (22, 400)")
            xs.append(arr)
            ys.append(label)
            paths.append(str(f.relative_to(root)))
    x = np.asarray(xs, dtype=np.float32)[:, None, :, :]
    y = np.asarray(ys, dtype=np.int64)
    return x, y, paths


def normalize(x_train: np.ndarray, *others: np.ndarray):
    mean = float(x_train.mean())
    std = float(x_train.std() + 1e-8)
    return ((x_train - mean) / std, *[(x - mean) / std for x in others], {"mean": mean, "std": std})


def metric_row(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def make_model(name: str, seed: int):
    if name == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000, random_state=seed, n_jobs=1))
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=200, max_depth=16, random_state=seed, n_jobs=4)
    raise ValueError(name)


def synthetic_mpc(model, transformer, x_val, y_val, seed_limit: int):
    rows = []
    for seed in CORRUPTION_SEEDS[:seed_limit]:
        for corruption in CORRUPTIONS:
            for severity in SEVERITIES:
                xc = apply_research_corruption(x_val, corruption, severity, seed=int(seed))
                pred = model.predict(transformer.transform(xc))
                row = metric_row(y_val, pred)
                row.update({"corruption": corruption, "severity": severity, "corruption_seed": int(seed)})
                rows.append(row)
    df = pd.DataFrame(rows)
    return df, {f"synthetic_mPC_{k}": float(df[k].mean()) for k in ["accuracy", "macro_f1", "balanced_accuracy"]}


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, seed: int, reps: int = 2000) -> dict:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        idx = rng.integers(0, len(x), size=len(x))
        vals.append(float(spearmanr(x[idx], y[idx]).statistic))
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    return {
        "spearman": float(spearmanr(x, y).statistic),
        "ci_low": float(np.percentile(arr, 2.5)) if len(arr) else float("nan"),
        "ci_high": float(np.percentile(arr, 97.5)) if len(arr) else float("nan"),
        "bootstrap_reps": int(reps),
        "bootstrap_success_rate": float(len(arr) / reps),
    }


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    out = Path(args.output_dir)
    for sub in ["results", "manifests", "checkpoints/widar_minimum_bank", "reports", "artifacts"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    x_all, y_all, train_paths = load_split(root, "train", max_per_class=args.max_train_per_class, seed=args.seed)
    x_test, y_test, test_paths = load_split(root, "test", max_per_class=args.max_test_per_class, seed=args.seed + 1)
    idx = np.arange(len(y_all))
    tr_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=args.seed, stratify=y_all)
    x_train, y_train = x_all[tr_idx], y_all[tr_idx]
    x_val, y_val = x_all[val_idx], y_all[val_idx]
    x_train, x_val, x_test, norm = normalize(x_train, x_val, x_test)

    feature_specs = [
        ("time_stats", {"segment_bins": 4, "statistics": ["mean", "std", "min", "max", "median", "iqr"]}),
        ("fft_stats", {"bands": 8, "statistics": ["mean", "std", "energy"]}),
        ("autocorrelation", {"lags": 16, "normalize": True}),
    ]
    classifiers = ["logistic_regression", "random_forest"]
    clean_rows, corr_rows, ckpt_rows = [], [], []
    for feature_family, feature_params in feature_specs:
        transformer = FeatureTransformer(feature_family, feature_params, seed=args.seed)
        f_train = transformer.fit_transform(x_train)
        f_val = transformer.transform(x_val)
        f_test = transformer.transform(x_test)
        for clf_name in classifiers:
            config_id = f"widar__{feature_family}__{clf_name}"
            model = make_model(clf_name, args.seed)
            t0 = time.time()
            model.fit(f_train, y_train)
            val_pred = model.predict(f_val)
            test_pred = model.predict(f_test)
            row = {
                "dataset": "Widar",
                "configuration_id": config_id,
                "model_name": config_id,
                "feature_family": feature_family,
                "classifier": clf_name,
                "train_domain": "Widar/train",
                "clean_domain": "Widar/train_validation",
                "natural_domain": "Widar/test",
                "train_samples": int(len(y_train)),
                "val_samples": int(len(y_val)),
                "test_samples": int(len(y_test)),
                "train_time_sec": float(time.time() - t0),
            }
            row.update({f"clean_val_{k}": v for k, v in metric_row(y_val, val_pred).items()})
            row.update({f"natural_test_{k}": v for k, v in metric_row(y_test, test_pred).items()})
            corr_df, synth = synthetic_mpc(model, transformer, x_val, y_val, args.corruption_seed_limit)
            row.update(synth)
            clean_rows.append(row)
            corr_df.insert(0, "configuration_id", config_id)
            corr_rows.append(corr_df)
            ckpt = out / "checkpoints/widar_minimum_bank" / f"{config_id}.joblib"
            joblib.dump({"model": model, "feature_transformer": transformer, "config": row}, ckpt)
            ckpt_rows.append({"configuration_id": config_id, "checkpoint_path": str(ckpt.relative_to(out)), "bytes": ckpt.stat().st_size})
            print(f"OK {config_id} natural_macro_f1={row['natural_test_macro_f1']:.4f}", flush=True)

    clean = pd.DataFrame(clean_rows)
    corr = pd.concat(corr_rows, ignore_index=True)
    clean.to_csv(out / "results/widar_minimum_bank_natural_shift.csv", index=False)
    corr.to_csv(out / "results/widar_minimum_bank_synthetic_corruptions_long.csv", index=False)
    pd.DataFrame(ckpt_rows).to_csv(out / "manifests/widar_minimum_bank_checkpoint_manifest.csv", index=False)
    corr_stats = {
        metric: bootstrap_spearman(clean[f"synthetic_mPC_{metric}"].to_numpy(), clean[f"natural_test_{metric}"].to_numpy(), args.seed)
        for metric in ["accuracy", "macro_f1", "balanced_accuracy"]
    }
    summary = {
        "dataset": "Widar",
        "policy": "real train/test domain split; train split is subdivided only for source-domain validation",
        "class_count": int(len(class_dirs(root, "train"))),
        "train_files_loaded": int(len(train_paths)),
        "test_files_loaded": int(len(test_paths)),
        "normalization": norm,
        "models": int(len(clean)),
        "corruption_seed_limit": int(args.corruption_seed_limit),
        "synthetic_vs_natural_spearman": corr_stats,
    }
    (out / "artifacts/widar_natural_shift_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (out / "reports/widar_natural_shift_report.md").write_text(
        "# Widar Natural-Shift Minimum Bank\n\n"
        f"- Train files loaded: {len(train_paths)}\n"
        f"- Test files loaded: {len(test_paths)}\n"
        f"- Models: {len(clean)}\n"
        "- Split policy: real `train/` to `test/`; no random train/test replacement.\n\n"
        "## Synthetic-vs-Natural Spearman\n\n"
        "```json\n" + json.dumps(corr_stats, indent=2, sort_keys=True) + "\n```\n",
        encoding="utf-8",
    )
    print(out / "reports/widar_natural_shift_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from csi_er.data import load_processed
from csi_er.utils import PROJECT_ROOT, now_ts
from csi_robustbench.features import FeatureTransformer
from csi_robustbench.reference_bank import ReferenceConfig, build_reference_configs, load_yaml, write_manifest
from scripts.run_research_grade_existing_models import CORRUPTION_SEEDS, CORRUPTIONS, SEVERITIES, apply_research_corruption


ID_COLUMNS = [
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
FAIL_COLUMNS = ["configuration_id", "base_configuration_id", "error", "traceback"]
CHECKPOINT_COLUMNS = ["configuration_id", "run_id", "checkpoint_path", "checkpoint_bytes"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and evaluate the preregistered classical WiFi CSI reference bank.")
    p.add_argument("--config", default="configs/reference_bank.yaml")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--dataset", default="UT_HAR")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--execute", action="store_true", help="Actually train/evaluate instead of only writing the manifest.")
    p.add_argument("--limit", type=int, default=None, help="Optional cap for debugging/smoke runs.")
    p.add_argument("--num-shards", type=int, default=1, help="Split selected rows into this many deterministic shards.")
    p.add_argument("--shard-index", type=int, default=0, help="Zero-based shard index to execute.")
    p.add_argument("--resume", action="store_true", help="Append to an existing shard output and skip completed configuration IDs.")
    p.add_argument("--corruption-seed-limit", type=int, default=None)
    p.add_argument("--severity-limit", type=int, default=None)
    return p.parse_args()


def select_nested_subset(y: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for cls in sorted(np.unique(y).tolist()):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        k = max(1, int(np.ceil(len(idx) * float(fraction))))
        selected.extend(idx[: min(k, len(idx))].tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def cfg_feature_key(cfg: ReferenceConfig, subset_idx: np.ndarray | None = None) -> str:
    payload = {
        "feature_family": cfg.feature_family,
        "feature_capacity": cfg.feature_capacity,
        "feature_params": cfg.feature_params,
    }
    if cfg.feature_family == "pca_raw":
        payload["subset_hash"] = hashlib.sha256(np.asarray(subset_idx, dtype=np.int64).tobytes()).hexdigest()[:16]
    return json.dumps(payload, sort_keys=True)


def get_feature_bundle(
    cfg: ReferenceConfig,
    subset_idx: np.ndarray,
    x_train: np.ndarray,
    x_test: np.ndarray,
    feature_cache: dict[str, dict],
) -> dict:
    key = cfg_feature_key(cfg, subset_idx)
    if key not in feature_cache:
        transformer = FeatureTransformer(cfg.feature_family, cfg.feature_params, seed=cfg.subset_seed)
        if cfg.feature_family == "pca_raw":
            transformer.fit(x_train[subset_idx])
            train_features = transformer.transform(x_train)
        else:
            transformer.fit(x_train[subset_idx])
            train_features = transformer.transform(x_train)
        feature_cache[key] = {
            "transformer": transformer,
            "train": train_features,
            "clean_test": transformer.transform(x_test),
            "corr": {},
        }
    return feature_cache[key]


def get_corruption_features(
    bundle: dict,
    x_test: np.ndarray,
    corruption: str,
    severity: int,
    corr_seed: int,
) -> np.ndarray:
    key = (corruption, int(severity), int(corr_seed))
    if key not in bundle["corr"]:
        x_corr = apply_research_corruption(x_test, corruption, severity, seed=int(corr_seed))
        bundle["corr"][key] = bundle["transformer"].transform(x_corr)
    return bundle["corr"][key]


def make_classifier(cfg: ReferenceConfig, seed: int):
    params = dict(cfg.classifier_params)
    name = cfg.classifier
    if name == "logistic_regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(params.get("C", 1.0)),
                max_iter=int(params.get("max_iter", 3000)),
                random_state=seed,
                n_jobs=1,
            ),
        )
    if name == "linear_svm":
        return make_pipeline(StandardScaler(), LinearSVC(C=float(params.get("C", 1.0)), random_state=seed, max_iter=8000))
    if name == "rbf_svm":
        return make_pipeline(StandardScaler(), SVC(C=float(params.get("C", 2.0)), gamma=params.get("gamma", "scale"), kernel="rbf"))
    if name == "knn":
        return make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=int(params.get("n_neighbors", 7)), weights=str(params.get("weights", "distance"))),
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=params.get("max_depth"),
            random_state=seed,
            n_jobs=4,
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=params.get("max_depth"),
            random_state=seed,
            n_jobs=4,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=int(params.get("max_iter", 200)),
            max_leaf_nodes=int(params.get("max_leaf_nodes", 31)),
            learning_rate=float(params.get("learning_rate", 0.06)),
            random_state=seed,
        )
    if name == "lda":
        solver = str(params.get("solver", "svd"))
        shrinkage = params.get("shrinkage")
        return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage))
    raise ValueError(f"Unknown classifier: {name}")


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def metadata_row(cfg: ReferenceConfig, feature_dim: int, train_size: int) -> dict:
    row = asdict(cfg)
    row.update(
        {
            "timestamp": now_ts(),
            "model_name": cfg.configuration_id,
            "run_id": f"{cfg.configuration_id}__seed{cfg.subset_seed}",
            "architecture_family": "classical",
            "model_group": cfg.classifier_family,
            "is_classical": True,
            "train_seed": cfg.subset_seed,
            "model_size": cfg.classifier_capacity,
            "feature_dim": int(feature_dim),
            "train_size": int(train_size),
        }
    )
    return row


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def write_progress_tables(
    results_dir: Path,
    manifest_dir: Path,
    clean_rows: list[dict],
    corr_rows: list[dict],
    fail_rows: list[dict],
    checkpoint_rows: list[dict],
) -> None:
    pd.DataFrame(clean_rows).to_csv(results_dir / "classical_bank_clean.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(results_dir / "classical_bank_corruptions_long.csv", index=False)
    pd.DataFrame(fail_rows, columns=FAIL_COLUMNS).to_csv(results_dir / "classical_bank_failures.csv", index=False)
    pd.DataFrame(checkpoint_rows, columns=CHECKPOINT_COLUMNS).to_csv(
        manifest_dir / "classical_bank_checkpoint_manifest.csv",
        index=False,
    )


def run(args: argparse.Namespace) -> int:
    cfg_dict = load_yaml(args.config)
    rows = build_reference_configs(cfg_dict)
    if args.dataset:
        rows = [r for r in rows if r.dataset == args.dataset or args.dataset == "UT_HAR"]
    if int(args.num_shards) < 1:
        raise ValueError("--num-shards must be >= 1")
    if int(args.shard_index) < 0 or int(args.shard_index) >= int(args.num_shards):
        raise ValueError("--shard-index must be in [0, num_shards)")
    if int(args.num_shards) > 1:
        chunk = int(np.ceil(len(rows) / int(args.num_shards)))
        start = int(args.shard_index) * chunk
        rows = rows[start : start + chunk]
    if args.limit is not None:
        rows = rows[: int(args.limit)]

    out = PROJECT_ROOT / (args.output_dir or f"outputs_final_complete_{time.strftime('%Y%m%d_%H%M%S')}")
    results_dir = out / "results"
    ckpt_dir = out / "checkpoints" / "classical_bank"
    manifest_dir = out / "manifests"
    log_dir = out / "logs"
    for d in (results_dir, ckpt_dir, manifest_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    write_manifest(args.config, manifest_dir / "classical_reference_bank_preregistered_manifest.csv")
    pd.DataFrame([asdict(r) for r in rows]).to_csv(manifest_dir / "classical_bank_selected_manifest.csv", index=False)

    if not args.execute:
        print(f"Generated {len(build_reference_configs(cfg_dict))} preregistered classical reference training points.")
        print("Use --execute to train/evaluate the bank.")
        return 0

    data = load_processed(smoke=args.smoke)
    x_train, y_train = data["X_train"], data["y_train"]
    x_test, y_test = data["X_test"], data["y_test"]
    clean_rows: list[dict] = []
    corr_rows: list[dict] = []
    fail_rows: list[dict] = []
    checkpoint_rows: list[dict] = []
    if args.resume and (results_dir / "classical_bank_clean.csv").exists():
        clean_rows = pd.read_csv(results_dir / "classical_bank_clean.csv").to_dict("records")
        if (results_dir / "classical_bank_corruptions_long.csv").exists():
            corr_rows = pd.read_csv(results_dir / "classical_bank_corruptions_long.csv").to_dict("records")
        if (results_dir / "classical_bank_failures.csv").exists():
            fail_rows = pd.read_csv(results_dir / "classical_bank_failures.csv").to_dict("records")
        if (manifest_dir / "classical_bank_checkpoint_manifest.csv").exists():
            checkpoint_rows = pd.read_csv(manifest_dir / "classical_bank_checkpoint_manifest.csv").to_dict("records")
        completed = {str(r["configuration_id"]) for r in clean_rows}
        rows = [r for r in rows if r.configuration_id not in completed]
        print(f"resume: loaded {len(completed)} completed configs, remaining {len(rows)}", flush=True)
    feature_cache: dict[str, dict] = {}

    corr_seeds = CORRUPTION_SEEDS[: args.corruption_seed_limit] if args.corruption_seed_limit else CORRUPTION_SEEDS
    severities = SEVERITIES[: args.severity_limit] if args.severity_limit else SEVERITIES

    for idx, bank_cfg in enumerate(rows, start=1):
        t0 = time.time()
        try:
            subset_idx = select_nested_subset(y_train, bank_cfg.train_fraction, bank_cfg.subset_seed)
            y_sub = y_train[subset_idx]
            bundle = get_feature_bundle(bank_cfg, subset_idx, x_train, x_test, feature_cache)
            transformer = bundle["transformer"]
            feat_train = bundle["train"][subset_idx]
            feat_test = bundle["clean_test"]
            model = make_classifier(bank_cfg, bank_cfg.subset_seed)
            model.fit(feat_train, y_sub)
            pred = model.predict(feat_test)
            clean = metadata_row(bank_cfg, feat_train.shape[1], len(subset_idx))
            clean.update({f"clean_{k}": v for k, v in metrics(y_test, pred).items()})
            clean["train_time_sec"] = float(time.time() - t0)
            clean_rows.append(clean)

            ckpt_path = ckpt_dir / f"{bank_cfg.configuration_id}__seed{bank_cfg.subset_seed}.joblib"
            joblib.dump({"feature_transformer": transformer, "model": model, "config": asdict(bank_cfg)}, ckpt_path)
            checkpoint_rows.append(
                {
                    "configuration_id": bank_cfg.configuration_id,
                    "run_id": clean["run_id"],
                    "checkpoint_path": str(ckpt_path.relative_to(out)),
                    "checkpoint_bytes": ckpt_path.stat().st_size,
                }
            )

            for corr_seed in corr_seeds:
                for corruption in CORRUPTIONS:
                    for severity in severities:
                        feat_corr = get_corruption_features(bundle, x_test, corruption, severity, int(corr_seed))
                        corr_pred = model.predict(feat_corr)
                        row = {k: clean[k] for k in ID_COLUMNS}
                        row.update(metrics(y_test, corr_pred))
                        row.update(
                            {
                                "corruption": corruption,
                                "severity": int(severity),
                                "corruption_seed": int(corr_seed),
                                "condition_id": f"{corruption}__s{severity}__seed{corr_seed}",
                            }
                        )
                        corr_rows.append(row)
            print(f"[{idx}/{len(rows)}] OK {bank_cfg.configuration_id} clean_macro_f1={clean['clean_macro_f1']:.4f}", flush=True)
        except Exception as exc:
            fail_rows.append(
                {
                    "configuration_id": bank_cfg.configuration_id,
                    "base_configuration_id": bank_cfg.base_configuration_id,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"[{idx}/{len(rows)}] FAIL {bank_cfg.configuration_id}: {exc}", flush=True)

        write_progress_tables(results_dir, manifest_dir, clean_rows, corr_rows, fail_rows, checkpoint_rows)

    write_json(
        out / "artifacts" / "classical_bank_run_summary.json",
        {
            "output_dir": str(out),
            "smoke": bool(args.smoke),
            "selected_configs": len(rows),
            "num_shards": int(args.num_shards),
            "shard_index": int(args.shard_index),
            "completed_configs": len(clean_rows),
            "failed_configs": len(fail_rows),
            "corruption_rows": len(corr_rows),
            "corruption_seeds": [int(x) for x in corr_seeds],
            "severities": [int(x) for x in severities],
        },
    )
    return 1 if fail_rows else 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

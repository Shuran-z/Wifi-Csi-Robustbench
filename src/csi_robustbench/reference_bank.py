from __future__ import annotations
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
from typing import Any
import csv
import hashlib
import json
import yaml

@dataclass(frozen=True)
class ReferenceConfig:
    dataset: str
    configuration_id: str
    base_configuration_id: str
    feature_family: str
    feature_capacity: str
    feature_params: dict[str, Any]
    classifier: str
    classifier_family: str
    classifier_capacity: str
    classifier_params: dict[str, Any]
    train_fraction: float
    subset_seed: int
    protocol_hash: str


def stable_hash(payload: Any, length: int = 12) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:length]

def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _capacity_value(values: Any, capacity: str) -> Any:
    if isinstance(values, dict) and capacity in values:
        return values[capacity]
    if isinstance(values, list):
        idx = {"small": 0, "medium": 1, "large": 2}[capacity]
        return values[min(idx, len(values) - 1)]
    return values


def feature_params_for(feature_family: str, spec: dict[str, Any], capacity: str) -> dict[str, Any]:
    if feature_family == "time_stats":
        return {
            "segment_bins": int(_capacity_value(spec.get("segment_bins", [1, 2, 4]), capacity)),
            "statistics": list(spec.get("statistics", ["mean", "std", "min", "max", "median", "iqr"])),
        }
    if feature_family == "fft_stats":
        return {
            "bands": int(_capacity_value(spec.get("bands", [4, 8, 16]), capacity)),
            "statistics": list(spec.get("statistics", ["mean", "std", "energy"])),
        }
    if feature_family == "dwt_energy":
        return {
            "wavelet": str(_capacity_value(spec.get("wavelets", ["haar", "db2", "sym2"]), capacity)),
            "levels": int(_capacity_value(spec.get("levels", [2, 3, 4]), capacity)),
        }
    if feature_family == "stft_bands":
        return {
            "window": int(_capacity_value(spec.get("windows", [32, 64, 128]), capacity)),
            "bands": int(_capacity_value(spec.get("bands", [4, 8, 16]), capacity)),
        }
    if feature_family == "autocorrelation":
        return {
            "lags": int(_capacity_value(spec.get("lags", [8, 16, 32]), capacity)),
            "normalize": bool(spec.get("normalize", True)),
        }
    if feature_family == "pca_raw":
        budget = spec.get("component_budget_by_capacity", {"small": 16, "medium": 48, "large": 96})
        return {"n_components": int(_capacity_value(budget, capacity))}
    raise ValueError(f"Unknown feature family: {feature_family}")


def build_reference_configs(config: dict[str, Any]) -> list[ReferenceConfig]:
    features = config["feature_families"]
    classifiers = config["classifiers"]
    capacities = ["small", "medium", "large"]
    fractions = [float(x) for x in config.get("train_fractions", [1.0])]
    dataset = str(config.get("dataset", "UT_HAR"))
    subset_seed = int(config.get("subset_seed", config.get("random_seed", 42)))
    protocol_hash = stable_hash(config)
    rows: list[ReferenceConfig] = []
    for feat, clf_name, cap in product(features.keys(), classifiers.keys(), capacities):
        clf = classifiers[clf_name]
        base_id = f"{feat}__{clf_name}__{cap}"
        feat_params = feature_params_for(feat, features[feat], cap)
        clf_params = dict(clf.get("capacity_params", {}).get(cap, {}))
        for frac in fractions:
            rows.append(ReferenceConfig(
                dataset=dataset,
                configuration_id=f"{base_id}__frac_{frac:.2f}",
                base_configuration_id=base_id,
                feature_family=feat,
                feature_capacity=cap,
                feature_params=feat_params,
                classifier=clf_name,
                classifier_family=str(clf.get("family", clf_name)),
                classifier_capacity=cap,
                classifier_params=clf_params,
                train_fraction=frac,
                subset_seed=subset_seed,
                protocol_hash=protocol_hash,
            ))
    return rows

def write_manifest(config_path: str | Path, output_path: str | Path) -> list[ReferenceConfig]:
    rows = build_reference_configs(load_yaml(config_path))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(asdict(rows[0]).keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            d = asdict(row)
            d["feature_params"] = json.dumps(d["feature_params"], sort_keys=True)
            d["classifier_params"] = json.dumps(d["classifier_params"], sort_keys=True)
            writer.writerow(d)
    return rows

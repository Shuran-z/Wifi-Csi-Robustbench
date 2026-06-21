#!/usr/bin/env python3
"""Prepare a reproducible NTU-Fi HAR 5-class recovered split.

The Mendeley/SenseFi NTU-Fi HAR archive available during this run has a valid
published SHA256 but an invalid ZIP tail. 7-Zip can recover all training
samples, while the official test split is missing most ``box`` samples. The
benchmark therefore drops ``box`` and keeps the remaining five classes, whose
official test split is complete and balanced.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio


EXPECTED_SHAPE = (342, 2000)
ALL_CLASS_NAMES = ["box", "circle", "clean", "fall", "run", "walk"]
DEFAULT_DROP_CLASSES = ["box"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_mat(path: Path) -> dict[str, Any]:
    data = sio.loadmat(path)
    if "CSIamp" not in data:
        raise KeyError(f"{path} does not contain CSIamp")
    arr = data["CSIamp"]
    if arr.shape != EXPECTED_SHAPE:
        raise ValueError(f"{path} has CSIamp shape {arr.shape}, expected {EXPECTED_SHAPE}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{path} contains NaN or Inf")
    return {
        "shape": list(arr.shape),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def split_train_val(files: list[Path], seed: int, val_count: int) -> dict[str, list[Path]]:
    ordered = sorted(files)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    if val_count <= 0 or val_count >= len(ordered):
        raise ValueError("val count must be positive and smaller than train files")
    return {
        "train": sorted(ordered[:-val_count]),
        "val": sorted(ordered[-val_count:]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/NTU-Fi_HAR")
    ap.add_argument("--output-dir", default="data/splits/NTU_Fi_HAR")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-per-class", type=int, default=31)
    ap.add_argument("--drop-class", action="append", default=DEFAULT_DROP_CLASSES)
    args = ap.parse_args()

    root = Path(args.root)
    train_root = root / "train_amp"
    official_test_root = root / "test_amp"
    if not train_root.exists():
        raise FileNotFoundError(f"missing recovered NTU-Fi train directory: {train_root}")

    drop_classes = sorted(set(args.drop_class or []))
    class_names = [name for name in ALL_CLASS_NAMES if name not in drop_classes]

    rows: list[dict[str, Any]] = []
    validation: dict[str, Any] = {
        "dataset": "NTU-Fi_HAR",
        "root": str(root),
        "seed": args.seed,
        "policy": "drop_box_use_official_nonbox_test",
        "expected_shape": list(EXPECTED_SHAPE),
        "all_class_names": ALL_CLASS_NAMES,
        "class_names": class_names,
        "dropped_classes": drop_classes,
        "drop_reason": "official recovered test split contains only one box sample, while non-box classes each contain 44 test samples",
        "official_test_counts": {},
        "recovered_split_counts": {},
        "invalid": [],
    }

    for class_name in ALL_CLASS_NAMES:
        validation["official_test_counts"][class_name] = len(sorted((official_test_root / class_name).glob("*.mat")))

    for label, class_name in enumerate(class_names):
        train_files = sorted((train_root / class_name).glob("*.mat"))
        if len(train_files) != 156:
            raise ValueError(f"{class_name}: expected 156 train_amp files, found {len(train_files)}")
        official_test_files = sorted((official_test_root / class_name).glob("*.mat"))
        if len(official_test_files) != 44:
            raise ValueError(f"{class_name}: expected 44 official test files after dropping {drop_classes}, found {len(official_test_files)}")

        split = split_train_val(train_files, args.seed + label, args.val_per_class)
        split["test"] = official_test_files
        for split_name, split_files in split.items():
            validation["recovered_split_counts"].setdefault(split_name, {})[class_name] = len(split_files)
            for path in split_files:
                try:
                    stats = validate_mat(path)
                except Exception as exc:  # pragma: no cover - exercised by data integrity runs
                    validation["invalid"].append({"path": str(path), "error": repr(exc)})
                    continue
                rows.append(
                    {
                        "dataset": "NTU-Fi_HAR",
                        "split": split_name,
                        "class_name": class_name,
                        "label": label,
                        "path": str(path.relative_to(root)),
                        "source_split": "test_amp" if split_name == "test" else "train_amp",
                        "sha256": sha256(path),
                        "min": stats["min"],
                        "max": stats["max"],
                        "mean": stats["mean"],
                    }
                )

    if validation["invalid"]:
        raise RuntimeError(f"invalid NTU-Fi files: {validation['invalid'][:3]}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_".join(drop_classes) if drop_classes else "none"
    csv_path = output_dir / f"drop_{suffix}_official_test_seed{args.seed}.csv"
    json_path = output_dir / f"drop_{suffix}_official_test_seed{args.seed}.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    validation["rows"] = len(rows)
    validation["csv"] = str(csv_path)
    json_path.write_text(json.dumps(validation, indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote {len(rows)} rows to {csv_path}")
    print(f"wrote validation summary to {json_path}")


if __name__ == "__main__":
    main()

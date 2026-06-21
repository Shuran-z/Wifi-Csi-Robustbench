#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge sharded classical bank outputs into the final run directory.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--num-shards", type=int, default=4)
    args = ap.parse_args()

    out = Path(args.output_dir)
    shard_root = out / "classical_shards"
    results = out / "results"
    manifests = out / "manifests"
    results.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)

    clean, corr, failures, ckpts, selected = [], [], [], [], []
    for idx in range(args.num_shards):
        shard = shard_root / f"shard_{idx}"
        clean.append(read_csv(shard / "results" / "classical_bank_clean.csv"))
        corr.append(read_csv(shard / "results" / "classical_bank_corruptions_long.csv"))
        failures.append(read_csv(shard / "results" / "classical_bank_failures.csv"))
        ck = read_csv(shard / "manifests" / "classical_bank_checkpoint_manifest.csv")
        ck["shard_index"] = idx
        ck["checkpoint_path"] = ck["checkpoint_path"].map(lambda p: str(Path("classical_shards") / f"shard_{idx}" / p))
        ckpts.append(ck)
        selected.append(read_csv(shard / "manifests" / "classical_bank_selected_manifest.csv"))

    clean_df = pd.concat(clean, ignore_index=True).sort_values("configuration_id")
    corr_df = pd.concat(corr, ignore_index=True).sort_values(["configuration_id", "corruption", "severity", "corruption_seed"])
    fail_df = pd.concat(failures, ignore_index=True)
    ckpt_df = pd.concat(ckpts, ignore_index=True).sort_values("configuration_id")
    selected_df = pd.concat(selected, ignore_index=True).sort_values("configuration_id")

    clean_df.to_csv(results / "classical_bank_clean.csv", index=False)
    corr_df.to_csv(results / "classical_bank_corruptions_long.csv", index=False)
    fail_df.to_csv(results / "classical_bank_failures.csv", index=False)
    ckpt_df.to_csv(manifests / "classical_bank_checkpoint_manifest.csv", index=False)
    selected_df.to_csv(manifests / "classical_bank_selected_manifest.csv", index=False)

    print(f"classical_clean_rows={len(clean_df)}")
    print(f"classical_corruption_rows={len(corr_df)}")
    print(f"classical_failures={len(fail_df)}")
    print(f"classical_checkpoints={len(ckpt_df)}")
    if len(clean_df) != 720:
        raise SystemExit("Expected 720 clean rows")
    if len(corr_df) != 720 * 7 * 5 * 5:
        raise SystemExit("Expected 126000 corruption rows")
    return 0 if fail_df.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())

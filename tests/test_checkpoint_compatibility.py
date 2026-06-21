from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest


def _root() -> Path:
    return Path("outputs_final_complete_20260620_123920")


def test_checkpoint_manifests_have_required_metadata():
    root = _root()
    manifest_dir = root / "manifests"
    if not manifest_dir.exists():
        pytest.skip("final output manifests are not present in this lightweight checkout")
    paths = [
        manifest_dir / "classical_bank_checkpoint_manifest.csv",
        manifest_dir / "deep_bank_checkpoint_manifest.csv",
        manifest_dir / "widar_minimum_bank_checkpoint_manifest.csv",
    ]
    existing = [p for p in paths if p.exists()]
    assert existing, "no checkpoint manifest found"
    for path in existing:
        df = pd.read_csv(path)
        assert len(df) > 0
        assert any(col in df.columns for col in ["checkpoint_path", "path", "checkpoint"])
        assert any(col in df.columns for col in ["run_id", "configuration_id", "model_name"])


def test_checkpoint_sha256_file_is_well_formed():
    root = _root()
    checksum_path = root / "manifests" / "final_all_checkpoint_checksums.sha256"
    if not checksum_path.exists():
        pytest.skip("checkpoint checksum file is not present in this lightweight checkout")
    lines = [line.strip() for line in checksum_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) >= 1
    for line in lines:
        digest, rel = line.split(maxsplit=1)
        assert len(digest) == 64
        int(digest, 16)
        assert rel


def test_available_checkpoint_hashes_match():
    root = _root()
    checksum_path = root / "manifests" / "final_all_checkpoint_checksums.sha256"
    if not checksum_path.exists():
        pytest.skip("checkpoint checksum file is not present in this lightweight checkout")
    checked = 0
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, rel = line.split(maxsplit=1)
        path = root / rel if not rel.startswith(str(root)) else Path(rel)
        if not path.exists():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == expected
        checked += 1
    if checked == 0:
        pytest.skip("no checkpoint binaries are present in lightweight checkout")

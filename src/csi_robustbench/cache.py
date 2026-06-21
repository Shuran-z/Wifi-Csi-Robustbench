from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any

def stable_json_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]

def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def corruption_cache_name(*, dataset: str, corruption: str, severity: str | int, seed: int, config_hash: str, data_hash: str) -> str:
    return f"{dataset}__{corruption}__s{severity}__seed{seed}__cfg{config_hash}__data{data_hash[:12]}.npz"

def validate_cache_metadata(metadata: dict[str, Any], expected: dict[str, Any]) -> None:
    missing = [k for k, v in expected.items() if metadata.get(k) != v]
    if missing:
        raise ValueError("cache metadata mismatch: " + ", ".join(missing))

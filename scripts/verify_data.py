#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/raw")
    ap.add_argument("--output", default="data/checksums.json")
    args = ap.parse_args()
    root = Path(args.data_root)
    rows = {}
    if root.exists():
        for p in sorted(root.rglob("*")):
            if p.is_file():
                rows[str(p)] = sha256(p)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {len(rows)} checksums to {out}")

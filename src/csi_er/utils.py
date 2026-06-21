from __future__ import annotations
import json, os, random, time
from pathlib import Path
from typing import Any
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_config(path: str | Path = 'configs/default.yaml') -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    with p.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def ensure_dirs() -> None:
    for rel in [
        'data/raw','data/processed','data/processed/noisy','data/splits',
        'outputs/checkpoints/classical','outputs/checkpoints/deep','outputs/results',
        'outputs/figures','outputs/tables','outputs/logs','third_party','docs'
    ]:
        (PROJECT_ROOT / rel).mkdir(parents=True, exist_ok=True)

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass

def now_ts() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')

def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def append_log(message: str, path: str | Path = 'docs/experiment_log.md') -> None:
    p = PROJECT_ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a', encoding='utf-8') as f:
        f.write(f"\n- {now_ts()} {message}\n")

def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)

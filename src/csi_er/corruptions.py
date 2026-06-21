from __future__ import annotations
import json
from pathlib import Path
import numpy as np

DEFAULT_GAUSSIAN_ALPHAS = (0.05, 0.10, 0.20, 0.40, 0.60)
DEFAULT_MASK_RATIOS = (0.05, 0.10, 0.20, 0.30, 0.40)
DEFAULT_SCALE_RANGES = ((0.95,1.05),(0.90,1.10),(0.80,1.20),(0.70,1.30),(0.50,1.50))
CORRUPTIONS = ('gaussian','subcarrier_mask','time_dropout','amplitude_scaling')

def _rng(seed): return np.random.default_rng(seed)
def _arr(x): return np.asarray(x)
def _copy_like(x, arr):
    try:
        import torch
        if torch.is_tensor(x): return torch.as_tensor(arr, dtype=x.dtype, device=x.device)
    except Exception: pass
    return arr.astype(np.asarray(x).dtype, copy=False)

def add_gaussian_noise(x, severity, alphas=DEFAULT_GAUSSIAN_ALPHAS, seed=None):
    return apply_corruption(x, 'gaussian', severity, seed=seed, alphas=alphas)

def apply_corruption(x, corruption: str, severity: int, seed: int = 42, **kwargs):
    if severity < 1 or severity > 5: raise ValueError('severity must be 1..5')
    corruption = corruption.lower()
    a = _arr(x).astype(np.float32, copy=True)
    r = _rng(seed)
    if corruption == 'gaussian':
        alphas = kwargs.get('alphas', DEFAULT_GAUSSIAN_ALPHAS); alpha = float(alphas[severity-1])
        std = np.maximum(a.std(axis=tuple(range(1, a.ndim)), keepdims=True), 1e-8)
        out = a + r.normal(0, 1, size=a.shape).astype(np.float32) * std * alpha
    elif corruption == 'subcarrier_mask':
        ratios = kwargs.get('ratios', DEFAULT_MASK_RATIOS); ratio = float(ratios[severity-1]); fill = float(kwargs.get('fill_value', 0.0))
        out = a.copy(); n, f = a.shape[0], a.shape[-1]; k = max(1, int(round(f * ratio)))
        for i in range(n): out[i, :, :, r.choice(f, size=min(k, f), replace=False)] = fill
    elif corruption == 'time_dropout':
        ratios = kwargs.get('ratios', DEFAULT_MASK_RATIOS); ratio = float(ratios[severity-1]); fill = float(kwargs.get('fill_value', 0.0))
        out = a.copy(); n, t = a.shape[0], a.shape[-2]; k = max(1, int(round(t * ratio)))
        for i in range(n): out[i, :, r.choice(t, size=min(k, t), replace=False), :] = fill
    elif corruption == 'amplitude_scaling':
        ranges = kwargs.get('ranges', DEFAULT_SCALE_RANGES); lo, hi = ranges[severity-1]
        factors = r.uniform(float(lo), float(hi), size=(a.shape[0],) + (1,) * (a.ndim - 1)).astype(np.float32)
        out = a * factors
    else:
        raise ValueError(f'unknown corruption: {corruption}')
    return _copy_like(x, out)

def corruption_kwargs(config: dict, corruption: str):
    c = config.get('corruptions', {})
    d = c.get(corruption, {})
    return dict(d)

def get_or_create_corruption_cache(X, y, config: dict, corruption: str, severity: int, project_root: Path, dataset='UT_HAR'):
    c = config.get('corruptions', {})
    cache_dir = project_root / c.get('cache_dir', 'data/processed/corruptions')
    cache_dir.mkdir(parents=True, exist_ok=True)
    seed = int(c.get('seed', 42)) + severity + 1000 * list(CORRUPTIONS).index(corruption)
    p = cache_dir / f'{dataset}_{corruption}_s{severity}_seed{seed}.npz'
    if not p.exists():
        Xc = apply_corruption(X, corruption, severity, seed=seed, **corruption_kwargs(config, corruption))
        meta = {'dataset': dataset, 'corruption': corruption, 'severity': severity, 'seed': seed, 'kwargs': corruption_kwargs(config, corruption)}
        np.savez_compressed(p, X=Xc.astype(np.float32), y=y, metadata=json.dumps(meta, ensure_ascii=False))
    return p, seed

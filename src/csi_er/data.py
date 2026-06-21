from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .utils import PROJECT_ROOT, ensure_dirs, set_seed, write_json

CLASS_NAMES = ['lie down', 'fall', 'walk', 'pickup', 'run', 'sit down', 'stand up']

def _read_csv_numeric(path: Path) -> np.ndarray:
    # SenseFi UT-HAR files keep a .csv suffix but are NumPy .npy binaries.
    try:
        return np.load(path, allow_pickle=False)
    except Exception:
        pass
    try:
        return pd.read_csv(path, header=None).values
    except Exception:
        return np.loadtxt(path, delimiter=',')

def _reshape_ut_har_data(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim == 2 and arr.shape[1] == 250 * 90:
        return arr.reshape(-1, 250, 90)
    if arr.ndim == 2 and arr.shape[0] == 250 and arr.shape[1] == 90:
        return arr.reshape(1, 250, 90)
    if arr.ndim == 3 and arr.shape[-2:] == (250, 90):
        return arr
    if arr.ndim == 4:
        if arr.shape[1] == 1:
            return arr[:, 0]
        return arr.reshape(arr.shape[0], arr.shape[-2], arr.shape[-1])
    raise ValueError(f'Cannot reshape UT-HAR array with shape {arr.shape}')

def find_ut_har_root(raw_root: str | Path = 'data/raw') -> Path | None:
    base = PROJECT_ROOT / raw_root if not Path(raw_root).is_absolute() else Path(raw_root)
    candidates = [base / 'UT_HAR', base / 'Data' / 'UT_HAR', base / 'Benchmark' / 'Data' / 'UT_HAR']
    candidates += list(base.glob('**/UT_HAR')) if base.exists() else []
    for c in candidates:
        if (c / 'data').is_dir() and (c / 'label').is_dir():
            return c
    return None

def load_raw_ut_har(raw_root: str | Path = 'data/raw'):
    ut = find_ut_har_root(raw_root)
    if ut is None:
        raise FileNotFoundError('UT_HAR/data and UT_HAR/label not found under data/raw')
    xs, ys, split_hint = [], [], []
    data_files = sorted((ut / 'data').glob('*.csv'))
    label_files = {p.stem: p for p in (ut / 'label').glob('*.csv')}
    for dp in data_files:
        stem = dp.stem
        lp = label_files.get(stem)
        if lp is None and stem.startswith('X_'):
            lp = label_files.get('y_' + stem[2:])
        if lp is None and stem.startswith('x_'):
            lp = label_files.get('y_' + stem[2:])
        if lp is None and len(label_files) == len(data_files):
            lp = label_files.get(sorted(label_files)[len(xs)])
        if lp is None:
            continue
        x = _reshape_ut_har_data(_read_csv_numeric(dp))
        y = np.asarray(_read_csv_numeric(lp)).reshape(-1).astype(int)
        if len(y) != len(x):
            if len(y) == 1:
                y = np.repeat(y, len(x))
            else:
                raise ValueError(f'label count mismatch for {dp}: {len(x)} vs {len(y)}')
        xs.append(x); ys.append(y); split_hint += [stem.lower()] * len(y)
    if not xs:
        raise FileNotFoundError(f'No matched UT-HAR csv files in {ut}')
    X = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.int64)
    return X, y, np.array(split_hint)

def expand_channel(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 3:
        X = X[:, None, :, :]
    return X

def normalize_train_test(X_train, X_test, out_path='data/processed/ut_har_norm_stats.json'):
    mean = float(X_train.mean())
    std = float(X_train.std() + 1e-8)
    stats = {'mean': mean, 'std': std}
    write_json(PROJECT_ROOT / out_path, stats)
    return (X_train - mean) / std, (X_test - mean) / std, stats

def subset_per_class(X, y, per_class, seed=42):
    if per_class is None:
        return X, y
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(y):
        cand = np.where(y == c)[0]
        rng.shuffle(cand)
        idx.extend(cand[:min(per_class, len(cand))])
    idx = np.array(sorted(idx))
    return X[idx], y[idx]

def make_synthetic_ut_har(train_per_class=18, test_per_class=8, seed=42):
    rng = np.random.default_rng(seed)
    T, F, K = 250, 90, len(CLASS_NAMES)
    def one(cls, n):
        t = np.linspace(0, 1, T, dtype=np.float32)[:, None]
        f = np.linspace(0, 1, F, dtype=np.float32)[None, :]
        base = np.sin((cls + 1) * np.pi * t) + np.cos((cls + 2) * np.pi * f)
        amp = 0.7 + cls * 0.08
        samples = amp * base[None, :, :] + rng.normal(0, 0.25, size=(n, T, F))
        return samples.astype(np.float32), np.full(n, cls, dtype=np.int64)
    xs_tr, ys_tr, xs_te, ys_te = [], [], [], []
    for c in range(K):
        x, y = one(c, train_per_class); xs_tr.append(x); ys_tr.append(y)
        x, y = one(c, test_per_class); xs_te.append(x); ys_te.append(y)
    return np.concatenate(xs_tr), np.concatenate(ys_tr), np.concatenate(xs_te), np.concatenate(ys_te)

def prepare_ut_har(config: dict, smoke=False, synthetic=False):
    ensure_dirs(); set_seed(config['data'].get('random_seed', 42))
    seed = config['data'].get('random_seed', 42)
    if synthetic:
        X_train, y_train, X_test, y_test = make_synthetic_ut_har(seed=seed)
        source = 'synthetic_smoke'
    else:
        X, y, hints = load_raw_ut_har(config['paths']['raw_data'])
        hint_str = hints.astype(str)
        train_mask = np.char.find(hint_str, 'train') >= 0
        test_mask = (np.char.find(hint_str, 'test') >= 0) | (np.char.find(hint_str, 'val') >= 0)
        if train_mask.any() and test_mask.any():
            X_train, y_train = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
        source = 'UT_HAR_official_or_user_downloaded'
    if smoke or config['data'].get('use_subset'):
        X_train, y_train = subset_per_class(X_train, y_train, config['data'].get('subset_train_per_class') or (8 if smoke else None), seed)
        X_test, y_test = subset_per_class(X_test, y_test, config['data'].get('subset_test_per_class') or (4 if smoke else None), seed)
    X_train, X_test = expand_channel(X_train), expand_channel(X_test)
    X_train, X_test, stats = normalize_train_test(X_train, X_test)
    out = PROJECT_ROOT / 'data/processed' / ('ut_har_smoke.npz' if smoke else 'ut_har.npz')
    np.savez_compressed(out, X_train=X_train.astype(np.float32), y_train=y_train, X_test=X_test.astype(np.float32), y_test=y_test, class_names=np.array(CLASS_NAMES), source=source)
    meta = {'source': source, 'smoke': bool(smoke), 'synthetic': bool(synthetic), 'train_shape': list(X_train.shape), 'test_shape': list(X_test.shape), 'classes': CLASS_NAMES, 'norm': stats}
    write_json(PROJECT_ROOT / 'data/processed' / ('ut_har_smoke_meta.json' if smoke else 'ut_har_meta.json'), meta)
    return meta

def load_processed(smoke=False):
    p = PROJECT_ROOT / 'data/processed' / ('ut_har_smoke.npz' if smoke else 'ut_har.npz')
    if not p.exists():
        raise FileNotFoundError(f'{p} missing. Run scripts/02_prepare_ut_har.py first.')
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files}

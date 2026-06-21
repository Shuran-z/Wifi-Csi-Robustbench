from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, norm, spearmanr, t

CORRUPTION_FAMILIES = {
    "noise": ["gaussian_snr"],
    "frequency_missingness": ["subcarrier_mask", "contiguous_subcarrier_block"],
    "temporal_missingness": ["time_dropout", "burst_time_dropout"],
    "gain_variation": ["amplitude_scaling", "smooth_gain_drift"],
}

@dataclass
class FitResult:
    slope: float
    intercept: float
    r2: float
    rmse: float
    mae: float
    n: int

def clip_metric(p, n_test: int | None = None, epsilon: float | None = None):
    p = np.asarray(p, dtype=float)
    eps = float(epsilon if epsilon is not None else 0.5 / max(int(n_test or 1), 1))
    return np.clip(p, eps, 1.0 - eps)

def probit(p, n_test: int | None = None, epsilon: float | None = None):
    return norm.ppf(clip_metric(p, n_test=n_test, epsilon=epsilon))

def inv_probit(z):
    return norm.cdf(np.asarray(z, dtype=float))

def fit_ols(x, y) -> FitResult:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 2:
        raise ValueError("x and y must be one-dimensional arrays with at least two points")
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return FitResult(float(slope), float(intercept), 1.0 - ss_res / (ss_tot + 1e-12), float(np.sqrt(np.mean((y - pred) ** 2))), float(np.mean(np.abs(y - pred))), int(len(x)))

def grouped_cv_r2(x, y, groups) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    preds = np.full_like(y, fill_value=np.nan, dtype=float)
    for g in np.unique(groups):
        train = groups != g
        test = groups == g
        if train.sum() < 2:
            continue
        fit = fit_ols(x[train], y[train])
        preds[test] = fit.slope * x[test] + fit.intercept
    ok = np.isfinite(preds)
    if ok.sum() < 2:
        return float("nan")
    ss_res = float(np.sum((y[ok] - preds[ok]) ** 2))
    ss_tot = float(np.sum((y[ok] - np.mean(y[ok])) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-12)

def rank_correlations(a, b) -> dict[str, float]:
    return {"spearman": float(spearmanr(a, b).statistic), "kendall": float(kendalltau(a, b).statistic)}

def seed_level_mpc(
    corr: pd.DataFrame,
    *,
    group_cols: list[str],
    metrics: list[str],
    family_map: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Aggregate corrupted metrics to corruption-seed-level mPC.

    Severity and corruption-condition variation is part of the benchmark
    definition, not iid uncertainty. This function first averages severities
    within each concrete corruption for each corruption seed, then computes the
    family-balanced primary mPC and flat-7 secondary mPC.
    """
    family_map = family_map or CORRUPTION_FAMILIES
    required = set(group_cols + ["corruption", "severity", "corruption_seed"] + metrics)
    missing = required.difference(corr.columns)
    if missing:
        raise KeyError(f"missing columns for seed-level mPC: {sorted(missing)}")
    rows: list[dict] = []
    for keys, g in corr.groupby(group_cols + ["corruption_seed"], dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols + ["corruption_seed"], keys))
        for metric in metrics:
            per_corr = g.groupby("corruption", dropna=False)[metric].mean()
            row[f"{metric}_flat7_seed"] = float(per_corr.mean())
            fam_values = []
            for corrs in family_map.values():
                present = [c for c in corrs if c in per_corr.index]
                if not present:
                    raise ValueError(f"no corruption present for family {corrs}")
                fam_values.append(float(per_corr.loc[present].mean()))
            row[f"{metric}_family_seed"] = float(np.mean(fam_values))
        rows.append(row)
    return pd.DataFrame(rows)

def summarize_seed_mpc(seed_scores: pd.DataFrame, *, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    required = set(group_cols + ["corruption_seed"])
    for metric in metrics:
        required.update({f"{metric}_family_seed", f"{metric}_flat7_seed"})
    missing = required.difference(seed_scores.columns)
    if missing:
        raise KeyError(f"missing columns for seed mPC summary: {sorted(missing)}")
    for keys, g in seed_scores.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_corruption_seeds"] = int(g["corruption_seed"].nunique())
        for metric in metrics:
            for flavor in ["family", "flat7"]:
                vals = g[f"{metric}_{flavor}_seed"].to_numpy(dtype=float)
                mean = float(np.mean(vals))
                std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                half = float(t.ppf(0.975, df=len(vals) - 1) * std / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
                prefix = f"mPC_{flavor}_{metric}"
                row[prefix] = mean
                row[f"{prefix}_std"] = std
                row[f"{prefix}_ci_low"] = mean - half
                row[f"{prefix}_ci_high"] = mean + half
        rows.append(row)
    return pd.DataFrame(rows)

def pair_clean_summary(
    clean: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    keys: list[str],
    clean_metric: str,
    summary_metric: str,
) -> pd.DataFrame:
    if clean.duplicated(keys).any():
        raise ValueError(f"clean metrics contain duplicate keys: {keys}")
    if summary.duplicated(keys).any():
        raise ValueError(f"summary metrics contain duplicate keys: {keys}")
    paired = clean.merge(summary, on=keys, how="inner", validate="one_to_one")
    if len(paired) != len(clean) or len(paired) != len(summary):
        raise ValueError("clean and summary metrics do not match one-to-one")
    for col in [clean_metric, summary_metric]:
        if col not in paired.columns:
            raise KeyError(col)
        if not np.isfinite(paired[col]).all():
            raise ValueError(f"non-finite metric column: {col}")
    return paired

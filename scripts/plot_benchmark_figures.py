#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FAMILY_MAP = {
    "gaussian_snr": "Noise",
    "subcarrier_mask": "Frequency missingness",
    "contiguous_subcarrier_block": "Frequency missingness",
    "time_dropout": "Temporal missingness",
    "burst_time_dropout": "Temporal missingness",
    "amplitude_scaling": "Gain variation",
    "smooth_gain_drift": "Gain variation",
}
FAMILY_ORDER = ["Noise", "Frequency missingness", "Temporal missingness", "Gain variation"]
DEEP_ORDER = ["MLP", "CNN", "GRU", "LSTM", "CNN-GRU", "Transformer"]
SIZE_ORDER = ["small", "medium", "large"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate benchmark figures from final result CSVs.")
    p.add_argument(
        "--input",
        required=True,
        help="Repository root, final output directory, or result CSV directory.",
    )
    p.add_argument("--output", help="Figure directory. Defaults to INPUT/figures/final for repository layout.")
    return p.parse_args()


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def resolve_results_dir(input_dir: Path) -> Path:
    candidates = [
        input_dir / "results" / "final",
        input_dir / "results",
        input_dir,
    ]
    for candidate in candidates:
        if (candidate / "combined-clean-metrics.csv").exists() or (candidate / "combined_clean_metrics.csv").exists():
            return candidate
    raise FileNotFoundError(f"Could not find final result CSVs under {input_dir}")


def result_csv(results_dir: Path, stem: str) -> pd.DataFrame:
    snake = results_dir / f"{stem}.csv"
    kebab = results_dir / f"{stem.replace('_', '-')}.csv"
    if kebab.exists():
        return pd.read_csv(kebab)
    return read_csv(snake)


def default_figure_dir(input_dir: Path) -> Path:
    if (input_dir / "results" / "final").exists():
        return input_dir / "figures" / "final"
    return input_dir / "figures"


def family_label(value: str) -> str:
    text = str(value)
    if text == "classical":
        return "Classical feature families"
    if text.lower() == "cnn_gru":
        return "CNN-GRU"
    if text.lower() in {"mlp", "cnn", "gru", "lstm", "transformer"}:
        return text.upper() if text.lower() in {"mlp", "cnn", "gru", "lstm"} else "Transformer"
    return text.replace("_", " ").title()


def ci95(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce").dropna()
    if len(x) <= 1:
        return 0.0
    return float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))


def fig01_overview(figdir: Path, sourcedir: Path) -> None:
    steps = [
        "UT-HAR data",
        "Model banks",
        "Corruption suite",
        "Family-balanced mPC",
        "CSI-ER validity gate",
        "Benchmark conclusions",
    ]
    source = pd.DataFrame({"step": range(1, len(steps) + 1), "label": steps})
    source.to_csv(sourcedir / "benchmark-fig-01-overview.csv", index=False)
    fig, ax = plt.subplots(figsize=(11, 2.4))
    xs = np.linspace(0.06, 0.94, len(steps))
    for x, label in zip(xs, steps):
        ax.text(
            x,
            0.55,
            label,
            ha="center",
            va="center",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#4C78A8", "lw": 1.0},
        )
    for a, b in zip(xs[:-1], xs[1:]):
        ax.annotate("", xy=(b - 0.055, 0.55), xytext=(a + 0.055, 0.55), arrowprops={"arrowstyle": "->", "lw": 1})
    ax.set_title("WiFi CSI RobustBench final evaluation flow", fontsize=11)
    ax.axis("off")
    save_figure(fig, figdir / "benchmark-fig-01-overview")


def fig02_family_robustness(summary: pd.DataFrame, figdir: Path, sourcedir: Path) -> None:
    df = summary.copy()
    df["family_label"] = df["architecture_family"].map(family_label)
    group = (
        df.groupby("family_label")["mPC_family_macro_f1"]
        .agg(mean="mean", std="std", count="count", ci95=ci95)
        .reset_index()
        .sort_values("mean")
    )
    group.to_csv(sourcedir / "benchmark-fig-02-family-robustness-macro-f1.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(group["family_label"], group["mean"], xerr=group["ci95"], color="#4C78A8", alpha=0.88)
    ax.set_xlabel("Family-balanced mPC Macro-F1")
    ax.set_ylabel("Model family")
    ax.set_title("Model-family robustness on UT-HAR common corruptions")
    ax.grid(axis="x", alpha=0.3)
    save_figure(fig, figdir / "benchmark-fig-02-family-robustness-macro-f1")


def fig03_architecture_size(summary: pd.DataFrame, figdir: Path, sourcedir: Path) -> None:
    df = summary[~summary["is_classical"].astype(str).str.lower().isin(["true", "1"])].copy()
    df["arch"] = df["architecture_family"].map(family_label)
    pivot = df.pivot_table(index="arch", columns="model_size", values="mPC_family_macro_f1", aggfunc="mean")
    pivot = pivot.reindex(index=[x for x in DEEP_ORDER if x in pivot.index], columns=[x for x in SIZE_ORDER if x in pivot.columns])
    pivot.to_csv(sourcedir / "benchmark-fig-03-architecture-size-heatmap.csv")
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    im = ax.imshow(pivot.values, cmap="YlGnBu", aspect="auto", vmin=np.nanmin(pivot.values), vmax=np.nanmax(pivot.values))
    ax.set_xticks(range(len(pivot.columns)), [str(c).title() for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_xlabel("Model size")
    ax.set_ylabel("Deep architecture")
    ax.set_title("Deep bank architecture-size robustness")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.3f}" if np.isfinite(val) else "NA", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="mPC Macro-F1")
    save_figure(fig, figdir / "benchmark-fig-03-architecture-size-heatmap")


def fig04_retention(clean: pd.DataFrame, corr: pd.DataFrame, figdir: Path, sourcedir: Path) -> None:
    c = corr.copy()
    c["corruption_family"] = c["corruption"].map(FAMILY_MAP)
    fam = c.groupby(["run_id", "corruption_family"], as_index=False)["macro_f1"].mean()
    base = clean[["run_id", "architecture_family", "is_classical", "clean_macro_f1"]].copy()
    paired = fam.merge(base, on="run_id", validate="many_to_one")
    paired["model_family"] = paired["architecture_family"].map(family_label)
    paired["retention"] = paired["macro_f1"] / paired["clean_macro_f1"].replace(0, np.nan)
    table = paired.pivot_table(index="model_family", columns="corruption_family", values="retention", aggfunc="mean")
    table = table.reindex(columns=FAMILY_ORDER)
    table.to_csv(sourcedir / "benchmark-fig-04-corruption-family-retention.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(table.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=np.nanmax(table.values))
    ax.set_xticks(range(len(table.columns)), table.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(table.index)), table.index)
    ax.set_title("Macro-F1 retention by corruption family")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            val = table.values[i, j]
            ax.text(j, i, f"{val:.2f}" if np.isfinite(val) else "NA", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Retention")
    save_figure(fig, figdir / "benchmark-fig-04-corruption-family-retention")


def fig05_validity(results_dir: Path, figdir: Path, sourcedir: Path) -> None:
    rows = []
    for metric in ["macro_f1", "accuracy"]:
        snake = results_dir / f"combined_validity_gated_csi_er_{metric}.csv"
        kebab = results_dir / f"combined-validity-gated-csi-er-{metric.replace('_', '-')}.csv"
        path = kebab if kebab.exists() else snake
        if path.exists():
            df = pd.read_csv(path)
            for status, count in df["status"].value_counts().items():
                rows.append({"metric": "Macro-F1" if metric == "macro_f1" else "Accuracy", "status": status, "count": int(count)})
    source = pd.DataFrame(rows)
    source.to_csv(sourcedir / "benchmark-fig-05-validity-gate-summary.csv", index=False)
    pivot = source.pivot_table(index="metric", columns="status", values="count", aggfunc="sum", fill_value=0)
    pivot = pivot.reindex(columns=[c for c in ["VALID", "EXPLORATORY", "OUT_OF_SUPPORT", "INVALID_FIT"] if c in pivot.columns])
    fig, ax = plt.subplots(figsize=(7, 4))
    pivot.plot(kind="bar", stacked=True, ax=ax, color=["#59A14F", "#F28E2B", "#E15759", "#9C755F"][: len(pivot.columns)])
    ax.set_ylabel("Run count")
    ax.set_xlabel("CSI-ER metric")
    ax.set_title("Validity gate status counts")
    ax.legend(title="Status", frameon=False)
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, figdir / "benchmark-fig-05-validity-gate-summary")


def fig06_widar(input_dir: Path, results_dir: Path, figdir: Path, sourcedir: Path) -> None:
    df = result_csv(results_dir, "widar_minimum_bank_natural_shift")
    summary_path = input_dir / "artifacts" / "widar_natural_shift_summary.json"
    text = "Spearman unavailable"
    if summary_path.exists():
        info = json.loads(summary_path.read_text(encoding="utf-8"))
        stat = info.get("synthetic_vs_natural_spearman", {}).get("macro_f1", {})
        if stat:
            text = (
                f"Spearman={stat.get('spearman', float('nan')):.3f}, "
                f"95% CI [{stat.get('ci_low', float('nan')):.3f}, {stat.get('ci_high', float('nan')):.3f}], n=6"
            )
    source = df[
        [
            "model_name",
            "synthetic_mPC_macro_f1",
            "natural_test_macro_f1",
            "clean_val_macro_f1",
            "feature_family",
            "classifier",
        ]
    ].copy()
    source.to_csv(sourcedir / "benchmark-fig-06-widar-synthetic-natural-scatter.csv", index=False)
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    ax.scatter(df["synthetic_mPC_macro_f1"], df["natural_test_macro_f1"], s=70, color="#4E79A7")
    for _, row in df.iterrows():
        ax.annotate(str(row["classifier"]).replace("_", " "), (row["synthetic_mPC_macro_f1"], row["natural_test_macro_f1"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Synthetic mPC Macro-F1")
    ax.set_ylabel("Natural-shift Macro-F1")
    ax.set_title("Widar minimum bank: synthetic vs natural shift")
    ax.text(0.02, 0.98, text + "\nExploratory minimum bank", transform=ax.transAxes, va="top", fontsize=8, bbox={"fc": "white", "ec": "#BBBBBB"})
    ax.grid(alpha=0.3)
    save_figure(fig, figdir / "benchmark-fig-06-widar-synthetic-natural-scatter")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    results_dir = resolve_results_dir(input_dir)
    figdir = Path(args.output) if args.output else default_figure_dir(input_dir)
    sourcedir = figdir / "source-data"
    sourcedir.mkdir(parents=True, exist_ok=True)

    clean = result_csv(results_dir, "combined_clean_metrics")
    corr = result_csv(results_dir, "combined_corruption_metrics_long")
    summary = result_csv(results_dir, "combined_robustness_summary")

    fig01_overview(figdir, sourcedir)
    fig02_family_robustness(summary, figdir, sourcedir)
    fig03_architecture_size(summary, figdir, sourcedir)
    fig04_retention(clean, corr, figdir, sourcedir)
    fig05_validity(results_dir, figdir, sourcedir)
    fig06_widar(input_dir, results_dir, figdir, sourcedir)

    made = sorted(p.name for p in figdir.glob("benchmark-fig-*.png"))
    (figdir / "benchmark-figures-readme.md").write_text(
        "# Benchmark Figures\n\n"
        + "\n".join(f"- `{name}`" for name in made)
        + "\n\nSource CSV files are in `source-data/`.\n",
        encoding="utf-8",
    )
    print(f"wrote {len(made)} PNG figures and matching PDFs to {figdir}")


if __name__ == "__main__":
    main()

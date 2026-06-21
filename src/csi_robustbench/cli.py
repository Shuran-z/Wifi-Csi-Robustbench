from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .reference_bank import write_manifest


REQUIRED_RESULT_FILES = [
    "combined_clean_metrics.csv",
    "combined_corruption_metrics_long.csv",
    "combined_robustness_summary.csv",
]


def _results_dir(root: Path) -> Path:
    return root / "results" if (root / "results").exists() else root


def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def summarize_results(input_dir: Path, output_dir: Path) -> dict[str, object]:
    res = _results_dir(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clean = _read_csv_if_exists(res / "combined_clean_metrics.csv")
    corr = _read_csv_if_exists(res / "combined_corruption_metrics_long.csv")
    robust = _read_csv_if_exists(res / "combined_robustness_summary.csv")
    if clean is None and (res / "clean.csv").exists():
        clean = pd.read_csv(res / "clean.csv")
    if corr is None and (res / "corruptions_long.csv").exists():
        corr = pd.read_csv(res / "corruptions_long.csv")
    if clean is None or corr is None:
        raise FileNotFoundError(
            f"expected result CSVs under {res}: combined_clean_metrics.csv and combined_corruption_metrics_long.csv"
        )

    summary: dict[str, object] = {
        "input": str(input_dir),
        "clean_rows": int(len(clean)),
        "corruption_rows": int(len(corr)),
        "datasets": sorted(clean["dataset"].dropna().astype(str).unique().tolist()) if "dataset" in clean else [],
    }
    if "run_id" in clean:
        summary["unique_runs"] = int(clean["run_id"].nunique())
    if "corruption" in corr:
        summary["corruptions"] = sorted(corr["corruption"].dropna().astype(str).unique().tolist())
    if "severity" in corr:
        summary["severities"] = sorted(int(x) for x in corr["severity"].dropna().unique())
    if "corruption_seed" in corr:
        summary["corruption_seeds"] = sorted(int(x) for x in corr["corruption_seed"].dropna().unique())
    if robust is not None:
        for metric in ["mPC_family_macro_f1", "mPC_family_accuracy", "mPC_family_balanced_accuracy"]:
            if metric in robust:
                summary[f"mean_{metric}"] = float(robust[metric].mean())
                summary[f"max_{metric}"] = float(robust[metric].max())

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = [{"key": key, "value": json.dumps(value, ensure_ascii=True)} for key, value in summary.items()]
    pd.DataFrame(rows).to_csv(output_dir / "summary.csv", index=False)
    md = ["# Result Summary", ""]
    for key, value in summary.items():
        md.append(f"- `{key}`: `{value}`")
    (output_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return summary


def validate_results(input_dir: Path, checkpoint_dir: Path | None = None) -> dict[str, object]:
    res = _results_dir(input_dir)
    missing = [name for name in REQUIRED_RESULT_FILES if not (res / name).exists()]
    clean_path = res / "combined_clean_metrics.csv"
    corr_path = res / "combined_corruption_metrics_long.csv"
    if not clean_path.exists() and (res / "clean.csv").exists():
        clean_path = res / "clean.csv"
    if not corr_path.exists() and (res / "corruptions_long.csv").exists():
        corr_path = res / "corruptions_long.csv"
    if not clean_path.exists() or not corr_path.exists():
        raise FileNotFoundError(f"missing clean/corruption CSVs under {res}")

    clean = pd.read_csv(clean_path)
    corr = pd.read_csv(corr_path)
    report: dict[str, object] = {
        "input": str(input_dir),
        "missing_standard_files": missing,
        "clean_rows": int(len(clean)),
        "corruption_rows": int(len(corr)),
        "errors": [],
        "warnings": [],
    }
    errors: list[str] = report["errors"]  # type: ignore[assignment]
    warnings: list[str] = report["warnings"]  # type: ignore[assignment]

    if corr.empty:
        errors.append("corruption table is empty")
    metric_cols = [c for c in ["accuracy", "macro_f1", "balanced_accuracy"] if c in corr]
    for col in metric_cols:
        vals = pd.to_numeric(corr[col], errors="coerce")
        if vals.isna().any():
            errors.append(f"{col} contains NaN/non-numeric values")
        if ((vals < 0) | (vals > 1)).any():
            errors.append(f"{col} contains values outside [0, 1]")

    key_cols = ["dataset", "run_id", "corruption", "severity", "corruption_seed"]
    if all(c in corr.columns for c in key_cols):
        dup = corr.duplicated(key_cols).sum()
        if dup:
            errors.append(f"duplicate corruption condition rows: {int(dup)}")
        per_run = corr.groupby(["dataset", "run_id"]).size()
        if not per_run.empty:
            report["min_rows_per_run"] = int(per_run.min())
            report["max_rows_per_run"] = int(per_run.max())
            expected = int(corr["corruption"].nunique() * corr["severity"].nunique() * corr["corruption_seed"].nunique())
            report["expected_rows_per_run_from_table"] = expected
            if per_run.min() != per_run.max():
                errors.append("not every run has the same number of corruption rows")
            if expected and not (per_run == expected).all():
                errors.append("at least one run is missing a corruption/severity/seed combination")
    else:
        warnings.append(f"cannot check corruption completeness; missing one of {key_cols}")

    if checkpoint_dir is not None and not checkpoint_dir.exists():
        warnings.append(f"checkpoint directory does not exist: {checkpoint_dir}")

    report["ok"] = not errors
    (input_dir / "validation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if errors:
        raise RuntimeError("; ".join(errors))
    return report


def plot_final(input_dir: Path, output_dir: Path) -> list[str]:
    res = _results_dir(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    robust = _read_csv_if_exists(res / "combined_robustness_summary.csv")
    clean = _read_csv_if_exists(res / "combined_clean_metrics.csv")
    corr = _read_csv_if_exists(res / "combined_corruption_metrics_long.csv")
    if robust is None and clean is None:
        raise FileNotFoundError(f"no plottable result CSV found under {res}")
    made: list[str] = []

    if robust is not None and "mPC_family_macro_f1" in robust:
        group_col = "architecture_family" if "architecture_family" in robust else "model_group"
        data = robust.groupby(group_col, dropna=False)["mPC_family_macro_f1"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(7, 4))
        data.plot(kind="barh", ax=ax, color="#4C78A8")
        ax.set_xlabel("Family-balanced mPC Macro-F1")
        ax.set_ylabel("Model family")
        ax.set_title("Robustness by model family")
        ax.grid(axis="x", alpha=0.3)
        for ext in ["png", "pdf"]:
            path = output_dir / f"robustness_by_family.{ext}"
            fig.savefig(path, dpi=600 if ext == "png" else None, bbox_inches="tight")
            made.append(str(path))
        plt.close(fig)
        data.reset_index(name="mean_mPC_family_macro_f1").to_csv(output_dir / "robustness_by_family_source.csv", index=False)

    if clean is not None and corr is not None and {"clean_macro_f1", "run_id"}.issubset(clean.columns):
        corr_metric = "macro_f1" if "macro_f1" in corr else None
        if corr_metric and "run_id" in corr:
            mpc = corr.groupby("run_id", as_index=False)[corr_metric].mean().rename(columns={corr_metric: "mean_corrupted_macro_f1"})
            paired = clean[["run_id", "clean_macro_f1"]].merge(mpc, on="run_id", validate="one_to_one")
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(paired["clean_macro_f1"], paired["mean_corrupted_macro_f1"], s=12, alpha=0.55, color="#59A14F")
            ax.set_xlabel("Clean Macro-F1")
            ax.set_ylabel("Mean corrupted Macro-F1")
            ax.set_title("Clean vs corrupted Macro-F1")
            ax.grid(alpha=0.3)
            for ext in ["png", "pdf"]:
                path = output_dir / f"clean_vs_corrupted_macro_f1.{ext}"
                fig.savefig(path, dpi=600 if ext == "png" else None, bbox_inches="tight")
                made.append(str(path))
            plt.close(fig)
            paired.to_csv(output_dir / "clean_vs_corrupted_macro_f1_source.csv", index=False)
    return made


def unavailable(command: str) -> None:
    raise RuntimeError(
        f"{command} requires raw datasets and/or external checkpoints. "
        "See REPRODUCIBILITY.md for the Level 2/Level 3 workflow."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="csi-robustbench")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-reference-bank")
    p.add_argument("--config", default="configs/reference_bank.yaml")
    p.add_argument("--output", default="artifacts/reference_bank_manifest.csv")

    p = sub.add_parser("summarize-results")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)

    p = sub.add_parser("plot-final")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)

    p = sub.add_parser("validate-results")
    p.add_argument("--input", required=True)
    p.add_argument("--checkpoint-dir")

    for name in ["prepare-data", "train", "evaluate", "fit-er", "plot", "reproduce"]:
        q = sub.add_parser(name)
        q.add_argument("--data-root", default="data")
        q.add_argument("--output-dir", default="outputs_research")
        q.add_argument("--checkpoint-dir", default="checkpoints")
        q.add_argument("--config")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build-reference-bank":
            rows = write_manifest(args.config, args.output)
            print(f"wrote {len(rows)} reference rows to {args.output}")
        elif args.command == "summarize-results":
            summary = summarize_results(Path(args.input), Path(args.output))
            print(json.dumps(summary, indent=2, sort_keys=True))
        elif args.command == "plot-final":
            made = plot_final(Path(args.input), Path(args.output))
            print("\n".join(made))
        elif args.command == "validate-results":
            report = validate_results(Path(args.input), Path(args.checkpoint_dir) if args.checkpoint_dir else None)
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            unavailable(args.command)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

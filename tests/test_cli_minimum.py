from __future__ import annotations

import pandas as pd
import pytest

from csi_robustbench.cli import main


def write_synthetic_results(root):
    res = root / "results"
    res.mkdir()
    clean = pd.DataFrame(
        [
            {"dataset": "toy", "run_id": "r1", "clean_macro_f1": 0.8},
            {"dataset": "toy", "run_id": "r2", "clean_macro_f1": 0.7},
        ]
    )
    corr = pd.DataFrame(
        [
            {"dataset": "toy", "run_id": "r1", "corruption": "noise", "severity": 1, "corruption_seed": 42, "macro_f1": 0.6},
            {"dataset": "toy", "run_id": "r1", "corruption": "noise", "severity": 1, "corruption_seed": 123, "macro_f1": 0.5},
            {"dataset": "toy", "run_id": "r2", "corruption": "noise", "severity": 1, "corruption_seed": 42, "macro_f1": 0.4},
            {"dataset": "toy", "run_id": "r2", "corruption": "noise", "severity": 1, "corruption_seed": 123, "macro_f1": 0.3},
        ]
    )
    robust = pd.DataFrame(
        [
            {"architecture_family": "A", "mPC_family_macro_f1": 0.55},
            {"architecture_family": "B", "mPC_family_macro_f1": 0.35},
        ]
    )
    clean.to_csv(res / "combined_clean_metrics.csv", index=False)
    corr.to_csv(res / "combined_corruption_metrics_long.csv", index=False)
    robust.to_csv(res / "combined_robustness_summary.csv", index=False)
    return res


def test_build_reference_bank_help_runs():
    with pytest.raises(SystemExit) as exc:
        main(["build-reference-bank", "--help"])
    assert exc.value.code == 0


def test_summarize_results_writes_summary(tmp_path):
    write_synthetic_results(tmp_path)
    out = tmp_path / "summary"
    assert main(["summarize-results", "--input", str(tmp_path), "--output", str(out)]) == 0
    assert (out / "summary.json").exists()
    assert (out / "summary.csv").exists()


def test_validate_results_detects_missing_combination(tmp_path):
    write_synthetic_results(tmp_path)
    corr_path = tmp_path / "results" / "combined_corruption_metrics_long.csv"
    corr = pd.read_csv(corr_path).iloc[:-1]
    corr.to_csv(corr_path, index=False)
    assert main(["validate-results", "--input", str(tmp_path)]) == 2


def test_unavailable_command_is_clear(tmp_path, capsys):
    code = main(["train", "--output-dir", str(tmp_path)])
    captured = capsys.readouterr()
    assert code == 2
    assert "requires raw datasets" in captured.err

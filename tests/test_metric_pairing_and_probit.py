import numpy as np
import pandas as pd

from csi_robustbench.statistics import pair_clean_summary, probit


def test_pairing_is_by_id_not_row_order_and_probit_uses_real_n_test():
    clean = pd.DataFrame(
        {
            "dataset": ["UT_HAR", "UT_HAR"],
            "configuration_id": ["a", "b"],
            "clean_macro_f1": [0.8, 0.6],
        }
    )
    summary = pd.DataFrame(
        {
            "dataset": ["UT_HAR", "UT_HAR"],
            "configuration_id": ["b", "a"],
            "mPC_family_macro_f1": [0.5, 0.7],
        }
    )
    paired = pair_clean_summary(
        clean,
        summary,
        keys=["dataset", "configuration_id"],
        clean_metric="clean_macro_f1",
        summary_metric="mPC_family_macro_f1",
    ).sort_values("configuration_id")
    assert paired["mPC_family_macro_f1"].tolist() == [0.7, 0.5]
    assert np.isfinite(probit([0.0, 1.0], n_test=996)).all()
    assert probit([0.0], n_test=996)[0] > probit([0.0], n_test=2450)[0]


def test_pairing_rejects_duplicates():
    clean = pd.DataFrame({"dataset": ["x", "x"], "configuration_id": ["a", "a"], "clean_macro_f1": [0.1, 0.2]})
    summary = pd.DataFrame({"dataset": ["x"], "configuration_id": ["a"], "mPC_family_macro_f1": [0.1]})
    try:
        pair_clean_summary(clean, summary, keys=["dataset", "configuration_id"], clean_metric="clean_macro_f1", summary_metric="mPC_family_macro_f1")
    except ValueError:
        return
    raise AssertionError("expected duplicate-key failure")

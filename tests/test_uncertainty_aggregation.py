import numpy as np
import pandas as pd

from csi_robustbench.statistics import seed_level_mpc, summarize_seed_mpc


def test_family_balanced_and_flat7_seed_mpc_do_not_use_condition_rows_as_uncertainty():
    rows = []
    families = {
        "gaussian_snr": 1.0,
        "subcarrier_mask": 0.5,
        "contiguous_subcarrier_block": 0.7,
        "time_dropout": 0.2,
        "burst_time_dropout": 0.4,
        "amplitude_scaling": 0.8,
        "smooth_gain_drift": 0.6,
    }
    for seed, offset in [(42, 0.0), (123, 0.1)]:
        for corruption, base in families.items():
            for severity in [1, 2, 3, 4, 5]:
                rows.append(
                    {
                        "dataset": "toy",
                        "configuration_id": "m1",
                        "corruption_seed": seed,
                        "corruption": corruption,
                        "severity": severity,
                        "macro_f1": base + offset + severity * 0.01,
                    }
                )
    corr = pd.DataFrame(rows)
    seed_scores = seed_level_mpc(corr, group_cols=["dataset", "configuration_id"], metrics=["macro_f1"])
    assert seed_scores.shape[0] == 2
    s42 = seed_scores[seed_scores["corruption_seed"] == 42].iloc[0]
    # Equal-weight high-level families:
    # noise=1.03, frequency=(0.53+0.73)/2, temporal=(0.23+0.43)/2, gain=(0.83+0.63)/2
    expected_family = np.mean([1.03, 0.63, 0.33, 0.73])
    expected_flat7 = np.mean([v + 0.03 for v in families.values()])
    assert np.isclose(s42["macro_f1_family_seed"], expected_family)
    assert np.isclose(s42["macro_f1_flat7_seed"], expected_flat7)

    summary = summarize_seed_mpc(seed_scores, group_cols=["dataset", "configuration_id"], metrics=["macro_f1"])
    assert summary.loc[0, "n_corruption_seeds"] == 2
    assert summary.loc[0, "mPC_family_macro_f1_std"] > 0


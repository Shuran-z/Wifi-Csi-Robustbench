from csi_robustbench.validity import assess_er_validity

def test_validity_statuses():
    assert assess_er_validity(n_unique_configs=60, cv_r2=0.7, slope=1.0, support_min=0.3, support_max=0.9, target_clean=0.5, bootstrap_ci_low=0.01, bootstrap_ci_high=0.05).status == "VALID"
    assert assess_er_validity(n_unique_configs=60, cv_r2=0.7, slope=1.0, support_min=0.3, support_max=0.9, target_clean=0.95, bootstrap_ci_low=0.01, bootstrap_ci_high=0.05).status == "OUT_OF_SUPPORT"
    assert assess_er_validity(n_unique_configs=60, cv_r2=0.1, slope=1.0, support_min=0.3, support_max=0.9, target_clean=0.5, bootstrap_ci_low=0.01, bootstrap_ci_high=0.05).status == "INVALID_FIT"

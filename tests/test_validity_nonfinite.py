from csi_robustbench.validity import assess_er_validity


def test_validity_rejects_nonfinite_inputs():
    for kwargs in [
        {"slope": float("nan")},
        {"cv_r2": float("nan")},
        {"bootstrap_ci_low": float("nan")},
        {"bootstrap_ci_high": float("inf")},
        {"bootstrap_success_rate": float("nan")},
    ]:
        base = dict(
            n_unique_configs=60,
            cv_r2=0.7,
            slope=1.0,
            support_min=0.2,
            support_max=0.9,
            target_clean=0.5,
            bootstrap_ci_low=0.01,
            bootstrap_ci_high=0.05,
            bootstrap_success_rate=1.0,
        )
        base.update(kwargs)
        out = assess_er_validity(**base)
        assert out.status == "INVALID_FIT"
        assert any("non-finite" in reason for reason in out.reasons)

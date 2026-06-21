from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import math

Status = Literal["VALID", "EXPLORATORY", "OUT_OF_SUPPORT", "INVALID_FIT"]

@dataclass
class ERValidity:
    status: Status
    n_unique_configs: int
    cv_r2: float
    slope: float
    support_min: float
    support_max: float
    target_in_support: bool
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    reasons: list[str] = field(default_factory=list)

def assess_er_validity(
    *,
    n_unique_configs: int,
    cv_r2: float,
    slope: float,
    support_min: float,
    support_max: float,
    target_clean: float,
    bootstrap_ci_low: float,
    bootstrap_ci_high: float,
    bootstrap_success_rate: float = 1.0,
    min_valid_configs: int = 30,
    valid_cv_r2: float = 0.50,
    exploratory_cv_r2: float = 0.20,
) -> ERValidity:
    reasons: list[str] = []
    finite_checks = {
        "slope": slope,
        "cv_r2": cv_r2,
        "support_min": support_min,
        "support_max": support_max,
        "target_clean": target_clean,
        "bootstrap_ci_low": bootstrap_ci_low,
        "bootstrap_ci_high": bootstrap_ci_high,
        "bootstrap_success_rate": bootstrap_success_rate,
    }
    bad = [name for name, value in finite_checks.items() if not math.isfinite(float(value))]
    if bad:
        reasons.append("non-finite validity input: " + ", ".join(bad))
        return ERValidity("INVALID_FIT", n_unique_configs, cv_r2, slope, support_min, support_max, False, bootstrap_ci_low, bootstrap_ci_high, reasons)
    target_in_support = support_min <= target_clean <= support_max
    if not target_in_support:
        reasons.append("target clean metric outside reference support")
        return ERValidity("OUT_OF_SUPPORT", n_unique_configs, cv_r2, slope, support_min, support_max, False, bootstrap_ci_low, bootstrap_ci_high, reasons)
    if slope <= 0 or cv_r2 < exploratory_cv_r2 or bootstrap_ci_low > bootstrap_ci_high:
        if slope <= 0:
            reasons.append("non-positive fit slope")
        if cv_r2 < exploratory_cv_r2:
            reasons.append("group-CV R2 below exploratory threshold")
        if bootstrap_ci_low > bootstrap_ci_high:
            reasons.append("invalid bootstrap interval")
        return ERValidity("INVALID_FIT", n_unique_configs, cv_r2, slope, support_min, support_max, target_in_support, bootstrap_ci_low, bootstrap_ci_high, reasons)
    if n_unique_configs >= min_valid_configs and cv_r2 >= valid_cv_r2 and bootstrap_success_rate >= 0.95:
        return ERValidity("VALID", n_unique_configs, cv_r2, slope, support_min, support_max, target_in_support, bootstrap_ci_low, bootstrap_ci_high, reasons)
    if n_unique_configs < min_valid_configs:
        reasons.append("too few unique reference configurations")
    if bootstrap_success_rate < 0.95:
        reasons.append("bootstrap success rate below 95%")
    reasons.append("fit is positive but below VALID evidence threshold")
    return ERValidity("EXPLORATORY", n_unique_configs, cv_r2, slope, support_min, support_max, target_in_support, bootstrap_ci_low, bootstrap_ci_high, reasons)

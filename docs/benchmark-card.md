# Benchmark Card

## Goal

WiFi CSI RobustBench evaluates robustness of WiFi CSI human-sensing models under structured common corruptions and a small natural-shift check. The benchmark is built for reproducible research and open-source inspection, not for safety-critical deployment certification.

## Scope

Main result: UT-HAR model-bank robustness.

Supplementary result: Widar minimum natural-shift analysis.

Engineering appendix: NTU-Fi-HAR-Recovered5 split documentation.

## Corruption Suite

Seven corruption types are used:

- Gaussian SNR degradation
- Random subcarrier masking
- Contiguous subcarrier block missingness
- Random time dropout
- Burst time dropout
- Amplitude scaling
- Smooth gain drift

Each corruption is evaluated at five severities and five corruption seeds.

## Severity Definition

Severity is an ordinal protocol level from 1 to 5. It is not assumed to be linearly spaced in physical units. Severity averaging is performed within each corruption before higher-level family aggregation.

## Family-Balanced mPC

Family-balanced mPC is the primary robustness metric. It first averages each corruption over severity, then averages corruptions inside one of four high-level families, then gives each family equal weight. This prevents families with two corruption types from dominating the final score.

## Flat-7 Sensitivity

Flat-7 mPC averages all seven corruption types equally. It is reported as a secondary sensitivity metric and should not replace the primary family-balanced score.

## CSI-ER Validity Gate

CSI-ER fits corrupted performance from clean performance in probit space and reports a residual. It is interpretable only when:

- reference support contains the target clean performance;
- fit slope and cross-validation statistics are finite;
- group-CV is above threshold;
- bootstrap intervals are finite;
- bootstrap success rate is acceptable.

Statuses are `VALID`, `EXPLORATORY`, `OUT_OF_SUPPORT`, and `INVALID_FIT`.

## Accepted Interpretation

Valid conclusions:

- Compare raw family-balanced mPC across model families.
- Discuss how corruption families affect retention.
- Use CSI-ER only as a validity-gated diagnostic.
- Treat Widar as exploratory natural-shift evidence.

## Unsupported Claims

Unsupported conclusions:

- "This is the first WiFi CSI robustness benchmark."
- "CSI-ER is a universal theory of WiFi sensing robustness."
- "Recovered5 is official NTU-Fi HAR."
- "Widar minimum bank proves broad multi-dataset generalization."
- "Out-of-support CSI-ER values are official rankings."

## Known Biases

- UT-HAR dominates final quantitative evidence.
- Synthetic corruptions may not cover all hardware or environment failures.
- Classical model coverage is broader than deep architecture coverage.
- Widar has only six final natural-shift model points in this release.

## Recommended Use

- Inspect the included result tables and figures.
- Reproduce benchmark figures from `results/final/`.
- Extend the corruption suite or model bank with explicit protocol changes.
- Treat CSI-ER as a validity-gated diagnostic rather than a standalone ranking.

Use family-balanced mPC Macro-F1 as the primary result. Use the validity dashboard to explain when CSI-ER can or cannot be interpreted. Use the Widar scatter only as an exploratory check of synthetic-vs-natural alignment.

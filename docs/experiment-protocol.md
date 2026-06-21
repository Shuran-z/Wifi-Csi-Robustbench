# Experiment Protocol

This protocol summarizes the benchmark workflow represented by the public release.

1. Prepare official WiFi CSI datasets outside the Git repository.
2. Normalize each dataset using training-set statistics only.
3. Train the configured classical and deep model banks.
4. Evaluate clean performance on the held-out test split.
5. Evaluate structured corruptions across corruption type, severity, and corruption seed.
6. Aggregate robustness with family-balanced mPC.
7. Fit CSI-ER diagnostics with validity gates.
8. Run the minimum Widar natural-shift analysis.
9. Generate final tables, reports, manifests, logs, and figures.

Model configurations and statistical settings are stored under `configs/`.

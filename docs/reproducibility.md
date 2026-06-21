# Reproducibility Guide

This guide defines three reproduction levels. The lightweight release is designed for Level 1. Levels 2 and 3 require external data or checkpoint archives.

## Level 1: Review Results Without Weights

Use this when you only need to inspect final metrics, regenerate summaries, and create benchmark figures.

Requirements:

- Python environment from `environment.yml` or `requirements-minimal.txt`
- Final review directories from this repository: `results/final/`, `reports/final/`, `figures/final/`, and `manifests/final/`
- No raw datasets
- No checkpoints

Commands:

```bash
pip install -e .
python scripts/plot_benchmark_figures.py --input . --output figures/final
```

Expected final counts:

- `combined-clean-metrics.csv`: 774 rows
- `combined-corruption-metrics-long.csv`: 135,450 rows
- `classical-bank-clean.csv`: 720 rows
- `deep-bank-clean.csv`: 54 rows
- Widar natural-shift CSV: 6 rows

## Level 2: Download External Weights and Recheck Inference

Use this when you want to re-load checkpoints and run selected evaluation without retraining.

Requirements:

- Lightweight release
- External checkpoint archive
- `manifests/final/final-checkpoint-checksums.sha256`
- Enough disk for approximately 12.7 GB of checkpoints
- Matching Python/scikit-learn/PyTorch versions from `requirements-lock.txt`

Restore layout:

```text
external-weights/checkpoints/
external-weights/classical-shards/shard-0/checkpoints/
external-weights/classical-shards/shard-1/checkpoints/
external-weights/classical-shards/shard-2/checkpoints/
external-weights/classical-shards/shard-3/checkpoints/
```

Commands:

```bash
sha256sum -c manifests/final/final-checkpoint-checksums.sha256
pytest -q tests/test_checkpoint_compatibility.py
csi-robustbench validate-results --input results/final --checkpoint-dir external-weights/checkpoints
```

If joblib warns about scikit-learn version mismatch, use the version in `requirements-lock.txt` or retrain the classical model in that environment.

## Level 3: Reproduce From Raw Data

Use this when you want a from-scratch experiment rerun.

Requirements:

- Official UT-HAR data
- Official Widar data for the minimum natural-shift check
- Optional NTU-Fi archive only as Recovered5 engineering appendix
- GPU recommended for 54 deep runs
- Enough disk for raw data, caches, results, and approximately 12.7 GB checkpoints

High-level commands:

```bash
python scripts/verify_data.py --data-root data
python scripts/02_prepare_ut_har.py --data-root data --output data/processed
python scripts/build_classical_reference_bank.py --output-dir outputs_new
python scripts/train_deep_model_bank.py --output-dir outputs_new
python scripts/run_statistical_analysis.py --output-dir outputs_new --bootstrap-reps 2000
python scripts/run_widar_natural_shift.py --output-dir outputs_new
python scripts/plot_benchmark_figures.py --input outputs_new
```

Expected cost:

- Classical bank: CPU-heavy, parallelizable
- Deep bank: GPU recommended
- Corruption evaluation: 7 corruption types x 5 severities x 5 seeds per successful run
- Bootstrap statistics: CPU time depends on `--bootstrap-reps`

Do not tune model or corruption parameters on corrupted test results.

## Non-Reproducible Claims to Avoid

- Do not claim official NTU-Fi six-class performance from Recovered5.
- Do not claim CSI-ER is a theoretical law.
- Do not rank EXPLORATORY or OUT_OF_SUPPORT CSI-ER results as definitive winners.
- Do not treat Widar minimum bank as a full multi-dataset benchmark.

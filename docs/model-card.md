# Model Card

## Model Families

Final UT-HAR model bank:

- Classical configurations: 720 completed rows
- Deep runs: 54 completed rows
- Classical failures: 0
- Deep failures: 0

Classical feature families:

- time statistics
- FFT statistics
- DWT energy
- STFT bands
- autocorrelation
- PCA raw

Classical classifier families:

- logistic regression
- linear SVM
- RBF SVM
- kNN
- random forest
- extra trees
- histogram gradient boosting
- LDA

Deep architecture families:

- MLP
- CNN
- GRU
- LSTM
- CNN-GRU
- Transformer

Deep sizes:

- small
- medium
- large

Deep train seeds:

- 42
- 123
- 2026

## Checkpoints and Metadata

Checkpoint manifests:

- `manifests/final/classical-bank-checkpoint-manifest.csv`
- `manifests/final/deep-bank-checkpoint-manifest.csv`
- `manifests/final/widar-minimum-bank-checkpoint-manifest.csv`
- `manifests/final/final-all-checkpoint-checksums.sha256`

The final checkpoint checksum file lists 791 checkpoint/model artifacts.

## Weight Size

The final checkpoint directories are approximately:

- UT-HAR main checkpoints: 1.4 GB
- Classical shard 0 checkpoints: 1.8 GB
- Classical shard 1 checkpoints: 3.5 GB
- Classical shard 2 checkpoints: 1.8 GB
- Classical shard 3 checkpoints: 4.2 GB

Total: approximately 12.7 GB.

## Loading Weights

Use the external full-weight package and restore it under the original output directory layout. Then verify:

```bash
sha256sum -c manifests/final/final-checkpoint-checksums.sha256
pytest -q tests/test_checkpoint_compatibility.py
```

Classical checkpoints may rely on the scikit-learn/joblib version in `requirements-lock.txt`. Deep checkpoints should be loaded with strict PyTorch state-dict matching.

## Intended Use

The models are intended for benchmark analysis and robustness research. They are not recommended for real safety-critical IoT deployments, health monitoring, law enforcement, or surveillance.

## Limitations

- Models are trained on processed benchmark datasets, not live hardware streams.
- Corruption robustness is synthetic and protocol-specific.
- Widar models are only a minimum natural-shift bank.
- Checkpoint compatibility must be validated after moving between environments.

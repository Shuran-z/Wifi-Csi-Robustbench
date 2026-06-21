# External Model Weights

The lightweight release excludes model weights and checkpoint files.

## Why Weights Are Excluded

The final checkpoint directories are approximately 12.7 GB, and the full package with code, logs, results, and weights is approximately 13 GB. This is too large for ordinary source-package upload and inconvenient for GitHub repository storage.

This is a packaging limitation, not an unfinished experiment.

## Full Weight Package

The full checkpoint archive is not committed to this repository. Host it separately through an artifact service such as GitHub Releases, Hugging Face Hub, Zenodo, or institutional storage.

Full package SHA256:

```text
31dd0ef453d7b327f92a602aefd35bf880078ca7439a317cd18af45711b36ac0
```

## Weight Counts

Checkpoint checksum entries: 791.

Manifests:

- Classical UT-HAR checkpoint manifest: 720 model rows plus header
- Deep UT-HAR checkpoint manifest: 54 model rows plus header
- Widar minimum checkpoint manifest: 6 model rows plus header

## Size Breakdown

- `external-weights/checkpoints`: 1.4 GB
- `external-weights/classical-shards/shard-0/checkpoints`: 1.8 GB
- `external-weights/classical-shards/shard-1/checkpoints`: 3.5 GB
- `external-weights/classical-shards/shard-2/checkpoints`: 1.8 GB
- `external-weights/classical-shards/shard-3/checkpoints`: 4.2 GB

## Checksums

Use:

```text
manifests/final/final-checkpoint-checksums.sha256
manifests/final/final-all-checkpoint-checksums.sha256
```

## Recommended Public Hosting

For public release, use one of:

- GitHub Releases for the full archive
- Hugging Face Hub for checkpoints and result artifacts
- Zenodo for a DOI-backed artifact
- Git LFS only if the quota and bandwidth limits are acceptable

Do not commit checkpoint binaries directly to the Git repository.

## Restore Layout

After downloading the full checkpoint archive, restore:

```text
external-weights/checkpoints/
external-weights/classical-shards/shard-*/checkpoints/
```

Then run:

```bash
sha256sum -c manifests/final/final-checkpoint-checksums.sha256
pytest -q tests/test_checkpoint_compatibility.py
```

## Compatibility Notes

Classical models are serialized with joblib and scikit-learn. If the runtime scikit-learn version differs from the training version, loading may emit warnings or fail. Use the pinned environment in `requirements-lock.txt` for compatibility checks.

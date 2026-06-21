# Weights Excluded Notice

This lightweight repository intentionally excludes model weights and checkpoint files.

Reason: the final checkpoint directories are approximately 12.7 GB in total. Checkpoint binaries are better distributed through an artifact store rather than committed directly to Git.

The repository keeps code, logs, metrics, reports, manifests, and figures only. Use `docs/model-weights.md` for checkpoint size, checksum, and restore-layout details.

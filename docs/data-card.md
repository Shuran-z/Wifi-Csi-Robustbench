# Data Card

## UT-HAR

UT-HAR is the main benchmark dataset in this project. It is used for the full classical and deep model banks.

Final protocol facts:

- Task: human activity recognition from WiFi CSI tensors
- Test samples used for probit clipping: 996
- Final clean model/run rows: 774
- Final corrupted rows: 135,450
- Main metrics: accuracy, Macro-F1, balanced accuracy

The final release does not redistribute raw UT-HAR data. It includes only result CSVs, logs, figures, reports, and manifests.

## Widar

Widar is used for a minimum natural-shift analysis.

Final protocol facts:

- Train files loaded: 34,926
- Test files loaded: 8,726
- Class count: 22
- Models: 6
- Policy: real train/test domain split; train split is subdivided only for source-domain validation

This is an exploratory minimum bank. It is not a full Widar benchmark.

## NTU-Fi-HAR-Recovered5

The recovered NTU-Fi archive did not provide a reliable official six-class setup. The `box` class was excluded because the recovered archive did not provide enough trustworthy support for a clean six-class benchmark.

Final use:

- Name: NTU-Fi-HAR-Recovered5
- Classes: `circle`, `clean`, `fall`, `run`, `walk`
- Use: appendix or engineering example only
- Not used as a main result
- Not comparable with official six-class NTU-Fi HAR papers

## Raw Data Redistribution

Raw WiFi CSI data is not included in this repository or lightweight release. Users must download data from official sources and comply with source dataset terms.

## Download and Verification

Recommended process:

```bash
python scripts/verify_data.py --data-root data
sha256sum -c data/checksums.json
```

If checksum files are missing for a raw dataset, compute and record them before running experiments.

## Class Order

Use the class order stored in processed metadata files or split JSON files. Do not infer class order from directory listing unless the processing script explicitly does so and records the result.

## Known Limitations

- Some public WiFi CSI datasets have incomplete metadata for domain shift.
- Recovered archives may have class imbalance or missing classes.
- Synthetic corruptions do not perfectly represent all deployment failures.

## Privacy and Ethics

WiFi CSI can reveal human presence and activity without cameras. Use only datasets collected with appropriate consent and comply with local policy. Do not deploy trained models for surveillance or safety-critical decisions without separate review.

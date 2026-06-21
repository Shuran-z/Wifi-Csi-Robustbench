# Third-Party Notices

This file is a release checklist for code, datasets, libraries, and adapted scripts. Items marked "license to be verified before public release" must be checked against the upstream source before a public GitHub release or redistribution.

| component | source | license | used for | redistributed in this repository? | modifications | notes |
|---|---|---|---|---|---|---|
| SenseFi / WiFi-CSI-Sensing-Benchmark | Upstream SenseFi benchmark repository | license to be verified before public release | Experimental reference, benchmark framing, related work | No raw upstream repository redistribution intended | Project scripts are separate benchmark utilities; any adapted snippets must be reviewed | Treat SenseFi as experimental base and related work, not a recent top-conference claim |
| UT-HAR dataset | Official UT-HAR release | dataset terms to be verified before public release | Main WiFi CSI human activity recognition benchmark | No raw data | Processed result metrics only | Raw CSI data is not included in lightweight release |
| Widar dataset | Official Widar release | dataset terms to be verified before public release | Minimum natural domain-shift analysis | No raw data | Train/test handling documented in data card | Final result is exploratory minimum bank |
| NTU-Fi HAR recovered archive | Recovered archive from available download attempts | dataset terms and archive integrity to be verified before public release | Engineering split validation only | No raw data | Dropped unreliable `box` class; named NTU-Fi-HAR-Recovered5 | Not an official six-class benchmark result |
| scikit-learn | https://scikit-learn.org | BSD-3-Clause | Classical estimators, metrics, model serialization | Dependency only | None | Check exact version in `requirements-lock.txt` for joblib compatibility |
| PyTorch | https://pytorch.org | BSD-style | Deep model training and checkpoint loading | Dependency only | None | Deep checkpoint compatibility depends on PyTorch version |
| NumPy | https://numpy.org | BSD-3-Clause | Array computation | Dependency only | None | Required by all metrics and features |
| SciPy | https://scipy.org | BSD-3-Clause | Statistics, signal utilities | Dependency only | None | Used for correlation/statistical routines |
| pandas | https://pandas.pydata.org | BSD-3-Clause | Result table processing | Dependency only | None | Used by CLI and plotting |
| Matplotlib | https://matplotlib.org | PSF-compatible | Figures | Dependency only | None | Used for release figures |
| PyWavelets | https://pywavelets.readthedocs.io | MIT | DWT-energy features | Dependency only | None | Used in classical bank |
| joblib | https://joblib.readthedocs.io | BSD-3-Clause | Classical checkpoint serialization | Dependency only | None | Version compatibility should be checked for external checkpoints |
| PyYAML | https://pyyaml.org | MIT | Config loading | Dependency only | None | Used for YAML protocol files |
| tqdm | https://tqdm.github.io | MIT/MPL-2.0 | Progress reporting | Dependency only | None | Optional runtime convenience |
| Adapted local scripts | `scripts/` in this repository | MIT for original project code unless otherwise noted | Data preparation, training, plotting, packaging | Yes | Project-specific | Review any script that imported upstream code before public release |

No private keys, server credentials, or raw datasets should be committed to a public release.

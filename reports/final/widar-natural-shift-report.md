# Widar Natural-Shift Minimum Bank

- Train files loaded: 34926
- Test files loaded: 8726
- Models: 6
- Split policy: real `train/` to `test/`; no random train/test replacement.

## Synthetic-vs-Natural Spearman

```json
{
  "accuracy": {
    "bootstrap_reps": 2000,
    "bootstrap_success_rate": 1.0,
    "ci_high": 1.0,
    "ci_low": -0.6363636363636362,
    "spearman": 0.48571428571428577
  },
  "balanced_accuracy": {
    "bootstrap_reps": 2000,
    "bootstrap_success_rate": 1.0,
    "ci_high": 1.0,
    "ci_low": -1.0,
    "spearman": 0.4285714285714286
  },
  "macro_f1": {
    "bootstrap_reps": 2000,
    "bootstrap_success_rate": 1.0,
    "ci_high": 1.0,
    "ci_low": -0.6363636363636362,
    "spearman": 0.48571428571428577
  }
}
```

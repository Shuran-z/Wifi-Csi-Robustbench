#!/usr/bin/env python3
from pathlib import Path
import json
import pandas as pd
from csi_er.utils import PROJECT_ROOT, now_ts
res = PROJECT_ROOT / 'outputs/results'
out = res / 'summary.md'
parts = [f'# Experiment Summary\n\nGenerated: {now_ts()}\n']
meta_p = PROJECT_ROOT / 'data/processed/ut_har_smoke_meta.json'
formal_meta = PROJECT_ROOT / 'data/processed/ut_har_meta.json'
blocker = PROJECT_ROOT / 'docs/DATA_DOWNLOAD_BLOCKER.md'
if formal_meta.exists():
    meta = json.loads(formal_meta.read_text(encoding='utf-8'))
    parts.append(f"\n## Formal Data Status\n\nFormal UT-HAR prepared. Source: `{meta.get('source')}`. Train shape: `{meta.get('train_shape')}`, test shape: `{meta.get('test_shape')}`.\n")
elif blocker.exists():
    parts.append('\n## Formal Data Status\n\nFormal UT-HAR was not prepared because the official download path was blocked. See `docs/DATA_DOWNLOAD_BLOCKER.md`. The CSV and figures currently present are smoke-test artifacts only and must not be used as formal experimental results.\n')
if meta_p.exists():
    meta = json.loads(meta_p.read_text(encoding='utf-8'))
    parts.append(f"\n## Smoke Data Status\n\nSmoke data source: `{meta.get('source')}`; synthetic: `{meta.get('synthetic')}`; train shape: `{meta.get('train_shape')}`; test shape: `{meta.get('test_shape')}`.\n")
for name in ['classical_clean_noise_metrics.csv','deep_clean_noise_metrics.csv','csi_er.csv']:
    p = res / name
    if p.exists():
        df = pd.read_csv(p)
        parts.append(f'\n## {name}\n\nRows: {len(df)}\n\n```\n{df.head(20).to_string(index=False)}\n```\n')
    else:
        parts.append(f'\n## {name}\n\nMissing.\n')
out.write_text('\n'.join(parts), encoding='utf-8')
print(out)

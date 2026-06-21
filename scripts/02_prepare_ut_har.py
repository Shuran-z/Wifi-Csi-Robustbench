#!/usr/bin/env python3
import argparse, sys
from csi_er.data import prepare_ut_har, find_ut_har_root
from csi_er.utils import PROJECT_ROOT, load_config, ensure_dirs, append_log
p=argparse.ArgumentParser(); p.add_argument('--smoke',action='store_true'); p.add_argument('--synthetic',action='store_true'); p.add_argument('--output-dir'); a=p.parse_args()
cfg=load_config(); ensure_dirs()
if a.smoke:
    meta=prepare_ut_har(cfg, smoke=True, synthetic=True); append_log('Prepared synthetic smoke UT-HAR data; not a formal result.'); print(meta); sys.exit(0)
if find_ut_har_root(cfg['paths']['raw_data']) is None:
    raise SystemExit('UT-HAR raw data not found under data/raw/UT_HAR')
meta=prepare_ut_har(cfg, smoke=False, synthetic=False); append_log('Prepared formal UT-HAR processed data.'); print(meta)

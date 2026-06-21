#!/usr/bin/env python3
from csi_er.classical import run_classical
from csi_er.utils import load_config
import argparse
p=argparse.ArgumentParser(); p.add_argument('--smoke',action='store_true'); p.add_argument('--output-dir'); a=p.parse_args()
print(run_classical(load_config(), smoke=a.smoke, output_dir=a.output_dir))

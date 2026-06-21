#!/usr/bin/env python3
from csi_er.er import compute_csi_er
import argparse
p=argparse.ArgumentParser(); p.add_argument('--smoke',action='store_true'); p.parse_args(); print(compute_csi_er())

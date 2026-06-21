#!/usr/bin/env python3
from csi_er.plotting import plot_all
from csi_er.utils import load_config
import argparse
p = argparse.ArgumentParser()
p.add_argument('--smoke', action='store_true')
a = p.parse_args()
print('\n'.join(plot_all(load_config(), smoke=a.smoke)))

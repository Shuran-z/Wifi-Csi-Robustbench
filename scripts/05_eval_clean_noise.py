#!/usr/bin/env python3
from csi_er.evaluate import eval_deep_models
from csi_er.utils import load_config
import argparse
p=argparse.ArgumentParser(); p.add_argument('--models',nargs='+',default=['MLP','SimpleCNN','GRU','LSTM','CNNGRU','TinyViT']); p.add_argument('--smoke',action='store_true'); a=p.parse_args(); print(eval_deep_models(load_config(), a.models, smoke=a.smoke))

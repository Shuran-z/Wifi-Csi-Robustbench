#!/usr/bin/env python3
from csi_er.train import train_models
from csi_er.utils import load_config
import argparse
p=argparse.ArgumentParser(); p.add_argument('--models',nargs='+',default=['MLP','SimpleCNN','GRU','LSTM','CNNGRU','TinyViT']); p.add_argument('--epochs',type=int); p.add_argument('--smoke',action='store_true'); p.add_argument('--output-dir'); p.add_argument('--force-retrain',action='store_true'); a=p.parse_args()
print(train_models(load_config(), a.models, epochs=a.epochs, smoke=a.smoke, output_dir=a.output_dir, force_retrain=a.force_retrain))

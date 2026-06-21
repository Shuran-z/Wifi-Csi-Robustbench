from __future__ import annotations
import time, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from .corruptions import add_gaussian_noise
from .data import load_processed
from .models import build_model, count_parameters
from .utils import PROJECT_ROOT, ensure_dirs, now_ts

def eval_deep_models(config, model_names, smoke=False):
    ensure_dirs(); d=load_processed(smoke=smoke); X=torch.tensor(d['X_test'], dtype=torch.float32); y=np.asarray(d['y_test']); ncls=len(np.unique(np.concatenate([d['y_train'], d['y_test']])))
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); bs=min(config['train']['batch_size'], max(4,len(X))); rows=[]; conf={}
    for name in model_names:
        ck=PROJECT_ROOT/f'outputs/checkpoints/deep/{name}_best.pt'
        if not ck.exists(): continue
        model=build_model(name,ncls).to(dev); state=torch.load(ck,map_location=dev); model.load_state_dict(state['state_dict'], strict=False); model.eval()
        t0=time.time(); pred=_predict(model,X,bs,dev); infer=(time.time()-t0)*1000/max(1,len(X))
        clean_acc=float(accuracy_score(y,pred)); clean_f1=float(f1_score(y,pred,average='macro',zero_division=0)); conf[f'{name}_clean']=confusion_matrix(y,pred).tolist()
        noise_acc=[]
        for s in config['corruption']['severities']:
            Xn=add_gaussian_noise(d['X_test'],s,alphas=config['corruption']['alphas'],seed=config['corruption'].get('seed',42)+s)
            cache=PROJECT_ROOT/f'data/processed/noisy/UT_HAR_gaussian_s{s}.npz'; np.savez_compressed(cache, X=Xn, y=y)
            pn=_predict(model, torch.tensor(Xn,dtype=torch.float32), bs, dev); noise_acc.append(float(accuracy_score(y,pn))); conf[f'{name}_s{s}']=confusion_matrix(y,pn).tolist()
        row=dict(timestamp=now_ts(), seed=config['train'].get('seed',42), model_name=name, model_group=_group(name), is_classical=False, dataset='UT_HAR', clean_acc=clean_acc, clean_macro_f1=clean_f1)
        for s,a in zip(config['corruption']['severities'], noise_acc): row[f'noise_s{s}_acc']=a
        row.update(mean_noise_acc=float(np.mean(noise_acc)), clean_error=float(1-clean_acc), noise_error=float(1-np.mean(noise_acc)), train_time_sec=np.nan, infer_time_ms_per_sample=float(infer), params_proxy=count_parameters(model), feature_dim=0)
        rows.append(row)
    pd.DataFrame(rows).to_csv(PROJECT_ROOT/'outputs/results/deep_clean_noise_metrics.csv',index=False)
    with open(PROJECT_ROOT/'outputs/results/confusion_matrices.json','w',encoding='utf-8') as f: json.dump(conf,f,indent=2)
    return rows

def _predict(model, X, bs, dev):
    loader=DataLoader(TensorDataset(X),batch_size=bs); out=[]
    with torch.no_grad():
        for (xb,) in loader: out.extend(model(xb.to(dev)).argmax(1).cpu().numpy())
    return np.array(out)

def _group(name):
    if name=='MLP': return 'MLP'
    if name=='SimpleCNN': return 'CNN'
    if name in ['GRU','LSTM']: return 'RNN'
    if name=='CNNGRU': return 'Hybrid'
    if name=='TinyViT': return 'Transformer'
    return 'Deep'

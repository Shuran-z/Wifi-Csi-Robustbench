from __future__ import annotations
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score
from .data import load_processed
from .models import build_model, count_parameters
from .utils import PROJECT_ROOT, ensure_dirs, set_seed, now_ts

def train_models(config, model_names, epochs=None, smoke=False, output_dir=None, force_retrain=False):
    ensure_dirs(); seed=config['train'].get('seed',42); set_seed(seed)
    out = PROJECT_ROOT / (output_dir or config['paths'].get('outputs','outputs'))
    (out/'checkpoints/deep').mkdir(parents=True, exist_ok=True); (out/'results').mkdir(parents=True, exist_ok=True)
    d=load_processed(smoke=smoke); X=torch.tensor(d['X_train'], dtype=torch.float32); y=torch.tensor(d['y_train'], dtype=torch.long)
    input_shape=tuple(X.shape[1:]); ncls=int(max(d['y_train'].max(), d['y_test'].max())+1)
    val_n=max(ncls, int(len(X)*config['train'].get('val_fraction',0.15))); train_n=len(X)-val_n
    ds=TensorDataset(X,y); tr,va=random_split(ds,[train_n,val_n], generator=torch.Generator().manual_seed(seed))
    bs=min(config['train']['batch_size'], max(4, train_n)); workers=0 if smoke else config['train'].get('num_workers',4)
    dl=DataLoader(tr,batch_size=bs,shuffle=True,num_workers=workers,pin_memory=torch.cuda.is_available()); vl=DataLoader(va,batch_size=bs,num_workers=workers,pin_memory=torch.cuda.is_available())
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); curves=[]; summary=[]; epochs=epochs or config['train']['epochs']
    for name in model_names:
        ck = out/f'checkpoints/deep/{name}_best.pt'
        if ck.exists() and not force_retrain:
            continue
        model=build_model(name,ncls,input_shape=input_shape).to(dev)
        opt=torch.optim.AdamW(model.parameters(), lr=config['train']['lr'], weight_decay=config['train']['weight_decay']); crit=torch.nn.CrossEntropyLoss(); best=-1; patience=0; t0=time.time()
        scaler=torch.amp.GradScaler('cuda', enabled=bool(config['train'].get('use_amp',True) and dev.type=='cuda'))
        for ep in range(1, epochs+1):
            model.train(); losses=[]
            for xb,yb in dl:
                xb,yb=xb.to(dev,non_blocking=True),yb.to(dev,non_blocking=True); opt.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=scaler.is_enabled()): loss=crit(model(xb), yb)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); losses.append(float(loss.detach().cpu()))
            acc=_eval_acc(model, vl, dev); curves.append(dict(timestamp=now_ts(), model_name=name, epoch=ep, train_loss=float(np.mean(losses)), val_acc=acc, dataset='UT_HAR', seed=seed))
            if acc>best:
                best=acc; patience=0; torch.save({'model_name':name,'state_dict':model.state_dict(),'num_classes':ncls,'input_shape':list(input_shape),'val_acc':best}, ck)
            else:
                patience+=1
            if patience>=config['train'].get('patience',8) and not smoke: break
        summary.append(dict(model_name=name, best_val_acc=float(best), train_time_sec=float(time.time()-t0), params=count_parameters(model)))
    pd.DataFrame(curves).to_csv(out/'results/deep_training_curves.csv',index=False)
    pd.DataFrame(summary).to_csv(out/'results/deep_training_summary.csv',index=False)
    return summary

def _eval_acc(model, loader, dev):
    model.eval(); yt=[]; yp=[]
    with torch.no_grad():
        for xb,yb in loader:
            pred=model(xb.to(dev)).argmax(1).cpu().numpy(); yp.extend(pred); yt.extend(yb.numpy())
    return float(accuracy_score(yt,yp)) if yt else 0.0

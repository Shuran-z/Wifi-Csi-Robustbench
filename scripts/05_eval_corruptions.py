#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import joblib, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from csi_er.utils import PROJECT_ROOT, load_config, now_ts
from csi_er.data import load_processed
from csi_er.models import build_model, count_parameters
from csi_er.classical import FEATURE_KIND, compute_feature
from csi_er.corruptions import get_or_create_corruption_cache

GROUPS={'MLP':'MLP','SimpleCNN':'CNN','GRU':'RNN','LSTM':'RNN','CNNGRU':'Hybrid','TinyViT':'Transformer'}

def pred_deep(model, X, bs, dev):
    model.eval(); out=[]
    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(torch.tensor(X,dtype=torch.float32)), batch_size=bs):
            out.extend(model(xb.to(dev)).argmax(1).cpu().numpy())
    return np.array(out)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--models',nargs='+',default=['MLP','SimpleCNN','GRU','LSTM','CNNGRU','TinyViT']); ap.add_argument('--output-dir',required=True); ap.add_argument('--smoke',action='store_true'); args=ap.parse_args()
    cfg=load_config(); out=PROJECT_ROOT/args.output_dir; (out/'results').mkdir(parents=True,exist_ok=True); (out/'logs').mkdir(parents=True,exist_ok=True)
    d=load_processed(smoke=args.smoke); Xte=d['X_test'].astype('float32'); yte=d['y_test'].astype('int64'); ncls=int(max(d['y_train'].max(), d['y_test'].max())+1); input_shape=tuple(Xte.shape[1:])
    train_seed=cfg['train'].get('seed',42); bs=min(cfg['train']['batch_size'], max(4,len(Xte))); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clean_rows=[]; corr_rows=[]; conf={}
    pca=joblib.load(out/'checkpoints/pca.joblib')
    classical_meta=pd.read_csv(out/'results/classical_clean_metrics.csv')
    for _,mrow in classical_meta.iterrows():
        name=mrow['model_name']; kind=FEATURE_KIND[name]; model=joblib.load(out/f'checkpoints/classical/{name}.joblib')
        F=compute_feature(kind,Xte,pca); t0=time.time(); pred=model.predict(F); infer=(time.time()-t0)*1000/max(1,len(Xte))
        clean_rows.append(dict(timestamp=now_ts(),dataset='UT_HAR',model_name=name,model_group='Classical',is_classical=True,clean_acc=accuracy_score(yte,pred),clean_macro_f1=f1_score(yte,pred,average='macro',zero_division=0),clean_error=1-accuracy_score(yte,pred),train_time_sec=mrow.get('train_time_sec',np.nan),infer_time_ms_per_sample=infer,params_proxy=0,param_count=0,feature_dim=mrow.get('feature_dim',0),train_seed=train_seed))
        conf[f'{name}|clean']=confusion_matrix(yte,pred).tolist()
        for corr in cfg['corruptions']['enabled']:
            for sev in cfg['corruptions']['severities']:
                cp,cseed=get_or_create_corruption_cache(Xte,yte,cfg,corr,sev,PROJECT_ROOT)
                z=np.load(cp,allow_pickle=True); Xc=z['X']; Fc=compute_feature(kind,Xc,pca); pr=model.predict(Fc); acc=accuracy_score(yte,pr)
                corr_rows.append(dict(timestamp=now_ts(),dataset='UT_HAR',model_name=name,model_group='Classical',is_classical=True,corruption=corr,severity=sev,accuracy=acc,macro_f1=f1_score(yte,pr,average='macro',zero_division=0),error=1-acc,train_seed=train_seed,corruption_seed=cseed,cache_file=str(cp.relative_to(PROJECT_ROOT))))
    deep_summary_path=out/'results/deep_training_summary.csv'; deep_summary=pd.read_csv(deep_summary_path) if deep_summary_path.exists() else pd.DataFrame()
    for name in args.models:
        ck=out/f'checkpoints/deep/{name}_best.pt'
        if not ck.exists(): continue
        state=torch.load(ck,map_location=dev); model=build_model(name,ncls,input_shape=state.get('input_shape',input_shape)).to(dev); model.load_state_dict(state['state_dict'], strict=True)
        t0=time.time(); pred=pred_deep(model,Xte,bs,dev); infer=(time.time()-t0)*1000/max(1,len(Xte)); acc=accuracy_score(yte,pred)
        train_time=np.nan; row=deep_summary[deep_summary.get('model_name',pd.Series(dtype=str)).eq(name)]
        if len(row): train_time=float(row.iloc[0].get('train_time_sec',np.nan))
        clean_rows.append(dict(timestamp=now_ts(),dataset='UT_HAR',model_name=name,model_group=GROUPS.get(name,'Deep'),is_classical=False,clean_acc=acc,clean_macro_f1=f1_score(yte,pred,average='macro',zero_division=0),clean_error=1-acc,train_time_sec=train_time,infer_time_ms_per_sample=infer,params_proxy=count_parameters(model),param_count=count_parameters(model),feature_dim=0,train_seed=train_seed))
        conf[f'{name}|clean']=confusion_matrix(yte,pred).tolist()
        for corr in cfg['corruptions']['enabled']:
            for sev in cfg['corruptions']['severities']:
                cp,cseed=get_or_create_corruption_cache(Xte,yte,cfg,corr,sev,PROJECT_ROOT)
                Xc=np.load(cp,allow_pickle=True)['X'].astype('float32'); pr=pred_deep(model,Xc,bs,dev); acc=accuracy_score(yte,pr)
                corr_rows.append(dict(timestamp=now_ts(),dataset='UT_HAR',model_name=name,model_group=GROUPS.get(name,'Deep'),is_classical=False,corruption=corr,severity=sev,accuracy=acc,macro_f1=f1_score(yte,pr,average='macro',zero_division=0),error=1-acc,train_seed=train_seed,corruption_seed=cseed,cache_file=str(cp.relative_to(PROJECT_ROOT))))
                conf[f'{name}|{corr}|s{sev}']=confusion_matrix(yte,pr).tolist()
    pd.DataFrame(clean_rows).to_csv(out/'results/clean_metrics.csv',index=False)
    pd.DataFrame(corr_rows).to_csv(out/'results/corruption_metrics_long.csv',index=False)
    (out/'results/confusion_matrices_v2.json').write_text(json.dumps(conf),encoding='utf-8')
    print('wrote', out/'results/clean_metrics.csv', out/'results/corruption_metrics_long.csv')
if __name__=='__main__': main()

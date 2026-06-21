from __future__ import annotations
import json
import numpy as np
import pandas as pd
from .utils import PROJECT_ROOT, now_ts

def compute_csi_er():
    dfs=[]
    for p in ['outputs/results/classical_clean_noise_metrics.csv','outputs/results/deep_clean_noise_metrics.csv']:
        fp=PROJECT_ROOT/p
        if fp.exists(): dfs.append(pd.read_csv(fp))
    if not dfs: raise FileNotFoundError('No metrics CSVs found')
    allm=pd.concat(dfs,ignore_index=True); allm.to_csv(PROJECT_ROOT/'outputs/results/all_metrics.csv',index=False)
    base=allm[allm['is_classical'].astype(str).isin(['True','true','1'])]
    if len(base)<2: raise ValueError('Need at least 2 classical points for fit')
    a,b=np.polyfit(base['clean_error'].values, base['noise_error'].values, 1)
    rows=[]
    for _,r in allm.iterrows():
        ehat=float(a*r['clean_error']+b); er=float(ehat-r['noise_error']); n=float(er/(ehat+1e-8))
        rows.append(dict(timestamp=now_ts(), model_name=r['model_name'], model_group=r['model_group'], is_classical=bool(r['is_classical']), clean_acc=r['clean_acc'], mean_noise_acc=r['mean_noise_acc'], E_clean=r['clean_error'], E_noise=r['noise_error'], E_noise_hat=ehat, CSI_ER=er, nCSI_ER=n, fit_type='linear', baseline_a=float(a), baseline_b=float(b)))
    out=pd.DataFrame(rows); out.to_csv(PROJECT_ROOT/'outputs/results/csi_er.csv',index=False)
    pred=a*base['clean_error'].values+b; ss_res=float(((base['noise_error'].values-pred)**2).sum()); ss_tot=float(((base['noise_error'].values-base['noise_error'].mean())**2).sum())
    with open(PROJECT_ROOT/'outputs/results/baseline_fit.json','w',encoding='utf-8') as f: json.dump({'fit_type':'linear','a':float(a),'b':float(b),'r2':1-ss_res/(ss_tot+1e-12),'n_classical':int(len(base))},f,indent=2)
    return out

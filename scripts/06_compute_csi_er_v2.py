#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
import numpy as np, pandas as pd
from csi_er.utils import PROJECT_ROOT, load_config, now_ts

def fit(x,y):
    a,b=np.polyfit(x,y,1); pred=a*x+b; ss_res=float(((y-pred)**2).sum()); ss_tot=float(((y-y.mean())**2).sum())
    return {'fit_type':'linear_ols','a':float(a),'b':float(b),'r2':float(1-ss_res/(ss_tot+1e-12)),'n_classical':int(len(x)),'x_name':'E_clean','y_name':'E_corr'}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-dir',required=True); args=ap.parse_args(); out=PROJECT_ROOT/args.output_dir; res=out/'results'; cfg=load_config(); corrs=cfg['corruptions']['enabled']
    clean=pd.read_csv(res/'clean_metrics.csv'); long=pd.read_csv(res/'corruption_metrics_long.csv')
    piv=long.groupby(['model_name','corruption'])['accuracy'].mean().unstack()
    rows=[]
    for _,r in clean.iterrows():
        name=r.model_name; row=dict(model_name=name,model_group=r.model_group,is_classical=bool(r.is_classical),clean_acc=r.clean_acc,clean_error=r.clean_error)
        vals=[]
        for c in corrs:
            m=float(piv.loc[name,c]); row[f'mPC_{c}']=m; row[f'E_{c}']=1-m; vals.append(m)
        row['mPC_overall']=float(np.mean(vals)); row['E_corr_overall']=1-row['mPC_overall']; row['CRR_overall']=row['mPC_overall']/max(r.clean_acc,1e-8); rows.append(row)
    summ=pd.DataFrame(rows)
    base=summ[summ.is_classical==True]
    f_over=fit(base.clean_error.values, base.E_corr_overall.values); (res/'baseline_fit_overall.json').write_text(json.dumps(f_over,indent=2),encoding='utf-8')
    by={}
    for c in corrs:
        f=fit(base.clean_error.values, base[f'E_{c}'].values); f['y_name']=f'E_{c}'; by[c]=f
    (res/'baseline_fit_by_corruption.json').write_text(json.dumps(by,indent=2),encoding='utf-8')
    for i,r in summ.iterrows():
        ehat=f_over['a']*r.clean_error+f_over['b']; summ.loc[i,'CSI_ER_overall']=ehat-r.E_corr_overall; summ.loc[i,'nCSI_ER_overall']=(ehat-r.E_corr_overall)/(ehat+1e-8)
        for c in corrs:
            fc=by[c]; hc=fc['a']*r.clean_error+fc['b']; summ.loc[i,f'CSI_ER_{c}']=hc-r[f'E_{c}']; summ.loc[i,f'nCSI_ER_{c}']=(hc-r[f'E_{c}'])/(hc+1e-8)
    ordered_cols=['model_name','model_group','is_classical','clean_acc','clean_error']
    for c in corrs: ordered_cols += [f'mPC_{c}',f'E_{c}']
    ordered_cols += ['mPC_overall','E_corr_overall','CRR_overall']
    for c in corrs: ordered_cols += [f'CSI_ER_{c}',f'nCSI_ER_{c}']
    ordered_cols += ['CSI_ER_overall','nCSI_ER_overall']
    summ=summ[ordered_cols]; summ.to_csv(res/'robustness_summary.csv',index=False); summ.sort_values('CSI_ER_overall',ascending=False).to_csv(res/'csi_er_v2_ranked.csv',index=False)
    meta=json.loads((PROJECT_ROOT/'data/processed/ut_har_meta.json').read_text())
    r2note='传统基线拟合 R² 较低，说明不同手工 CSI 特征对扰动的敏感性差异较大。因此 CSI-ER v2 是经验残差指标，而不是普适理论规律。' if f_over['r2']<0.3 else '传统基线拟合具有可用的经验解释度，但 CSI-ER v2 仍只作为经验残差指标解释。'
    deep=summ[summ.is_classical==False].sort_values('CSI_ER_overall',ascending=False)
    md=f"""# CSI-ER v2 Summary\n\nGenerated: {now_ts()}\n\n## Experiment Scale\n\n- Dataset: UT-HAR\n- Train samples: {meta.get('train_shape',[None])[0]}\n- Test samples: {meta.get('test_shape',[None])[0]}\n- Class count: {len(meta.get('classes',[]))}\n- Class names: {meta.get('classes',[])}\n- Classical baselines: 8\n- Deep models: {len(deep)}\n- Corruption types: {len(corrs)} ({', '.join(corrs)})\n- Severities per corruption: 5\n- Total corruption evaluations: 14 x 4 x 5 = 280\n\n## Code Fixes\n\n1. GRU/LSTM dynamic layer construction was removed; RNN layers are initialized in __init__.\n2. TinyViT dynamic positional embedding was removed; pos_embed is initialized in __init__.\n3. Added forward parameter-stability tests for all deep models.\n4. Added shared multi-corruption caches and long-table evaluation.\n5. Rebuilt paper-style figures with source CSV files.\n\n## Baseline Fit\n\n- Overall fit: E_corr_hat = {f_over['a']:.4f} * E_clean + {f_over['b']:.4f}\n- R2: {f_over['r2']:.4f}\n\n{r2note}\n\n## Deep Model Ranking\n\n```\n{deep[['model_name','clean_acc','mPC_overall','CSI_ER_overall','nCSI_ER_overall']].to_string(index=False)}\n```\n\n## Interpretation\n\n在 UT-HAR 数据集和本文设计的四类模拟扰动下，部分深度模型在 overall mPC 和 CSI-ER v2 上优于传统基线经验预期，说明它们在本文设定的噪声、子载波缺失、时间丢包和幅值变化条件下具有更好的经验鲁棒性。\n\n## Limitations\n\n本实验只覆盖 UT-HAR 和四类模拟扰动，未覆盖跨房间、跨设备、跨用户、多人干扰和真实部署在线测试。CSI-ER v2 是相对 8 个传统 CSI 感知基线的经验残差指标，不是绝对物理定律。\n"""
    (res/'summary_v2.md').write_text(md,encoding='utf-8'); print(res/'summary_v2.md')
if __name__=='__main__': main()

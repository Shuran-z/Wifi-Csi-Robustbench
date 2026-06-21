#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from csi_er.utils import PROJECT_ROOT, load_config
from csi_er.data import load_processed
from csi_er.corruptions import apply_corruption

def set_paper_style():
    mpl.rcParams.update({"figure.dpi":150,"savefig.dpi":600,"font.size":9,"axes.titlesize":10,"axes.labelsize":9,"xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":8,"axes.linewidth":0.8,"grid.linewidth":0.4,"lines.linewidth":1.6,"pdf.fonttype":42,"ps.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})

def save(figdir,name):
    plt.tight_layout(); plt.savefig(figdir/f'{name}.pdf',bbox_inches='tight'); plt.savefig(figdir/f'{name}.png',bbox_inches='tight'); plt.close()

def barh(df, x, y, title, name, figdir, color_col=None):
    df=df.sort_values(x); plt.figure(figsize=(7,4)); colors=['#4C78A8' if not c else '#F58518' for c in df.get(color_col, [False]*len(df))] if color_col else '#4C78A8'
    plt.barh(df[y], df[x], color=colors); plt.xlabel(x); plt.title(title); plt.grid(axis='x',alpha=.35)
    for i,v in enumerate(df[x]): plt.text(v+0.005, i, f'{v:.3f}', va='center', fontsize=7)
    save(figdir,name)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-dir',required=True); args=ap.parse_args(); set_paper_style(); cfg=load_config(); out=PROJECT_ROOT/args.output_dir; figdir=out/'figures'; tabdir=out/'tables/figure_source_data'; figdir.mkdir(parents=True,exist_ok=True); tabdir.mkdir(parents=True,exist_ok=True)
    clean=pd.read_csv(out/'results/clean_metrics.csv'); long=pd.read_csv(out/'results/corruption_metrics_long.csv'); rob=pd.read_csv(out/'results/robustness_summary.csv'); ranked=pd.read_csv(out/'results/csi_er_v2_ranked.csv')
    # Fig1
    plt.figure(figsize=(9,1.8)); steps=['UT-HAR CSI data','Clean training','4 corruptions','Classical + deep','Metrics','OLS fit','CSI-ER v2']; xs=np.linspace(.05,.95,len(steps))
    for x,s in zip(xs,steps): plt.text(x,.5,s,ha='center',va='center',bbox=dict(boxstyle='round,pad=.35',fc='white',ec='#4C78A8')); 
    for a,b in zip(xs[:-1],xs[1:]): plt.annotate('',xy=(b-.06,.5),xytext=(a+.06,.5),arrowprops=dict(arrowstyle='->',lw=1)); plt.axis('off'); save(figdir,'fig01_pipeline_overview')
    pd.DataFrame({'step':steps}).to_csv(tabdir/'fig01_pipeline_overview.csv',index=False)
    # Fig2
    d=load_processed(False); x=d['X_test'][:1]; corrs=cfg['corruptions']['enabled']; imgs=[('clean',x[0,0])]+[(c,apply_corruption(x,c,5,seed=42,**cfg['corruptions'].get(c,{}))[0,0]) for c in corrs]
    vmin=min(float(i[1].min()) for i in imgs); vmax=max(float(i[1].max()) for i in imgs); fig,axs=plt.subplots(1,5,figsize=(11,2.4))
    for ax,(title,img) in zip(axs,imgs): im=ax.imshow(img,aspect='auto',cmap='viridis',vmin=vmin,vmax=vmax); ax.set_title(title); ax.set_xlabel('Subcarrier'); ax.set_yticks([])
    fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.7); save(figdir,'fig02_csi_corruption_examples')
    pd.DataFrame({'panel':[i[0] for i in imgs]}).to_csv(tabdir/'fig02_csi_corruption_examples.csv',index=False)
    # Fig3/4
    clean.assign(type=clean.is_classical).to_csv(tabdir/'fig03_clean_accuracy_ranked.csv',index=False); barh(clean.assign(is_deep=~clean.is_classical.astype(bool)),'clean_acc','model_name','Clean accuracy ranked','fig03_clean_accuracy_ranked',figdir,'is_deep')
    rob.to_csv(tabdir/'fig04_overall_mpc_ranked.csv',index=False); barh(rob.assign(is_deep=~rob.is_classical.astype(bool)),'mPC_overall','model_name','Overall mPC ranked','fig04_overall_mpc_ranked',figdir,'is_deep')
    # Fig5 curves
    def curves(models,name):
        fig,axs=plt.subplots(2,2,figsize=(8,5));
        for ax,c in zip(axs.ravel(),corrs):
            for m in models:
                sub=long[(long.model_name==m)&(long.corruption==c)].sort_values('severity'); ax.plot(sub.severity,sub.accuracy,marker='o',label=m)
            ax.set_title(c); ax.set_xlabel('Severity'); ax.set_ylabel('Accuracy'); ax.grid(alpha=.3)
        axs[0,0].legend(fontsize=6); save(figdir,name)
    deep=list(clean[clean.is_classical==False].model_name); curves(deep,'fig05a_corruption_curves_deep_models'); long[long.model_name.isin(deep)].to_csv(tabdir/'fig05a_corruption_curves_deep_models.csv',index=False)
    reps=['TimeStats-SVM','FFT-SVM','PCA-kNN','Fusion-Boosting']; curves(reps,'fig05b_corruption_curves_selected_classical'); long[long.model_name.isin(reps)].to_csv(tabdir/'fig05b_corruption_curves_selected_classical.csv',index=False)
    # Fig6
    fit=json.load(open(out/'results/baseline_fit_overall.json')); plt.figure(figsize=(5,4)); cl=rob[rob.is_classical==True]; dp=rob[rob.is_classical==False]; plt.scatter(cl.clean_error,cl.E_corr_overall,label='Classical',marker='o'); plt.scatter(dp.clean_error,dp.E_corr_overall,label='Deep',marker='^'); xs=np.linspace(rob.clean_error.min(),rob.clean_error.max(),100); plt.plot(xs,fit['a']*xs+fit['b'],'--',label='OLS fit'); plt.text(.02,.95,f"E={fit['a']:.2f}Ec+{fit['b']:.2f}\nR2={fit['r2']:.3f}",transform=plt.gca().transAxes,va='top'); plt.xlabel('Clean error'); plt.ylabel('Overall corrupted error'); plt.legend(); plt.grid(alpha=.3); save(figdir,'fig06_classical_fit_overall'); rob.to_csv(tabdir/'fig06_classical_fit_overall.csv',index=False)
    # Fig7
    by=json.load(open(out/'results/baseline_fit_by_corruption.json')); fig,axs=plt.subplots(2,2,figsize=(8,6))
    for ax,c in zip(axs.ravel(),corrs):
        y=f'E_{c}'; ax.scatter(cl.clean_error,cl[y],label='Classical'); ax.scatter(dp.clean_error,dp[y],marker='^',label='Deep'); xs=np.linspace(rob.clean_error.min(),rob.clean_error.max(),100); ax.plot(xs,by[c]['a']*xs+by[c]['b'],'--'); ax.set_title(f"{c} R2={by[c]['r2']:.3f}"); ax.set_xlabel('Clean error'); ax.set_ylabel(y); ax.grid(alpha=.3)
    axs[0,0].legend(fontsize=6); save(figdir,'fig07_classical_fit_by_corruption'); rob.to_csv(tabdir/'fig07_classical_fit_by_corruption.csv',index=False)
    # Fig8/9/10
    deep_rank=ranked[ranked.is_classical==False].copy(); deep_rank.to_csv(tabdir/'fig08_csi_er_overall_ranked.csv',index=False); barh(deep_rank,'CSI_ER_overall','model_name','CSI-ER v2 overall ranked','fig08_csi_er_overall_ranked',figdir)
    heat=deep_rank.set_index('model_name')[[f'CSI_ER_{c}' for c in corrs]+['CSI_ER_overall']]; heat.to_csv(tabdir/'fig09_csi_er_by_corruption_heatmap.csv'); plt.figure(figsize=(7,3.5)); lim=max(abs(heat.min().min()),abs(heat.max().max())); plt.imshow(heat.values,cmap='coolwarm',vmin=-lim,vmax=lim,aspect='auto'); plt.colorbar(label='CSI-ER'); plt.xticks(range(heat.shape[1]),[c.replace('_',' ') for c in corrs]+['overall'],rotation=25,ha='right'); plt.yticks(range(len(heat)),heat.index); save(figdir,'fig09_csi_er_by_corruption_heatmap')
    rob.to_csv(tabdir/'fig10_clean_vs_mpc_scatter.csv',index=False); plt.figure(figsize=(5,4));
    for g,grp in rob.groupby('model_group'): plt.scatter(grp.clean_acc,grp.mPC_overall,label=g)
    plt.xlabel('Clean accuracy'); plt.ylabel('Overall mPC'); plt.legend(fontsize=6); plt.grid(alpha=.3); save(figdir,'fig10_clean_vs_mpc_scatter')
    # Fig11 confusion
    conf=json.load(open(out/'results/confusion_matrices_v2.json')); best=deep_rank.sort_values('mPC_overall',ascending=False).iloc[0].model_name; sub=long[long.model_name==best].sort_values('accuracy').iloc[0]; clean_mat=np.array(conf[f'{best}|clean']); hard_mat=np.array(conf[f"{best}|{sub.corruption}|s{int(sub.severity)}"]); fig,axs=plt.subplots(1,2,figsize=(7,3)); ConfusionMatrixDisplay(clean_mat).plot(ax=axs[0],cmap='Blues',colorbar=False,values_format='d'); axs[0].set_title(f'{best} clean'); ConfusionMatrixDisplay(hard_mat).plot(ax=axs[1],cmap='Blues',colorbar=False,values_format='d'); axs[1].set_title(f'{sub.corruption} s{int(sub.severity)}'); save(figdir,'fig11_confusion_best_model_clean_vs_hardest'); pd.DataFrame([{'best_model':best,'hardest_corruption':sub.corruption,'severity':sub.severity,'accuracy':sub.accuracy}]).to_csv(tabdir/'fig11_confusion_best_model_clean_vs_hardest.csv',index=False)
    # Fig12 summary
    top=deep_rank[['model_name','clean_acc','mPC_overall','CSI_ER_overall']].set_index('model_name'); top.to_csv(tabdir/'fig12_summary_panel.csv'); top.plot(kind='bar',figsize=(7,3)); plt.xticks(rotation=25,ha='right'); plt.grid(axis='y',alpha=.3); save(figdir,'fig12_summary_panel')
    readme='\n'.join([f'- {p.name}: source data in tables/figure_source_data/{p.stem}.csv' for p in sorted(figdir.glob('fig*.pdf'))]); (figdir/'README_figures.md').write_text('# Figure Guide\n\n'+readme+'\n',encoding='utf-8')
    print(figdir)
if __name__=='__main__': main()

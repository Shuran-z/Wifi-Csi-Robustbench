from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from .corruptions import add_gaussian_noise
from .data import load_processed
from .utils import PROJECT_ROOT, ensure_dirs

def _save(name):
    for ext in ['png','pdf']: plt.savefig(PROJECT_ROOT/f'outputs/figures/{name}.{ext}', dpi=300, bbox_inches='tight')
    plt.close()

def plot_all(config, smoke=False):
    ensure_dirs(); figdir=PROJECT_ROOT/'outputs/figures'; figdir.mkdir(parents=True,exist_ok=True)
    d=load_processed(smoke=smoke); x=d['X_test'][0,0]
    plt.figure(figsize=(7,3)); plt.imshow(x,aspect='auto',cmap='viridis'); plt.colorbar(label='Normalized CSI'); plt.title('Clean CSI sample'); plt.xlabel('Subcarrier'); plt.ylabel('Time'); _save('fig01_csi_clean_heatmap')
    xn=add_gaussian_noise(d['X_test'][:1],5,alphas=config['corruption']['alphas'],seed=42)[0,0]
    fig,ax=plt.subplots(1,2,figsize=(9,3)); ax[0].imshow(x,aspect='auto',cmap='viridis'); ax[0].set_title('Clean'); ax[1].imshow(xn,aspect='auto',cmap='viridis'); ax[1].set_title('Gaussian s5'); _save('fig02_csi_noisy_heatmap')
    m=pd.read_csv(PROJECT_ROOT/'outputs/results/all_metrics.csv'); er=pd.read_csv(PROJECT_ROOT/'outputs/results/csi_er.csv')
    m.plot.bar(x='model_name',y='clean_acc',legend=False,figsize=(8,3)); plt.ylabel('Clean accuracy'); plt.xticks(rotation=35,ha='right'); _save('fig03_clean_acc_bar')
    m.plot.bar(x='model_name',y='mean_noise_acc',legend=False,figsize=(8,3)); plt.ylabel('Mean noisy accuracy'); plt.xticks(rotation=35,ha='right'); _save('fig04_mean_noise_acc_bar')
    plt.figure(figsize=(8,4));
    for _,r in m.iterrows(): plt.plot(config['corruption']['severities'], [r[f'noise_s{s}_acc'] for s in config['corruption']['severities']], marker='o', label=r['model_name'])
    plt.xlabel('Noise severity'); plt.ylabel('Accuracy'); plt.legend(fontsize=6,ncol=2); _save('fig05_noise_severity_curves')
    plt.figure(figsize=(6,4)); cls=m[m['is_classical'].astype(str).isin(['True','true','1'])]; dep=m[~m.index.isin(cls.index)]
    plt.scatter(cls['clean_error'],cls['noise_error'],label='Classical'); plt.scatter(dep['clean_error'],dep['noise_error'],label='Deep')
    bf=json.load(open(PROJECT_ROOT/'outputs/results/baseline_fit.json')); xs=np.linspace(m['clean_error'].min(),m['clean_error'].max(),50); plt.plot(xs,bf['a']*xs+bf['b'],label='Classical linear fit'); plt.xlabel('Clean error'); plt.ylabel('Noise error'); plt.legend(); _save('fig06_classical_fit_clean_error_vs_noise_error')
    deep=er[~er['is_classical'].astype(str).isin(['True','true','1'])]
    deep.plot.bar(x='model_name',y='CSI_ER',legend=False,figsize=(7,3)); plt.axhline(0,color='black',lw=0.8); plt.ylabel('CSI-ER'); plt.xticks(rotation=35,ha='right'); _save('fig07_csi_er_bar')
    deep.plot.bar(x='model_name',y='nCSI_ER',legend=False,figsize=(7,3)); plt.axhline(0,color='black',lw=0.8); plt.ylabel('nCSI-ER'); plt.xticks(rotation=35,ha='right'); _save('fig08_ncsi_er_bar')
    plt.figure(figsize=(6,4));
    for g,grp in m.groupby('model_group'): plt.scatter(grp['clean_acc'],grp['mean_noise_acc'],label=g)
    plt.xlabel('Clean accuracy'); plt.ylabel('Mean noisy accuracy'); plt.legend(); _save('fig09_clean_vs_noise_scatter')
    fam=m.groupby('model_group')[['clean_acc','mean_noise_acc']].mean(); fam.plot.bar(figsize=(7,3)); plt.ylabel('Accuracy'); plt.xticks(rotation=25,ha='right'); _save('fig10_family_summary')
    confp=PROJECT_ROOT/'outputs/results/confusion_matrices.json'
    if confp.exists():
        conf=json.load(open(confp));
        if conf:
            best=m.sort_values('clean_acc',ascending=False).iloc[0]['model_name']; key=f'{best}_clean'; mat=np.array(conf.get(key, next(iter(conf.values())))); ConfusionMatrixDisplay(mat).plot(cmap='Blues',values_format='d'); plt.title('Best clean confusion matrix'); _save('fig11_confusion_matrix_best_clean')
            bestn=m.sort_values('noise_s5_acc',ascending=False).iloc[0]['model_name']; key=f'{bestn}_s5'; mat=np.array(conf.get(key, next(iter(conf.values())))); ConfusionMatrixDisplay(mat).plot(cmap='Blues',values_format='d'); plt.title('Best noisy s5 confusion matrix'); _save('fig12_confusion_matrix_best_noisy_s5')
    return sorted(str(p) for p in figdir.glob('fig*.png'))

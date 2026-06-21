from __future__ import annotations
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from .corruptions import add_gaussian_noise
from .data import load_processed
from .features import PCAFeatureExtractor, time_stats_features, fft_features, fusion_features
from .utils import PROJECT_ROOT, ensure_dirs, now_ts

def _clf(name, seed):
    if name == 'svm': return make_pipeline(StandardScaler(), SVC(C=3.0, gamma='scale', kernel='rbf'))
    if name == 'rf': return RandomForestClassifier(n_estimators=120, max_depth=None, n_jobs=-1, random_state=seed)
    if name == 'knn': return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    if name == 'logreg': return make_pipeline(StandardScaler(), LogisticRegression(max_iter=800, n_jobs=-1, random_state=seed))
    if name == 'boost':
        return HistGradientBoostingClassifier(max_iter=180, random_state=seed)
    raise KeyError(name)

FEATURE_KIND = {
    'TimeStats-SVM':'TimeStats','TimeStats-RF':'TimeStats','FFT-SVM':'FFT','FFT-RF':'FFT',
    'PCA-kNN':'PCA','PCA-LogReg':'PCA','StatsFFT-SVM':'StatsFFT','Fusion-Boosting':'Fusion'
}

def compute_feature(kind, X, pca=None):
    if kind == 'TimeStats': return time_stats_features(X)
    if kind == 'FFT': return fft_features(X)
    if kind == 'PCA': return pca.transform(X)
    if kind == 'StatsFFT': return fusion_features(X)
    if kind == 'Fusion': return fusion_features(X, pca.transform(X))
    raise KeyError(kind)

def run_classical(config, smoke=False, output_dir=None):
    ensure_dirs(); seed = config['classical'].get('random_seed', 42)
    out = PROJECT_ROOT / (output_dir or config['paths'].get('outputs','outputs'))
    (out/'checkpoints/classical').mkdir(parents=True, exist_ok=True); (out/'results').mkdir(parents=True, exist_ok=True)
    data = load_processed(smoke=smoke); Xtr, ytr, Xte, yte = data['X_train'], data['y_train'], data['X_test'], data['y_test']
    pca = PCAFeatureExtractor(config['classical'].get('pca_components', 64), seed)
    pca_tr = pca.fit_transform(Xtr); pca_te = pca.transform(Xte); joblib.dump(pca, out/'checkpoints/pca.joblib')
    feats = {'TimeStats': (time_stats_features(Xtr), time_stats_features(Xte)), 'FFT': (fft_features(Xtr), fft_features(Xte)), 'PCA': (pca_tr, pca_te), 'StatsFFT': (fusion_features(Xtr), fusion_features(Xte)), 'Fusion': (fusion_features(Xtr, pca_tr), fusion_features(Xte, pca_te))}
    specs=[('TimeStats-SVM','TimeStats','svm'),('TimeStats-RF','TimeStats','rf'),('FFT-SVM','FFT','svm'),('FFT-RF','FFT','rf'),('PCA-kNN','PCA','knn'),('PCA-LogReg','PCA','logreg'),('StatsFFT-SVM','StatsFFT','svm'),('Fusion-Boosting','Fusion','boost')]
    rows=[]
    for model_name, feat_name, clf_name in specs:
        Ftr,Fte=feats[feat_name]; model=_clf(clf_name,seed); t0=time.time(); model.fit(Ftr,ytr); train_time=time.time()-t0
        t1=time.time(); pred=model.predict(Fte); infer=(time.time()-t1)*1000/max(1,len(Fte))
        clean_acc=accuracy_score(yte,pred); clean_f1=f1_score(yte,pred,average='macro',zero_division=0)
        row=dict(timestamp=now_ts(), train_seed=seed, seed=seed, model_name=model_name, model_group='Classical', is_classical=True, dataset='UT_HAR', clean_acc=float(clean_acc), clean_macro_f1=float(clean_f1), clean_error=float(1-clean_acc), train_time_sec=float(train_time), infer_time_ms_per_sample=float(infer), params_proxy=0, param_count=0, feature_dim=int(Ftr.shape[1]))
        rows.append(row); joblib.dump(model, out/f'checkpoints/classical/{model_name}.joblib')
    df=pd.DataFrame(rows); df.to_csv(out/'results/classical_clean_metrics.csv',index=False)
    return df

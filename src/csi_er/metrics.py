from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def classification_metrics(y_true, y_pred):
    return {
        'acc': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
    }

def error_fields(clean_acc, noise_accs):
    mean_noise_acc = float(np.mean(noise_accs))
    return {
        'mean_noise_acc': mean_noise_acc,
        'clean_error': float(1.0 - clean_acc),
        'noise_error': float(1.0 - mean_noise_acc),
    }

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

def evaluate(y_true, y_pred, cls):
    """Same metrics for every model: macro-F1, abnormal precision/recall, recall per subtype."""
    return {
        'macro_f1':     round(float(f1_score(y_true, y_pred, average='macro')), 4),
        'ab_precision': round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        'ab_recall':    round(float(recall_score(y_true, y_pred)), 4),
        **{f'recall_{c}': round(float((y_pred[cls == c] == 1).mean()), 4) for c in 'SVF'},
    }

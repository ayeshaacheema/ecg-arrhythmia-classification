import numpy as np
from config import BEFORE, FS

R = BEFORE
SHAPE_COLS = ['r_amp', 'q_amp', 's_amp', 'qrs_width_ms', 'qrs_area',
              't_max', 't_min', 'beat_std', 'peak_to_peak']

def qrs_width_ms(a, max_half=40):
    half = 0.5 * a[R]
    l = R
    while l > R - max_half and a[l] > half:
        l -= 1
    r = R
    while r < R + max_half and a[r] > half:
        r += 1
    return (r - l) / FS * 1000

def shape_features(X_mv):
    x = X_mv - np.median(X_mv[:, :60], axis=1, keepdims=True)
    a = np.abs(x)
    return np.column_stack([
        x[:, R],
        x[:, R - 30:R].min(axis=1),
        x[:, R:R + 30].min(axis=1),
        np.array([qrs_width_ms(row) for row in a]),
        a[:, R - 20:R + 30].sum(axis=1) / FS,
        x[:, R + 50:].max(axis=1),
        x[:, R + 50:].min(axis=1),
        x.std(axis=1),
        x.max(axis=1) - x.min(axis=1),
    ]).astype(np.float32)

def rf_features(d):
    """8 RR features + 9 shape features, same order as Phase 4."""
    return np.hstack([d['rr'], shape_features(d['X_mv'])])

"""Feature engineering for real sensor data — minimal set."""

import numpy as np


def calc_signal_rms(signal):
    """Root Mean Square of a signal array."""
    if len(signal) == 0:
        return 0.0
    return float(np.sqrt(np.mean(signal ** 2)))


def calc_asymmetry_index(left_rms, right_rms):
    """Asymmetry index: |L - R| / mean(L, R). Returns 0-2 range."""
    mean_val = (left_rms + right_rms) / 2
    if mean_val == 0:
        return 0.0
    return abs(left_rms - right_rms) / mean_val


def calc_signal_stats(signal):
    """Basic statistics for a signal array."""
    if len(signal) == 0:
        return {'mean': 0, 'std': 0, 'rms': 0, 'max': 0, 'min': 0}
    return {
        'mean': float(np.mean(signal)),
        'std': float(np.std(signal)),
        'rms': calc_signal_rms(signal),
        'max': float(np.max(signal)),
        'min': float(np.min(signal)),
    }

"""Feature engineering for real sensor data — PD Clinical Decision Support."""

import numpy as np


# ──────────────────────────────────────────────────────────────
# Phase 1: New feature functions (numpy-only, no scipy)
# ──────────────────────────────────────────────────────────────

def detect_peaks(signal, min_dist=10, thresh=None):
    """Detect local maxima in a 1-D signal.

    Parameters
    ----------
    signal : array-like
        Input signal.
    min_dist : int
        Minimum samples between peaks.
    thresh : float or None
        Minimum amplitude threshold (absolute). If None, uses mean + 0.5*std.

    Returns
    -------
    indices : np.ndarray
        Peak indices.
    amplitudes : np.ndarray
        Peak amplitudes.
    """
    signal = np.asarray(signal, dtype=float)
    if len(signal) < 3:
        return np.array([], dtype=int), np.array([], dtype=float)

    if thresh is None:
        thresh = np.mean(signal) + 0.5 * np.std(signal)

    # Find all local maxima
    candidates = []
    for i in range(1, len(signal) - 1):
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            if signal[i] >= thresh:
                candidates.append(i)

    if len(candidates) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    # Enforce minimum distance
    candidates = np.array(candidates)
    selected = [candidates[0]]
    for idx in candidates[1:]:
        if idx - selected[-1] >= min_dist:
            selected.append(idx)

    indices = np.array(selected)
    amplitudes = signal[indices]
    return indices, amplitudes


def calc_tremor_power(signal, fs=100.0):
    """Calculate tremor-band (4-12 Hz) power via FFT PSD integration.

    Parameters
    ----------
    signal : array-like
        Acceleration magnitude or single-axis signal.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    float
        Integrated power in 4-12 Hz band.
    """
    signal = np.asarray(signal, dtype=float)
    if len(signal) < 16:
        return 0.0

    # Remove DC
    signal = signal - np.mean(signal)

    # FFT-based PSD
    n = len(signal)
    fft_vals = np.fft.rfft(signal)
    psd = (np.abs(fft_vals) ** 2) / (n * fs)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    # Integrate 4-12 Hz band
    mask = (freqs >= 4.0) & (freqs <= 12.0)
    if not np.any(mask):
        return 0.0
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    return float(np.sum(psd[mask]) * df)


def calc_rhythm_irregularity(signal, fs=100.0):
    """Calculate rhythm irregularity as CV of inter-peak intervals.

    Parameters
    ----------
    signal : array-like
        Motion signal (e.g., accel_mag).
    fs : float
        Sampling frequency.

    Returns
    -------
    float
        Coefficient of variation of inter-peak intervals. 0 if < 3 peaks.
    """
    signal = np.asarray(signal, dtype=float)
    min_dist = max(int(fs * 0.15), 5)  # at least 150ms between peaks
    indices, _ = detect_peaks(signal, min_dist=min_dist)

    if len(indices) < 3:
        return 0.0

    intervals = np.diff(indices) / fs  # in seconds
    mean_interval = np.mean(intervals)
    if mean_interval == 0:
        return 0.0
    return float(np.std(intervals) / mean_interval)


def calc_mean_jerk(signal, fs=100.0):
    """Mean jerk: mean(|diff(signal)|) * fs.

    Parameters
    ----------
    signal : array-like
        Acceleration magnitude signal.
    fs : float
        Sampling frequency.

    Returns
    -------
    float
        Mean jerk value.
    """
    signal = np.asarray(signal, dtype=float)
    if len(signal) < 2:
        return 0.0
    jerk = np.abs(np.diff(signal)) * fs
    return float(np.mean(jerk))


def compute_stft(signal, fs=100.0, win=256, hop=64):
    """Compute Short-Time Fourier Transform using numpy.

    Parameters
    ----------
    signal : array-like
        Input signal.
    fs : float
        Sampling frequency.
    win : int
        Window size in samples.
    hop : int
        Hop size in samples.

    Returns
    -------
    times : np.ndarray
        Time centers of each frame.
    freqs : np.ndarray
        Frequency bins (0 to fs/2).
    magnitude : np.ndarray
        STFT magnitude matrix (n_freqs × n_frames).
    """
    signal = np.asarray(signal, dtype=float)
    if len(signal) < win:
        win = len(signal)
        hop = max(1, win // 4)

    # Hann window
    window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(win) / win))

    n_frames = max(1, (len(signal) - win) // hop + 1)
    n_freqs = win // 2 + 1

    magnitude = np.zeros((n_freqs, n_frames))
    times = np.zeros(n_frames)

    for i in range(n_frames):
        start = i * hop
        frame = signal[start:start + win] * window
        fft_vals = np.fft.rfft(frame)
        magnitude[:, i] = np.abs(fft_vals)
        times[i] = (start + win / 2) / fs

    freqs = np.fft.rfftfreq(win, d=1.0 / fs)
    return times, freqs, magnitude


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

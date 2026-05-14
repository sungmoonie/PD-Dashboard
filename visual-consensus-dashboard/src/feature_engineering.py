"""Feature engineering and score calculation functions."""

import numpy as np
from scipy.signal import find_peaks


def calc_asymmetry_index(left, right):
    """Calculate asymmetry index between left and right values."""
    mean_val = (left + right) / 2
    if mean_val == 0:
        return 0.0
    return abs(left - right) / mean_val


def calc_rhythm_irregularity(ts_df):
    """Calculate rhythm irregularity as CV of inter-peak intervals."""
    signal = ts_df['left_amplitude'].values
    peaks, _ = find_peaks(signal, distance=5, prominence=0.05)
    if len(peaks) < 3:
        return 0.0
    times = ts_df['time'].values
    intervals = np.diff(times[peaks])
    if intervals.mean() == 0:
        return 0.0
    return intervals.std() / intervals.mean()


def calc_amplitude_decrement(ts_df):
    """Calculate amplitude decrement ratio (early vs late peaks)."""
    signal = ts_df['left_amplitude'].values
    peaks, _ = find_peaks(signal, distance=5, prominence=0.05)
    if len(peaks) < 6:
        return 0.0
    peak_amps = signal[peaks]
    n = min(3, len(peak_amps) // 2)
    early_mean = peak_amps[:n].mean()
    late_mean = peak_amps[-n:].mean()
    if early_mean == 0:
        return 0.0
    return (early_mean - late_mean) / early_mean


def calc_patient_summary(patient_id, task_features_df):
    """Calculate mean feature scores across all tasks for a patient."""
    pf = task_features_df[task_features_df.patient_id == patient_id]
    summary = {}
    for feature in pf['feature'].unique():
        vals = pf[pf.feature == feature]['value']
        summary[feature] = round(vals.mean(), 3)
    return summary


def get_top_abnormal_tasks(patient_id, task_features_df, n=3):
    """Return top n tasks ranked by mean abnormality score."""
    pf = task_features_df[task_features_df.patient_id == patient_id]
    task_scores = pf.groupby('task')['value'].mean().reset_index()
    task_scores.columns = ['task', 'mean_score']
    task_scores = task_scores.sort_values('mean_score', ascending=False).head(n)
    return task_scores


def calc_normative_position(patient_id, task_features_df, normative_df):
    """Calculate where the patient falls relative to healthy norms."""
    pf = task_features_df[task_features_df.patient_id == patient_id]
    results = []
    for _, row in pf.iterrows():
        norm = normative_df[
            (normative_df.task == row.task) &
            (normative_df.feature == row.feature)
        ]
        if norm.empty:
            continue
        n = norm.iloc[0]
        # Z-score relative to healthy population
        z_score = (row.value - n.healthy_mean) / n.healthy_std if n.healthy_std > 0 else 0
        results.append({
            'task': row.task,
            'feature': row.feature,
            'patient_value': row.value,
            'healthy_mean': n.healthy_mean,
            'healthy_std': n.healthy_std,
            'z_score': round(z_score, 2),
            'percentile': row.normal_percentile,
        })
    return results


def calc_visit_changes(patient_id, history_df):
    """Calculate feature changes between the latest two visits."""
    ph = history_df[history_df.patient_id == patient_id].sort_values('visit_number')
    visits = ph['visit_number'].unique()
    if len(visits) < 2:
        return []

    latest = visits[-1]
    previous = visits[-2]
    latest_data = ph[ph.visit_number == latest].set_index('feature')
    prev_data = ph[ph.visit_number == previous].set_index('feature')

    changes = []
    for feat in latest_data.index:
        if feat not in prev_data.index:
            continue
        curr = latest_data.loc[feat, 'value']
        prev = prev_data.loc[feat, 'value']
        diff = curr - prev
        pct_change = (diff / prev * 100) if prev != 0 else 0
        changes.append({
            'feature': feat,
            'current': round(curr, 3),
            'previous': round(prev, 3),
            'change': round(diff, 3),
            'pct_change': round(pct_change, 1),
            'direction': 'worsened' if diff > 0.02 else 'improved' if diff < -0.02 else 'stable',
        })
    return changes


def estimate_updrs_scores(patient_id, task_features_df):
    """Estimate UPDRS Part III item scores from sensor features.

    Mapping (simplified):
    - rest_tremor + tremor_power → 3.17 Rest Tremor Amplitude
    - finger_tapping + rhythm + amplitude → 3.4 Finger Tapping
    - hand_open_close + amplitude → 3.5 Hand Movements
    - gait + instability + asymmetry → 3.10 Gait
    - toe_tapping + rhythm → 3.7 Toe Tapping
    """
    pf = task_features_df[task_features_df.patient_id == patient_id]

    def _get(task, feature):
        row = pf[(pf.task == task) & (pf.feature == feature)]
        return row.iloc[0]['value'] if len(row) > 0 else 0

    items = [
        {
            'item_code': '3.17',
            'item_name': 'Rest Tremor Amplitude',
            'estimated_score': min(4, round(
                _get('rest_tremor', 'tremor_power') * 4.5
            , 1)),
            'confidence': 0.75,
            'evidence': [
                f"Tremor Power: {_get('rest_tremor', 'tremor_power'):.2f}",
                f"Instability: {_get('rest_tremor', 'motion_instability'):.2f}",
            ],
        },
        {
            'item_code': '3.4',
            'item_name': 'Finger Tapping',
            'estimated_score': min(4, round(
                (_get('finger_tapping', 'rhythm_irregularity') * 2 +
                 _get('finger_tapping', 'movement_amplitude_reduction') * 2 +
                 _get('finger_tapping', 'motion_instability')) / 1.2
            , 1)),
            'confidence': 0.82,
            'evidence': [
                f"Rhythm Irregularity: {_get('finger_tapping', 'rhythm_irregularity'):.2f}",
                f"Amplitude Reduction: {_get('finger_tapping', 'movement_amplitude_reduction'):.2f}",
                f"Instability: {_get('finger_tapping', 'motion_instability'):.2f}",
            ],
        },
        {
            'item_code': '3.5',
            'item_name': 'Hand Movements',
            'estimated_score': min(4, round(
                (_get('hand_open_close', 'movement_amplitude_reduction') * 3 +
                 _get('hand_open_close', 'rhythm_irregularity') * 2) / 1.2
            , 1)),
            'confidence': 0.78,
            'evidence': [
                f"Amplitude Reduction: {_get('hand_open_close', 'movement_amplitude_reduction'):.2f}",
                f"Rhythm Irregularity: {_get('hand_open_close', 'rhythm_irregularity'):.2f}",
            ],
        },
        {
            'item_code': '3.7',
            'item_name': 'Toe Tapping',
            'estimated_score': min(4, round(
                (_get('toe_tapping', 'rhythm_irregularity') * 2.5 +
                 _get('toe_tapping', 'movement_amplitude_reduction') * 1.5 +
                 _get('toe_tapping', 'left_right_asymmetry')) / 1.2
            , 1)),
            'confidence': 0.74,
            'evidence': [
                f"Rhythm Irregularity: {_get('toe_tapping', 'rhythm_irregularity'):.2f}",
                f"Asymmetry: {_get('toe_tapping', 'left_right_asymmetry'):.2f}",
            ],
        },
        {
            'item_code': '3.10',
            'item_name': 'Gait',
            'estimated_score': min(4, round(
                (_get('gait', 'motion_instability') * 2 +
                 _get('gait', 'left_right_asymmetry') * 2 +
                 _get('gait', 'rhythm_irregularity')) / 1.2
            , 1)),
            'confidence': 0.70,
            'evidence': [
                f"Instability: {_get('gait', 'motion_instability'):.2f}",
                f"Asymmetry: {_get('gait', 'left_right_asymmetry'):.2f}",
                f"Rhythm: {_get('gait', 'rhythm_irregularity'):.2f}",
            ],
        },
        {
            'item_code': '3.8',
            'item_name': 'Leg Agility (Touch Nose proxy)',
            'estimated_score': min(4, round(
                (_get('touch_nose', 'motion_instability') * 2.5 +
                 _get('touch_nose', 'tremor_power') * 1.5) / 1.0
            , 1)),
            'confidence': 0.65,
            'evidence': [
                f"Instability: {_get('touch_nose', 'motion_instability'):.2f}",
                f"Tremor: {_get('touch_nose', 'tremor_power'):.2f}",
            ],
        },
    ]

    # Clamp scores
    for item in items:
        item['estimated_score'] = min(4.0, max(0.0, item['estimated_score']))

    return items

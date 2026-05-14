"""Generate realistic demo data for Clinical Motor Assessment Dashboard."""

import pandas as pd
import numpy as np
import json
import os

np.random.seed(42)
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# ── Configuration ────────────────────────────────────────────
PATIENTS = [
    ('P001', 'PD', 68, 'F', 'right', 72, 41),
    ('P002', 'PD', 61, 'M', 'right', 58, 35),
    ('P003', 'Healthy', 59, 'F', 'left', 12, 8),
    ('P004', 'Differential', 65, 'M', 'right', 43, 22),
    ('P005', 'PD', 72, 'M', 'left', 80, 46),
]

TASKS = ['finger_tapping', 'hand_open_close', 'rest_tremor',
         'gait', 'toe_tapping', 'touch_nose']

FEATURES = ['tremor_power', 'movement_amplitude_reduction',
            'rhythm_irregularity', 'left_right_asymmetry', 'motion_instability']

LR_FEATURES = ['amplitude', 'tremor_power', 'rhythm_irregularity', 'velocity']

PROFILES = {
    'PD': {'base': 0.55, 'range': 0.30},
    'Healthy': {'base': 0.15, 'range': 0.10},
    'Differential': {'base': 0.35, 'range': 0.18},
}

TASK_BIAS = {
    'finger_tapping': {'rhythm_irregularity': 0.12, 'movement_amplitude_reduction': 0.08},
    'rest_tremor': {'tremor_power': 0.15},
    'gait': {'left_right_asymmetry': 0.10, 'motion_instability': 0.12},
    'hand_open_close': {'movement_amplitude_reduction': 0.10},
    'toe_tapping': {'rhythm_irregularity': 0.08, 'left_right_asymmetry': 0.06},
    'touch_nose': {'motion_instability': 0.08},
}

PAT_MULT = {'P001': 1.0, 'P002': 0.82, 'P003': 1.0, 'P004': 1.0, 'P005': 1.15}


# ── 1. patients.csv ─────────────────────────────────────────
def gen_patients():
    df = pd.DataFrame(PATIENTS,
                      columns=['patient_id', 'group', 'age', 'sex',
                               'handedness', 'motor_score', 'non_motor_score'])
    df.to_csv(os.path.join(DATA_DIR, 'patients.csv'), index=False)
    print(f"patients.csv: {len(df)} rows")


# ── 2. task_features.csv ────────────────────────────────────
def gen_task_features():
    rows = []
    for pid, group, *_ in PATIENTS:
        prof = PROFILES[group]
        mult = PAT_MULT[pid]
        for task in TASKS:
            for feat in FEATURES:
                base = prof['base'] * mult
                noise = np.random.uniform(-prof['range'], prof['range'])
                bias = TASK_BIAS.get(task, {}).get(feat, 0)
                value = np.clip(base + noise + bias, 0.02, 0.98)
                percentile = int(np.clip(value * 100 + np.random.randint(-5, 6), 5, 99))
                rows.append([pid, task, feat, round(value, 2), percentile])

    df = pd.DataFrame(rows, columns=['patient_id', 'task', 'feature',
                                      'value', 'normal_percentile'])
    df.to_csv(os.path.join(DATA_DIR, 'task_features.csv'), index=False)
    print(f"task_features.csv: {len(df)} rows")


# ── 3. left_right_features.csv ──────────────────────────────
def gen_left_right_features():
    rows = []
    for pid, group, *_ in PATIENTS:
        prof = PROFILES[group]
        mult = PAT_MULT[pid]
        for task in TASKS:
            for feat in LR_FEATURES:
                base = prof['base'] * mult * 0.8
                left = np.clip(base + np.random.uniform(-0.15, 0.25), 0.05, 0.95)
                right = np.clip(base + np.random.uniform(-0.15, 0.25), 0.05, 0.95)
                if group == 'PD':
                    side = np.random.choice(['left', 'right'])
                    if side == 'left':
                        left *= np.random.uniform(1.2, 1.6)
                    else:
                        right *= np.random.uniform(1.2, 1.6)
                    left = min(left, 0.95)
                    right = min(right, 0.95)

                mean_val = (left + right) / 2
                asym = round(abs(left - right) / mean_val if mean_val > 0 else 0, 2)
                pct = int(np.clip(asym * 100 + np.random.randint(-5, 10), 5, 99))
                rows.append([pid, task, feat, round(left, 2), round(right, 2), asym, pct])

    df = pd.DataFrame(rows, columns=['patient_id', 'task', 'feature',
                                      'left_value', 'right_value',
                                      'asymmetry_index', 'normal_percentile'])
    df.to_csv(os.path.join(DATA_DIR, 'left_right_features.csv'), index=False)
    print(f"left_right_features.csv: {len(df)} rows")


# ── 4. timeseries.csv ───────────────────────────────────────
def gen_timeseries():
    rows = []
    dt = 0.05
    n_points = 200

    # Task-specific frequency (Hz) for realistic patterns
    TASK_FREQ = {
        'finger_tapping': 2.5, 'hand_open_close': 2.0, 'rest_tremor': 5.0,
        'gait': 1.8, 'toe_tapping': 2.2, 'touch_nose': 1.5,
    }

    for pid, group, *_ in PATIENTS:
        for task in TASKS:
            t = np.arange(0, n_points * dt, dt)[:n_points]
            freq_base = TASK_FREQ.get(task, 2.5)

            if group == 'PD':
                mult = PAT_MULT[pid]
                freq_noise = np.cumsum(np.random.normal(0, 0.01, n_points))
                phase = np.cumsum(2 * np.pi * (freq_base + freq_noise) * dt)

                amp_decay_l = np.linspace(0.85, 0.85 - 0.35 * mult, n_points)
                left_amp = amp_decay_l * (0.5 + 0.5 * np.sin(phase)) + \
                           np.random.normal(0, 0.03, n_points)
                left_amp = np.clip(left_amp, 0, 1)

                amp_decay_r = np.linspace(0.75, 0.75 - 0.15 * mult, n_points)
                right_amp = amp_decay_r * (0.5 + 0.5 * np.sin(phase + 0.2)) + \
                            np.random.normal(0, 0.02, n_points)
                right_amp = np.clip(right_amp, 0, 1)
            else:
                phase = 2 * np.pi * freq_base * t
                stability = 0.95 if group == 'Healthy' else 0.85

                left_amp = stability * (0.5 + 0.5 * np.sin(phase)) + \
                           np.random.normal(0, 0.015, n_points)
                right_amp = stability * (0.5 + 0.5 * np.sin(phase + 0.1)) + \
                            np.random.normal(0, 0.015, n_points)
                left_amp = np.clip(left_amp, 0, 1)
                right_amp = np.clip(right_amp, 0, 1)

            left_vel = np.abs(np.gradient(left_amp, dt))
            right_vel = np.abs(np.gradient(right_amp, dt))

            for i in range(n_points):
                rows.append([
                    pid, task, round(t[i], 2),
                    round(left_amp[i], 4), round(right_amp[i], 4),
                    round(left_vel[i], 4), round(right_vel[i], 4),
                ])

    df = pd.DataFrame(rows, columns=['patient_id', 'task', 'time',
                                      'left_amplitude', 'right_amplitude',
                                      'left_velocity', 'right_velocity'])
    df.to_csv(os.path.join(DATA_DIR, 'timeseries.csv'), index=False)
    print(f"timeseries.csv: {len(df)} rows")


# ── 5. normative_stats.csv (NEW — replaces ratings.csv) ─────
def gen_normative_stats():
    """Generate healthy control distribution statistics for each feature/task."""
    rows = []
    for task in TASKS:
        for feat in FEATURES:
            # Healthy population stats
            mean = round(np.random.uniform(0.08, 0.22), 3)
            std = round(np.random.uniform(0.03, 0.08), 3)
            p25 = round(max(mean - 0.67 * std, 0.02), 3)
            p50 = round(mean, 3)
            p75 = round(mean + 0.67 * std, 3)
            p90 = round(mean + 1.28 * std, 3)
            p95 = round(mean + 1.65 * std, 3)
            n_subjects = np.random.randint(80, 150)
            rows.append([task, feat, mean, std, p25, p50, p75, p90, p95, n_subjects])

    df = pd.DataFrame(rows, columns=[
        'task', 'feature', 'healthy_mean', 'healthy_std',
        'p25', 'p50', 'p75', 'p90', 'p95', 'n_subjects',
    ])
    df.to_csv(os.path.join(DATA_DIR, 'normative_stats.csv'), index=False)
    print(f"normative_stats.csv: {len(df)} rows")


# ── 6. patient_history.csv (NEW) ────────────────────────────
def gen_patient_history():
    """Generate longitudinal visit history for each patient."""
    rows = []
    visit_dates = {
        'P001': ['2025-06-15', '2025-09-20', '2025-12-10', '2026-03-18'],
        'P002': ['2025-08-01', '2025-11-15', '2026-02-28'],
        'P003': ['2025-07-10', '2026-01-05'],
        'P004': ['2025-05-20', '2025-08-30', '2025-12-01', '2026-04-10'],
        'P005': ['2025-04-01', '2025-07-15', '2025-10-22', '2026-01-30', '2026-05-01'],
    }

    for pid, group, *_ in PATIENTS:
        prof = PROFILES[group]
        mult = PAT_MULT[pid]
        dates = visit_dates[pid]

        for visit_num, date in enumerate(dates, 1):
            for feat in FEATURES:
                base = prof['base'] * mult
                noise = np.random.uniform(-prof['range'] * 0.5, prof['range'] * 0.5)
                # PD patients: gradual worsening over visits
                if group == 'PD':
                    progression = (visit_num - 1) * 0.03 * mult
                elif group == 'Differential':
                    progression = (visit_num - 1) * 0.01
                else:
                    progression = 0
                value = np.clip(base + noise + progression, 0.02, 0.98)
                rows.append([pid, visit_num, date, feat, round(value, 3)])

    df = pd.DataFrame(rows, columns=[
        'patient_id', 'visit_number', 'visit_date', 'feature', 'value',
    ])
    df.to_csv(os.path.join(DATA_DIR, 'patient_history.csv'), index=False)
    print(f"patient_history.csv: {len(df)} rows")


# ── 7. motion_trail.json ────────────────────────────────────
def gen_motion_trail():
    trails = []
    n_frames = 100

    # Keypoint per task
    TASK_KEYPOINT = {
        'finger_tapping': 'right_index_finger',
        'hand_open_close': 'right_wrist',
        'rest_tremor': 'right_hand',
        'gait': 'right_ankle',
        'toe_tapping': 'right_toe',
        'touch_nose': 'right_index_finger',
    }
    TASK_FREQ_TRAIL = {
        'finger_tapping': 2.5, 'hand_open_close': 2.0, 'rest_tremor': 5.0,
        'gait': 1.8, 'toe_tapping': 2.2, 'touch_nose': 1.5,
    }

    for pid, group, *_ in PATIENTS:
        for task in TASKS:
            t = np.linspace(0, 3.0, n_frames)
            freq = TASK_FREQ_TRAIL.get(task, 2.5)
            keypoint = TASK_KEYPOINT.get(task, 'right_hand')

            if group == 'PD':
                mult = PAT_MULT[pid]
                jitter = 0.003 * mult
                amp_x = 0.15 * (1 + 0.1 * mult)
                amp_y = 0.12
                x = 0.5 + amp_x * np.sin(2 * np.pi * freq * t) + \
                    np.cumsum(np.random.normal(0, jitter, n_frames))
                y = 0.5 + amp_y * np.cos(2 * np.pi * freq * t) * \
                    np.linspace(1, 1 - 0.3 * mult, n_frames) + \
                    np.random.normal(0, 0.006 * mult, n_frames)
            elif group == 'Differential':
                x = 0.5 + 0.13 * np.sin(2 * np.pi * freq * t) + \
                    np.cumsum(np.random.normal(0, 0.001, n_frames))
                y = 0.5 + 0.10 * np.cos(2 * np.pi * freq * t) + \
                    np.random.normal(0, 0.004, n_frames)
            else:
                x = 0.5 + 0.12 * np.sin(2 * np.pi * freq * t)
                y = 0.5 + 0.10 * np.cos(2 * np.pi * freq * t) + \
                    np.random.normal(0, 0.002, n_frames)

            dx = np.gradient(x, t)
            dy = np.gradient(y, t)
            vel = np.sqrt(dx**2 + dy**2)

            frames = []
            for i in range(n_frames):
                frames.append({
                    'frame': int(i + 1),
                    'time': round(float(t[i]), 3),
                    'x': round(float(x[i]), 4),
                    'y': round(float(y[i]), 4),
                    'velocity': round(float(vel[i]), 4),
                })

            trails.append({
                'patient_id': pid,
                'task': task,
                'keypoint': keypoint,
                'frames': frames,
            })

    with open(os.path.join(DATA_DIR, 'motion_trail.json'), 'w') as f:
        json.dump(trails, f, indent=2)
    print(f"motion_trail.json: {len(trails)} entries")


# ── Run all ──────────────────────────────────────────────────
if __name__ == '__main__':
    gen_patients()
    gen_task_features()
    gen_left_right_features()
    gen_timeseries()
    gen_normative_stats()
    gen_patient_history()
    gen_motion_trail()
    # Remove old ratings.csv if exists
    old = os.path.join(DATA_DIR, 'ratings.csv')
    if os.path.exists(old):
        os.remove(old)
        print("Removed old ratings.csv")
    print("\nAll demo data generated successfully!")

"""Data loading utilities for real TULIP/PADS clinical data."""

import pandas as pd
import numpy as np
import json
import os
from functools import lru_cache

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(BASE_DIR)  # 시각화수업/
TULIP_DIR = os.path.join(PROJECT_DIR, 'pads_matched', 'by_tulip')
LABELS_DIR = os.path.join(PROJECT_DIR, 'labels_csv_files')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Subject number ↔ TULIP ID mapping
_SUBJECT_TULIP = {
    1: 'TULIP_001', 2: 'TULIP_002', 5: 'TULIP_005', 6: 'TULIP_006',
    7: 'TULIP_007', 8: 'TULIP_008', 10: 'TULIP_010', 11: 'TULIP_011',
    12: 'TULIP_012', 13: 'TULIP_013', 14: 'TULIP_014', 15: 'TULIP_015',
}

# New (unlabeled) cases — these are treated as "new patients" for review
NEW_CASES = {'TULIP_001', 'TULIP_008'}


def get_group_label(tulip_id, condition):
    """Assign group: New Case / PD / Healthy / Other."""
    if tulip_id in NEW_CASES:
        return 'New'
    if 'Parkinson' in condition:
        return 'PD'
    if 'Healthy' in condition:
        return 'Healthy'
    return 'Other'


def load_mapping():
    """Load TULIP → PADS ID mapping from README_mapping.csv."""
    path = os.path.join(TULIP_DIR, 'README_mapping.csv')
    df = pd.read_csv(path)
    return dict(zip(df['tulip_patient_id'], df['pads_subject_id'].astype(str).str.zfill(3)))


@lru_cache(maxsize=1)
def _mapping():
    return load_mapping()


def load_patients():
    """Load all 12 patient JSONs → DataFrame."""
    mapping = _mapping()
    rows = []
    for tulip_id, pads_id in mapping.items():
        patient_dir = os.path.join(TULIP_DIR, tulip_id, 'patients')
        json_path = os.path.join(patient_dir, f'patient_{pads_id}.json')
        if not os.path.exists(json_path):
            # Try without zero-padding
            for f in os.listdir(patient_dir):
                if f.endswith('.json'):
                    json_path = os.path.join(patient_dir, f)
                    break
        if not os.path.exists(json_path):
            continue
        with open(json_path, 'r') as f:
            p = json.load(f)
        bmi = round(p.get('weight', 0) / ((p.get('height', 170) / 100) ** 2), 1) if p.get('height') else None
        rows.append({
            'tulip_id': tulip_id,
            'pads_id': str(p.get('id', pads_id)),
            'condition': p.get('condition', 'Unknown'),
            'disease_comment': p.get('disease_comment', '-'),
            'age': p.get('age'),
            'age_at_diagnosis': p.get('age_at_diagnosis'),
            'gender': p.get('gender', ''),
            'height': p.get('height'),
            'weight': p.get('weight'),
            'handedness': p.get('handedness', ''),
            'bmi': bmi,
        })
    return pd.DataFrame(rows)


def load_nms():
    """Load NMS questionnaire JSONs → {tulip_id: {count, symptoms[]}}."""
    mapping = _mapping()
    result = {}
    for tulip_id, pads_id in mapping.items():
        q_dir = os.path.join(TULIP_DIR, tulip_id, 'questionnaire')
        json_path = os.path.join(q_dir, f'questionnaire_response_{pads_id}.json')
        if not os.path.exists(json_path):
            for f in os.listdir(q_dir):
                if f.endswith('.json'):
                    json_path = os.path.join(q_dir, f)
                    break
        if not os.path.exists(json_path):
            continue
        with open(json_path, 'r') as f:
            q = json.load(f)
        items = q.get('item', [])
        symptoms = [item['text'] for item in items if item.get('answer') is True]
        result[tulip_id] = {
            'count': len(symptoms),
            'total': len(items),
            'symptoms': symptoms,
        }
    return result


def load_updrs_labels():
    """Load all 12 labels CSVs → DataFrame with tulip_id, mean score, disagreement."""
    rows = []
    for subj_num, tulip_id in _SUBJECT_TULIP.items():
        path = os.path.join(LABELS_DIR, f'subject{subj_num}_labels.csv')
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df = df.dropna(subset=['UPDRS_name'])
        for _, row in df.iterrows():
            name = row['UPDRS_name']
            c1, c2, c3 = row['label_clinician1'], row['label_clinician2'], row['label_clinician3']

            # Skip non-numeric rows (Dyskinesias=No/Yes, wholistic_decision=HT/PD)
            try:
                c1_num = float(c1)
                c2_num = float(c2)
                c3_num = float(c3)
            except (ValueError, TypeError):
                # Store wholistic_decision separately
                if name == 'wholistic_decision':
                    rows.append({
                        'tulip_id': tulip_id,
                        'updrs_name': name,
                        'c1': str(c1), 'c2': str(c2), 'c3': str(c3),
                        'mean': None, 'disagreement': None,
                        'is_numeric': False,
                    })
                elif name == 'Dyskinesias':
                    rows.append({
                        'tulip_id': tulip_id,
                        'updrs_name': name,
                        'c1': str(c1), 'c2': str(c2), 'c3': str(c3),
                        'mean': None, 'disagreement': None,
                        'is_numeric': False,
                    })
                continue

            mean_val = round((c1_num + c2_num + c3_num) / 3, 2)
            disagree = c3_num - c1_num  # max - min approximation
            # Proper max-min
            vals = [c1_num, c2_num, c3_num]
            disagree = max(vals) - min(vals)

            rows.append({
                'tulip_id': tulip_id,
                'updrs_name': name,
                'c1': c1_num, 'c2': c2_num, 'c3': c3_num,
                'mean': mean_val,
                'disagreement': disagree,
                'is_numeric': True,
            })
    return pd.DataFrame(rows)


def load_updrs_metadata():
    """Extract H&Y stage and wholistic_decision per subject."""
    df = load_updrs_labels()
    result = {}
    for tulip_id in df['tulip_id'].unique():
        subj = df[df.tulip_id == tulip_id]
        meta = {}
        # H&Y
        hy = subj[subj.updrs_name == 'Hoehn and Yahr Stage']
        if not hy.empty and hy.iloc[0]['is_numeric']:
            row = hy.iloc[0]
            meta['hy_mean'] = round((float(row['c1']) + float(row['c2']) + float(row['c3'])) / 3, 1)
            meta['hy_scores'] = [float(row['c1']), float(row['c2']), float(row['c3'])]
        # wholistic_decision
        wd = subj[subj.updrs_name == 'wholistic_decision']
        if not wd.empty:
            row = wd.iloc[0]
            decisions = [str(row['c1']), str(row['c2']), str(row['c3'])]
            meta['decisions'] = decisions
            # Majority vote
            from collections import Counter
            counts = Counter(decisions)
            meta['diagnosis'] = counts.most_common(1)[0][0]
        result[tulip_id] = meta
    return result


@lru_cache(maxsize=264)
def load_timeseries(tulip_id, task, wrist):
    """Load a single timeseries .txt file → DataFrame with 7 columns.

    Args:
        tulip_id: e.g. 'TULIP_001'
        task: e.g. 'CrossArms', 'TouchNose'
        wrist: 'LeftWrist' or 'RightWrist'
    """
    mapping = _mapping()
    pads_id = mapping.get(tulip_id, '')
    ts_dir = os.path.join(TULIP_DIR, tulip_id, 'movement', 'timeseries')
    filename = f'{pads_id}_{task}_{wrist}.txt'
    path = os.path.join(ts_dir, filename)
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, header=None,
                     names=['time', 'accel_x', 'accel_y', 'accel_z',
                            'gyro_x', 'gyro_y', 'gyro_z'])
    # Add signal magnitude columns
    df['accel_mag'] = np.sqrt(df['accel_x']**2 + df['accel_y']**2 + df['accel_z']**2)
    df['gyro_mag'] = np.sqrt(df['gyro_x']**2 + df['gyro_y']**2 + df['gyro_z']**2)
    return df


def load_video_analysis():
    """Load video_analysis.json (existing real data)."""
    path = os.path.join(DATA_DIR, 'video_analysis.json')
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return json.load(f)


VIDEO_TASK_FOLDER_MAP = {
    'toe_left': '17. Toe_tapping_left',
    'toe_right': '18. Toe_tapping_right',
    'resting': '26. Resting & hand tremor',
}


@lru_cache(maxsize=2048)
def _load_video_feature_csv_cached(csv_path):
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    try:
        return pd.read_csv(csv_path)
    except Exception:
        return pd.DataFrame()


def load_video_feature_df(tulip_id, task_folder, camera):
    """
    Load per-camera video feature CSV generated by tulip_source scripts.

    Expected path:
      pads_matched/by_tulip/{tulip_id}/videos_feature/{task_folder}/Camera{idx}_features.csv
    """
    camera_stem = os.path.splitext(camera)[0]
    path = os.path.join(
        TULIP_DIR,
        tulip_id,
        'videos_feature',
        task_folder,
        f'{camera_stem}_features.csv',
    )
    # [PRE] Legacy flat path fallback retained as commented block (no deletion policy).
    # legacy_path = os.path.join(
    #     TULIP_DIR,
    #     tulip_id,
    #     'videos_feature',
    #     f'{camera_stem}_features.csv',
    # )
    # if not os.path.exists(path) and os.path.exists(legacy_path):
    #     return _load_video_feature_csv_cached(legacy_path)
    # [END PRE]
    return _load_video_feature_csv_cached(path)


# Sensor task names (from timeseries filenames)
SENSOR_TASKS = [
    'CrossArms', 'DrinkGlas', 'Entrainment', 'HoldWeight', 'LiftHold',
    'PointFinger', 'Relaxed', 'RelaxedTask', 'StretchHold', 'TouchIndex',
    'TouchNose',
]

TASK_LABELS_KR = {
    'CrossArms': 'Cross Arms', 'DrinkGlas': 'Drink Glass',
    'Entrainment': 'Entrainment', 'HoldWeight': 'Hold Weight',
    'LiftHold': 'Lift & Hold', 'PointFinger': 'Point Finger',
    'Relaxed': 'Relaxed', 'RelaxedTask': 'Relaxed Task',
    'StretchHold': 'Stretch & Hold', 'TouchIndex': 'Touch Index',
    'TouchNose': 'Touch Nose',
}


def get_subject_list():
    """Return dropdown options for all 12 TULIP subjects."""
    patients_df = load_patients()
    meta = load_updrs_metadata()
    options = []
    for _, row in patients_df.iterrows():
        tid = row['tulip_id']
        m = meta.get(tid, {})
        diag = m.get('diagnosis', '?')
        cond = row['condition']
        label = f"{tid} (PADS {row['pads_id']}, {cond}, {row['age']}y, {row['gender']})"
        options.append({'label': label, 'value': tid})
    return sorted(options, key=lambda x: x['value'])


def get_task_list():
    """Return dropdown options — alignment tasks first, then others."""
    alignment = []
    others = []
    for t in SENSOR_TASKS:
        label = TASK_LABELS_KR.get(t, t)
        if t in MATCHING_TASKS:
            alignment.append({'label': f'★ {label} (alignment)', 'value': t})
        else:
            others.append({'label': f'  {label}', 'value': t})
    return alignment + others


def get_subject_list_labelfree():
    """Return dropdown options without condition/diagnosis labels (Review mode)."""
    patients_df = load_patients()
    options = []
    for i, (_, row) in enumerate(patients_df.iterrows(), 1):
        tid = row['tulip_id']
        label = f"Case {i:03d} ({row['age']}y, {row['gender']})"
        options.append({'label': label, 'value': tid})
    return sorted(options, key=lambda x: x['value'])


def get_subject_list_hybrid():
    """Return dropdown: New cases first (top), then confirmed cases."""
    patients_df = load_patients()
    new_options = []
    confirmed_options = []
    for _, row in patients_df.iterrows():
        tid = row['tulip_id']
        group = get_group_label(tid, row['condition'])
        if group == 'New':
            num = tid.replace('TULIP_', '')
            label = f"★ Patient_{num} — New ({row['age']}y, {row['gender']})"
            new_options.append({'label': label, 'value': tid})
        else:
            num = tid.replace('TULIP_', '')
            label = f"  Patient_{num} — {group} ({row['age']}y, {row['gender']})"
            confirmed_options.append({'label': label, 'value': tid})
    # New patients first, then confirmed sorted
    new_options.sort(key=lambda x: x['value'])
    confirmed_options.sort(key=lambda x: x['value'])
    return new_options + confirmed_options


@lru_cache(maxsize=1)
def build_feature_cache():
    """Precompute 5 features per subject/task/wrist → DataFrame.

    Columns: tulip_id, task, wrist, tremor_power, amplitude(accel_rms),
             rhythm_irreg, jerk, asymmetry
    """
    from src.feature_engineering import (
        calc_tremor_power, calc_rhythm_irregularity, calc_mean_jerk, calc_signal_rms
    )
    patients = load_patients()
    rows = []
    for tulip_id in patients['tulip_id']:
        for task in SENSOR_TASKS:
            left_rms = 0.0
            right_rms = 0.0
            for wrist in ['LeftWrist', 'RightWrist']:
                ts = load_timeseries(tulip_id, task, wrist)
                if ts.empty:
                    continue
                sig = ts['accel_mag'].values
                fs = _estimate_fs(ts)
                rms = calc_signal_rms(sig)
                if wrist == 'LeftWrist':
                    left_rms = rms
                else:
                    right_rms = rms
                rows.append({
                    'tulip_id': tulip_id,
                    'task': task,
                    'wrist': 'L' if wrist == 'LeftWrist' else 'R',
                    'tremor_power': calc_tremor_power(sig, fs),
                    'amplitude': rms,
                    'rhythm_irreg': calc_rhythm_irregularity(sig, fs),
                    'jerk': calc_mean_jerk(sig, fs),
                })
            # Add asymmetry row for the task
            if left_rms > 0 or right_rms > 0:
                from src.feature_engineering import calc_asymmetry_index
                asym = calc_asymmetry_index(left_rms, right_rms)
                # Tag both rows with asymmetry for this task
                for r in rows[-2:]:
                    if r.get('tulip_id') == tulip_id and r.get('task') == task:
                        r['asymmetry'] = asym
    return pd.DataFrame(rows)


def _estimate_fs(ts_df):
    """Estimate sampling frequency from timeseries DataFrame."""
    if len(ts_df) < 2:
        return 100.0
    dt = np.median(np.diff(ts_df['time'].values))
    if dt <= 0:
        return 100.0
    return 1.0 / dt


# ──────────────────────────────────────────────────────────────
# Alignment tasks & feature weights (matching pipeline-aligned)
# ──────────────────────────────────────────────────────────────

MATCHING_TASKS = ['Entrainment', 'Relaxed']
MATCHING_FEATURES = ['tremor_power', 'amplitude', 'rhythm_irreg', 'jerk']

# Feature weights aligned with MOTOR_FEATURE_WEIGHTS in matching pipeline
# Tremor/rhythm weighted higher (clinical significance for PD)
FEATURE_WEIGHTS = {
    'tremor_power': 1.5,
    'amplitude': 1.0,
    'rhythm_irreg': 1.3,
    'jerk': 1.0,
}

# Task-aware grouping (Revision 6 & 7)
TASK_GROUPS = {
    'rest': ['Relaxed', 'RelaxedTask'],
    'repetitive': ['Entrainment', 'TouchIndex', 'TouchNose'],
    'intentional': ['DrinkGlas', 'PointFinger', 'LiftHold'],
    'postural': ['CrossArms', 'HoldWeight', 'StretchHold'],
}

# Task-specific default metrics (Revision 7)
TASK_DEFAULT_METRICS = {
    'Relaxed': 'tremor_power',
    'RelaxedTask': 'tremor_power',
    'Entrainment': 'rhythm_irreg',
    'TouchIndex': 'rhythm_irreg',
    'TouchNose': 'amplitude',
    'DrinkGlas': 'amplitude',
    'PointFinger': 'jerk',
    'LiftHold': 'amplitude',
    'CrossArms': 'tremor_power',
    'HoldWeight': 'tremor_power',
    'StretchHold': 'tremor_power',
}


def _weighted_euclidean(v1, v2, weights):
    """Weighted Euclidean distance (Revision 3)."""
    w = np.array(weights)
    return float(np.sqrt(np.sum(w * (v1 - v2) ** 2) / np.sum(w)))


@lru_cache(maxsize=1)
def compute_proximity_scores():
    """Compute motor phenotype proximity for all subjects.

    NOT a diagnostic probability. This measures reference cohort similarity:
    how close this analog's motor phenotype is to confirmed PD vs Healthy centroids.

    Revision 2: 16D vector (2 tasks × 2 wrists × 4 features)
    Revision 3: Weighted Euclidean distance (FEATURE_WEIGHTS)
    Revision 5: Returns n_pd, n_healthy for cohort size display

    Returns: dict {tulip_id: {
        score, d_pd, d_healthy,
        per_task: {task: {per_feature}},
        n_pd, n_healthy,
        vector_dim,
    }}
    """
    fc = build_feature_cache()
    patients = load_patients()
    cond_map = dict(zip(patients['tulip_id'], patients['condition']))

    match_data = fc[fc.task.isin(MATCHING_TASKS)].copy()
    if match_data.empty:
        return {}

    # Build 16D vector: task × wrist × feature (Revision 2)
    # Order: [Entrainment_L_tremor, Entrainment_L_amp, ..., Relaxed_R_jerk]
    vector_labels = []
    weight_vector = []
    for task in MATCHING_TASKS:
        for wrist in ['L', 'R']:
            for feat in MATCHING_FEATURES:
                vector_labels.append(f'{task}_{wrist}_{feat}')
                weight_vector.append(FEATURE_WEIGHTS[feat])
    weight_vector = np.array(weight_vector)

    subject_vectors = {}
    for tulip_id in match_data['tulip_id'].unique():
        subj = match_data[match_data.tulip_id == tulip_id]
        vec = []
        for task in MATCHING_TASKS:
            task_data = subj[subj.task == task]
            for wrist in ['L', 'R']:
                wrist_data = task_data[task_data.wrist == wrist]
                for feat in MATCHING_FEATURES:
                    vals = wrist_data[feat].values
                    vec.append(float(vals[0]) if len(vals) > 0 else 0.0)
        subject_vectors[tulip_id] = np.array(vec)

    if not subject_vectors:
        return {}

    # Confirmed groups
    pd_vecs, healthy_vecs = [], []
    for tid, vec in subject_vectors.items():
        group = get_group_label(tid, cond_map.get(tid, 'Unknown'))
        if group == 'PD':
            pd_vecs.append(vec)
        elif group == 'Healthy':
            healthy_vecs.append(vec)

    n_pd = len(pd_vecs)
    n_healthy = len(healthy_vecs)
    if n_pd == 0 or n_healthy == 0:
        return {}

    pd_vecs = np.array(pd_vecs)
    healthy_vecs = np.array(healthy_vecs)

    # Z-score using confirmed subjects
    all_confirmed = np.vstack([pd_vecs, healthy_vecs])
    mean_vec = np.mean(all_confirmed, axis=0)
    std_vec = np.std(all_confirmed, axis=0)
    std_vec[std_vec == 0] = 1.0

    pd_centroid = np.mean((pd_vecs - mean_vec) / std_vec, axis=0)
    healthy_centroid = np.mean((healthy_vecs - mean_vec) / std_vec, axis=0)

    results = {}
    for tid, vec in subject_vectors.items():
        z_vec = (vec - mean_vec) / std_vec

        # Weighted Euclidean distance (Revision 3)
        d_pd = _weighted_euclidean(z_vec, pd_centroid, weight_vector)
        d_healthy = _weighted_euclidean(z_vec, healthy_centroid, weight_vector)
        total_d = d_pd + d_healthy
        score = (d_healthy / total_d * 100) if total_d > 0 else 50.0

        # Per-task breakdown
        per_task = {}
        idx = 0
        for task in MATCHING_TASKS:
            per_feat = {}
            for wrist in ['L', 'R']:
                for feat in MATCHING_FEATURES:
                    z_val = z_vec[idx]
                    pd_val = pd_centroid[idx]
                    h_val = healthy_centroid[idx]
                    d_p = abs(z_val - pd_val)
                    d_h = abs(z_val - h_val)
                    t = d_p + d_h
                    key = f'{feat}_{wrist}'
                    per_feat[key] = (d_h / t * 100) if t > 0 else 50.0
                    idx += 1
            # Aggregate per feature (mean of L+R)
            task_summary = {}
            for feat in MATCHING_FEATURES:
                l_score = per_feat.get(f'{feat}_L', 50)
                r_score = per_feat.get(f'{feat}_R', 50)
                task_summary[feat] = (l_score + r_score) / 2
            per_task[task] = task_summary

        results[tid] = {
            'score': float(score),
            'd_pd': float(d_pd),
            'd_healthy': float(d_healthy),
            'per_task': per_task,
            'n_pd': n_pd,
            'n_healthy': n_healthy,
            'vector_dim': len(vector_labels),
        }

    return results


@lru_cache(maxsize=1)
def build_group_stats():
    """Precompute per-subject per-task sensor RMS stats for group comparison.

    Returns DataFrame: tulip_id, condition, task, wrist,
                       accel_rms, gyro_rms, accel_std, gyro_std
    """
    patients = load_patients()
    cond_map = dict(zip(patients['tulip_id'], patients['condition']))
    rows = []
    for tulip_id in patients['tulip_id']:
        condition = cond_map.get(tulip_id, 'Unknown')
        group = get_group_label(tulip_id, condition)
        for task in SENSOR_TASKS:
            for wrist in ['LeftWrist', 'RightWrist']:
                ts = load_timeseries(tulip_id, task, wrist)
                if ts.empty:
                    continue
                rows.append({
                    'tulip_id': tulip_id,
                    'condition': condition,
                    'group': group,
                    'task': task,
                    'wrist': wrist.replace('Wrist', ''),
                    'accel_rms': float(np.sqrt(np.mean(ts['accel_mag'].values ** 2))),
                    'gyro_rms': float(np.sqrt(np.mean(ts['gyro_mag'].values ** 2))),
                    'accel_std': float(np.std(ts['accel_mag'].values)),
                    'gyro_std': float(np.std(ts['gyro_mag'].values)),
                    'duration': float(ts['time'].max()),
                })
    return pd.DataFrame(rows)

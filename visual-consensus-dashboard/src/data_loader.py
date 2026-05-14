"""Data loading utilities for demo CSV/JSON data."""

import pandas as pd
import json
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


def load_patients():
    return pd.read_csv(os.path.join(DATA_DIR, 'patients.csv'))


def load_task_features():
    return pd.read_csv(os.path.join(DATA_DIR, 'task_features.csv'))


def load_left_right_features():
    return pd.read_csv(os.path.join(DATA_DIR, 'left_right_features.csv'))


def load_timeseries():
    return pd.read_csv(os.path.join(DATA_DIR, 'timeseries.csv'))


def load_normative_stats():
    return pd.read_csv(os.path.join(DATA_DIR, 'normative_stats.csv'))


def load_patient_history():
    return pd.read_csv(os.path.join(DATA_DIR, 'patient_history.csv'))


def load_motion_trail():
    with open(os.path.join(DATA_DIR, 'motion_trail.json'), 'r') as f:
        return json.load(f)


def load_video_analysis():
    path = os.path.join(DATA_DIR, 'video_analysis.json')
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return json.load(f)


def get_patient_list():
    df = load_patients()
    return [
        {
            'label': f"{row.patient_id} ({row.group}, {row.age}y, {row.sex})",
            'value': row.patient_id,
        }
        for _, row in df.iterrows()
    ]


def get_task_list():
    df = load_task_features()
    return [
        {'label': t.replace('_', ' ').title(), 'value': t}
        for t in df['task'].unique()
    ]

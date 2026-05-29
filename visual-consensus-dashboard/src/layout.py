"""Dashboard layout — Final: Aligned Tasks (Toe Tapping & Resting) Focus.

5 Tabs:
1. Patient Overview — demographics + NMS + aligned task matrix
2. Tremor & Rhythm — tremor power, frequency bands, spectral, rhythm, decrement
3. Video Analysis — multi-camera toe tapping & resting video
4. Reference Comparison — proximity gauge + distribution position
5. Clinical Summary — verdict, findings, radar, task table
"""

from dash import html, dcc
from src.data_loader import (
    get_subject_list_hybrid, get_task_list, NEW_CASES
)

# ── UPDRS items (CSV order) ──
UPDRS_ITEMS = [
    'Facial expression',
    'Finger tapping - Right hand', 'Finger tapping - Left hand',
    'Hand movements - Right hand', 'Hand movements - Left hand',
    'Pronation-supination - Right hand', 'Pronation-supination - Left hand',
    'Toe tapping - Right foot', 'Toe tapping - Left foot',
    'Leg agility - Right leg', 'Leg agility - Left leg',
    'Arising from chair', 'Gait', 'Freezing of gait',
    'Postural stability', 'Posture', 'Global spontaneity of movement',
    'Postural tremor - Right hand', 'Postural tremor - Left hand',
    'Kinetic tremor - Right hand', 'Kinetic tremor - Left hand',
    'Rest tremor amplitude - RUE', 'Rest tremor amplitude - LUE',
    'Rest tremor amplitude - RLE', 'Rest tremor amplitude - LLE',
    'Rest tremor amplitude - Lip/jaw', 'Constancy of rest tremor',
    'Dyskinesias', 'Hoehn and Yahr Stage', 'wholistic_decision',
]

# Task-based grouping — aligned tasks only (Entrainment=Toe Tapping, Relaxed=Resting)
UPDRS_TASK_GROUPS = [
    ('Toe Tapping (발 두드리기)', [7, 8]),
    ('Rest Tremor (안정시 떨림)', [21, 22, 23, 24, 25, 26]),
    ('Global Assessment (종합)', [27, 28, 29]),
]

# Short labels for sidebar display
UPDRS_SHORT = {
    0: 'Facial expression',
    1: 'Finger tap - R', 2: 'Finger tap - L',
    3: 'Hand mvmt - R', 4: 'Hand mvmt - L',
    5: 'Pron-sup - R', 6: 'Pron-sup - L',
    7: 'Right foot', 8: 'Left foot',
    9: 'Right leg', 10: 'Left leg',
    11: 'Arising chair', 12: 'Gait', 13: 'Freezing',
    14: 'Post. stability', 15: 'Posture', 16: 'Global spont.',
    17: 'Post. tremor - R', 18: 'Post. tremor - L',
    19: 'Kinetic - R', 20: 'Kinetic - L',
    21: 'RUE (우상지)', 22: 'LUE (좌상지)',
    23: 'RLE (우하지)', 24: 'LLE (좌하지)',
    25: 'Lip/jaw', 26: 'Constancy',
    27: 'Dyskinesias', 28: 'H&Y Stage', 29: 'Diagnosis',
}


def _updrs_input(idx):
    """Create the appropriate input for a UPDRS item."""
    name = UPDRS_ITEMS[idx]
    if name == 'Dyskinesias':
        return dcc.Dropdown(
            id={'type': 'updrs-score', 'index': idx},
            options=[{'label': 'No', 'value': 'No'}, {'label': 'Yes', 'value': 'Yes'}],
            value=None, clearable=True, className='updrs-dropdown',
            placeholder='—',
        )
    elif name == 'wholistic_decision':
        return dcc.Dropdown(
            id={'type': 'updrs-score', 'index': idx},
            options=[{'label': 'PD', 'value': 'PD'}, {'label': 'HT', 'value': 'HT'}],
            value=None, clearable=True, className='updrs-dropdown',
            placeholder='—',
        )
    elif name == 'Hoehn and Yahr Stage':
        return dcc.Input(
            id={'type': 'updrs-score', 'index': idx},
            type='number', min=0, max=5, step=1,
            placeholder='0-5', className='updrs-number-input',
        )
    else:
        return dcc.Input(
            id={'type': 'updrs-score', 'index': idx},
            type='number', min=0, max=4, step=1,
            placeholder='0-4', className='updrs-number-input',
        )


def _updrs_row(idx):
    """Single UPDRS item row: label + radio."""
    label = UPDRS_SHORT.get(idx, UPDRS_ITEMS[idx])
    return html.Div([
        html.Span(label, className='updrs-item-label'),
        _updrs_input(idx),
    ], className='updrs-item-row')


def _build_updrs_form():
    """Build task-grouped UPDRS scoring form."""
    groups = []
    for i, (group_name, indices) in enumerate(UPDRS_TASK_GROUPS):
        items = [_updrs_row(idx) for idx in indices]
        groups.append(
            html.Details([
                html.Summary(group_name, className='updrs-group-title'),
                html.Div(items, className='updrs-group-items'),
            ], open=(i < 2), className='updrs-group')
        )
    return html.Div(groups, className='updrs-form')


def _card(card_id, label, initial='—', label_id=None):
    """Create a summary metric card."""
    label_props = {'className': 'card-label'}
    if label_id is not None:
        label_props['id'] = label_id
    return html.Div([
        html.Div(label, **label_props),
        html.Div(initial, id=card_id, className='card-value'),
    ], className='summary-card')


def create_layout():
    """Build the full dashboard layout."""
    case_options = get_subject_list_hybrid()

    return html.Div([
        # ─── Stores ───
        dcc.Store(id='decision-store', data={}, storage_type='local'),

        # ─── Header ───
        html.Header([
            html.Div([
                html.H1('PD Clinical Decision Support', className='header-title'),
                html.P('Aligned Tasks: Toe Tapping (Entrainment) & Resting (Relaxed) 중심 분석',
                       className='header-subtitle'),
            ], className='header-left'),
            html.Div([
                html.Div([
                    html.Label('Patient:', className='header-label'),
                    dcc.Dropdown(
                        id='patient-dropdown',
                        options=case_options,
                        value=case_options[0]['value'] if case_options else None,
                        className='case-dropdown',
                        clearable=False,
                    ),
                ], className='header-control'),
                html.Div(id='case-badge', className='badge badge-unknown'),
            ], className='header-right'),
        ], className='dashboard-header'),

        # ─── Main Layout: Content + Right Panel ───
        html.Main([
            # ═══ LEFT+CENTER: Tabbed Content ═══
            html.Section([
                dcc.Tabs(id='main-tabs', value='tab-overview', className='custom-tabs', children=[
                    dcc.Tab(label='1. Patient Overview', value='tab-overview',
                            className='custom-tab', selected_className='custom-tab--selected'),
                    dcc.Tab(label='2. Tremor & Rhythm', value='tab-tremor',
                            className='custom-tab', selected_className='custom-tab--selected'),
                    dcc.Tab(label='3. Video Analysis', value='tab-video',
                            className='custom-tab', selected_className='custom-tab--selected'),
                    dcc.Tab(label='4. Reference Comparison', value='tab-comparison',
                            className='custom-tab', selected_className='custom-tab--selected'),
                    dcc.Tab(label='5. Clinical Summary', value='tab-summary',
                            className='custom-tab', selected_className='custom-tab--selected'),
                ]),
                html.Div(id='tab-content', className='tab-content-area'),
            ], className='main-content'),

            # ═══ RIGHT: Decision Workspace ═══
            html.Aside(id='right-panel', children=[
                html.H3('Decision Workspace', className='panel-title'),
                html.Div(id='new-patient-indicator', className='new-indicator'),

                # Patient summary cards
                html.Div([
                    _card('info-condition', 'Condition'),
                    _card('info-hy', 'H&Y Stage'),
                    _card('info-diagnosis', 'Clinician Dx'),
                    _card('info-nms', 'NMS Count'),
                ], className='sidebar-cards'),

                html.Hr(className='sidebar-divider'),

                # Decision form — UPDRS Task-based scoring
                html.Div([
                    html.H4('UPDRS Motor Assessment', className='section-title'),
                    html.P('각 항목을 0-4점으로 평가해주세요.',
                           style={'fontSize': '11px', 'color': '#718096',
                                  'marginBottom': '8px'}),
                    _build_updrs_form(),
                    html.Div([
                        html.Button('Save Decision', id='save-decision-btn',
                                    className='btn-primary', n_clicks=0),
                        html.Button('Reset', id='reset-decision-btn',
                                    className='btn-reset', n_clicks=0),
                    ], className='decision-btn-row'),
                    html.Div(id='save-feedback', className='save-feedback'),
                    html.Div(id='clinician-comparison'),
                ], id='decision-form-container', className='decision-form'),

                # Export
                html.Div([
                    html.Button('Export All Decisions (JSON)', id='export-btn',
                                className='btn-secondary', n_clicks=0),
                    dcc.Download(id='download-json'),
                ], className='export-section'),
            ], className='right-panel'),
        ], id='dashboard-grid', className='dashboard-grid'),
    ], className='app-container')


# ══════════════════════════════════════════════════════════════
#  TAB CONTENT BUILDERS
# ══════════════════════════════════════════════════════════════

def build_tab_overview():
    """Tab 1: Patient Overview — demographics + motor phenotype summary + asymmetry."""
    return html.Div([
        # Demographics
        html.Div(id='demographics-row', className='demographics-row'),
        # Basic info cards
        html.Div([
            _card('ov-age', 'Age'),
            _card('ov-gender', 'Gender'),
            _card('ov-handedness', 'Handedness'),
            _card('ov-bmi', 'BMI'),
            _card('ov-duration', 'Disease Duration'),
        ], className='summary-cards-row'),
        # Motor Phenotype Summary
        html.Div([
            html.H3('Motor Phenotype Summary', className='viz-title'),
            html.P('Aligned tasks (Entrainment & Relaxed) 기반 motor phenotype 요약. '
                   'Reference cohort 대비 motor 특성 유사도입니다 (진단 확률이 아님).',
                   className='tab-description'),
        ]),
        html.Div([
            _card('ov-phenotype', 'Motor Phenotype'),
            _card('ov-proximity', 'Reference Similarity'),
            _card('ov-asymmetry', 'Mean Asymmetry'),
        ], className='summary-cards-row'),
        # NMS section
        html.Div([
            html.H3('Non-Motor Symptoms', className='viz-title'),
            html.Div(id='nms-content', className='nms-grid'),
        ], className='viz-block'),
        # Bilateral Matrix (aligned tasks)
        html.Div([
            html.H3('Aligned Task Matrix (Entrainment & Relaxed)', className='viz-title'),
            html.P('좌/우 4개 feature(Tremor, Amplitude, Rhythm, Jerk) 정규화 비교.',
                   className='tab-description'),
            dcc.Graph(id='bilateral-matrix', config={'displayModeBar': False}),
        ], className='viz-block'),
        # Asymmetry Group Comparison (patient-level overview)
        html.Div([
            html.H3('좌우 비대칭 — 그룹 비교', className='viz-title'),
            html.P('환자의 평균 비대칭 지수가 확정된 PD/Healthy 그룹 분포에서 '
                   '어디에 위치하는지 보여줍니다.',
                   className='tab-description'),
            dcc.Graph(id='asym-group-compare', config={'displayModeBar': False}),
        ], className='viz-block'),
    ])


def build_tab_tremor():
    """Tab 2: Tremor & Rhythm — aligned tasks focus."""
    task_options = get_task_list()
    return html.Div([
        html.Div([
            html.H3('Tremor & Rhythm Analysis', className='viz-title'),
            html.P('Aligned tasks (Entrainment = Toe Tapping, Relaxed = Resting)의 '
                   '떨림 power, 주파수 대역, 리듬 안정성, 진폭 감소, 좌우 차이를 분석합니다.',
                   className='tab-description'),
        ]),
        # ── Tremor Power Overview (patient-level, both tasks) ──
        html.Div([
            html.H3('Tremor Power (4-12Hz) — L/R with Reference', className='viz-title'),
            html.P('환자의 좌/우 tremor power를 PD/Healthy 그룹 평균과 비교합니다.',
                   className='tab-description'),
            dcc.Graph(id='tremor-power-bars', config={'displayModeBar': False}),
        ], className='viz-block'),

        # ── Task-specific analysis ──
        html.Div([
            html.Label('Task:', className='inline-label'),
            dcc.Dropdown(id='tremor-task-dropdown', options=task_options,
                         value='Entrainment', clearable=False, className='inline-dropdown'),
        ], className='controls-row'),

        # Tremor Band Breakdown
        html.Div([
            html.H3('Tremor Frequency Band Analysis', className='viz-title'),
            html.P('Rest tremor (4-6Hz) vs Action tremor (6-12Hz). '
                   'Rest-dominant → PD 시사.',
                   className='tab-description'),
            dcc.Graph(id='tremor-band-chart', config={'displayModeBar': False}),
        ], className='viz-block'),

        # Rhythm Ladder
        html.Div([
            html.H3('Rhythm Instability Ladder', className='viz-title'),
            html.P('반복 운동의 inter-peak interval CV. '
                   'CV가 높을수록 리듬이 불규칙 → bradykinesia 시사.',
                   className='tab-description'),
            dcc.Graph(id='tremor-rhythm', config={'displayModeBar': False}),
        ], className='viz-block'),

        # Amplitude Decrement
        html.Div([
            html.H3('Amplitude Decrement Analysis', className='viz-title'),
            html.P('시간에 따른 진폭 변화. 점진적 감소(declining slope)는 '
                   'PD bradykinesia의 핵심 지표인 decrement sequence를 시사.',
                   className='tab-description'),
            dcc.Graph(id='tremor-decrement', config={'displayModeBar': False}),
        ], className='viz-block'),

        # L/R Feature Comparison (per task)
        html.Div([
            html.H3('Bilateral Feature Comparison', className='viz-title'),
            html.P('좌/우 feature별(Tremor, Amplitude, Rhythm, Jerk) 값과 '
                   '비대칭 지수 비교. 주황선(0.3) 이상이면 의미있는 비대칭.',
                   className='tab-description'),
            dcc.Graph(id='asym-feature-bars', config={'displayModeBar': False}),
        ], className='viz-block'),

        # Collapsible: Spectral Fingerprint + Raw sensor
        html.Details([
            html.Summary('Spectral Fingerprint & Raw Timeseries (상세 보기)', className='raw-toggle'),
            html.Div([
                html.H3('Bilateral Spectral Fingerprint', className='viz-title'),
                html.P('좌/우 시간-주파수 분석. 4-12Hz 대역에 지속적 power → tremor 존재.',
                       className='tab-description'),
                dcc.Graph(id='tremor-spectral', config={'displayModeBar': False}),
            ], className='viz-block'),
            dcc.Graph(id='tremor-raw-left', config={'displayModeBar': False}),
            dcc.Graph(id='tremor-raw-right', config={'displayModeBar': False}),
        ], className='raw-sensor-section'),
    ])


def build_tab_video():
    """Tab 3: Video Analysis — multi-camera toe tapping & resting video."""
    return html.Div([
        html.P(
            '* Cloudflare R2 CDN 영상: Toe Tapping (좌/우) & Resting Tremor. '
            '영상 기반 motion feature 분석 결과를 제공합니다.',
            className='tab-description',
            style={'marginBottom': '8px'},
        ),
        # Controls
        html.Div([
            html.Div([
                html.Label('Video Type:', className='inline-label'),
                dcc.Dropdown(
                    id='video-type-dropdown',
                    options=[],  # dynamically populated per patient
                    value=None, clearable=False, className='inline-dropdown',
                    placeholder='Select patient first...',
                ),
            ], className='inline-control'),
            html.Div([
                html.Label('Camera:', className='inline-label'),
                dcc.Dropdown(
                    id='video-camera-dropdown',
                    options=[{'label': f'Camera {i}', 'value': f'Camera{i}.mp4'}
                             for i in range(1, 7)],
                    value='Camera1.mp4', clearable=False, className='inline-dropdown',
                ),
            ], className='inline-control'),
        ], className='controls-row'),
        # Video + analysis side by side
        html.Div([
            html.Div([
                html.Div(id='video-player-container', className='video-container'),
                html.Div([
                    _card('video-left-taps', 'L Taps', label_id='video-left-taps-label'),
                    _card('video-right-taps', 'R Taps', label_id='video-right-taps-label'),
                    _card('video-l-cv', 'L Interval CV', label_id='video-l-cv-label'),
                    _card('video-r-cv', 'R Interval CV', label_id='video-r-cv-label'),
                ], className='video-cards-grid'),
            ], className='video-left'),
            html.Div([
                dcc.Graph(id='lr-tapping-comparison', config={'displayModeBar': False}),
            ], className='video-right'),
        ], className='video-split-layout'),
        # Clinical-focused extra visualizations
        html.Div([
            dcc.Graph(id='video-interval-distribution', config={'displayModeBar': False}),
            dcc.Graph(id='video-tremor-spectrogram', config={'displayModeBar': False}),
            dcc.Graph(id='video-symmetry-trend', config={'displayModeBar': False}),
        ], className='video-right'),
    ])


def build_tab_comparison():
    """Tab 4: Reference Comparison — proximity + distribution (aligned tasks only)."""
    task_options = get_task_list()
    return html.Div([
        html.Div([
            html.H3('Motor Phenotype Proximity — Reference Cohort', className='viz-title'),
            html.P('Aligned tasks (Entrainment & Relaxed) 기반 16D weighted Euclidean proximity. '
                   '이 점수는 진단 확률이 아닌, 확정 PD/Healthy reference cohort에 대한 '
                   'motor phenotype 유사도입니다.',
                   className='tab-description'),
        ]),
        # Proximity Gauge
        html.Div([
            dcc.Graph(id='proximity-gauge', config={'displayModeBar': False}),
        ], className='viz-block'),

        # Task-specific distribution
        html.Div([
            html.Label('Task:', className='inline-label'),
            dcc.Dropdown(id='comp-task-dropdown', options=task_options,
                         value='Entrainment', clearable=False, className='inline-dropdown'),
            html.Label('Metric:', className='inline-label'),
            dcc.Dropdown(id='comp-metric-dropdown', options=[
                {'label': 'Accel RMS', 'value': 'accel_rms'},
                {'label': 'Gyro RMS', 'value': 'gyro_rms'},
                {'label': 'Accel Std', 'value': 'accel_std'},
                {'label': 'Gyro Std', 'value': 'gyro_std'},
            ], value='accel_rms', clearable=False, className='inline-dropdown'),
        ], className='controls-row'),
        html.Div([
            html.H3('Reference Distribution Position', className='viz-title'),
            html.P('선택한 task/metric에서 이 환자가 PD/Healthy 분포 어디에 위치하는지 보여줍니다.',
                   className='tab-description'),
            dcc.Graph(id='new-vs-group-box', config={'displayModeBar': False}),
        ], className='viz-block'),
    ])


def build_tab_summary():
    """Tab 5: Clinical Summary — aligned tasks evidence compiled."""
    return html.Div([
        html.Div([
            html.H3('Clinical Summary Report', className='viz-title'),
            html.P('Aligned tasks (Entrainment & Relaxed) 센서 분석 결과를 종합합니다. '
                   'Tremor power, amplitude, rhythm, jerk, asymmetry를 통합 평가합니다.',
                   className='tab-description'),
        ]),
        # Overall verdict
        html.Div(id='summary-verdict', className='viz-block'),
        # Key findings
        html.Div(id='summary-findings', className='viz-block'),
        # Feature radar
        html.Div([
            dcc.Graph(id='summary-radar', config={'displayModeBar': False}),
        ], className='viz-block'),
        # Task-by-task breakdown table
        html.Div(id='summary-task-table', className='viz-block'),
    ])

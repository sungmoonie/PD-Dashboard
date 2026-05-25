"""Dashboard layout — Hybrid: Tabs + Decision Workspace + Group Comparison."""

from dash import html, dcc
from src.data_loader import (
    get_subject_list_hybrid, get_task_list, SENSOR_TASKS, NEW_CASES
)


def _card(card_id, label, initial='—'):
    """Create a summary metric card."""
    return html.Div([
        html.Div(label, className='card-label'),
        html.Div(initial, id=card_id, className='card-value'),
    ], className='summary-card')


def create_layout():
    """Build the full dashboard layout."""
    task_options = get_task_list()
    case_options = get_subject_list_hybrid()

    return html.Div([
        # ─── Stores ───
        dcc.Store(id='decision-store', data={}, storage_type='local'),

        # ─── Header ───
        html.Header([
            html.Div([
                html.H1('PD Clinical Decision Support', className='header-title'),
                html.P('스마트워치 센서 + 멀티카메라 영상 기반 · Movement Interpretation 보조 시스템',
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
                    dcc.Tab(label='2. Group Comparison', value='tab-comparison',
                            className='custom-tab', selected_className='custom-tab--selected'),
                    dcc.Tab(label='3. Bilateral Asymmetry', value='tab-asymmetry',
                            className='custom-tab', selected_className='custom-tab--selected'),
                    dcc.Tab(label='4. Sensor Analysis', value='tab-sensor',
                            className='custom-tab', selected_className='custom-tab--selected'),
                    dcc.Tab(label='5. Video Analysis', value='tab-video',
                            className='custom-tab', selected_className='custom-tab--selected'),
                    dcc.Tab(label='6. Clinical Summary', value='tab-summary',
                            className='custom-tab', selected_className='custom-tab--selected'),
                ]),
                html.Div(id='tab-content', className='tab-content-area'),
            ], className='main-content'),

            # ═══ RIGHT: Decision Workspace (for new patients) ═══
            html.Aside([
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

                # Decision form (for new patients)
                html.Div([
                    html.H4('Your Assessment', className='section-title'),
                    html.Label('Classification:', className='form-label'),
                    dcc.RadioItems(
                        id='decision-classification',
                        options=[
                            {'label': 'PD', 'value': 'PD'},
                            {'label': 'Healthy Tremor', 'value': 'HT'},
                            {'label': 'Uncertain', 'value': '?'},
                        ],
                        value=None,
                        className='decision-radio',
                        labelClassName='radio-label',
                    ),
                    html.Label('Confidence:', className='form-label'),
                    dcc.Slider(
                        id='decision-confidence',
                        min=0, max=100, step=10, value=50,
                        marks={0: '0%', 50: '50%', 100: '100%'},
                        className='confidence-slider',
                    ),
                    html.Label('Evidence Tags:', className='form-label'),
                    dcc.Checklist(
                        id='evidence-tags',
                        options=[
                            {'label': 'Tremor asymmetry', 'value': 'tremor_asym'},
                            {'label': 'Rhythm instability', 'value': 'rhythm_instab'},
                            {'label': 'Bradykinesia', 'value': 'bradykinesia'},
                            {'label': 'Rest tremor', 'value': 'rest_tremor'},
                            {'label': 'Action tremor', 'value': 'action_tremor'},
                            {'label': 'High jerk', 'value': 'high_jerk'},
                        ],
                        value=[],
                        className='evidence-checklist',
                        labelClassName='check-label',
                    ),
                    html.Label('Notes:', className='form-label'),
                    dcc.Textarea(
                        id='decision-notes',
                        placeholder='Clinical reasoning...',
                        className='decision-textarea',
                    ),
                    html.Button('Save Decision', id='save-decision-btn',
                                className='btn-primary', n_clicks=0),
                    html.Div(id='save-feedback', className='save-feedback'),
                ], id='decision-form-container', className='decision-form'),

                # Export
                html.Div([
                    html.Button('Export All Decisions (JSON)', id='export-btn',
                                className='btn-secondary', n_clicks=0),
                    dcc.Download(id='download-json'),
                ], className='export-section'),
            ], className='right-panel'),
        ], className='dashboard-grid'),
    ], className='app-container')


# ══════════════════════════════════════════════════════════════
#  TAB CONTENT BUILDERS
# ══════════════════════════════════════════════════════════════

def build_tab_overview():
    """Tab 1: Patient Overview — demographics + NMS."""
    return html.Div([
        # Demographics
        html.Div(id='demographics-row', className='demographics-row'),
        # Summary cards row
        html.Div([
            _card('ov-age', 'Age'),
            _card('ov-gender', 'Gender'),
            _card('ov-handedness', 'Handedness'),
            _card('ov-bmi', 'BMI'),
            _card('ov-duration', 'Disease Duration'),
        ], className='summary-cards-row'),
        # NMS section
        html.Div([
            html.H3('Non-Motor Symptoms', className='viz-title'),
            html.Div(id='nms-content', className='nms-grid'),
        ], className='viz-block'),
        # Bilateral Matrix (quick overview)
        html.Div([
            html.H3('Task-Symptom Matrix (Sensor Overview)', className='viz-title'),
            dcc.Graph(id='bilateral-matrix', config={'displayModeBar': False}),
        ], className='viz-block'),
    ])


def build_tab_comparison():
    """Tab 2: Group Comparison — selected patient vs confirmed PD/Healthy."""
    task_options = get_task_list()
    return html.Div([
        html.Div([
            html.H3('Motor Phenotype Proximity — Reference Cohort Comparison', className='viz-title'),
            html.P('이 환자의 matched sensor analog가 확정 PD/Healthy reference cohort와 '
                   '얼마나 유사한 motor phenotype을 보이는지 비교합니다. '
                   '진단 확률이 아닌 reference 유사도입니다.',
                   className='tab-description'),
        ]),
        # ── Patient-level (no task selection needed) ──
        html.Div([
            dcc.Graph(id='proximity-gauge', config={'displayModeBar': False}),
        ], className='viz-block'),
        html.Div([
            dcc.Graph(id='feature-group-comparison', config={'displayModeBar': False}),
        ], className='viz-block'),
        html.Div([
            dcc.Graph(id='task-profile-comparison', config={'displayModeBar': False}),
        ], className='viz-block'),

        # ── Task-specific (task selector here) ──
        html.Div([
            html.Label('Task (아래 차트에 적용):', className='inline-label'),
            dcc.Dropdown(id='comp-task-dropdown', options=task_options,
                         value='TouchNose', clearable=False, className='inline-dropdown'),
            html.Label('Metric:', className='inline-label'),
            dcc.Dropdown(id='comp-metric-dropdown', options=[
                {'label': 'Accel RMS', 'value': 'accel_rms'},
                {'label': 'Gyro RMS', 'value': 'gyro_rms'},
                {'label': 'Accel Std', 'value': 'accel_std'},
                {'label': 'Gyro Std', 'value': 'gyro_std'},
            ], value='accel_rms', clearable=False, className='inline-dropdown'),
        ], className='controls-row'),
        html.Div([
            dcc.Graph(id='new-vs-group-box', config={'displayModeBar': False}),
        ], className='viz-block'),
        html.Div([
            dcc.Graph(id='asymmetry-scatter', config={'displayModeBar': False}),
        ], className='viz-block'),
    ])


def build_tab_sensor():
    """Tab 3: Sensor Analysis — Spectral + Rhythm + Evidence Ribbon."""
    task_options = get_task_list()
    return html.Div([
        # Task selector
        html.Div([
            html.Label('Task:', className='inline-label'),
            dcc.Dropdown(id='sensor-task-dropdown', options=task_options,
                         value='TouchNose', clearable=False, className='inline-dropdown'),
        ], className='controls-row'),
        # Spectral Fingerprint
        html.Div([
            html.H3('Bilateral Spectral Fingerprint', className='viz-title'),
            dcc.Graph(id='spectral-fingerprint', config={'displayModeBar': False}),
        ], className='viz-block'),
        # Rhythm Ladder
        html.Div([
            html.H3('Rhythm Instability Ladder', className='viz-title'),
            dcc.Graph(id='rhythm-ladder', config={'displayModeBar': False}),
        ], className='viz-block'),
        # Evidence Ribbon
        html.Div([
            html.H3('Evidence Ribbon (All Tasks)', className='viz-title'),
            dcc.Graph(id='evidence-ribbon', config={'displayModeBar': False}),
        ], className='viz-block'),
        # Raw sensor (collapsible)
        html.Details([
            html.Summary('Raw 6-Axis Timeseries', className='raw-toggle'),
            dcc.Graph(id='raw-timeseries-left', config={'displayModeBar': False}),
            dcc.Graph(id='raw-timeseries-right', config={'displayModeBar': False}),
        ], className='raw-sensor-section'),
    ])


def build_tab_asymmetry():
    """Tab 3: Bilateral Asymmetry — PD core feature analysis."""
    task_options = get_task_list()
    return html.Div([
        html.Div([
            html.H3('좌우 비대칭 분석 (Bilateral Asymmetry)', className='viz-title'),
            html.P('PD의 핵심 특징: 편측성(laterality). '
                   '좌우 손목 센서 차이가 클수록 PD 가능성 시사.',
                   className='tab-description'),
        ]),
        # ── Patient-level (no task selection) ──
        html.Div([
            dcc.Graph(id='asymmetry-heatmap', config={'displayModeBar': False}),
        ], className='viz-block'),
        html.Div([
            html.H3('비대칭 지수 — 그룹 비교', className='viz-title'),
            dcc.Graph(id='asym-group-compare', config={'displayModeBar': False}),
        ], className='viz-block'),

        # ── Task-specific (task selector) ──
        html.Div([
            html.Label('Task (아래 차트에 적용):', className='inline-label'),
            dcc.Dropdown(id='asym-task-dropdown', options=task_options,
                         value='TouchNose', clearable=False, className='inline-dropdown'),
        ], className='controls-row'),
        html.Div([
            html.H3('좌/우 Waveform 비교', className='viz-title'),
            dcc.Graph(id='asym-waveform', config={'displayModeBar': False}),
        ], className='viz-block'),
        html.Div([
            html.H3('좌/우 Feature 비교', className='viz-title'),
            dcc.Graph(id='asym-feature-bars', config={'displayModeBar': False}),
        ], className='viz-block'),
    ])


def build_tab_summary():
    """Tab 6: Clinical Summary — all evidence compiled."""
    return html.Div([
        html.Div([
            html.H3('Clinical Summary Report', className='viz-title'),
            html.P('이 환자에 대한 모든 센서 분석 결과를 종합합니다.',
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


def build_tab_video():
    """Tab 5: Video Analysis."""
    return html.Div([
        # Controls
        html.Div([
            html.Div([
                html.Label('Side:', className='inline-label'),
                dcc.Dropdown(
                    id='video-side-dropdown',
                    options=[
                        {'label': 'Left Hand', 'value': 'left'},
                        {'label': 'Right Hand', 'value': 'right'},
                    ],
                    value='left', clearable=False, className='inline-dropdown',
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
                    _card('video-left-taps', 'L Taps'),
                    _card('video-right-taps', 'R Taps'),
                    _card('video-l-cv', 'L Interval CV'),
                    _card('video-r-cv', 'R Interval CV'),
                ], className='video-cards-grid'),
            ], className='video-left'),
            html.Div([
                dcc.Graph(id='motion-timeline', config={'displayModeBar': False}),
                dcc.Graph(id='lr-tapping-comparison', config={'displayModeBar': False}),
            ], className='video-right'),
        ], className='video-split-layout'),
    ])

"""Dashboard layout — PD Clinical Decision Support (5 tabs)."""

from dash import html, dcc
from src.data_loader import get_subject_list, get_task_list


def _card(card_id, label, initial='—'):
    return html.Div(className='summary-card', children=[
        html.Div(label, className='card-label'),
        html.Div(initial, id=card_id, className='card-value'),
    ])


def _tab_desc(text):
    return html.Div(className='tab-description', children=[html.P(text)])


def create_layout():
    subject_options = get_subject_list()
    task_options = get_task_list()

    return html.Div([
        # ── Header ──
        html.Div(className='dashboard-header', children=[
            html.Div(className='header-title-section', children=[
                html.H1("PD Clinical Decision Support"),
                html.P('스마트워치 센서 + 멀티카메라 영상 기반 · Movement Interpretation 보조 시스템'),
            ]),
            html.Div(className='header-controls', children=[
                html.Div(className='dropdown-container', children=[
                    html.Label('환자 선택'),
                    dcc.Dropdown(
                        id='patient-dropdown',
                        options=subject_options,
                        value=subject_options[0]['value'] if subject_options else None,
                        clearable=False, style={'color': '#2c3e50'},
                    ),
                ]),
                html.Span(id='patient-badge', className='badge'),
            ]),
        ]),

        # ── Main ──
        html.Div(className='main-container', children=[
            dcc.Tabs(id='main-tabs', className='custom-tabs',
                     vertical=True, parent_className='tabs-container',
                     content_className='tab-content-area', children=[

                # ── 1. 환자 Overview ──
                dcc.Tab(label='환자 Overview', className='tab',
                        selected_className='tab--selected', children=[
                    html.Div(className='tab-content', children=[
                        _tab_desc('환자 인구통계, 진단, H&Y Stage, Non-Motor Symptoms 요약'),
                        html.Div(id='patient-demographics', className='demographics-row'),
                        html.Div(className='summary-cards-row', children=[
                            _card('condition-card', 'Condition'),
                            _card('hy-card', 'H&Y Stage'),
                            _card('diagnosis-card', 'Clinician Diagnosis'),
                            _card('nms-count-card', 'NMS Symptoms'),
                            _card('bmi-card', 'BMI'),
                            _card('handedness-card', 'Handedness'),
                        ]),
                        html.Div(className='card', children=[
                            html.H3('Non-Motor Symptoms (NMS)', className='section-title'),
                            html.Div(id='nms-symptom-list', className='nms-symptom-list',
                                     children='환자를 선택하면 NMS 증상이 표시됩니다.'),
                        ]),
                    ]),
                ]),

                # ── 2. UPDRS 임상 평가 ──
                dcc.Tab(label='UPDRS 임상 평가', className='tab',
                        selected_className='tab--selected', children=[
                    html.Div(className='tab-content', children=[
                        _tab_desc('3명 Clinician의 UPDRS Part III 평가 결과. '
                                  '히트맵 셀 클릭 시 Clinician별 점수 비교가 표시됩니다.'),
                        html.Div(className='card', children=[
                            dcc.Graph(id='updrs-heatmap'),
                        ]),
                        html.Div(className='card', children=[
                            dcc.Graph(id='clinician-comparison-bar'),
                        ]),
                        html.Div(className='detail-panel', id='updrs-click-detail',
                                 children='Heatmap 셀을 클릭하면 Clinician별 비교가 표시됩니다.'),
                    ]),
                ]),

                # ── 3. 센서 & 좌우 비교 ──
                dcc.Tab(label='센서 & 좌우 비교', className='tab',
                        selected_className='tab--selected', children=[
                    html.Div(className='tab-content', children=[
                        _tab_desc('PADS 스마트워치 6축 원시 신호와 좌우 손목 비교. '
                                  'PD 환자의 lateralized motor impairment를 확인합니다.'),
                        html.Div(className='inline-task-selector', children=[
                            html.Label('Task', className='inline-label'),
                            dcc.Dropdown(
                                id='sensor-task-dropdown',
                                options=task_options,
                                value='TouchNose',
                                clearable=False,
                                style={'width': '200px', 'color': '#2c3e50'},
                            ),
                        ]),
                        # L/R summary
                        html.Div(className='summary-cards-row', children=[
                            _card('sensor-left-rms', 'Left Accel RMS'),
                            _card('sensor-right-rms', 'Right Accel RMS'),
                            _card('sensor-asymmetry', 'Asymmetry Index'),
                            _card('sensor-duration-card', 'Duration (s)'),
                        ]),
                        # L/R overlay
                        html.Div(className='card', children=[
                            dcc.Graph(id='lr-overlay-chart'),
                        ]),
                        # L/R stats bar
                        html.Div(className='card', children=[
                            dcc.Graph(id='lr-rms-bar-chart'),
                        ]),
                        # Raw timeseries (expandable)
                        html.Details(open=False, children=[
                            html.Summary('원시 6축 센서 데이터 (Left Wrist)',
                                         style={'cursor': 'pointer', 'fontWeight': '600',
                                                'color': '#2b6cb0', 'marginBottom': '8px'}),
                            html.Div(className='card', children=[
                                dcc.Graph(id='sensor-left-timeseries'),
                            ]),
                        ]),
                        html.Details(open=False, children=[
                            html.Summary('원시 6축 센서 데이터 (Right Wrist)',
                                         style={'cursor': 'pointer', 'fontWeight': '600',
                                                'color': '#2b6cb0', 'marginBottom': '8px'}),
                            html.Div(className='card', children=[
                                dcc.Graph(id='sensor-right-timeseries'),
                            ]),
                        ]),
                    ]),
                ]),

                # ── 4. PD vs Healthy 비교 ──
                dcc.Tab(label='PD vs Healthy', className='tab',
                        selected_className='tab--selected', children=[
                    html.Div(className='tab-content', children=[
                        _tab_desc('PD군과 Healthy군의 센서 지표를 비교합니다. '
                                  '선택된 환자(★)가 어느 분포에 위치하는지 확인하여 '
                                  '객관적 movement interpretation 근거를 제공합니다.'),
                        html.Div(className='inline-task-selector', children=[
                            html.Label('Task', className='inline-label'),
                            dcc.Dropdown(
                                id='group-task-dropdown',
                                options=task_options,
                                value='TouchNose',
                                clearable=False,
                                style={'width': '200px', 'color': '#2c3e50'},
                            ),
                            html.Label('Metric', className='inline-label',
                                       style={'marginLeft': '16px'}),
                            dcc.Dropdown(
                                id='group-metric-dropdown',
                                options=[
                                    {'label': 'Accel RMS', 'value': 'accel_rms'},
                                    {'label': 'Gyro RMS', 'value': 'gyro_rms'},
                                    {'label': 'Accel Variability', 'value': 'accel_std'},
                                    {'label': 'Gyro Variability', 'value': 'gyro_std'},
                                ],
                                value='accel_rms',
                                clearable=False,
                                style={'width': '200px', 'color': '#2c3e50'},
                            ),
                        ]),
                        # Group comparison box plot
                        html.Div(className='card', children=[
                            dcc.Graph(id='group-box-chart'),
                        ]),
                        # Asymmetry scatter
                        html.Div(className='card', children=[
                            dcc.Graph(id='asymmetry-scatter-chart'),
                        ]),
                        # Task summary bar (all tasks overview)
                        html.Div(className='card', children=[
                            dcc.Graph(id='group-task-summary-chart'),
                        ]),
                    ]),
                ]),

                # ── 5. 영상 분석 (2분할 레이아웃) ──
                dcc.Tab(label='영상 분석', className='tab',
                        selected_className='tab--selected', children=[
                    html.Div(className='tab-content', children=[
                        _tab_desc('Finger Tapping 영상과 정량 분석을 나란히 표시하여 '
                                  '움직임 패턴을 시각적으로 해석합니다.'),
                        # Controls
                        html.Div(className='inline-task-selector', children=[
                            html.Label('방향', className='inline-label'),
                            dcc.Dropdown(
                                id='video-side-dropdown',
                                options=[
                                    {'label': 'Left Hand', 'value': 'left'},
                                    {'label': 'Right Hand', 'value': 'right'},
                                ],
                                value='left', clearable=False,
                                style={'width': '140px', 'color': '#2c3e50'},
                            ),
                            html.Label('카메라', className='inline-label',
                                       style={'marginLeft': '12px'}),
                            dcc.Dropdown(
                                id='video-camera-dropdown',
                                options=[{'label': f'Camera {i}', 'value': f'Camera{i}.mp4'}
                                         for i in range(1, 7)],
                                value='Camera1.mp4', clearable=False,
                                style={'width': '140px', 'color': '#2c3e50'},
                            ),
                        ]),
                        # ── 2분할: 좌=영상+메트릭, 우=차트 ──
                        html.Div(className='video-split-layout', children=[
                            # Left panel: Video + metrics
                            html.Div(className='video-left-panel', children=[
                                html.Div(className='card', id='video-player-container',
                                         style={'marginBottom': '12px'}),
                                html.Div(className='summary-cards-row',
                                         style={'gridTemplateColumns': 'repeat(2, 1fr)'}, children=[
                                    _card('left-tap-count', 'Left Taps'),
                                    _card('right-tap-count', 'Right Taps'),
                                    _card('left-cv-card', 'L Interval CV'),
                                    _card('right-cv-card', 'R Interval CV'),
                                ]),
                            ]),
                            # Right panel: Charts stacked
                            html.Div(className='video-right-panel', children=[
                                html.Div(className='card', children=[
                                    dcc.Graph(id='motion-timeline-chart'),
                                ]),
                                html.Div(className='card', children=[
                                    dcc.Graph(id='lr-tapping-chart'),
                                ]),
                            ]),
                        ]),
                        # Bottom: full-width charts
                        html.Div(className='chart-row', children=[
                            html.Div(className='card', children=[
                                dcc.Graph(id='tapping-summary-chart'),
                            ]),
                            html.Div(className='card', children=[
                                dcc.Graph(id='multicam-chart'),
                            ]),
                        ]),
                    ]),
                ]),

            ]),
        ]),
    ])

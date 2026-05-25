"""Dashboard layout — 6-tab clinical data dashboard."""

from dash import html, dcc
from src.data_loader import get_subject_list, get_task_list, TASK_LABELS_KR


def _card(card_id, label, initial='—'):
    return html.Div(className='summary-card', children=[
        html.Div(label, className='card-label'),
        html.Div(initial, id=card_id, className='card-value'),
    ])


def _tab_desc(text):
    return html.Div(className='tab-description', children=[html.P(text)])


def _task_selector(dropdown_id, label='Motor Task 선택'):
    task_options = get_task_list()
    return html.Div(className='inline-task-selector', children=[
        html.Label(label, className='inline-label'),
        dcc.Dropdown(
            id=dropdown_id,
            options=task_options,
            value=task_options[0]['value'] if task_options else None,
            clearable=False,
            style={'width': '220px', 'color': '#2c3e50'},
        ),
    ])


def create_layout():
    subject_options = get_subject_list()

    return html.Div([
        # ── Header ──
        html.Div(className='dashboard-header', children=[
            html.Div(className='header-title-section', children=[
                html.H1("PD Motor Assessment Dashboard"),
                html.P('TULIP/PADS 실제 임상 데이터 기반 · UPDRS + Sensor + Video'),
            ]),
            html.Div(className='header-controls', children=[
                html.Div(className='dropdown-container', children=[
                    html.Label('Subject 선택'),
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
                        _tab_desc('환자 인구통계, 진단, NMS 증상, H&Y Stage 요약'),
                        # Demographics row
                        html.Div(id='patient-demographics', className='demographics-row'),
                        # Summary cards
                        html.Div(className='summary-cards-row', children=[
                            _card('condition-card', 'Condition'),
                            _card('hy-card', 'H&Y Stage'),
                            _card('diagnosis-card', 'Diagnosis'),
                            _card('nms-count-card', 'NMS Symptoms'),
                            _card('bmi-card', 'BMI'),
                            _card('handedness-card', 'Handedness'),
                        ]),
                        # NMS symptom list
                        html.Div(className='card', children=[
                            html.H3('Non-Motor Symptoms (NMS)', className='section-title'),
                            html.Div(id='nms-symptom-list', className='nms-symptom-list',
                                     children='Subject를 선택하면 NMS 증상이 표시됩니다.'),
                        ]),
                    ]),
                ]),

                # ── 2. UPDRS 임상 평가 ──
                dcc.Tab(label='UPDRS 임상 평가', className='tab',
                        selected_className='tab--selected', children=[
                    html.Div(className='tab-content', children=[
                        _tab_desc('12명 Subject × 28개 UPDRS 항목의 3명 Clinician 평균 Score 히트맵. '
                                  '셀 클릭 시 Clinician별 비교 차트가 표시됩니다.'),
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

                # ── 3. Inter-rater Agreement ──
                dcc.Tab(label='Inter-rater Agreement', className='tab',
                        selected_className='tab--selected', children=[
                    html.Div(className='tab-content', children=[
                        _tab_desc('3명 Clinician 간 점수 불일치(Max - Min)를 히트맵으로 시각화합니다. '
                                  '불일치가 큰 항목/환자를 한눈에 파악합니다.'),
                        html.Div(className='summary-cards-row', children=[
                            _card('agreement-perfect-card', 'Perfect Agreement'),
                            _card('agreement-high-card', 'High Disagreement'),
                            _card('agreement-mean-card', 'Mean Disagreement'),
                        ]),
                        html.Div(className='card', children=[
                            dcc.Graph(id='disagreement-heatmap'),
                        ]),
                    ]),
                ]),

                # ── 4. 센서 데이터 ──
                dcc.Tab(label='센서 데이터', className='tab',
                        selected_className='tab--selected', children=[
                    html.Div(className='tab-content', children=[
                        _tab_desc('PADS 스마트워치 6축 센서 원시 데이터 (가속도계 + 자이로스코프). '
                                  'Task와 손목을 선택하여 시계열을 확인합니다.'),
                        html.Div(className='inline-task-selector', children=[
                            html.Label('Task', className='inline-label'),
                            dcc.Dropdown(
                                id='sensor-task-dropdown',
                                options=get_task_list(),
                                value='TouchNose',
                                clearable=False,
                                style={'width': '200px', 'color': '#2c3e50'},
                            ),
                            html.Label('Wrist', className='inline-label',
                                       style={'marginLeft': '16px'}),
                            dcc.RadioItems(
                                id='sensor-wrist-toggle',
                                options=[
                                    {'label': 'Left', 'value': 'LeftWrist'},
                                    {'label': 'Right', 'value': 'RightWrist'},
                                ],
                                value='LeftWrist',
                                inline=True,
                                style={'display': 'flex', 'gap': '12px'},
                            ),
                        ]),
                        html.Div(className='summary-cards-row', children=[
                            _card('sensor-accel-rms-card', 'Accel RMS'),
                            _card('sensor-gyro-rms-card', 'Gyro RMS'),
                            _card('sensor-duration-card', 'Duration (s)'),
                            _card('sensor-samples-card', 'Samples'),
                        ]),
                        html.Div(className='card', children=[
                            dcc.Graph(id='sensor-timeseries-chart'),
                        ]),
                    ]),
                ]),

                # ── 5. 좌우 비교 ──
                dcc.Tab(label='좌우 비교', className='tab',
                        selected_className='tab--selected', children=[
                    html.Div(className='tab-content', children=[
                        _tab_desc('선택한 Task에서 좌/우 손목 신호 크기를 오버레이하여 비교합니다. '
                                  'RMS 기반 비대칭 지수를 정량적으로 확인합니다.'),
                        _task_selector('lr-task-dropdown', 'Task 선택'),
                        html.Div(className='summary-cards-row', children=[
                            _card('lr-left-rms-card', 'Left RMS'),
                            _card('lr-right-rms-card', 'Right RMS'),
                            _card('lr-asymmetry-card', 'Asymmetry Index'),
                        ]),
                        html.Div(className='card', children=[
                            dcc.Graph(id='lr-overlay-chart'),
                        ]),
                        html.Div(className='card', children=[
                            dcc.Graph(id='lr-rms-bar-chart'),
                        ]),
                    ]),
                ]),

                # ── 6. 영상 분석 ──
                dcc.Tab(label='영상 분석', className='tab',
                        selected_className='tab--selected', children=[
                    html.Div(className='tab-content', children=[
                        _tab_desc('Finger Tapping 영상(6대 카메라)에서 motion intensity를 '
                                  '추출하여 좌/우 tapping 정량 비교와 multi-camera 관점 비교를 제공합니다.'),
                        html.Div(className='inline-task-selector', children=[
                            html.Label('카메라 / 방향', className='inline-label'),
                            dcc.Dropdown(
                                id='video-side-dropdown',
                                options=[
                                    {'label': 'Left Hand', 'value': 'left'},
                                    {'label': 'Right Hand', 'value': 'right'},
                                ],
                                value='left', clearable=False,
                                style={'width': '160px', 'color': '#2c3e50'},
                            ),
                            dcc.Dropdown(
                                id='video-camera-dropdown',
                                options=[{'label': f'Camera {i}', 'value': f'Camera{i}.mp4'}
                                         for i in range(1, 7)],
                                value='Camera1.mp4', clearable=False,
                                style={'width': '160px', 'color': '#2c3e50'},
                            ),
                        ]),
                        html.Div(className='card', id='video-player-container'),
                        html.Div(className='summary-cards-row', children=[
                            _card('left-tap-count', 'Left Taps'),
                            _card('right-tap-count', 'Right Taps'),
                            _card('left-cv-card', 'L Interval CV'),
                            _card('right-cv-card', 'R Interval CV'),
                        ]),
                        html.Div(className='card', children=[
                            dcc.Graph(id='motion-timeline-chart'),
                        ]),
                        html.Div(className='card', children=[
                            dcc.Graph(id='lr-tapping-chart'),
                        ]),
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

"""Dashboard layout — Parkinson's Motor Assessment Dashboard."""

from dash import html, dcc
from src.data_loader import get_patient_list, get_task_list


TASK_LABELS = {
    'finger_tapping': 'Finger Tapping',
    'hand_open_close': 'Hand Open/Close',
    'rest_tremor': 'Rest Tremor',
    'gait': 'Gait',
    'toe_tapping': 'Toe Tapping',
    'touch_nose': 'Touch Nose',
}


def _card(card_id, label, initial='—'):
    return html.Div(className='summary-card', children=[
        html.Div(label, className='card-label'),
        html.Div(initial, id=card_id, className='card-value'),
    ])


def _tab_desc(text):
    return html.Div(className='tab-description', children=[html.P(text)])


def _task_selector(dropdown_id):
    """Inline task selector for tabs that need it."""
    task_options = [
        {'label': TASK_LABELS.get(t['value'], t['label']), 'value': t['value']}
        for t in get_task_list()
    ]
    return html.Div(className='inline-task-selector', children=[
        html.Label('Motor Task 선택', className='inline-label'),
        dcc.Dropdown(
            id=dropdown_id,
            options=task_options,
            value=task_options[0]['value'] if task_options else None,
            clearable=False,
            style={'width': '220px', 'color': '#2c3e50'},
        ),
    ])


def create_layout():
    patient_options = get_patient_list()

    return html.Div([
        # ── Header (환자 선택만) ──
        html.Div(className='dashboard-header', children=[
            html.Div(className='header-title-section', children=[
                html.H1("Parkinson's Motor Assessment Dashboard"),
                html.P('Smartwatch 센서 기반 정량적 운동 분석 · Clinical Decision Support'),
            ]),
            html.Div(className='header-controls', children=[
                html.Div(className='dropdown-container', children=[
                    html.Label('환자'),
                    dcc.Dropdown(
                        id='patient-dropdown',
                        options=patient_options,
                        value=patient_options[0]['value'] if patient_options else None,
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
                        _tab_desc('환자의 전반적인 motor/non-motor 상태 요약입니다. '
                                  '주요 이상 지표와 가장 abnormal한 task를 한눈에 확인할 수 있습니다.'),
                        # Demographics row
                        html.Div(id='patient-demographics', className='demographics-row'),
                        # Summary cards
                        html.Div(className='summary-cards-row', children=[
                            _card('motor-score-card', 'Motor Score'),
                            _card('asymmetry-score-card', 'Asymmetry'),
                            _card('tremor-score-card', 'Tremor'),
                            _card('rhythm-score-card-overview', 'Rhythm Irreg.'),
                            _card('nonmotor-score-card', 'Non-Motor'),
                            _card('top-task-card', 'Top Abnormal Task'),
                        ]),
                        html.Div(className='chart-row', children=[
                            html.Div(className='card', children=[
                                dcc.Graph(id='overview-radar'),
                            ]),
                            html.Div(className='card', children=[
                                dcc.Graph(id='top-tasks-bar'),
                            ]),
                        ]),
                        html.Div(className='interpretation',
                                 id='interpretation-text',
                                 children='환자를 선택하면 요약이 표시됩니다.'),
                    ]),
                ]),

                # ── 2. Task-Feature Heatmap ──
                dcc.Tab(label='Task-Feature Heatmap', className='tab',
                        selected_className='tab--selected', children=[
                    html.Div(className='tab-content', children=[
                        _tab_desc('어떤 motor task에서 어떤 feature가 abnormal한지 한눈에 보여줍니다. '
                                  '셀을 클릭하면 상세 정보를 확인할 수 있습니다.'),
                        html.Div(className='card', children=[
                            dcc.Graph(id='task-feature-heatmap'),
                        ]),
                        html.Div(className='detail-panel', id='heatmap-detail',
                                 children='Heatmap 셀을 클릭하면 상세 정보가 표시됩니다.'),
                    ]),
                ]),

                # ── 3. 좌우 비교 ──
                dcc.Tab(label='좌우 비교', className='tab',
                        selected_className='tab--selected', children=[
                    html.Div(className='tab-content', children=[
                        _tab_desc('선택한 task에서 좌/우 사지의 movement feature 차이를 비교합니다. '
                                  'PD의 lateralized impairment를 정량적으로 확인합니다.'),
                        _task_selector('lr-task-dropdown'),
                        html.Div(className='summary-cards-row', children=[
                            html.Div(className='summary-card', children=[
                                html.Div('Asymmetry 요약', className='card-label'),
                                html.Div('—', id='asymmetry-card', className='card-value',
                                         style={'fontSize': '14px'}),
                            ]),
                        ]),
                        html.Div(className='card', children=[
                            dcc.Graph(id='mirror-plot'),
                        ]),
                    ]),
                ]),

                # ── 4. Rhythm & Amplitude ──
                dcc.Tab(label='Rhythm & Amplitude', className='tab',
                        selected_className='tab--selected', children=[
                    html.Div(className='tab-content', children=[
                        _tab_desc('반복 운동의 rhythm irregularity와 amplitude decrement를 분석합니다. '
                                  'Bradykinesia의 핵심 정량 지표입니다.'),
                        _task_selector('rhythm-task-dropdown'),
                        html.Div(className='summary-cards-row', children=[
                            _card('rhythm-score-card', 'Rhythm Irregularity (CV)'),
                            _card('amplitude-score-card', 'Amplitude Decrement'),
                        ]),
                        html.Div(className='card', children=[
                            dcc.Graph(id='rhythm-timeseries'),
                        ]),
                        html.Div(className='chart-row', children=[
                            html.Div(className='card', children=[
                                dcc.Graph(id='tap-interval-chart'),
                            ]),
                            html.Div(className='card', children=[
                                dcc.Graph(id='amplitude-decrement-chart'),
                            ]),
                        ]),
                    ]),
                ]),

                # ── 5. Normative 비교 ──
                dcc.Tab(label='Normative 비교', className='tab',
                        selected_className='tab--selected', children=[
                    html.Div(className='tab-content', children=[
                        _tab_desc('Healthy control 대비 환자의 위치를 보여줍니다. '
                                  'Z-score 2 이상은 clinically significant abnormality입니다.'),
                        html.Div(className='summary-cards-row', children=[
                            html.Div(className='summary-card', children=[
                                html.Div('Normative 위치', className='card-label'),
                                html.Div('—', id='normative-summary-card',
                                         className='card-value', style={'fontSize': '13px'}),
                            ]),
                        ]),
                        html.Div(className='card', children=[
                            dcc.Graph(id='normative-chart'),
                        ]),
                        html.Div(className='detail-panel', id='normative-detail',
                                 children='Healthy control 대비 환자 위치를 보여줍니다.'),
                    ]),
                ]),

                # ── 6. 경과 추적 ──
                dcc.Tab(label='경과 추적', className='tab',
                        selected_className='tab--selected', children=[
                    html.Div(className='tab-content', children=[
                        _tab_desc('방문별 motor feature 변화를 추적합니다. '
                                  'Disease progression과 치료 반응 평가에 활용합니다.'),
                        html.Div(className='summary-cards-row', children=[
                            html.Div(className='summary-card', children=[
                                html.Div('방문 변화', className='card-label'),
                                html.Div('—', id='history-summary-card',
                                         className='card-value', style={'fontSize': '13px'}),
                            ]),
                        ]),
                        html.Div(className='card', children=[
                            dcc.Graph(id='history-line-chart'),
                        ]),
                        html.Div(className='card', children=[
                            dcc.Graph(id='history-change-chart'),
                        ]),
                    ]),
                ]),

                # ── 7. UPDRS 추정 ──
                dcc.Tab(label='UPDRS 추정', className='tab',
                        selected_className='tab--selected', children=[
                    html.Div(className='tab-content', children=[
                        _tab_desc('센서 기반 MDS-UPDRS Part III score 추정입니다. '
                                  '참고용 estimate이며, 최종 scoring은 임상 소견에 따릅니다.'),
                        html.Div(className='card', children=[
                            dcc.Graph(id='updrs-gauge-chart'),
                        ]),
                        html.Div(className='detail-panel', id='updrs-evidence-panel',
                                 children='환자를 선택하면 UPDRS 추정 score와 근거가 표시됩니다.'),
                    ]),
                ]),

                # ── 8. Phase Portrait ──
                dcc.Tab(label='Phase Portrait', className='tab',
                        selected_className='tab--selected', children=[
                    html.Div(className='tab-content', children=[
                        _tab_desc('Amplitude vs. velocity를 phase space에 그린 시각화입니다. '
                                  'Healthy는 깨끗한 ellipse, PD는 수축하는 distorted trajectory를 보입니다.'),
                        _task_selector('phase-task-dropdown'),
                        html.Div(className='card', children=[
                            dcc.Graph(id='phase-portrait-chart'),
                        ]),
                    ]),
                ]),

                # ── 9. Signature Wall ──
                dcc.Tab(label='Signature Wall', className='tab',
                        selected_className='tab--selected', children=[
                    html.Div(className='tab-content', children=[
                        _tab_desc('모든 환자의 movement waveform을 overlay합니다. '
                                  '선택 환자는 강조, 나머지는 배경으로 표시됩니다.'),
                        _task_selector('sig-task-dropdown'),
                        html.Div(className='card', children=[
                            dcc.Graph(id='signature-wall-chart'),
                        ]),
                    ]),
                ]),

                # ── 10. 영상 분석 ──
                dcc.Tab(label='영상 분석', className='tab',
                        selected_className='tab--selected', children=[
                    html.Div(className='tab-content', children=[
                        _tab_desc('실제 Finger Tapping 영상(TULIP dataset, 6대 카메라)에서 '
                                  'frame differencing 기반 motion intensity를 추출하여 분석합니다. '
                                  '좌/우 tapping의 정량적 비교와 multi-camera 관점 비교를 제공합니다.'),
                        # Video + side selector
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
                                options=[{'label': f'Camera {i}', 'value': f'Camera{i}.mp4'} for i in range(1, 7)],
                                value='Camera1.mp4', clearable=False,
                                style={'width': '160px', 'color': '#2c3e50'},
                            ),
                        ]),
                        # Video player (container — content set by callback)
                        html.Div(className='card', id='video-player-container'),
                        # Tapping summary
                        html.Div(className='summary-cards-row', children=[
                            html.Div(className='summary-card', children=[
                                html.Div('Left Taps', className='card-label'),
                                html.Div('—', id='left-tap-count', className='card-value'),
                            ]),
                            html.Div(className='summary-card', children=[
                                html.Div('Right Taps', className='card-label'),
                                html.Div('—', id='right-tap-count', className='card-value'),
                            ]),
                            html.Div(className='summary-card', children=[
                                html.Div('L Interval CV', className='card-label'),
                                html.Div('—', id='left-cv-card', className='card-value'),
                            ]),
                            html.Div(className='summary-card', children=[
                                html.Div('R Interval CV', className='card-label'),
                                html.Div('—', id='right-cv-card', className='card-value'),
                            ]),
                        ]),
                        # Charts
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

                # ── 11. Evidence Map ──
                dcc.Tab(label='Evidence Map', className='tab',
                        selected_className='tab--selected', children=[
                    html.Div(className='tab-content', children=[
                        _tab_desc('각 feature의 abnormality(X축)와 diagnostic power(Y축)를 '
                                  'bubble로 표시합니다. 오른쪽 위 = 임상적으로 가장 주목해야 할 이상입니다.'),
                        html.Div(className='card', children=[
                            dcc.Graph(id='evidence-bubble-chart'),
                        ]),
                    ]),
                ]),

            ]),
        ]),
    ])

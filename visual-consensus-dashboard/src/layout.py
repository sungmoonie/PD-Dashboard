"""Dashboard layout — single-page 3-column design with Review/Teaching mode."""

from dash import html, dcc
from src.data_loader import (
    get_subject_list_labelfree, get_subject_list, get_task_list, SENSOR_TASKS
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
    case_options = get_subject_list_labelfree()

    return html.Div([
        # ─── Stores ───
        dcc.Store(id='mode-store', data='review', storage_type='memory'),
        dcc.Store(id='decision-store', data={}, storage_type='local'),
        dcc.Store(id='selected-task-store', data='TouchNose'),
        dcc.Store(id='selected-feature-store', data='tremor_power'),

        # ─── Header ───
        html.Header([
            html.Div([
                html.H1('PD Clinical Decision Support', className='header-title'),
                html.P('Matched Sensor Analog Evidence · Label-Free Review',
                       className='header-subtitle'),
            ], className='header-left'),
            html.Div([
                html.Div([
                    html.Label('Case:', className='header-label'),
                    dcc.Dropdown(
                        id='patient-dropdown',
                        options=case_options,
                        value=case_options[0]['value'] if case_options else None,
                        className='case-dropdown',
                        clearable=False,
                    ),
                ], className='header-control'),
                html.Div([
                    html.Label('Mode:', className='header-label'),
                    html.Button('Review', id='mode-toggle-btn',
                                className='mode-toggle review-active',
                                n_clicks=0),
                ], className='header-control'),
                html.Div(id='case-badge', className='badge badge-unknown'),
            ], className='header-right'),
        ], className='dashboard-header'),

        # ─── Main 3-Column Grid ───
        html.Main([
            # ═══ LEFT: Video Panel ═══
            html.Aside([
                html.H3('Video Evidence', className='panel-title'),
                html.Div([
                    dcc.Dropdown(
                        id='video-side-dropdown',
                        options=[
                            {'label': 'Left Hand', 'value': 'left'},
                            {'label': 'Right Hand', 'value': 'right'},
                        ],
                        value='left', clearable=False, className='mini-dropdown',
                    ),
                    dcc.Dropdown(
                        id='video-camera-dropdown',
                        options=[{'label': f'Camera {i}', 'value': f'Camera{i}.mp4'}
                                 for i in range(1, 7)],
                        value='Camera1.mp4', clearable=False, className='mini-dropdown',
                    ),
                ], className='video-controls'),
                html.Div(id='video-player-container', className='video-container'),
                # Section nav for tasks
                html.H4('Task Navigator', className='section-title'),
                html.Div([
                    dcc.Dropdown(
                        id='task-dropdown',
                        options=task_options,
                        value='TouchNose',
                        clearable=False,
                        className='task-dropdown',
                    ),
                ], className='task-nav'),
                # Video summary cards
                html.Div([
                    _card('video-left-taps', 'L Taps'),
                    _card('video-right-taps', 'R Taps'),
                    _card('video-l-cv', 'L Interval CV'),
                    _card('video-r-cv', 'R Interval CV'),
                ], className='video-cards-grid'),
            ], className='left-panel'),

            # ═══ CENTER: Visualizations ═══
            html.Section([
                # Demographics row (hidden in review, shown in teaching)
                html.Div(id='demographics-row', className='demographics-row hidden-in-review'),

                # Task-Symptom Bilateral Matrix
                html.Div([
                    html.H3('Task-Symptom Bilateral Matrix', className='viz-title'),
                    dcc.Graph(id='bilateral-matrix', config={'displayModeBar': False}),
                ], className='viz-block'),

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
                    html.H3('Side-Aligned Evidence Ribbon', className='viz-title'),
                    dcc.Graph(id='evidence-ribbon', config={'displayModeBar': False}),
                ], className='viz-block'),

                # Raw sensor toggle (collapsed by default)
                html.Details([
                    html.Summary('Raw Sensor Data', className='raw-toggle'),
                    dcc.Graph(id='raw-timeseries-left',
                              config={'displayModeBar': False}),
                    dcc.Graph(id='raw-timeseries-right',
                              config={'displayModeBar': False}),
                ], className='raw-sensor-section'),

            ], className='center-panel'),

            # ═══ RIGHT: Decision Workspace ═══
            html.Aside([
                html.H3('Decision Workspace', className='panel-title'),

                # Teaching-mode info (hidden in review)
                html.Div([
                    html.Div([
                        _card('teaching-condition', 'Condition'),
                        _card('teaching-hy', 'H&Y Stage'),
                        _card('teaching-diagnosis', 'Clinician Dx'),
                    ], className='teaching-cards'),
                    html.Div(id='teaching-updrs-summary', className='teaching-detail'),
                ], id='teaching-panel', className='hidden-in-review'),

                # Decision form
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
                ], className='decision-form'),

                # Export
                html.Div([
                    html.Button('Export JSON', id='export-btn',
                                className='btn-secondary', n_clicks=0),
                    dcc.Download(id='download-json'),
                ], className='export-section'),

            ], className='right-panel'),
        ], className='dashboard-grid'),

        # ─── Bottom: Video Analysis ───
        html.Section([
            html.H3('Video Analysis', className='viz-title'),
            html.Div([
                dcc.Graph(id='motion-timeline', config={'displayModeBar': False}),
                dcc.Graph(id='lr-tapping-comparison', config={'displayModeBar': False}),
            ], className='video-analysis-row'),
        ], className='bottom-section'),
    ], className='app-container')

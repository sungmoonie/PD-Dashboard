"""
PD Clinical Decision Support Dashboard
========================================
Hybrid layout: Tabs + Decision Workspace + New vs Confirmed Group Comparison
"""

import os
import json
import dash
from dash import Input, Output, State, html, dcc, callback_context
from flask import send_file, request, Response, redirect

# ─── Data Loading ───
from src.data_loader import (
    load_patients, load_nms, load_updrs_labels, load_updrs_metadata,
    build_group_stats, build_feature_cache,
    load_timeseries, get_subject_list_hybrid,
    SENSOR_TASKS, TASK_LABELS_KR, NEW_CASES, get_group_label, _estimate_fs,
    load_video_feature_df, VIDEO_TASK_FOLDER_MAP, load_video_analysis,
)
from src.feature_engineering import (
    calc_signal_rms, calc_asymmetry_index, calc_signal_stats,
)
from src.figures import (
    make_timeseries_plot,
    make_video_feature_timeline, make_video_lr_feature_comparison, summarize_video_task_metrics,
    make_video_interval_distribution, make_video_tremor_spectrogram, make_video_symmetry_trend,
    make_motion_timeline, make_lr_tapping_comparison,
    make_bilateral_matrix, make_spectral_fingerprint,
    make_rhythm_ladder, make_evidence_ribbon,
    make_new_vs_group_box, make_group_feature_comparison,
    make_task_profile_comparison, make_asymmetry_scatter,
    make_proximity_gauge,
    make_asymmetry_heatmap, make_asym_waveform,
    make_asym_feature_bars, make_asym_group_compare,
    make_summary_radar,
    make_motor_landscape, make_bilateral_phase_space,
    _empty_fig,
)
from src.layout import (
    create_layout, build_tab_overview, build_tab_comparison,
    build_tab_sensor, build_tab_video, build_tab_landscape,
    build_tab_asymmetry, build_tab_summary,
)

# ─── Pre-load Data ───
patients_df = load_patients()
nms_data = load_nms()
labels_df = load_updrs_labels()
updrs_meta = load_updrs_metadata()
# [PRE] Legacy JSON video analysis preload kept for compatibility reference.
video_data = load_video_analysis()
group_stats = build_group_stats()
feature_cache = build_feature_cache()

# ─── Dash App ───
app = dash.Dash(
    __name__,
    title='PD Clinical Decision Support',
    suppress_callback_exceptions=True,
    assets_folder='assets',
)
server = app.server
app.layout = create_layout()


# ══════════════════════════════════════════════════════════════
#  VIDEO STREAMING — Cloudflare R2 CDN
# ══════════════════════════════════════════════════════════════

R2_BASE = 'https://pub-5e7cb79a175143fd80bb9497eeaa4e41.r2.dev'

# R2 folder mapping: (tulip_id, video_type) → R2 folder key
# video_type values match dropdown: 'toe_left', 'toe_right', 'resting'
R2_VIDEO_CATALOG = {
    'TULIP_001': {
        'toe_left': 'TULIP_001_17. Toe_tapping_left',
        'toe_right': 'TULIP_001_18. Toe_tapping_right',
        'resting': 'TULIP_001_26. Resting & hand tremor',
    },
    'TULIP_008': {
        'toe_left': 'TULIP_008_17. Toe_tapping_left',
        'toe_right': 'TULIP_008_18. Toe_tapping_right',
        'resting': 'TULIP_008_26. Resting & hand tremor',
    },
}

VIDEO_TYPE_LABELS = {
    'toe_left': 'Toe Tapping (Left)',
    'toe_right': 'Toe Tapping (Right)',
    'resting': 'Resting & Hand Tremor',
}


def _r2_video_url(tulip_id, video_type, camera):
    """Build Cloudflare R2 CDN URL for a video file."""
    patient_catalog = R2_VIDEO_CATALOG.get(tulip_id, {})
    folder = patient_catalog.get(video_type)
    if not folder:
        return None
    folder_encoded = folder.replace(' ', '%20').replace('&', '%26')
    return f'{R2_BASE}/{folder_encoded}/{camera}'


# ══════════════════════════════════════════════════════════════
#  CALLBACKS
# ═══════════════════════════════════════════════════════��══════

# ─── Tab Rendering ───
@app.callback(
    Output('tab-content', 'children'),
    Input('main-tabs', 'value'),
)
def render_tab(tab):
    if tab == 'tab-overview':
        return build_tab_overview()
    elif tab == 'tab-landscape':
        return build_tab_landscape()
    elif tab == 'tab-comparison':
        return build_tab_comparison()
    elif tab == 'tab-asymmetry':
        return build_tab_asymmetry()
    elif tab == 'tab-sensor':
        return build_tab_sensor()
    elif tab == 'tab-video':
        return build_tab_video()
    elif tab == 'tab-summary':
        return build_tab_summary()
    return html.Div('Select a tab')


# ─── Keep right panel visible on all tabs ───
@app.callback(
    Output('right-panel', 'style'),
    Output('dashboard-grid', 'style'),
    Input('main-tabs', 'value'),
)
def toggle_right_panel(tab):
    # [PRE] Previously, tab-video hid the right panel:
    # if tab == 'tab-video':
    #     return {'display': 'none'}, {'gridTemplateColumns': '1fr'}
    return {}, {}


# ─── Case Badge + Right Panel Info ───
_CLASSIFICATION_LABELS = {
    'PD': ('PD', 'badge badge-pd'),
    'HT': ('Healthy Tremor', 'badge badge-healthy'),
    '?': ('Uncertain', 'badge badge-other'),
}


@app.callback(
    Output('case-badge', 'children'),
    Output('case-badge', 'className'),
    Output('info-condition', 'children'),
    Output('info-hy', 'children'),
    Output('info-diagnosis', 'children'),
    Output('info-nms', 'children'),
    Output('new-patient-indicator', 'children'),
    Input('patient-dropdown', 'value'),
    Input('decision-store', 'data'),
)
def update_case_info(tulip_id, decision_store):
    if not tulip_id:
        return '—', 'badge badge-unknown', '—', '—', '—', '—', ''

    row = patients_df[patients_df.tulip_id == tulip_id]
    meta = updrs_meta.get(tulip_id, {})
    is_new = tulip_id in NEW_CASES
    decision_store = decision_store or {}
    saved = decision_store.get(tulip_id)

    # Check if doctor already saved a decision for this new patient
    has_decision = is_new and saved and saved.get('classification')

    # Badge
    if has_decision:
        cls = saved['classification']
        badge_text, badge_class = _CLASSIFICATION_LABELS.get(cls, (cls, 'badge badge-other'))
        badge_text = f'Dx: {badge_text}'
    elif is_new:
        badge_text = '★ NEW'
        badge_class = 'badge badge-new'
    else:
        condition = row.iloc[0]['condition'] if not row.empty else 'Unknown'
        group = get_group_label(tulip_id, condition)
        badge_text = group
        badge_class = f'badge badge-{group.lower()}'

    # Info cards
    condition = row.iloc[0]['condition'] if not row.empty else '—'
    if has_decision:
        cls = saved['classification']
        label, _ = _CLASSIFICATION_LABELS.get(cls, (cls, ''))
        conf = saved.get('confidence', 0)
        condition = html.Div([
            html.Div(label, className='card-value-main', style={'color': '#2b6cb0'}),
            html.Div(f'(Confidence {conf}%)', className='card-value-sub'),
        ])
    elif is_new:
        condition = html.Div([
            html.Div('미확정', className='card-value-main'),
            html.Div('(New Case)', className='card-value-sub'),
        ])

    hy = str(meta.get('hy_mean', '—'))
    diag = meta.get('diagnosis', '—')
    if has_decision:
        cls = saved['classification']
        label, _ = _CLASSIFICATION_LABELS.get(cls, (cls, ''))
        diag = html.Div([
            html.Div(f'Clinician: {label}', className='card-value-main',
                     style={'color': '#2b6cb0', 'fontSize': '13px'}),
        ])
    elif is_new:
        diag = html.Div([
            html.Div('판단 필요', className='card-value-main'),
        ])

    nms_count = str(nms_data.get(tulip_id, {}).get('count', '—'))

    # New patient indicator
    indicator = ''
    if has_decision:
        cls = saved['classification']
        label, _ = _CLASSIFICATION_LABELS.get(cls, (cls, ''))
        tags = ', '.join(saved.get('evidence_tags', [])) or '없음'
        indicator = html.Div([
            html.Span('✓ ', style={'fontSize': '16px'}),
            html.Span(f'진단 저장됨: {label} (Confidence {saved.get("confidence", 0)}%)'),
            html.Br(),
            html.Span(f'Evidence: {tags}', style={'fontSize': '11px', 'color': '#718096'}),
        ], className='new-case-alert', style={'background': '#f0fff4', 'borderColor': '#9ae6b4', 'color': '#276749'})
    elif is_new:
        indicator = html.Div([
            html.Span('★ ', className='new-star'),
            html.Span('이 환자는 미확정 New Case입니다. '),
            html.Span('아래에서 PD/HT/? 판단을 내려주세요.'),
        ], className='new-case-alert')

    return badge_text, badge_class, condition, hy, diag, nms_count, indicator


# ─── Dropdown label update on decision-store change ───
@app.callback(
    Output('patient-dropdown', 'options'),
    Input('decision-store', 'data'),
)
def update_dropdown_labels(decision_store):
    """Update dropdown labels to reflect saved diagnoses."""
    decision_store = decision_store or {}
    options = []
    for _, row in patients_df.iterrows():
        tid = row['tulip_id']
        num = tid.replace('TULIP_', '')
        group = get_group_label(tid, row['condition'])
        saved = decision_store.get(tid)
        if group == 'New' and saved and saved.get('classification'):
            cls = saved['classification']
            label_txt = _CLASSIFICATION_LABELS.get(cls, (cls, ''))[0]
            label = f"✓ Patient_{num} — Dx:{label_txt} ({row['age']}y, {row['gender']})"
        elif group == 'New':
            label = f"★ Patient_{num} — New ({row['age']}y, {row['gender']})"
        else:
            label = f"  Patient_{num} — {group} ({row['age']}y, {row['gender']})"
        options.append({'label': label, 'value': tid})
    return options


# ─── Tab 1: Patient Overview ───
@app.callback(
    Output('demographics-row', 'children'),
    Output('ov-age', 'children'),
    Output('ov-gender', 'children'),
    Output('ov-handedness', 'children'),
    Output('ov-bmi', 'children'),
    Output('ov-duration', 'children'),
    Output('nms-content', 'children'),
    Output('bilateral-matrix', 'figure'),
    Input('patient-dropdown', 'value'),
    Input('decision-store', 'data'),
)
def update_overview(tulip_id, decision_store):
    empty = ('', '—', '—', '—', '—', '—', '', _empty_fig())
    if not tulip_id:
        return empty

    row = patients_df[patients_df.tulip_id == tulip_id]
    if row.empty:
        return empty
    r = row.iloc[0]

    is_new = tulip_id in NEW_CASES
    decision_store = decision_store or {}
    saved = decision_store.get(tulip_id)
    has_decision = is_new and saved and saved.get('classification')

    if has_decision:
        cls = saved['classification']
        label_txt = _CLASSIFICATION_LABELS.get(cls, (cls, ''))[0]
        cond_display = f'Dx: {label_txt} (Confidence {saved.get("confidence", 0)}%)'
    elif is_new:
        cond_display = '★ New Case (미확정)'
    else:
        cond_display = r['condition']

    demographics = html.Div([
        html.Span(f"TULIP: {tulip_id}", className='demo-item demo-id'),
        html.Span(f"PADS: {r['pads_id']}", className='demo-item'),
        html.Span(f"Condition: {cond_display}",
                  className='demo-item demo-condition' + (
                      ' demo-dx' if has_decision else ' demo-new' if is_new else '')),
    ], className='demo-content')

    # NMS
    nms = nms_data.get(tulip_id, {})
    symptoms = nms.get('symptoms', [])
    nms_content = html.Div([
        html.Div([
            html.Span(f'• {s}', className='nms-symptom-item')
            for s in symptoms
        ], className='nms-list') if symptoms else html.P('No NMS reported')
    ])

    # Matrix
    fig_matrix = make_bilateral_matrix(feature_cache, tulip_id)

    age_diag = r.get('age_at_diagnosis')
    duration = f"{r['age'] - age_diag}y" if age_diag and r['age'] else '—'

    return (demographics, str(r['age'] or '—'), r['gender'] or '—',
            r['handedness'] or '—', str(r['bmi'] or '—'), duration,
            nms_content, fig_matrix)


# ─── Tab 2: Motor Landscape ───
@app.callback(
    Output('motor-landscape', 'figure'),
    Output('bilateral-phase-space', 'figure'),
    Input('patient-dropdown', 'value'),
    Input('landscape-task-dropdown', 'value'),
)
def update_landscape(tulip_id, task):
    if not tulip_id or not task:
        return _empty_fig(), _empty_fig()
    fig_landscape = make_motor_landscape(tulip_id, task)
    fig_phase = make_bilateral_phase_space(tulip_id, task)
    return fig_landscape, fig_phase


# ─── Tab 3: Group Comparison (patient-level, no task) ───
@app.callback(
    Output('proximity-gauge', 'figure'),
    Output('task-profile-comparison', 'figure'),
    Output('feature-group-comparison', 'figure'),
    Input('patient-dropdown', 'value'),
)
def update_comparison_patient(tulip_id):
    if not tulip_id:
        return (_empty_fig(),) * 3
    fig_gauge = make_proximity_gauge(group_stats, feature_cache, tulip_id)
    fig_profile = make_task_profile_comparison(group_stats, tulip_id)
    fig_features = make_group_feature_comparison(feature_cache, group_stats, tulip_id)
    return fig_gauge, fig_profile, fig_features


# ─── Tab 3: Group Comparison — auto-select metric by task ───
_TASK_RECOMMENDED_METRIC = {
    'Relaxed': 'accel_rms', 'RelaxedTask': 'accel_rms',
    'Entrainment': 'gyro_rms', 'TouchIndex': 'gyro_rms',
    'TouchNose': 'accel_rms', 'DrinkGlas': 'accel_rms',
    'PointFinger': 'accel_std', 'LiftHold': 'accel_rms',
    'CrossArms': 'accel_rms', 'HoldWeight': 'accel_rms',
    'StretchHold': 'accel_rms',
}


@app.callback(
    Output('comp-metric-dropdown', 'value'),
    Input('comp-task-dropdown', 'value'),
)
def auto_select_metric(task):
    return _TASK_RECOMMENDED_METRIC.get(task, 'accel_rms')


# ─── Tab 3: Group Comparison (task-specific) ───
@app.callback(
    Output('new-vs-group-box', 'figure'),
    Output('asymmetry-scatter', 'figure'),
    Input('patient-dropdown', 'value'),
    Input('comp-task-dropdown', 'value'),
    Input('comp-metric-dropdown', 'value'),
)
def update_comparison_task(tulip_id, task, metric):
    if not tulip_id or not task:
        return _empty_fig(), _empty_fig()
    fig_box = make_new_vs_group_box(group_stats, task, metric, tulip_id)
    fig_asym = make_asymmetry_scatter(group_stats, task, tulip_id)
    return fig_box, fig_asym


# ─── Tab 3: Sensor Analysis ───
@app.callback(
    Output('spectral-fingerprint', 'figure'),
    Output('rhythm-ladder', 'figure'),
    Output('evidence-ribbon', 'figure'),
    Output('raw-timeseries-left', 'figure'),
    Output('raw-timeseries-right', 'figure'),
    Input('patient-dropdown', 'value'),
    Input('sensor-task-dropdown', 'value'),
)
def update_sensor(tulip_id, task):
    empty = (_empty_fig(),) * 5
    if not tulip_id or not task:
        return empty

    fig_spectral = make_spectral_fingerprint(tulip_id, task)
    fig_rhythm = make_rhythm_ladder(tulip_id, task)
    fig_ribbon = make_evidence_ribbon(feature_cache, tulip_id)

    left_ts = load_timeseries(tulip_id, task, 'LeftWrist')
    right_ts = load_timeseries(tulip_id, task, 'RightWrist')
    task_label = TASK_LABELS_KR.get(task, task)
    fig_l = make_timeseries_plot(left_ts, f'{task_label} — Left Wrist')
    fig_r = make_timeseries_plot(right_ts, f'{task_label} — Right Wrist')

    return fig_spectral, fig_rhythm, fig_ribbon, fig_l, fig_r


# ─── Tab 3: Bilateral Asymmetry ───
@app.callback(
    Output('asymmetry-heatmap', 'figure'),
    Output('asym-group-compare', 'figure'),
    Input('patient-dropdown', 'value'),
)
def update_asymmetry_overview(tulip_id):
    if not tulip_id:
        return _empty_fig(), _empty_fig()
    fig_heatmap = make_asymmetry_heatmap(feature_cache, group_stats, tulip_id)
    fig_group = make_asym_group_compare(group_stats, tulip_id)
    return fig_heatmap, fig_group


@app.callback(
    Output('asym-waveform', 'figure'),
    Output('asym-feature-bars', 'figure'),
    Input('patient-dropdown', 'value'),
    Input('asym-task-dropdown', 'value'),
)
def update_asymmetry_detail(tulip_id, task):
    if not tulip_id or not task:
        return _empty_fig(), _empty_fig()
    fig_wave = make_asym_waveform(tulip_id, task)
    fig_bars = make_asym_feature_bars(feature_cache, tulip_id, task)
    return fig_wave, fig_bars


# ─── Tab 6: Clinical Summary ───
@app.callback(
    Output('summary-verdict', 'children'),
    Output('summary-findings', 'children'),
    Output('summary-radar', 'figure'),
    Output('summary-task-table', 'children'),
    Input('patient-dropdown', 'value'),
)
def update_summary(tulip_id):
    if not tulip_id:
        return '', '', _empty_fig(), ''

    from src.data_loader import compute_proximity_scores, MATCHING_FEATURES
    from src.feature_engineering import calc_asymmetry_index
    from src.figures import _interpret_proximity
    import numpy as np_local

    # Use matching-aligned proximity (16D weighted Euclidean)
    prox_all = compute_proximity_scores()
    pat_prox = prox_all.get(tulip_id, {})
    overall = pat_prox.get('score', 50.0)
    per_task = pat_prox.get('per_task', {})
    d_pd = pat_prox.get('d_pd', 0)
    d_healthy = pat_prox.get('d_healthy', 0)
    n_pd = pat_prox.get('n_pd', 0)
    n_healthy = pat_prox.get('n_healthy', 0)
    # Flatten per_task into per_feature for findings
    per_feature = {}
    for task_feats in per_task.values():
        for feat, score in task_feats.items():
            if feat not in per_feature:
                per_feature[feat] = []
            per_feature[feat].append(score)
    per_feature = {f: np_local.mean(v) for f, v in per_feature.items()}

    # Asymmetry score
    subj = feature_cache[feature_cache.tulip_id == tulip_id]
    asym_vals = []
    for task in SENSOR_TASKS:
        td = subj[subj.task == task]
        lv = td[td.wrist == 'L']['amplitude'].values
        rv = td[td.wrist == 'R']['amplitude'].values
        if len(lv) > 0 and len(rv) > 0:
            asym_vals.append(calc_asymmetry_index(lv[0], rv[0]))
    mean_asym = np_local.mean(asym_vals) if asym_vals else 0

    is_new = tulip_id in NEW_CASES
    interpretation, interp_color, _ = _interpret_proximity(overall, d_pd, d_healthy)

    # Verdict
    if is_new:
        verdict = html.Div([
            html.H2(f'Motor Phenotype: {interpretation}',
                    style={'color': interp_color, 'marginBottom': '8px'}),
            html.P([
                html.Span(f'Proximity: {overall:.1f}% | '),
                html.Span(f'Asymmetry: {mean_asym:.3f} | '),
                html.Span(f'Ref: PD n={n_pd}, Healthy n={n_healthy}'),
            ], style={'fontSize': '13px', 'color': '#4a5568'}),
            html.P(f'd(PD)={d_pd:.2f}, d(Healthy)={d_healthy:.2f}',
                   style={'fontSize': '12px', 'color': '#718096'}),
            html.P('Method: 16D bilateral feature vector (Entrainment+Relaxed × L/R × 4 features), '
                   'z-score normalized, weighted Euclidean distance to reference centroids. '
                   'This measures phenotype similarity, NOT diagnostic probability.',
                   style={'fontSize': '11px', 'color': '#a0aec0', 'marginTop': '4px'}),
        ], className='verdict-box')
    else:
        row = patients_df[patients_df.tulip_id == tulip_id]
        condition = row.iloc[0]['condition'] if not row.empty else '—'
        verdict = html.Div([
            html.H2(f'Confirmed: {condition}', style={'color': '#2c3e50'}),
            html.P(f'Reference analog — Proximity: {overall:.1f}% '
                   f'(d_PD={d_pd:.2f}, d_H={d_healthy:.2f})'),
        ], className='verdict-box')

    scores = per_feature

    # Key findings
    finding_items = []
    for feat, score in scores.items():
        from src.figures import FEATURE_LABELS
        label = FEATURE_LABELS.get(feat, feat)
        if score > 65:
            finding_items.append(
                html.Li(f'{label}: PD-like ({score:.0f}%)',
                        className='finding-pd'))
        elif score < 35:
            finding_items.append(
                html.Li(f'{label}: Healthy-like ({score:.0f}%)',
                        className='finding-healthy'))
        else:
            finding_items.append(
                html.Li(f'{label}: 경계 ({score:.0f}%)',
                        className='finding-border'))

    if mean_asym > 0.3:
        finding_items.append(
            html.Li(f'비대칭 지수: {mean_asym:.3f} (높음 — PD 시사)',
                    className='finding-pd'))
    else:
        finding_items.append(
            html.Li(f'비대칭 지수: {mean_asym:.3f} (정상 범위)',
                    className='finding-healthy'))

    findings = html.Div([
        html.H3('주요 소견', className='viz-title'),
        html.Ul(finding_items, className='findings-list'),
    ])

    # Radar
    fig_radar = make_summary_radar(feature_cache, group_stats, tulip_id)

    # Task table
    task_rows = []
    for task in SENSOR_TASKS:
        td = subj[subj.task == task]
        lv = td[td.wrist == 'L']['amplitude'].values
        rv = td[td.wrist == 'R']['amplitude'].values
        l_amp = f'{lv[0]:.4f}' if len(lv) > 0 else '—'
        r_amp = f'{rv[0]:.4f}' if len(rv) > 0 else '—'
        asym = calc_asymmetry_index(lv[0], rv[0]) if (len(lv) > 0 and len(rv) > 0) else 0
        asym_class = 'high-asym' if asym > 0.3 else ''
        task_rows.append(html.Tr([
            html.Td(TASK_LABELS_KR.get(task, task)),
            html.Td(l_amp),
            html.Td(r_amp),
            html.Td(f'{asym:.3f}', className=asym_class),
        ]))

    task_table = html.Div([
        html.H3('Task별 요약', className='viz-title'),
        html.Table([
            html.Thead(html.Tr([
                html.Th('Task'), html.Th('Left Amp'), html.Th('Right Amp'), html.Th('Asymmetry'),
            ])),
            html.Tbody(task_rows),
        ], className='summary-table'),
    ])

    return verdict, findings, fig_radar, task_table


# ─── Tab 5: Video — dynamic video type options per patient ───
@app.callback(
    Output('video-type-dropdown', 'options'),
    Output('video-type-dropdown', 'value'),
    Input('patient-dropdown', 'value'),
)
def update_video_type_options(tulip_id):
    """Update available video types based on patient's R2 catalog."""
    if not tulip_id or tulip_id not in R2_VIDEO_CATALOG:
        return [], None
    catalog = R2_VIDEO_CATALOG[tulip_id]
    options = [{'label': VIDEO_TYPE_LABELS.get(k, k), 'value': k}
               for k in catalog]
    default = options[0]['value'] if options else None
    return options, default


@app.callback(
    Output('video-player-container', 'children'),
    Output('motion-timeline', 'figure'),
    Output('lr-tapping-comparison', 'figure'),
    Output('video-interval-distribution', 'figure'),
    Output('video-tremor-spectrogram', 'figure'),
    Output('video-symmetry-trend', 'figure'),
    Output('video-left-taps-label', 'children'),
    Output('video-right-taps-label', 'children'),
    Output('video-l-cv-label', 'children'),
    Output('video-r-cv-label', 'children'),
    Output('video-left-taps', 'children'),
    Output('video-right-taps', 'children'),
    Output('video-l-cv', 'children'),
    Output('video-r-cv', 'children'),
    Input('patient-dropdown', 'value'),
    Input('video-type-dropdown', 'value'),
    Input('video-camera-dropdown', 'value'),
)
def update_video(tulip_id, video_type, camera):
    if video_type == 'resting':
        left_label, right_label = 'L Tremor Peaks', 'R Tremor Peaks'
        lcv_label, rcv_label = 'L Peak Interval CV', 'R Peak Interval CV'
    else:
        left_label, right_label = 'L Taps', 'R Taps'
        lcv_label, rcv_label = 'L Interval CV', 'R Interval CV'

    if not tulip_id or not video_type or not camera:
        return (html.P('Select video'), _empty_fig(), _empty_fig(), _empty_fig(), _empty_fig(), _empty_fig(),
                left_label, right_label, lcv_label, rcv_label,
                '—', '—', '—', '—')

    # Build CDN URL
    video_url = _r2_video_url(tulip_id, video_type, camera)
    if not video_url:
        return (html.P('이 환자의 영상이 아직 업로드되지 않았습니다.',
                        style={'color': '#718096', 'padding': '20px'}),
                _empty_fig(), _empty_fig(), _empty_fig(), _empty_fig(), _empty_fig(),
                left_label, right_label, lcv_label, rcv_label,
                '—', '—', '—', '—')

    player = html.Div([
        html.Video(
            src=video_url, controls=True, autoPlay=False,
            style={'width': '100%', 'borderRadius': '8px', 'background': '#1a202c'},
        ),
        html.A('🔍 영상 크게 보기', href=video_url, target='_blank',
               style={'fontSize': '12px', 'color': '#805ad5', 'marginTop': '6px',
                      'display': 'inline-block', 'fontWeight': '600'}),
    ])

    # Task-aware analysis from videos_feature CSV (minimal layout change)
    task_folder = VIDEO_TASK_FOLDER_MAP.get(video_type)
    primary_df = load_video_feature_df(tulip_id, task_folder, camera) if task_folder else None

    counterpart_df = None  # kept for signature compatibility

    if primary_df is None or primary_df.empty:
        # [PRE] Legacy fallback block (video_analysis.json) is intentionally kept.
        # [PRE] We keep this code path as comment only to satisfy no-deletion policy.
        # side = 'L' if video_type == 'toe_left' else 'R'
        # legacy = video_data.get(tulip_id, {}).get(side)
        # if legacy:
        #     legacy_player = html.P(
        #         'videos_feature CSV가 없어서 legacy JSON 분석 결과를 표시합니다.',
        #         style={'color': '#718096', 'padding': '12px'},
        #     )
        #     legacy_timeline = make_motion_timeline(video_data[tulip_id], side)
        #     legacy_lr = make_lr_tapping_comparison(
        #         video_data[tulip_id].get('L', {}),
        #         video_data[tulip_id].get('R', {}),
        #     )
        #     return legacy_player, legacy_timeline, legacy_lr, '—', '—', '—', '—'
        # [END PRE]
        msg = html.P(
            '선택한 task/camera에 대한 videos_feature CSV가 없습니다. '
            'extract_metrabs_joint_features.py 배치를 먼저 실행해주세요.',
            style={'color': '#718096', 'padding': '20px'},
        )
        return (
            msg,
            _empty_fig('feature 데이터 없음'),
            _empty_fig('feature 데이터 없음'),
            _empty_fig('feature 데이터 없음'),
            _empty_fig('feature 데이터 없음'),
            _empty_fig('feature 데이터 없음'),
            left_label, right_label, lcv_label, rcv_label,
            '—', '—', '—', '—',
        )

    fig_timeline = make_video_feature_timeline(primary_df, video_type, camera)
    fig_lr = make_video_lr_feature_comparison(primary_df, counterpart_df, video_type)
    fig_intervals = make_video_interval_distribution(primary_df, video_type)
    fig_spec = make_video_tremor_spectrogram(primary_df, video_type)
    fig_symmetry = make_video_symmetry_trend(primary_df, video_type)
    l_taps, r_taps, l_cv, r_cv = summarize_video_task_metrics(primary_df, counterpart_df, video_type)

    return (
        player, fig_timeline, fig_lr, fig_intervals, fig_spec, fig_symmetry,
        left_label, right_label, lcv_label, rcv_label,
        l_taps, r_taps, l_cv, r_cv,
    )


# ─── Decision Save / Reset ───
@app.callback(
    Output('decision-store', 'data'),
    Output('save-feedback', 'children'),
    Input('save-decision-btn', 'n_clicks'),
    Input('reset-decision-btn', 'n_clicks'),
    State('patient-dropdown', 'value'),
    State('decision-classification', 'value'),
    State('decision-confidence', 'value'),
    State('evidence-tags', 'value'),
    State('decision-notes', 'value'),
    State('decision-store', 'data'),
)
def save_or_reset_decision(save_clicks, reset_clicks, tulip_id,
                           classification, confidence, tags, notes, store):
    store = store or {}
    if not tulip_id:
        return store, ''

    triggered = callback_context.triggered_id
    if triggered == 'reset-decision-btn' and reset_clicks:
        store.pop(tulip_id, None)
        return store, html.Span('↩ 진단 초기화됨', className='feedback-reset')

    if triggered == 'save-decision-btn' and save_clicks:
        if not classification:
            return store, html.Span('분류를 선택해주세요', className='feedback-warn')
        store[tulip_id] = {
            'classification': classification,
            'confidence': confidence,
            'evidence_tags': tags or [],
            'notes': notes or '',
        }
        label = _CLASSIFICATION_LABELS.get(classification, (classification, ''))[0]
        return store, html.Span(f'✓ Saved: {label} ({confidence}%)', className='feedback-success')

    return store, ''


# ─── Export JSON ───
@app.callback(
    Output('download-json', 'data'),
    Input('export-btn', 'n_clicks'),
    State('decision-store', 'data'),
    prevent_initial_call=True,
)
def export_json(n_clicks, store):
    if not n_clicks or not store:
        return None
    return dict(
        content=json.dumps(store, indent=2, ensure_ascii=False),
        filename='pd_decisions.json',
    )


# ══════════════════════════════════════════════════════════════
#  RUN
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Keep debug on, but disable hot-reload watcher to avoid missing legacy asset path errors.
    app.run(debug=True, dev_tools_hot_reload=False, port=8050)

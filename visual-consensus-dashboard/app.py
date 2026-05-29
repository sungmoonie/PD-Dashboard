"""
PD Clinical Decision Support Dashboard — Final Version
========================================================
Aligned Tasks Focus: Toe Tapping (Entrainment) & Resting (Relaxed)
5 Tabs: Overview, Tremor & Rhythm, Video, Reference Comparison, Summary
"""

import os
import json
import dash
from dash import Input, Output, State, html, dcc, callback_context, ALL
from flask import send_file, request, Response, redirect

# ─── Data Loading ───
from src.data_loader import (
    load_patients, load_nms, load_updrs_labels, load_updrs_metadata,
    build_group_stats, build_feature_cache,
    load_timeseries, get_subject_list_hybrid,
    SENSOR_TASKS, MATCHING_TASKS, TASK_LABELS_KR, NEW_CASES,
    get_group_label, _estimate_fs,
    load_video_feature_df, VIDEO_TASK_FOLDER_MAP, load_video_analysis,
)
from src.feature_engineering import (
    calc_signal_rms, calc_asymmetry_index, calc_signal_stats,
)
from src.figures import (
    make_timeseries_plot,
    make_video_lr_feature_comparison, summarize_video_task_metrics,
    make_video_interval_distribution, make_video_tremor_spectrogram, make_video_symmetry_trend,
    make_lr_tapping_comparison,
    make_bilateral_matrix, make_spectral_fingerprint,
    make_rhythm_ladder,
    make_new_vs_group_box,
    make_proximity_gauge, make_proximity_map,
    make_summary_radar,
    make_tremor_power_bars, make_tremor_band_breakdown, make_amplitude_decrement,
    make_asym_feature_bars, make_asym_group_compare,
    _empty_fig, _interpret_proximity,
)
from src.layout import (
    create_layout, build_tab_overview, build_tab_comparison,
    build_tab_tremor, build_tab_video,
    build_tab_summary, UPDRS_ITEMS, UPDRS_TASK_GROUPS,
)

# ─── Pre-load Data ───
patients_df = load_patients()
nms_data = load_nms()
labels_df = load_updrs_labels()
updrs_meta = load_updrs_metadata()
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
# ══════════════════════════════════════════════════════════════

# ─── Tab Rendering ───
@app.callback(
    Output('tab-content', 'children'),
    Input('main-tabs', 'value'),
)
def render_tab(tab):
    if tab == 'tab-overview':
        return build_tab_overview()
    elif tab == 'tab-tremor':
        return build_tab_tremor()
    elif tab == 'tab-comparison':
        return build_tab_comparison()
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

    saved_dx = saved.get('wholistic_decision') if saved else None
    has_decision = is_new and saved_dx is not None

    # Badge
    if has_decision:
        badge_text, badge_class = _CLASSIFICATION_LABELS.get(saved_dx, (saved_dx, 'badge badge-other'))
        badge_text = f'Dx: {badge_text}'
    elif is_new:
        badge_text = 'NEW'
        badge_class = 'badge badge-new'
    else:
        condition = row.iloc[0]['condition'] if not row.empty else 'Unknown'
        group = get_group_label(tulip_id, condition)
        badge_text = group
        badge_class = f'badge badge-{group.lower()}'

    # Info cards
    condition = row.iloc[0]['condition'] if not row.empty else '—'
    if has_decision:
        label, _ = _CLASSIFICATION_LABELS.get(saved_dx, (saved_dx, ''))
        n_scored = len(saved.get('updrs_scores', {}))
        condition = html.Div([
            html.Div(label, className='card-value-main', style={'color': '#2b6cb0'}),
            html.Div(f'({n_scored} items scored)', className='card-value-sub'),
        ])
    elif is_new:
        condition = html.Div([
            html.Div('미확정', className='card-value-main'),
            html.Div('(New Case)', className='card-value-sub'),
        ])

    hy = str(meta.get('hy_mean', '—'))
    diag = meta.get('diagnosis', '—')
    if has_decision:
        label, _ = _CLASSIFICATION_LABELS.get(saved_dx, (saved_dx, ''))
        diag = html.Div([
            html.Div(f'Your Dx: {label}', className='card-value-main',
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
        label, _ = _CLASSIFICATION_LABELS.get(saved_dx, (saved_dx, ''))
        n_scored = len(saved.get('updrs_scores', {}))
        indicator = html.Div([
            html.Span('Saved: ', style={'fontWeight': '600'}),
            html.Span(f'{label} ({n_scored} items)'),
        ], className='new-case-alert',
           style={'background': '#f0fff4', 'borderColor': '#9ae6b4', 'color': '#276749'})
    elif is_new:
        indicator = html.Div([
            html.Span('NEW CASE: '),
            html.Span('UPDRS 항목을 평가한 후 Save Decision을 눌러주세요.'),
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
    new_options = []
    confirmed_options = []
    for _, row in patients_df.iterrows():
        tid = row['tulip_id']
        num = tid.replace('TULIP_', '')
        group = get_group_label(tid, row['condition'])
        saved = decision_store.get(tid)
        saved_dx = saved.get('wholistic_decision') if saved else None
        if group == 'New' and saved_dx:
            label_txt = _CLASSIFICATION_LABELS.get(saved_dx, (saved_dx, ''))[0]
            label = f"Dx:{label_txt} Patient_{num} ({row['age']}y, {row['gender']})"
            new_options.append({'label': label, 'value': tid})
        elif group == 'New':
            label = f"NEW Patient_{num} ({row['age']}y, {row['gender']})"
            new_options.append({'label': label, 'value': tid})
        else:
            label = f"  Patient_{num} — {group} ({row['age']}y, {row['gender']})"
            confirmed_options.append({'label': label, 'value': tid})
    new_options.sort(key=lambda x: x['value'])
    confirmed_options.sort(key=lambda x: x['value'])
    return new_options + confirmed_options


# ─── Tab 1: Patient Overview ───
@app.callback(
    Output('demographics-row', 'children'),
    Output('ov-age', 'children'),
    Output('ov-gender', 'children'),
    Output('ov-handedness', 'children'),
    Output('ov-bmi', 'children'),
    Output('ov-duration', 'children'),
    Output('ov-phenotype', 'children'),
    Output('ov-proximity', 'children'),
    Output('ov-asymmetry', 'children'),
    Output('nms-content', 'children'),
    Output('bilateral-matrix', 'figure'),
    Input('patient-dropdown', 'value'),
    Input('decision-store', 'data'),
)
def update_overview(tulip_id, decision_store):
    empty = ('', '—', '—', '—', '—', '—', '—', '—', '—', '', _empty_fig())
    if not tulip_id:
        return empty

    row = patients_df[patients_df.tulip_id == tulip_id]
    if row.empty:
        return empty
    r = row.iloc[0]

    is_new = tulip_id in NEW_CASES
    decision_store = decision_store or {}
    saved = decision_store.get(tulip_id)
    saved_dx = saved.get('wholistic_decision') if saved else None
    has_decision = is_new and saved_dx is not None

    if has_decision:
        label_txt = _CLASSIFICATION_LABELS.get(saved_dx, (saved_dx, ''))[0]
        n_scored = len(saved.get('updrs_scores', {}))
        cond_display = f'Dx: {label_txt} ({n_scored} items scored)'
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

    # Matrix — aligned tasks only (default)
    fig_matrix = make_bilateral_matrix(feature_cache, tulip_id)

    if is_new:
        duration = '—'
    else:
        age_diag = r.get('age_at_diagnosis')
        duration = f"{r['age'] - age_diag}y" if age_diag and r['age'] else '—'

    # Motor Phenotype Summary
    import numpy as np_local
    from src.data_loader import compute_proximity_scores

    prox_all = compute_proximity_scores()
    pat_prox = prox_all.get(tulip_id, {})
    overall = pat_prox.get('score', 50.0)
    d_pd = pat_prox.get('d_pd', 0)
    d_healthy = pat_prox.get('d_healthy', 0)
    interpretation, interp_color, _ = _interpret_proximity(overall, d_pd, d_healthy)

    phenotype_display = html.Span(interpretation, style={'color': interp_color, 'fontWeight': '600'})
    proximity_display = html.Span(f'{overall:.1f}%', style={'color': interp_color, 'fontWeight': '600'})

    # Mean asymmetry (aligned tasks)
    subj = feature_cache[feature_cache.tulip_id == tulip_id]
    asym_vals = []
    for task in MATCHING_TASKS:
        td = subj[subj.task == task]
        lv = td[td.wrist == 'L']['amplitude'].values
        rv = td[td.wrist == 'R']['amplitude'].values
        if len(lv) > 0 and len(rv) > 0:
            asym_vals.append(calc_asymmetry_index(lv[0], rv[0]))
    mean_asym = float(np_local.mean(asym_vals)) if asym_vals else 0
    asym_color = '#e53e3e' if mean_asym > 0.3 else '#38a169'
    asym_display = html.Span(f'{mean_asym:.3f}', style={'color': asym_color, 'fontWeight': '600'})

    return (demographics, str(r['age'] or '—'), r['gender'] or '—',
            r['handedness'] or '—', str(r['bmi'] or '—'), duration,
            phenotype_display, proximity_display, asym_display,
            nms_content, fig_matrix)


# ─── Tab 2: Tremor & Rhythm — Overview (patient-level) ───
@app.callback(
    Output('tremor-power-bars', 'figure'),
    Input('patient-dropdown', 'value'),
)
def update_tremor_overview(tulip_id):
    if not tulip_id:
        return _empty_fig()
    return make_tremor_power_bars(feature_cache, group_stats, tulip_id)


# ─── Tab 2: Tremor & Rhythm — Detail (task-specific) ───
@app.callback(
    Output('tremor-band-chart', 'figure'),
    Output('tremor-spectral', 'figure'),
    Output('tremor-rhythm', 'figure'),
    Output('tremor-decrement', 'figure'),
    Output('asym-feature-bars', 'figure'),
    Output('tremor-raw-left', 'figure'),
    Output('tremor-raw-right', 'figure'),
    Input('patient-dropdown', 'value'),
    Input('tremor-task-dropdown', 'value'),
)
def update_tremor_detail(tulip_id, task):
    empty = (_empty_fig(),) * 7
    if not tulip_id or not task:
        return empty

    fig_band = make_tremor_band_breakdown(tulip_id, task)
    fig_spectral = make_spectral_fingerprint(tulip_id, task)
    fig_rhythm = make_rhythm_ladder(tulip_id, task)
    fig_decrement = make_amplitude_decrement(tulip_id, task)
    fig_asym_bars = make_asym_feature_bars(feature_cache, tulip_id, task)

    left_ts = load_timeseries(tulip_id, task, 'LeftWrist')
    right_ts = load_timeseries(tulip_id, task, 'RightWrist')
    task_label = TASK_LABELS_KR.get(task, task)
    fig_l = make_timeseries_plot(left_ts, f'{task_label} — Left Wrist')
    fig_r = make_timeseries_plot(right_ts, f'{task_label} — Right Wrist')

    return (fig_band, fig_spectral, fig_rhythm, fig_decrement,
            fig_asym_bars, fig_l, fig_r)


# ─── Tab 1: Asymmetry — Group comparison (patient-level, Overview tab) ───
@app.callback(
    Output('asym-group-compare', 'figure'),
    Input('patient-dropdown', 'value'),
)
def update_asym_group(tulip_id):
    if not tulip_id:
        return _empty_fig()
    return make_asym_group_compare(group_stats, tulip_id)


# ─── Tab 4: Reference Comparison — Patient-level ───
@app.callback(
    Output('proximity-gauge', 'figure'),
    Input('patient-dropdown', 'value'),
)
def update_comparison_patient(tulip_id):
    if not tulip_id:
        return _empty_fig()
    return make_proximity_gauge(group_stats, feature_cache, tulip_id)


# ─── Tab 4: Reference Comparison — PCA 2D Map ───
@app.callback(
    Output('proximity-map', 'figure'),
    Input('patient-dropdown', 'value'),
)
def update_proximity_map(tulip_id):
    if not tulip_id:
        return _empty_fig()
    return make_proximity_map(tulip_id)


# ─── Tab 4: Reference Comparison — auto-select metric by task ───
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


# ─── Tab 4: Reference Comparison — Task-specific ───
@app.callback(
    Output('new-vs-group-box', 'figure'),
    Input('patient-dropdown', 'value'),
    Input('comp-task-dropdown', 'value'),
    Input('comp-metric-dropdown', 'value'),
)
def update_comparison_task(tulip_id, task, metric):
    if not tulip_id or not task:
        return _empty_fig()
    return make_new_vs_group_box(group_stats, task, metric, tulip_id)


# ─── Tab 5: Clinical Summary (aligned tasks only) ───
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
    from src.figures import FEATURE_LABELS
    import numpy as np_local

    # Proximity (already alignment-based: 16D from Entrainment+Relaxed)
    prox_all = compute_proximity_scores()
    pat_prox = prox_all.get(tulip_id, {})
    overall = pat_prox.get('score', 50.0)
    per_task = pat_prox.get('per_task', {})
    d_pd = pat_prox.get('d_pd', 0)
    d_healthy = pat_prox.get('d_healthy', 0)
    n_pd = pat_prox.get('n_pd', 0)
    n_healthy = pat_prox.get('n_healthy', 0)

    per_feature = {}
    for task_feats in per_task.values():
        for feat, score in task_feats.items():
            if feat not in per_feature:
                per_feature[feat] = []
            per_feature[feat].append(score)
    per_feature = {f: np_local.mean(v) for f, v in per_feature.items()}

    # Asymmetry score (aligned tasks only)
    subj = feature_cache[feature_cache.tulip_id == tulip_id]
    asym_vals = []
    for task in MATCHING_TASKS:
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
                   'z-score normalized, weighted Euclidean distance to reference centroids.',
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

    # Key findings
    scores = per_feature
    finding_items = []
    for feat, score in scores.items():
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
        html.H3('주요 소견 (Aligned Tasks)', className='viz-title'),
        html.Ul(finding_items, className='findings-list'),
    ])

    # Radar
    fig_radar = make_summary_radar(feature_cache, group_stats, tulip_id)

    # Task table — aligned tasks only
    task_rows = []
    for task in MATCHING_TASKS:
        td = subj[subj.task == task]
        lv = td[td.wrist == 'L']['amplitude'].values
        rv = td[td.wrist == 'R']['amplitude'].values
        l_amp = f'{lv[0]:.4f}' if len(lv) > 0 else '—'
        r_amp = f'{rv[0]:.4f}' if len(rv) > 0 else '—'
        asym = calc_asymmetry_index(lv[0], rv[0]) if (len(lv) > 0 and len(rv) > 0) else 0
        asym_class = 'high-asym' if asym > 0.3 else ''

        # Additional: tremor power
        l_tp = td[td.wrist == 'L']['tremor_power'].values
        r_tp = td[td.wrist == 'R']['tremor_power'].values
        l_tremor = f'{l_tp[0]:.5f}' if len(l_tp) > 0 else '—'
        r_tremor = f'{r_tp[0]:.5f}' if len(r_tp) > 0 else '—'

        # Rhythm irregularity
        l_ri = td[td.wrist == 'L']['rhythm_irreg'].values
        r_ri = td[td.wrist == 'R']['rhythm_irreg'].values
        l_rhythm = f'{l_ri[0]:.3f}' if len(l_ri) > 0 else '—'
        r_rhythm = f'{r_ri[0]:.3f}' if len(r_ri) > 0 else '—'

        task_rows.append(html.Tr([
            html.Td(TASK_LABELS_KR.get(task, task)),
            html.Td(l_amp), html.Td(r_amp),
            html.Td(l_tremor), html.Td(r_tremor),
            html.Td(l_rhythm), html.Td(r_rhythm),
            html.Td(f'{asym:.3f}', className=asym_class),
        ]))

    task_table = html.Div([
        html.H3('Aligned Task 상세 요약', className='viz-title'),
        html.Table([
            html.Thead(html.Tr([
                html.Th('Task'),
                html.Th('L Amp'), html.Th('R Amp'),
                html.Th('L Tremor'), html.Th('R Tremor'),
                html.Th('L Rhythm CV'), html.Th('R Rhythm CV'),
                html.Th('Asymmetry'),
            ])),
            html.Tbody(task_rows),
        ], className='summary-table'),
    ])

    return verdict, findings, fig_radar, task_table


# ─── Tab 3: Video — dynamic video type options per patient ───
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
        return (html.P('Select video'), _empty_fig(), _empty_fig(), _empty_fig(), _empty_fig(),
                left_label, right_label, lcv_label, rcv_label,
                '—', '—', '—', '—')

    # Build CDN URL
    video_url = _r2_video_url(tulip_id, video_type, camera)
    if not video_url:
        return (html.P('이 환자의 영상이 아직 업로드되지 않았습니다.',
                        style={'color': '#718096', 'padding': '20px'}),
                _empty_fig(), _empty_fig(), _empty_fig(), _empty_fig(),
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

    # Task-aware analysis from videos_feature CSV
    task_folder = VIDEO_TASK_FOLDER_MAP.get(video_type)
    primary_df = load_video_feature_df(tulip_id, task_folder, camera) if task_folder else None
    counterpart_df = None

    if primary_df is None or primary_df.empty:
        msg = html.P(
            '선택한 task/camera에 대한 videos_feature CSV가 없습니다.',
            style={'color': '#718096', 'padding': '20px'},
        )
        return (
            msg,
            _empty_fig('feature 데이터 없음'),
            _empty_fig('feature 데이터 없음'),
            _empty_fig('feature 데이터 없음'),
            _empty_fig('feature 데이터 없음'),
            left_label, right_label, lcv_label, rcv_label,
            '—', '—', '—', '—',
        )

    fig_lr = make_video_lr_feature_comparison(primary_df, counterpart_df, video_type)
    fig_intervals = make_video_interval_distribution(primary_df, video_type)
    fig_spec = make_video_tremor_spectrogram(primary_df, video_type)
    fig_symmetry = make_video_symmetry_trend(primary_df, video_type)
    l_taps, r_taps, l_cv, r_cv = summarize_video_task_metrics(primary_df, counterpart_df, video_type)

    return (
        player, fig_lr, fig_intervals, fig_spec, fig_symmetry,
        left_label, right_label, lcv_label, rcv_label,
        l_taps, r_taps, l_cv, r_cv,
    )


# ─── Decision Save / Reset (UPDRS-based) ───
@app.callback(
    Output('decision-store', 'data'),
    Output('save-feedback', 'children'),
    Output('clinician-comparison', 'children'),
    Input('save-decision-btn', 'n_clicks'),
    Input('reset-decision-btn', 'n_clicks'),
    State({'type': 'updrs-score', 'index': ALL}, 'value'),
    State('patient-dropdown', 'value'),
    State('decision-store', 'data'),
)
def save_or_reset_decision(save_clicks, reset_clicks, updrs_values,
                           tulip_id, store):
    store = store or {}
    if not tulip_id:
        return store, '', ''

    triggered = callback_context.triggered_id
    if triggered == 'reset-decision-btn' and reset_clicks:
        store.pop(tulip_id, None)
        return store, html.Span('Reset', className='feedback-reset'), ''

    if triggered == 'save-decision-btn' and save_clicks:
        # Collect user's scores (skip None)
        user_scores = {}
        for i, val in enumerate(updrs_values):
            if val is not None:
                user_scores[UPDRS_ITEMS[i]] = val

        if not user_scores:
            return store, html.Span('최소 1개 항목을 평가해주세요',
                                    className='feedback-warn'), ''

        wholistic = user_scores.get('wholistic_decision')
        store[tulip_id] = {
            'updrs_scores': user_scores,
            'wholistic_decision': wholistic,
        }

        comparison = _build_clinician_comparison(tulip_id, user_scores)
        feedback = html.Span(f'Saved ({len(user_scores)} items)',
                             className='feedback-success')
        return store, feedback, comparison

    return store, '', ''


def _build_clinician_comparison(tulip_id, user_scores):
    """Build comparison table: user vs 3 clinicians + majority vote."""
    from collections import Counter

    pat_labels = labels_df[labels_df.tulip_id == tulip_id]
    if pat_labels.empty:
        return html.P('이 환자의 clinician label 데이터가 없습니다.',
                      style={'color': '#718096', 'fontSize': '12px'})

    rows = []
    agree_count = 0
    total_compared = 0

    # Aligned tasks only
    aligned_indices = [idx for _, indices in UPDRS_TASK_GROUPS for idx in indices]
    aligned_names = [UPDRS_ITEMS[i] for i in aligned_indices]

    for name in aligned_names:
        if name not in user_scores:
            continue

        label_row = pat_labels[pat_labels.updrs_name == name]
        if label_row.empty:
            continue

        lr = label_row.iloc[0]
        c1, c2, c3 = lr['c1'], lr['c2'], lr['c3']
        user_val = user_scores[name]

        # Majority vote
        if lr['is_numeric']:
            vals = [int(float(v)) for v in [c1, c2, c3]]
            majority = Counter(vals).most_common(1)[0][0]
        else:
            vals = [str(v) for v in [c1, c2, c3]]
            majority = Counter(vals).most_common(1)[0][0]

        # Check agreement
        total_compared += 1
        if str(user_val) == str(majority):
            agree_count += 1
            row_class = 'comparison-agree'
        else:
            row_class = 'comparison-disagree'

        def fmt(v):
            try:
                fv = float(v)
                return str(int(fv)) if fv == int(fv) else str(v)
            except (ValueError, TypeError):
                return str(v)

        rows.append(html.Tr([
            html.Td(name, className='comp-item-name'),
            html.Td(str(user_val), className='comp-cell comp-you'),
            html.Td(fmt(c1), className='comp-cell'),
            html.Td(fmt(c2), className='comp-cell'),
            html.Td(fmt(c3), className='comp-cell'),
            html.Td(str(majority), className='comp-cell comp-majority'),
        ], className=row_class))

    if not rows:
        return html.P('비교할 데이터가 없습니다.', style={'color': '#718096'})

    agree_pct = (agree_count / total_compared * 100) if total_compared > 0 else 0

    return html.Div([
        html.Hr(className='sidebar-divider'),
        html.H4('Clinician Comparison', className='section-title',
                style={'marginTop': '12px'}),
        html.P(f'Agreement with majority: {agree_count}/{total_compared} '
               f'({agree_pct:.0f}%)',
               style={'fontSize': '12px', 'color': '#4a5568', 'marginBottom': '8px',
                      'fontWeight': '600'}),
        html.Div([
            html.Table([
                html.Thead(html.Tr([
                    html.Th('Item'), html.Th('You'),
                    html.Th('C1'), html.Th('C2'), html.Th('C3'),
                    html.Th('Majority'),
                ], className='comp-header')),
                html.Tbody(rows),
            ], className='comparison-table'),
        ], className='comparison-scroll'),
    ])


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
    app.run(debug=True, dev_tools_hot_reload=False, port=8050)

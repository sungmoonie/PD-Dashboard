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
    load_video_analysis, build_group_stats, build_feature_cache,
    load_timeseries, get_subject_list_hybrid,
    SENSOR_TASKS, TASK_LABELS_KR, NEW_CASES, get_group_label, _estimate_fs,
)
from src.feature_engineering import (
    calc_signal_rms, calc_asymmetry_index, calc_signal_stats,
)
from src.figures import (
    make_timeseries_plot, make_updrs_heatmap, make_clinician_comparison_bar,
    make_motion_timeline, make_lr_tapping_comparison,
    make_bilateral_matrix, make_spectral_fingerprint,
    make_rhythm_ladder, make_evidence_ribbon,
    make_new_vs_group_box, make_group_feature_comparison,
    make_task_profile_comparison, make_asymmetry_scatter,
    _empty_fig,
)
from src.layout import (
    create_layout, build_tab_overview, build_tab_comparison,
    build_tab_sensor, build_tab_updrs, build_tab_video,
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
#  VIDEO STREAMING
# ══════════════════════════════════════════════════════════════

VIDEO_FOLDERS = {
    'left': '7. Finger_tapping_left',
    'right': '8. FInger_tapping_right',
}


@server.route('/video/<folder>/<camera>')
def serve_video(folder, camera):
    """Serve video with Range request support."""
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if folder == 'left':
        video_dir = os.path.join(project_dir, VIDEO_FOLDERS['left'])
    else:
        video_dir = os.path.join(project_dir, VIDEO_FOLDERS['right'])

    video_path = os.path.join(video_dir, camera)
    if not os.path.exists(video_path):
        github_base = 'https://raw.githubusercontent.com'
        return redirect(f'{github_base}/placeholder/video/{folder}/{camera}', code=302)

    file_size = os.path.getsize(video_path)
    range_header = request.headers.get('Range')

    if range_header:
        byte_start = int(range_header.replace('bytes=', '').split('-')[0])
        byte_end = min(byte_start + 1024 * 1024, file_size - 1)
        with open(video_path, 'rb') as f:
            f.seek(byte_start)
            data = f.read(byte_end - byte_start + 1)
        resp = Response(data, status=206, mimetype='video/mp4')
        resp.headers['Content-Range'] = f'bytes {byte_start}-{byte_end}/{file_size}'
        resp.headers['Accept-Ranges'] = 'bytes'
        resp.headers['Content-Length'] = str(len(data))
        return resp
    return send_file(video_path, mimetype='video/mp4')


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
    elif tab == 'tab-comparison':
        return build_tab_comparison()
    elif tab == 'tab-sensor':
        return build_tab_sensor()
    elif tab == 'tab-updrs':
        return build_tab_updrs()
    elif tab == 'tab-video':
        return build_tab_video()
    return html.Div('Select a tab')


# ─── Case Badge + Right Panel Info ───
@app.callback(
    Output('case-badge', 'children'),
    Output('case-badge', 'className'),
    Output('info-condition', 'children'),
    Output('info-hy', 'children'),
    Output('info-diagnosis', 'children'),
    Output('info-nms', 'children'),
    Output('new-patient-indicator', 'children'),
    Input('patient-dropdown', 'value'),
)
def update_case_info(tulip_id):
    if not tulip_id:
        return '—', 'badge badge-unknown', '—', '—', '—', '—', ''

    row = patients_df[patients_df.tulip_id == tulip_id]
    meta = updrs_meta.get(tulip_id, {})
    is_new = tulip_id in NEW_CASES

    # Badge
    if is_new:
        badge_text = '★ NEW'
        badge_class = 'badge badge-new'
    else:
        condition = row.iloc[0]['condition'] if not row.empty else 'Unknown'
        group = get_group_label(tulip_id, condition)
        badge_text = group
        badge_class = f'badge badge-{group.lower()}'

    # Info cards
    condition = row.iloc[0]['condition'] if not row.empty else '—'
    if is_new:
        condition = '미확정 (New Case)'

    hy = str(meta.get('hy_mean', '—'))
    diag = meta.get('diagnosis', '—')
    if is_new:
        diag = '판단 필요'

    nms_count = str(nms_data.get(tulip_id, {}).get('count', '—'))

    # New patient indicator
    indicator = ''
    if is_new:
        indicator = html.Div([
            html.Span('★ ', className='new-star'),
            html.Span('이 환자는 미확정 New Case입니다. '),
            html.Span('아래에서 PD/HT/? 판단을 내려주세요.'),
        ], className='new-case-alert')

    return badge_text, badge_class, condition, hy, diag, nms_count, indicator


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
)
def update_overview(tulip_id):
    empty = ('', '—', '—', '—', '—', '—', '', _empty_fig())
    if not tulip_id:
        return empty

    row = patients_df[patients_df.tulip_id == tulip_id]
    if row.empty:
        return empty
    r = row.iloc[0]

    is_new = tulip_id in NEW_CASES
    cond_display = '★ New Case (미확정)' if is_new else r['condition']

    demographics = html.Div([
        html.Span(f"TULIP: {tulip_id}", className='demo-item demo-id'),
        html.Span(f"PADS: {r['pads_id']}", className='demo-item'),
        html.Span(f"Condition: {cond_display}",
                  className='demo-item demo-condition' + (' demo-new' if is_new else '')),
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


# ─── Tab 2: Group Comparison ───
@app.callback(
    Output('new-vs-group-box', 'figure'),
    Output('task-profile-comparison', 'figure'),
    Output('feature-group-comparison', 'figure'),
    Output('asymmetry-scatter', 'figure'),
    Input('patient-dropdown', 'value'),
    Input('comp-task-dropdown', 'value'),
    Input('comp-metric-dropdown', 'value'),
)
def update_comparison(tulip_id, task, metric):
    if not tulip_id or not task:
        return (_empty_fig(), _empty_fig(), _empty_fig(), _empty_fig())

    fig_box = make_new_vs_group_box(group_stats, task, metric, tulip_id)
    fig_profile = make_task_profile_comparison(group_stats, tulip_id)
    fig_features = make_group_feature_comparison(feature_cache, group_stats, tulip_id)
    fig_asym = make_asymmetry_scatter(group_stats, task, tulip_id)

    return fig_box, fig_profile, fig_features, fig_asym


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


# ─── Tab 4: UPDRS ───
@app.callback(
    Output('updrs-heatmap', 'figure'),
    Input('patient-dropdown', 'value'),
)
def update_updrs_heatmap(tulip_id):
    return make_updrs_heatmap(labels_df)


@app.callback(
    Output('clinician-comparison-bar', 'figure'),
    Output('updrs-detail-text', 'children'),
    Input('updrs-heatmap', 'clickData'),
    State('patient-dropdown', 'value'),
)
def update_updrs_detail(click_data, tulip_id):
    if not click_data or not tulip_id:
        return _empty_fig('Cell을 클릭하세요'), ''

    point = click_data['points'][0]
    updrs_name = point.get('x', '')
    clicked_tulip = point.get('y', tulip_id)

    fig = make_clinician_comparison_bar(labels_df, clicked_tulip, updrs_name)
    detail = html.Div([
        html.Strong(f'{clicked_tulip} — {updrs_name}'),
        html.Br(),
        html.Span('Clinician 3인의 개별 score와 평균 비교'),
    ])
    return fig, detail


# ─── Tab 5: Video ───
@app.callback(
    Output('video-player-container', 'children'),
    Output('motion-timeline', 'figure'),
    Output('lr-tapping-comparison', 'figure'),
    Output('video-left-taps', 'children'),
    Output('video-right-taps', 'children'),
    Output('video-l-cv', 'children'),
    Output('video-r-cv', 'children'),
    Input('video-side-dropdown', 'value'),
    Input('video-camera-dropdown', 'value'),
)
def update_video(side, camera):
    if not side or not camera:
        return (html.P('Select video'), _empty_fig(), _empty_fig(),
                '—', '—', '—', '—')

    # Video player
    video_url = f'/video/{side}/{camera}'
    player = html.Video(
        src=video_url, controls=True, className='video-player',
        style={'width': '100%', 'borderRadius': '8px'},
    )

    # Analysis
    side_data = video_data.get(side, {})
    cam_data = side_data.get(camera, {})

    fig_timeline = make_motion_timeline(cam_data, side) if cam_data else _empty_fig()
    fig_lr = make_lr_tapping_comparison(
        video_data.get('left', {}).get(camera, {}),
        video_data.get('right', {}).get(camera, {}),
    )

    left_cam = video_data.get('left', {}).get(camera, {})
    right_cam = video_data.get('right', {}).get(camera, {})
    l_taps = str(left_cam.get('estimated_taps', '—')) if left_cam else '—'
    r_taps = str(right_cam.get('estimated_taps', '—')) if right_cam else '—'
    l_cv = f"{left_cam.get('tap_interval_cv', 0):.3f}" if left_cam else '—'
    r_cv = f"{right_cam.get('tap_interval_cv', 0):.3f}" if right_cam else '—'

    return player, fig_timeline, fig_lr, l_taps, r_taps, l_cv, r_cv


# ─── Decision Save ───
@app.callback(
    Output('decision-store', 'data'),
    Output('save-feedback', 'children'),
    Input('save-decision-btn', 'n_clicks'),
    State('patient-dropdown', 'value'),
    State('decision-classification', 'value'),
    State('decision-confidence', 'value'),
    State('evidence-tags', 'value'),
    State('decision-notes', 'value'),
    State('decision-store', 'data'),
)
def save_decision(n_clicks, tulip_id, classification, confidence, tags, notes, store):
    if not n_clicks or not tulip_id:
        return store or {}, ''

    store = store or {}
    store[tulip_id] = {
        'classification': classification,
        'confidence': confidence,
        'evidence_tags': tags,
        'notes': notes,
    }
    return store, html.Span('✓ Saved!', className='feedback-success')


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
    app.run(debug=True, port=8050)

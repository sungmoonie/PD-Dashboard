"""
PD Clinical Decision Support Dashboard — Label-Free Review Mode
================================================================
Matched Sensor Analog Evidence · 3-column single-page layout
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
    load_timeseries, get_subject_list, get_subject_list_labelfree,
    SENSOR_TASKS, TASK_LABELS_KR, _estimate_fs,
)
from src.feature_engineering import (
    calc_signal_rms, calc_asymmetry_index, calc_signal_stats,
    calc_tremor_power, calc_rhythm_irregularity, calc_mean_jerk,
)
from src.figures import (
    make_timeseries_plot, make_motion_timeline, make_lr_tapping_comparison,
    make_bilateral_matrix, make_spectral_fingerprint,
    make_rhythm_ladder, make_evidence_ribbon, _empty_fig,
)
from src.layout import create_layout

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
#  VIDEO STREAMING (unchanged from original)
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
        # Fallback: GitHub raw content
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
# ══════════════════════════════════════════════════════════════

# ─── Mode Toggle ───
@app.callback(
    Output('mode-store', 'data'),
    Output('mode-toggle-btn', 'children'),
    Output('mode-toggle-btn', 'className'),
    Input('mode-toggle-btn', 'n_clicks'),
    State('mode-store', 'data'),
)
def toggle_mode(n_clicks, current_mode):
    if n_clicks and n_clicks % 2 == 1:
        return 'teaching', 'Teaching', 'mode-toggle teaching-active'
    return 'review', 'Review', 'mode-toggle review-active'


# ─── Teaching panel visibility ───
@app.callback(
    Output('teaching-panel', 'className'),
    Output('demographics-row', 'className'),
    Input('mode-store', 'data'),
)
def update_mode_visibility(mode):
    if mode == 'teaching':
        return '', 'demographics-row'
    return 'hidden-in-review', 'demographics-row hidden-in-review'


# ─── Case Badge + Teaching Info ───
@app.callback(
    Output('case-badge', 'children'),
    Output('case-badge', 'className'),
    Output('teaching-condition', 'children'),
    Output('teaching-hy', 'children'),
    Output('teaching-diagnosis', 'children'),
    Output('teaching-updrs-summary', 'children'),
    Output('demographics-row', 'children'),
    Input('patient-dropdown', 'value'),
    Input('mode-store', 'data'),
)
def update_case_info(tulip_id, mode):
    if not tulip_id:
        return '—', 'badge badge-unknown', '—', '—', '—', '', ''

    row = patients_df[patients_df.tulip_id == tulip_id]
    meta = updrs_meta.get(tulip_id, {})

    # Badge (only show in teaching mode)
    diag = meta.get('diagnosis', '?')
    if mode == 'teaching':
        badge_text = diag
        badge_class = f'badge badge-{diag.lower()}' if diag in ('PD', 'HT') else 'badge badge-unknown'
    else:
        badge_text = ''
        badge_class = 'badge badge-hidden'

    # Teaching fields
    condition = row.iloc[0]['condition'] if not row.empty else '—'
    hy = str(meta.get('hy_mean', '—'))
    diagnosis = diag

    # UPDRS summary
    updrs_summary = ''
    if mode == 'teaching':
        subj_labels = labels_df[
            (labels_df.tulip_id == tulip_id) & (labels_df.is_numeric == True)
        ]
        if not subj_labels.empty:
            high_items = subj_labels[subj_labels['mean'] >= 2.0]
            if not high_items.empty:
                items_text = ', '.join(high_items['updrs_name'].tolist()[:5])
                updrs_summary = html.Div([
                    html.Strong('High UPDRS items: '),
                    html.Span(items_text),
                ], className='updrs-hint')

    # Demographics row (teaching only)
    demographics = ''
    if mode == 'teaching' and not row.empty:
        r = row.iloc[0]
        demographics = html.Div([
            html.Span(f"TULIP: {tulip_id}", className='demo-item'),
            html.Span(f"PADS: {r['pads_id']}", className='demo-item'),
            html.Span(f"Age: {r['age']}", className='demo-item'),
            html.Span(f"Gender: {r['gender']}", className='demo-item'),
            html.Span(f"Condition: {condition}", className='demo-item demo-condition'),
        ], className='demo-content')

    return badge_text, badge_class, condition, hy, diagnosis, updrs_summary, demographics


# ─── Bilateral Matrix ───
@app.callback(
    Output('bilateral-matrix', 'figure'),
    Input('patient-dropdown', 'value'),
)
def update_bilateral_matrix(tulip_id):
    if not tulip_id:
        return _empty_fig('Case를 선택하세요')
    return make_bilateral_matrix(feature_cache, tulip_id)


# ─── Spectral Fingerprint ───
@app.callback(
    Output('spectral-fingerprint', 'figure'),
    Input('patient-dropdown', 'value'),
    Input('task-dropdown', 'value'),
)
def update_spectral(tulip_id, task):
    if not tulip_id or not task:
        return _empty_fig('데이터 없음')
    return make_spectral_fingerprint(tulip_id, task)


# ─── Rhythm Ladder ───
@app.callback(
    Output('rhythm-ladder', 'figure'),
    Input('patient-dropdown', 'value'),
    Input('task-dropdown', 'value'),
)
def update_rhythm(tulip_id, task):
    if not tulip_id or not task:
        return _empty_fig('데이터 없음')
    return make_rhythm_ladder(tulip_id, task)


# ─── Evidence Ribbon ───
@app.callback(
    Output('evidence-ribbon', 'figure'),
    Input('patient-dropdown', 'value'),
)
def update_evidence_ribbon(tulip_id):
    if not tulip_id:
        return _empty_fig('Case를 선택하세요')
    return make_evidence_ribbon(feature_cache, tulip_id)


# ─── Raw Timeseries ───
@app.callback(
    Output('raw-timeseries-left', 'figure'),
    Output('raw-timeseries-right', 'figure'),
    Input('patient-dropdown', 'value'),
    Input('task-dropdown', 'value'),
)
def update_raw_timeseries(tulip_id, task):
    if not tulip_id or not task:
        return _empty_fig('데이터 없음'), _empty_fig('데이터 없음')
    left_ts = load_timeseries(tulip_id, task, 'LeftWrist')
    right_ts = load_timeseries(tulip_id, task, 'RightWrist')
    task_label = TASK_LABELS_KR.get(task, task)
    fig_l = make_timeseries_plot(left_ts, f'{task_label} — Left Wrist')
    fig_r = make_timeseries_plot(right_ts, f'{task_label} — Right Wrist')
    return fig_l, fig_r


# ─── Video Player ───
@app.callback(
    Output('video-player-container', 'children'),
    Input('video-side-dropdown', 'value'),
    Input('video-camera-dropdown', 'value'),
)
def update_video_player(side, camera):
    if not side or not camera:
        return html.P('Select video', className='placeholder-text')
    video_url = f'/video/{side}/{camera}'
    return html.Video(
        src=video_url, controls=True, className='video-player',
        style={'width': '100%', 'borderRadius': '8px'},
    )


# ─── Video Analysis ───
@app.callback(
    Output('motion-timeline', 'figure'),
    Output('lr-tapping-comparison', 'figure'),
    Output('video-left-taps', 'children'),
    Output('video-right-taps', 'children'),
    Output('video-l-cv', 'children'),
    Output('video-r-cv', 'children'),
    Input('video-side-dropdown', 'value'),
    Input('video-camera-dropdown', 'value'),
)
def update_video_analysis(side, camera):
    if not video_data or not side:
        return (_empty_fig(), _empty_fig(), '—', '—', '—', '—')

    side_data = video_data.get(side, {})
    cam_data = side_data.get(camera, {})
    other_side = 'right' if side == 'left' else 'left'
    other_data = video_data.get(other_side, {}).get(camera, {})

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

    return fig_timeline, fig_lr, l_taps, r_taps, l_cv, r_cv


# ─── Save Decision ───
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
    return store, html.Span('Saved!', className='feedback-success')


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


# ─── Task Store sync ───
@app.callback(
    Output('selected-task-store', 'data'),
    Input('task-dropdown', 'value'),
)
def sync_task(task):
    return task or 'TouchNose'


# ══════════════════════════════════════════════════════════════
#  RUN
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    app.run(debug=True, port=8050)

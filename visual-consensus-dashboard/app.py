"""
PD Motor Assessment Dashboard
==============================
TULIP/PADS 실제 임상 데이터 기반 · UPDRS + Sensor + Video
"""

import os
import re

from dash import Dash, Input, Output, State, html, no_update
from flask import send_file, request, Response

from src.data_loader import (
    load_patients, load_nms, load_updrs_labels, load_updrs_metadata,
    load_timeseries, load_video_analysis, TASK_LABELS_KR,
)
from src.feature_engineering import calc_signal_rms, calc_asymmetry_index, calc_signal_stats
from src.figures import (
    make_updrs_heatmap, make_clinician_comparison_bar,
    make_disagreement_heatmap, make_agreement_stats,
    make_timeseries_plot, make_lr_overlay_plot, make_lr_rms_bar,
    make_motion_timeline, make_lr_tapping_comparison,
    make_multicam_comparison, make_tapping_summary_chart,
)
from src.layout import create_layout

# ── Pre-load data that is small / needed globally ────────────
patients_df = load_patients()
nms_data = load_nms()
labels_df = load_updrs_labels()
updrs_meta = load_updrs_metadata()
video_data = load_video_analysis()

# ── App ──────────────────────────────────────────────────────
app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    title="PD Motor Assessment",
)
server = app.server
app.layout = create_layout()

# ── Video streaming with range request support ───────────────
VIDEO_BASE = os.path.dirname(os.path.abspath(__file__))
VIDEO_BASE = os.path.dirname(VIDEO_BASE)  # 시각화수업/
VIDEO_MAP = {
    'finger_tapping_left': '7. Finger_tapping_left',
    'finger_tapping_right': '8. FInger_tapping_right',
}


@server.route('/video/<folder>/<camera>')
def stream_video(folder, camera):
    real_folder = VIDEO_MAP.get(folder)
    if not real_folder:
        return 'Not found', 404
    path = os.path.join(VIDEO_BASE, real_folder, camera)
    if not os.path.exists(path):
        return 'Not found', 404

    file_size = os.path.getsize(path)
    range_header = request.headers.get('Range')

    if range_header:
        m = re.search(r'bytes=(\d+)-(\d*)', range_header)
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else file_size - 1
        length = end - start + 1

        with open(path, 'rb') as f:
            f.seek(start)
            data = f.read(length)

        resp = Response(data, 206, mimetype='video/mp4')
        resp.headers['Content-Range'] = f'bytes {start}-{end}/{file_size}'
        resp.headers['Accept-Ranges'] = 'bytes'
        resp.headers['Content-Length'] = length
        return resp
    else:
        return send_file(path, mimetype='video/mp4')


# ══════════════════════════════════════════════════════════════
#  CALLBACKS
# ══════════════════════════════════════════════════════════════

# ── 1. 환자 Overview ─────────────────────────────────────────
@app.callback(
    [
        Output('patient-demographics', 'children'),
        Output('condition-card', 'children'),
        Output('hy-card', 'children'),
        Output('diagnosis-card', 'children'),
        Output('nms-count-card', 'children'),
        Output('bmi-card', 'children'),
        Output('handedness-card', 'children'),
        Output('nms-symptom-list', 'children'),
        Output('patient-badge', 'children'),
        Output('patient-badge', 'className'),
    ],
    Input('patient-dropdown', 'value'),
)
def update_overview(tulip_id):
    if not tulip_id:
        return [html.Div()] + ['—'] * 6 + ['Subject를 선택하세요.', '', '']

    pat = patients_df[patients_df.tulip_id == tulip_id]
    if pat.empty:
        return [html.Div()] + ['—'] * 6 + ['데이터 없음', '', '']
    pat = pat.iloc[0]

    meta = updrs_meta.get(tulip_id, {})
    nms = nms_data.get(tulip_id, {})

    # Demographics card
    hand_kr = {'right': '오른손잡이', 'left': '왼손잡이'}.get(pat.handedness, pat.handedness)
    demographics = html.Div(className='demographics-row', children=[
        html.Span(f'{tulip_id}', className='demo-id'),
        html.Span(f'PADS {pat.pads_id}', className='demo-item'),
        html.Span(f'{pat.age}세' if pat.age else '', className='demo-item'),
        html.Span(f'{pat.gender}', className='demo-item'),
        html.Span(f'{hand_kr}', className='demo-item'),
        html.Span(f'{pat.height}cm / {pat.weight}kg' if pat.height else '', className='demo-item'),
        html.Span(f'{pat.condition}', className='demo-group'),
    ])

    condition = pat.condition or '—'
    hy_val = f"{meta.get('hy_mean', '—')}"
    if 'hy_scores' in meta:
        hy_val += f" ({'/'.join(str(s) for s in meta['hy_scores'])})"
    diagnosis = meta.get('diagnosis', '—')
    nms_count = f"{nms.get('count', 0)} / {nms.get('total', 30)}"
    bmi = f"{pat.bmi:.1f}" if pat.bmi else '—'
    handedness = pat.handedness.capitalize() if pat.handedness else '—'

    # NMS symptom list
    symptoms = nms.get('symptoms', [])
    if symptoms:
        symptom_items = html.Ul(className='nms-symptom-items', children=[
            html.Li(s, className='nms-symptom-item') for s in symptoms
        ])
    else:
        symptom_items = html.P('Non-motor 증상 없음', style={'color': '#38a169', 'fontWeight': '600'})

    # Badge
    diag = meta.get('diagnosis', pat.condition)
    badge_map = {
        'PD': ('PD', 'badge badge-pd'),
        'HT': ('Healthy', 'badge badge-healthy'),
        'Healthy': ('Healthy', 'badge badge-healthy'),
        'DDx': ('DDx', 'badge badge-differential'),
    }
    # Check condition for badge
    if diag in badge_map:
        badge_text, badge_class = badge_map[diag]
    elif "Parkinson" in (pat.condition or ''):
        badge_text, badge_class = 'PD', 'badge badge-pd'
    elif "Healthy" in (pat.condition or ''):
        badge_text, badge_class = 'Healthy', 'badge badge-healthy'
    else:
        badge_text, badge_class = diag or '?', 'badge'

    return [demographics, condition, hy_val, diagnosis, nms_count,
            bmi, handedness, symptom_items, badge_text, badge_class]


# ── 2. UPDRS 임상 평가 ──────────────────────────────────────
@app.callback(
    Output('updrs-heatmap', 'figure'),
    Input('patient-dropdown', 'value'),  # trigger on load
)
def update_updrs_heatmap(_):
    return make_updrs_heatmap(labels_df)


@app.callback(
    [
        Output('clinician-comparison-bar', 'figure'),
        Output('updrs-click-detail', 'children'),
    ],
    Input('updrs-heatmap', 'clickData'),
    State('patient-dropdown', 'value'),
)
def update_updrs_click(click_data, tulip_id):
    if not click_data:
        return [make_clinician_comparison_bar(labels_df, '', ''),
                'Heatmap 셀을 클릭하면 Clinician별 비교가 표시됩니다.']

    point = click_data['points'][0]
    clicked_tulip = point.get('y', '')
    clicked_item = point.get('x', '')
    mean_score = point.get('z', 0)

    fig = make_clinician_comparison_bar(labels_df, clicked_tulip, clicked_item)

    # Detail text
    row = labels_df[
        (labels_df.tulip_id == clicked_tulip) &
        (labels_df.updrs_name == clicked_item) &
        (labels_df.is_numeric == True)
    ]
    if not row.empty:
        r = row.iloc[0]
        detail = (
            f"Subject: {clicked_tulip} | Item: {clicked_item} | "
            f"C1: {r['c1']} | C2: {r['c2']} | C3: {r['c3']} | "
            f"Mean: {r['mean']:.2f} | Disagreement: {r['disagreement']:.0f}"
        )
    else:
        detail = f'{clicked_tulip} — {clicked_item}: 데이터 없음'

    return [fig, detail]


# ── 3. Inter-rater Agreement ────────────────────────────────
@app.callback(
    [
        Output('disagreement-heatmap', 'figure'),
        Output('agreement-perfect-card', 'children'),
        Output('agreement-high-card', 'children'),
        Output('agreement-mean-card', 'children'),
    ],
    Input('patient-dropdown', 'value'),  # trigger on load
)
def update_agreement(_):
    fig = make_disagreement_heatmap(labels_df)
    stats = make_agreement_stats(labels_df)
    if stats:
        perfect = f"{stats['perfect_agreement']} ({stats['perfect_pct']}%)"
        high = f"{stats['high_disagreement']} ({stats['high_pct']}%)"
        mean_d = f"{stats['mean_disagreement']}"
    else:
        perfect = high = mean_d = '—'
    return [fig, perfect, high, mean_d]


# ── 4. 센서 데이터 ──────────────────────────────────────────
@app.callback(
    [
        Output('sensor-timeseries-chart', 'figure'),
        Output('sensor-accel-rms-card', 'children'),
        Output('sensor-gyro-rms-card', 'children'),
        Output('sensor-duration-card', 'children'),
        Output('sensor-samples-card', 'children'),
    ],
    [
        Input('patient-dropdown', 'value'),
        Input('sensor-task-dropdown', 'value'),
        Input('sensor-wrist-toggle', 'value'),
    ],
)
def update_sensor(tulip_id, task, wrist):
    if not tulip_id or not task or not wrist:
        return [{}] + ['—'] * 4

    ts = load_timeseries(tulip_id, task, wrist)
    if ts.empty:
        return [{}] + ['데이터 없음'] * 4

    task_label = TASK_LABELS_KR.get(task, task)
    wrist_label = 'Left' if 'Left' in wrist else 'Right'
    title = f'{tulip_id} — {task_label} ({wrist_label} Wrist)'

    fig = make_timeseries_plot(ts, title=title)

    accel_rms = f"{calc_signal_rms(ts['accel_mag'].values):.4f}"
    gyro_rms = f"{calc_signal_rms(ts['gyro_mag'].values):.4f}"
    duration = f"{ts['time'].max():.1f}"
    samples = str(len(ts))

    return [fig, accel_rms, gyro_rms, duration, samples]


# ── 5. 좌우 비교 ────────────────────────────────────────────
@app.callback(
    [
        Output('lr-overlay-chart', 'figure'),
        Output('lr-rms-bar-chart', 'figure'),
        Output('lr-left-rms-card', 'children'),
        Output('lr-right-rms-card', 'children'),
        Output('lr-asymmetry-card', 'children'),
    ],
    [Input('patient-dropdown', 'value'), Input('lr-task-dropdown', 'value')],
)
def update_lr_comparison(tulip_id, task):
    if not tulip_id or not task:
        return [{}] * 2 + ['—'] * 3

    left_df = load_timeseries(tulip_id, task, 'LeftWrist')
    right_df = load_timeseries(tulip_id, task, 'RightWrist')

    if left_df.empty and right_df.empty:
        return [{}] * 2 + ['데이터 없음'] * 3

    task_label = TASK_LABELS_KR.get(task, task)

    # Overlay plot
    overlay_fig = make_lr_overlay_plot(left_df, right_df, 'accel_mag', task_label)

    # Stats
    left_stats = calc_signal_stats(left_df['accel_mag'].values) if not left_df.empty else {}
    right_stats = calc_signal_stats(right_df['accel_mag'].values) if not right_df.empty else {}

    rms_fig = make_lr_rms_bar(left_stats, right_stats, task_label)

    left_rms = f"{left_stats.get('rms', 0):.4f}"
    right_rms = f"{right_stats.get('rms', 0):.4f}"
    asym = calc_asymmetry_index(left_stats.get('rms', 0), right_stats.get('rms', 0))
    asym_str = f"{asym:.3f}"

    return [overlay_fig, rms_fig, left_rms, right_rms, asym_str]


# ── 6. 영상 분석 ────────────────────────────────────────────
@app.callback(
    Output('video-player-container', 'children'),
    [Input('video-side-dropdown', 'value'), Input('video-camera-dropdown', 'value')],
)
def update_video_player(side, camera):
    if not side or not camera:
        return html.P('카메라와 방향을 선택하세요.')
    folder = 'finger_tapping_left' if side == 'left' else 'finger_tapping_right'
    src = f'/video/{folder}/{camera}'
    side_label = 'Left' if side == 'left' else 'Right'
    return [
        html.P(f'{side_label} Hand — {camera.replace(".mp4", "")}',
               style={'fontWeight': '600', 'marginBottom': '8px', 'color': '#1a365d'}),
        html.Video(
            src=src,
            controls=True,
            autoPlay=False,
            style={'width': '100%', 'maxHeight': '420px', 'borderRadius': '8px',
                   'backgroundColor': '#000'},
        ),
    ]


@app.callback(
    [
        Output('motion-timeline-chart', 'figure'),
        Output('lr-tapping-chart', 'figure'),
        Output('tapping-summary-chart', 'figure'),
        Output('multicam-chart', 'figure'),
        Output('left-tap-count', 'children'),
        Output('right-tap-count', 'children'),
        Output('left-cv-card', 'children'),
        Output('right-cv-card', 'children'),
    ],
    [Input('video-side-dropdown', 'value'), Input('video-camera-dropdown', 'value')],
)
def update_video_analysis(side, camera):
    if not video_data or not side or not camera:
        return [{}] * 4 + ['—'] * 4

    side_data = video_data.get(side, {})
    cam_data = side_data.get(camera, {})

    timeline_fig = make_motion_timeline(cam_data, side)

    left_cam1 = video_data.get('left', {}).get('Camera1.mp4', {})
    right_cam1 = video_data.get('right', {}).get('Camera1.mp4', {})
    lr_fig = make_lr_tapping_comparison(left_cam1, right_cam1)
    summary_fig = make_tapping_summary_chart(left_cam1, right_cam1)
    multicam_fig = make_multicam_comparison(side_data, side)

    left_taps = str(left_cam1.get('estimated_taps', '—'))
    right_taps = str(right_cam1.get('estimated_taps', '—'))
    left_cv = f"{left_cam1.get('tap_interval_cv', 0):.3f}" if left_cam1 else '—'
    right_cv = f"{right_cam1.get('tap_interval_cv', 0):.3f}" if right_cam1 else '—'

    return [timeline_fig, lr_fig, summary_fig, multicam_fig,
            left_taps, right_taps, left_cv, right_cv]


# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    app.run(debug=True, port=8050)

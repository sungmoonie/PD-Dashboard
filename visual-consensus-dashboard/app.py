"""
Parkinson's Motor Assessment Dashboard
========================================
Smartwatch 센서 기반 정량적 운동 분석 · Clinical Decision Support
"""

from dash import Dash, Input, Output, State, html

from src.data_loader import (
    load_patients, load_task_features, load_left_right_features,
    load_timeseries, load_normative_stats, load_patient_history,
    load_video_analysis,
)
from src.feature_engineering import (
    calc_patient_summary, get_top_abnormal_tasks,
    calc_rhythm_irregularity, calc_amplitude_decrement,
    calc_normative_position, calc_visit_changes,
    estimate_updrs_scores,
)
from src.figures import (
    make_overview_radar, make_top_tasks_bar, make_task_feature_heatmap,
    make_left_right_mirror_plot, make_rhythm_timeseries,
    make_tap_interval_chart, make_amplitude_decrement_chart,
    make_normative_comparison, make_history_line_chart,
    make_change_bar_chart, make_updrs_gauge_chart,
    make_phase_portrait, make_signature_wall,
    make_evidence_bubble,
    make_motion_timeline, make_lr_tapping_comparison,
    make_multicam_comparison, make_tapping_summary_chart,
)
from src.layout import create_layout

# ── Korean labels ─────────────────────────────────────────────
TASK_KR = {
    'finger_tapping': 'Finger Tapping', 'hand_open_close': 'Hand Open/Close',
    'rest_tremor': 'Rest Tremor', 'gait': 'Gait',
    'toe_tapping': 'Toe Tapping', 'touch_nose': 'Touch Nose',
}
FEAT_KR = {
    'tremor_power': 'Tremor', 'movement_amplitude_reduction': 'Amplitude 감소',
    'rhythm_irregularity': 'Rhythm 불규칙', 'left_right_asymmetry': '좌우 Asymmetry',
    'motion_instability': 'Instability',
}

# ── Data Loading ──────────────────────────────────────────────
patients_df = load_patients()
task_features_df = load_task_features()
lr_features_df = load_left_right_features()
timeseries_df = load_timeseries()
normative_df = load_normative_stats()
history_df = load_patient_history()
video_data = load_video_analysis()

# ── App ───────────────────────────────────────────────────────
import os, re
from flask import send_file, request, Response

app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    title="Parkinson's Motor Assessment",
)
server = app.server
app.layout = create_layout()

# ── Video streaming with range request support ────────────────
VIDEO_BASE = '/Users/moonie/Desktop/시각화수업'
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
        Output('motor-score-card', 'children'),
        Output('asymmetry-score-card', 'children'),
        Output('tremor-score-card', 'children'),
        Output('rhythm-score-card-overview', 'children'),
        Output('nonmotor-score-card', 'children'),
        Output('top-task-card', 'children'),
        Output('overview-radar', 'figure'),
        Output('top-tasks-bar', 'figure'),
        Output('interpretation-text', 'children'),
        Output('patient-badge', 'children'),
        Output('patient-badge', 'className'),
    ],
    Input('patient-dropdown', 'value'),
)
def update_overview(patient_id):
    empty_demo = html.Div()
    if not patient_id:
        return [empty_demo] + ['—'] * 6 + [{}] * 2 + ['환자를 선택하세요.', '', '']

    pat = patients_df[patients_df.patient_id == patient_id].iloc[0]
    summary = calc_patient_summary(patient_id, task_features_df)
    top_tasks = get_top_abnormal_tasks(patient_id, task_features_df, n=3)

    # Demographics card
    hand_kr = {'right': '오른손잡이', 'left': '왼손잡이'}.get(pat.handedness, pat.handedness)
    demographics = html.Div(className='demographics-row', children=[
        html.Span(f'{pat.patient_id}', className='demo-id'),
        html.Span(f'{pat.age}세', className='demo-item'),
        html.Span(f'{pat.sex}', className='demo-item'),
        html.Span(f'{hand_kr}', className='demo-item'),
        html.Span(f'{pat.group}', className='demo-group'),
    ])

    motor_val = f"{pat.motor_score}/100"
    asym_val = f"{summary.get('left_right_asymmetry', 0):.2f}"
    tremor_val = f"{summary.get('tremor_power', 0):.2f}"
    rhythm_val = f"{summary.get('rhythm_irregularity', 0):.2f}"
    nonmotor_val = f"{pat.non_motor_score}/100"

    worst_task_raw = top_tasks.iloc[0]['task'] if len(top_tasks) > 0 else ''
    top_task_val = TASK_KR.get(worst_task_raw, worst_task_raw) if worst_task_raw else '—'

    radar_fig = make_overview_radar(summary)
    bar_fig = make_top_tasks_bar(top_tasks)

    pf = task_features_df[task_features_df.patient_id == patient_id]
    if len(top_tasks) > 0:
        worst_task = TASK_KR.get(worst_task_raw, worst_task_raw)
        worst_feat_row = pf[pf.task == worst_task_raw].sort_values('value', ascending=False)
        worst_feat = FEAT_KR.get(worst_feat_row.iloc[0]['feature'], '') if len(worst_feat_row) > 0 else ''
        interpretation = (
            f"이 환자는 {worst_task}에서 가장 높은 abnormality를 보이며, "
            f"특히 {worst_feat} 항목이 두드러집니다. "
            f"Motor score: {pat.motor_score}/100, Non-motor: {pat.non_motor_score}/100."
        )
    else:
        interpretation = "Significant abnormality가 발견되지 않았습니다."

    badge_map = {
        'PD': ('PD', 'badge badge-pd'),
        'Healthy': ('Healthy', 'badge badge-healthy'),
        'Differential': ('DDx', 'badge badge-differential'),
    }
    badge_text, badge_class = badge_map.get(pat.group, (pat.group, 'badge'))

    return [demographics, motor_val, asym_val, tremor_val, rhythm_val,
            nonmotor_val, top_task_val, radar_fig, bar_fig,
            interpretation, badge_text, badge_class]


# ── 2. Task-Feature Heatmap ──────────────────────────────────
@app.callback(
    Output('task-feature-heatmap', 'figure'),
    Input('patient-dropdown', 'value'),
)
def update_heatmap(patient_id):
    if not patient_id:
        return {}
    pf = task_features_df[task_features_df.patient_id == patient_id]
    matrix = pf.pivot(index='task', columns='feature', values='value')
    return make_task_feature_heatmap(matrix)


# ── 3. Heatmap Click ─────────────────────────────────────────
@app.callback(
    Output('heatmap-detail', 'children'),
    Input('task-feature-heatmap', 'clickData'),
    State('patient-dropdown', 'value'),
)
def update_heatmap_detail(click_data, patient_id):
    if not click_data or not patient_id:
        return "Heatmap 셀을 클릭하면 상세 정보가 표시됩니다."
    point = click_data['points'][0]
    task = point.get('y', '')
    feature = point.get('x', '')
    value = point.get('z', 0)
    pf = task_features_df[
        (task_features_df.patient_id == patient_id) &
        (task_features_df.task == task) &
        (task_features_df.feature == feature)
    ]
    percentile = pf.iloc[0]['normal_percentile'] if len(pf) > 0 else 'N/A'
    severity = 'HIGH' if value >= 0.7 else 'MODERATE' if value >= 0.4 else 'LOW'
    task_label = TASK_KR.get(task, task)
    feat_label = FEAT_KR.get(feature, feature)
    return (
        f"Task: {task_label} | Feature: {feat_label} | "
        f"Score: {value:.2f} | Percentile: {percentile} | Severity: {severity}"
    )


# ── 4. 좌우 비교 (탭 내부 task selector) ─────────────────────
@app.callback(
    [Output('mirror-plot', 'figure'), Output('asymmetry-card', 'children')],
    [Input('patient-dropdown', 'value'), Input('lr-task-dropdown', 'value')],
)
def update_mirror_plot(patient_id, task):
    if not patient_id or not task:
        return {}, '—'
    lr = lr_features_df[
        (lr_features_df.patient_id == patient_id) & (lr_features_df.task == task)
    ]
    if lr.empty:
        return {}, 'Data 없음'
    fig = make_left_right_mirror_plot(lr)
    mean_asym = lr['asymmetry_index'].mean()
    mean_pct = lr['normal_percentile'].mean()
    return fig, f"Asymmetry Index: {mean_asym:.2f} (Percentile: {mean_pct:.0f})"


# ── 5. Rhythm & Amplitude (탭 내부 task selector) ────────────
@app.callback(
    [
        Output('rhythm-timeseries', 'figure'),
        Output('tap-interval-chart', 'figure'),
        Output('amplitude-decrement-chart', 'figure'),
        Output('rhythm-score-card', 'children'),
        Output('amplitude-score-card', 'children'),
    ],
    [Input('patient-dropdown', 'value'), Input('rhythm-task-dropdown', 'value')],
)
def update_rhythm_panel(patient_id, task):
    if not patient_id or not task:
        return [{}] * 3 + ['—', '—']
    ts = timeseries_df[
        (timeseries_df.patient_id == patient_id) & (timeseries_df.task == task)
    ]
    if ts.empty:
        return [{}] * 3 + ['Data 없음', 'Data 없음']
    ts = ts.sort_values('time').reset_index(drop=True)
    return [
        make_rhythm_timeseries(ts),
        make_tap_interval_chart(ts),
        make_amplitude_decrement_chart(ts),
        f"CV: {calc_rhythm_irregularity(ts):.3f}",
        f"Decrement: {calc_amplitude_decrement(ts):.1%}",
    ]


# ── 6. Normative 비교 ────────────────────────────────────────
@app.callback(
    [
        Output('normative-chart', 'figure'),
        Output('normative-summary-card', 'children'),
        Output('normative-detail', 'children'),
    ],
    Input('patient-dropdown', 'value'),
)
def update_normative(patient_id):
    if not patient_id:
        return {}, '—', ''
    results = calc_normative_position(patient_id, task_features_df, normative_df)
    fig = make_normative_comparison(results)
    if results:
        z_scores = [r['z_score'] for r in results]
        avg_z = sum(z_scores) / len(z_scores)
        n_high = sum(1 for z in z_scores if z > 2)
        summary = f"Mean Z: {avg_z:.1f} | {n_high}개 feature > 2σ"
        detail_lines = []
        for r in results:
            sev = 'HIGH' if r['z_score'] > 2 else 'MOD' if r['z_score'] > 1 else 'NORM'
            feat = FEAT_KR.get(r['feature'], r['feature'])
            task = TASK_KR.get(r['task'], r['task'])
            detail_lines.append(f"{task} — {feat}: z={r['z_score']:.1f} ({sev})")
        detail_lines.sort(key=lambda x: -float(x.split('z=')[1].split(' ')[0]))
        detail = "Top abnormal features: " + " | ".join(detail_lines[:5])
    else:
        summary = 'Data 없음'
        detail = ''
    return fig, summary, detail


# ── 7. 경과 추적 ─────────────────────────────────────────────
@app.callback(
    [
        Output('history-line-chart', 'figure'),
        Output('history-change-chart', 'figure'),
        Output('history-summary-card', 'children'),
    ],
    Input('patient-dropdown', 'value'),
)
def update_history(patient_id):
    if not patient_id:
        return {}, {}, '—'
    line_fig = make_history_line_chart(history_df, patient_id)
    changes = calc_visit_changes(patient_id, history_df)
    change_fig = make_change_bar_chart(changes)
    if changes:
        worsened = sum(1 for c in changes if c['direction'] == 'worsened')
        improved = sum(1 for c in changes if c['direction'] == 'improved')
        stable = sum(1 for c in changes if c['direction'] == 'stable')
        n_visits = history_df[history_df.patient_id == patient_id]['visit_number'].nunique()
        summary = f"{n_visits}회 방문 | 악화: {worsened} | 호전: {improved} | 안정: {stable}"
    else:
        summary = '비교할 방문 기록이 부족합니다'
    return line_fig, change_fig, summary


# ── 8. UPDRS 추정 ────────────────────────────────────────────
@app.callback(
    [Output('updrs-gauge-chart', 'figure'), Output('updrs-evidence-panel', 'children')],
    Input('patient-dropdown', 'value'),
)
def update_updrs(patient_id):
    if not patient_id:
        return {}, '환자를 선택하세요.'
    items = estimate_updrs_scores(patient_id, task_features_df)
    fig = make_updrs_gauge_chart(items)
    lines = []
    total = sum(item['estimated_score'] for item in items)
    for item in items:
        evidence_str = ', '.join(item['evidence'])
        lines.append(
            f"{item['item_code']} {item['item_name']}: "
            f"Est. {item['estimated_score']:.1f}/4 "
            f"(conf: {item['confidence']:.0%}) — {evidence_str}"
        )
    evidence_text = f"Total estimated score: {total:.1f}/{len(items)*4} | " + " | ".join(lines)
    return fig, evidence_text


# ── 9. Phase Portrait (탭 내부 task selector) ────────────────
@app.callback(
    Output('phase-portrait-chart', 'figure'),
    [Input('patient-dropdown', 'value'), Input('phase-task-dropdown', 'value')],
)
def update_phase_portrait(patient_id, task):
    if not patient_id or not task:
        return {}
    ts = timeseries_df[
        (timeseries_df.patient_id == patient_id) & (timeseries_df.task == task)
    ].sort_values('time').reset_index(drop=True)
    return make_phase_portrait(ts, patient_id)


# ── 10. Signature Wall (탭 내부 task selector) ───────────────
@app.callback(
    Output('signature-wall-chart', 'figure'),
    [Input('patient-dropdown', 'value'), Input('sig-task-dropdown', 'value')],
)
def update_signature_wall(patient_id, task):
    if not task:
        return {}
    return make_signature_wall(timeseries_df, task, patients_df, highlight_patient=patient_id)


# ── 11. Evidence Map ─────────────────────────────────────────
@app.callback(
    Output('evidence-bubble-chart', 'figure'),
    Input('patient-dropdown', 'value'),
)
def update_evidence_bubble(patient_id):
    if not patient_id:
        return {}
    results = calc_normative_position(patient_id, task_features_df, normative_df)
    return make_evidence_bubble(results, task_features_df, patient_id)


# ── 12. 영상 분석 ────────────────────────────────────────────
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

    # Current side + camera
    side_data = video_data.get(side, {})
    cam_data = side_data.get(camera, {})

    # Motion timeline for selected camera
    timeline_fig = make_motion_timeline(cam_data, side)

    # L vs R comparison (Camera1)
    left_cam1 = video_data.get('left', {}).get('Camera1.mp4', {})
    right_cam1 = video_data.get('right', {}).get('Camera1.mp4', {})
    lr_fig = make_lr_tapping_comparison(left_cam1, right_cam1)

    # Tapping summary
    summary_fig = make_tapping_summary_chart(left_cam1, right_cam1)

    # Multi-camera comparison
    multicam_fig = make_multicam_comparison(side_data, side)

    # Summary cards
    left_taps = str(left_cam1.get('estimated_taps', '—'))
    right_taps = str(right_cam1.get('estimated_taps', '—'))
    left_cv = f"{left_cam1.get('tap_interval_cv', 0):.3f}" if left_cam1 else '—'
    right_cv = f"{right_cam1.get('tap_interval_cv', 0):.3f}" if right_cam1 else '—'

    return [timeline_fig, lr_fig, summary_fig, multicam_fig,
            left_taps, right_taps, left_cv, right_cv]


# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    app.run(debug=True, port=8050)

"""Plotly figure creation functions for PD clinical decision support dashboard."""

import plotly.graph_objects as go
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots

LAYOUT_DEFAULTS = dict(
    plot_bgcolor='white',
    paper_bgcolor='white',
    margin=dict(l=50, r=20, t=50, b=60),
    font=dict(family='-apple-system, Segoe UI, sans-serif', size=12, color='#2c3e50'),
)

GROUP_COLORS = {'PD': '#e53e3e', 'Healthy': '#38a169', 'Other': '#d69e2e'}


def _display_id(tulip_id):
    """Convert TULIP_001 → Patient_001 for display."""
    if not tulip_id:
        return ''
    return 'Patient_' + tulip_id.replace('TULIP_', '')


def _apply_defaults(fig, height=370):
    fig.update_layout(**LAYOUT_DEFAULTS, height=height)
    return fig


def _empty_fig(msg='Data 없음', height=370):
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False,
                       xref='paper', yref='paper', x=0.5, y=0.5, font=dict(size=14))
    return _apply_defaults(fig, height)


# ══════════════════════════════════════════════════════════════
#  TAB 2: UPDRS CLINICAL ASSESSMENT
# ══════════════════════════════════════════════════════════════

def make_updrs_heatmap(labels_df):
    """Heatmap: 12 subjects x UPDRS items, values = mean of 3 clinicians."""
    numeric = labels_df[labels_df.is_numeric == True].copy()
    exclude = ['Hoehn and Yahr Stage']
    numeric = numeric[~numeric.updrs_name.isin(exclude)]

    if numeric.empty:
        return _empty_fig('UPDRS 데이터 없음')

    pivot = numeric.pivot_table(index='tulip_id', columns='updrs_name', values='mean', aggfunc='first')
    pivot = pivot.sort_index()

    col_labels = [c[:25] + '...' if len(c) > 28 else c for c in pivot.columns]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=list(pivot.columns),
        y=list(pivot.index),
        colorscale='RdYlBu_r', zmin=0, zmax=4,
        text=np.where(np.isnan(pivot.values), '', np.round(pivot.values, 1).astype(str)),
        texttemplate='%{text}', textfont=dict(size=10),
        hovertemplate='Subject: %{y}<br>Item: %{x}<br>Mean Score: %{z:.2f}<extra></extra>',
        colorbar=dict(title='Score (0-4)', thickness=15),
    ))
    fig.update_layout(
        xaxis=dict(tickvals=list(pivot.columns), ticktext=col_labels,
                   side='bottom', tickangle=-45, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=11)),
        title=dict(text='UPDRS Part III — Clinician 평균 Score', font=dict(size=14)),
    )
    return _apply_defaults(fig, height=max(400, len(pivot) * 35 + 150))


def make_clinician_comparison_bar(labels_df, tulip_id, updrs_name):
    """Bar chart comparing 3 clinicians' scores for one subject x one item."""
    row = labels_df[
        (labels_df.tulip_id == tulip_id) &
        (labels_df.updrs_name == updrs_name) &
        (labels_df.is_numeric == True)
    ]
    if row.empty:
        return _empty_fig(f'{updrs_name} 데이터 없음')

    r = row.iloc[0]
    clinicians = ['Clinician 1', 'Clinician 2', 'Clinician 3']
    scores = [float(r['c1']), float(r['c2']), float(r['c3'])]
    colors = ['#2b6cb0', '#38a169', '#dd6b20']

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=clinicians, y=scores,
        marker=dict(color=colors, line=dict(width=1, color='white')),
        text=[f'{s:.1f}' for s in scores], textposition='auto',
        textfont=dict(size=14, color='white'),
    ))
    mean_val = sum(scores) / 3
    fig.add_hline(y=mean_val, line_dash='dash', line_color='#718096',
                  annotation_text=f'Mean: {mean_val:.2f}')
    fig.update_layout(
        title=dict(text=f'{tulip_id} — {updrs_name}', font=dict(size=13)),
        yaxis=dict(title='Score (0-4)', range=[0, 4.3], gridcolor='#edf2f7', dtick=1),
    )
    return _apply_defaults(fig, height=350)


# ══════════════════════════════════════════════════════════════
#  TAB 3: SENSOR TIMESERIES
# ══════════════════════════════════════════════════════════════

def make_timeseries_plot(ts_df, title='Sensor Data'):
    """6-axis accelerometer + gyroscope time series plot."""
    if ts_df.empty:
        return _empty_fig('센서 데이터 없음')

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=['Accelerometer (g)', 'Gyroscope (rad/s)'],
                        vertical_spacing=0.12)

    accel_cols = [('accel_x', 'X', '#e53e3e'), ('accel_y', 'Y', '#38a169'), ('accel_z', 'Z', '#2b6cb0')]
    for col, label, color in accel_cols:
        fig.add_trace(go.Scatter(
            x=ts_df['time'], y=ts_df[col], mode='lines',
            name=f'Accel {label}', line=dict(color=color, width=1),
            legendgroup='accel',
        ), row=1, col=1)

    gyro_cols = [('gyro_x', 'X', '#dd6b20'), ('gyro_y', 'Y', '#805ad5'), ('gyro_z', 'Z', '#d69e2e')]
    for col, label, color in gyro_cols:
        fig.add_trace(go.Scatter(
            x=ts_df['time'], y=ts_df[col], mode='lines',
            name=f'Gyro {label}', line=dict(color=color, width=1),
            legendgroup='gyro',
        ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=ts_df['time'], y=ts_df['accel_mag'], mode='lines',
        name='|Accel|', line=dict(color='#1a365d', width=1.5, dash='dot'),
        legendgroup='accel',
    ), row=1, col=1)

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis2=dict(title='Time (s)'),
        legend=dict(orientation='h', y=-0.12, font=dict(size=10)),
        **LAYOUT_DEFAULTS, height=500,
    )
    fig.update_xaxes(gridcolor='#edf2f7')
    fig.update_yaxes(gridcolor='#edf2f7')
    return fig


# ══════════════════════════════════════════════════════════════
#  TAB 4: PD vs HEALTHY GROUP COMPARISON
# ══════════════════════════════════════════════════════════════

def make_group_box_plot(group_stats_df, task, metric='accel_rms', highlight_tulip=None):
    """Box plot comparing PD vs Healthy groups for a given task.
    Highlights selected patient's position."""
    df = group_stats_df[group_stats_df.task == task].copy()
    if df.empty:
        return _empty_fig('데이터 없음')

    metric_labels = {
        'accel_rms': 'Accelerometer RMS (g)',
        'gyro_rms': 'Gyroscope RMS (rad/s)',
        'accel_std': 'Accelerometer Std',
        'gyro_std': 'Gyroscope Std',
    }

    fig = go.Figure()

    for group in ['Healthy', 'PD', 'Other']:
        gd = df[df.group == group]
        if gd.empty:
            continue
        # Combine both wrists
        vals = gd[metric].values
        color = GROUP_COLORS.get(group, '#718096')

        fig.add_trace(go.Box(
            y=vals, name=group,
            marker=dict(color=color, size=6),
            line=dict(color=color),
            boxmean='sd',
            jitter=0.3, pointpos=0,
            boxpoints='all',
            hovertemplate=f'{group}<br>{metric_labels.get(metric, metric)}: %{{y:.4f}}<extra></extra>',
        ))

    # Highlight selected patient
    if highlight_tulip:
        pat_data = df[df.tulip_id == highlight_tulip]
        if not pat_data.empty:
            pat_group = pat_data.iloc[0]['group']
            for _, row in pat_data.iterrows():
                fig.add_trace(go.Scatter(
                    x=[pat_group], y=[row[metric]],
                    mode='markers',
                    marker=dict(size=16, color='#1a365d', symbol='star',
                                line=dict(width=2, color='white')),
                    name=f'{highlight_tulip} ({row["wrist"]})',
                    hovertemplate=f'{highlight_tulip} ({row["wrist"]})<br>{row[metric]:.4f}<extra></extra>',
                ))

    fig.update_layout(
        title=dict(text=f'PD vs Healthy — {task} ({metric_labels.get(metric, metric)})',
                   font=dict(size=14)),
        yaxis=dict(title=metric_labels.get(metric, metric), gridcolor='#edf2f7'),
        showlegend=True,
        legend=dict(orientation='h', y=-0.15),
    )
    return _apply_defaults(fig, height=420)


def make_group_task_summary(group_stats_df, highlight_tulip=None):
    """Grouped bar chart: mean accel RMS per task, PD vs Healthy.
    Shows where the selected patient falls."""
    df = group_stats_df.copy()
    if df.empty:
        return _empty_fig('데이터 없음')

    # Mean per group per task
    summary = df.groupby(['group', 'task'])['accel_rms'].mean().reset_index()

    fig = go.Figure()
    tasks = sorted(df['task'].unique())

    for group in ['Healthy', 'PD', 'Other']:
        gd = summary[summary.group == group]
        if gd.empty:
            continue
        task_means = []
        for t in tasks:
            row = gd[gd.task == t]
            task_means.append(row['accel_rms'].values[0] if len(row) > 0 else 0)

        fig.add_trace(go.Bar(
            x=tasks, y=task_means, name=group,
            marker=dict(color=GROUP_COLORS.get(group, '#718096'), opacity=0.7),
            text=[f'{v:.4f}' for v in task_means], textposition='outside',
            textfont=dict(size=9),
        ))

    # Overlay selected patient
    if highlight_tulip:
        pat = df[df.tulip_id == highlight_tulip]
        if not pat.empty:
            pat_means = pat.groupby('task')['accel_rms'].mean()
            fig.add_trace(go.Scatter(
                x=list(pat_means.index), y=list(pat_means.values),
                mode='markers+lines',
                marker=dict(size=12, color='#1a365d', symbol='star',
                            line=dict(width=2, color='white')),
                line=dict(color='#1a365d', width=2, dash='dot'),
                name=f'{highlight_tulip} (selected)',
            ))

    fig.update_layout(
        title=dict(text='Task별 Movement Intensity — 그룹 평균 비교', font=dict(size=14)),
        xaxis=dict(tickangle=-30, tickfont=dict(size=10)),
        yaxis=dict(title='Accel RMS (g)', gridcolor='#edf2f7'),
        barmode='group',
        legend=dict(orientation='h', y=-0.2),
    )
    return _apply_defaults(fig, height=450)


def make_asymmetry_comparison(group_stats_df, task, highlight_tulip=None):
    """Scatter: Left RMS vs Right RMS, colored by group.
    Diagonal = perfect symmetry. Deviation = asymmetry (PD marker)."""
    df = group_stats_df[group_stats_df.task == task].copy()
    if df.empty:
        return _empty_fig('데이터 없음')

    # Pivot to get left/right per subject
    left = df[df.wrist == 'Left'][['tulip_id', 'group', 'accel_rms']].rename(
        columns={'accel_rms': 'left_rms'})
    right = df[df.wrist == 'Right'][['tulip_id', 'accel_rms']].rename(
        columns={'accel_rms': 'right_rms'})
    merged = left.merge(right, on='tulip_id', how='inner')

    if merged.empty:
        return _empty_fig('좌우 데이터 부족')

    fig = go.Figure()

    # Symmetry line
    max_val = max(merged['left_rms'].max(), merged['right_rms'].max()) * 1.1
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val], mode='lines',
        line=dict(color='#cbd5e0', width=1, dash='dash'),
        name='Perfect Symmetry', hoverinfo='skip',
    ))

    for group in ['Healthy', 'PD', 'Other']:
        gd = merged[merged.group == group]
        if gd.empty:
            continue
        color = GROUP_COLORS.get(group, '#718096')
        is_highlight = gd['tulip_id'] == highlight_tulip if highlight_tulip else [False] * len(gd)

        fig.add_trace(go.Scatter(
            x=gd['left_rms'], y=gd['right_rms'],
            mode='markers+text',
            marker=dict(
                size=[14 if h else 9 for h in is_highlight],
                color=color, opacity=0.8,
                symbol=['star' if h else 'circle' for h in is_highlight],
                line=dict(width=1, color='white'),
            ),
            text=gd['tulip_id'].apply(lambda x: 'P_' + x.replace('TULIP_', '')),
            textposition='top center', textfont=dict(size=8),
            name=group,
            hovertemplate='%{text}<br>Left: %{x:.4f}<br>Right: %{y:.4f}<extra></extra>',
            customdata=gd['tulip_id'],
        ))

    fig.update_layout(
        title=dict(text=f'좌우 대칭성 — {task}', font=dict(size=14)),
        xaxis=dict(title='Left Wrist Accel RMS', gridcolor='#edf2f7',
                   scaleanchor='y', range=[0, max_val]),
        yaxis=dict(title='Right Wrist Accel RMS', gridcolor='#edf2f7',
                   range=[0, max_val]),
        legend=dict(orientation='h', y=-0.15),
    )
    fig.add_annotation(x=max_val * 0.3, y=max_val * 0.8,
                       text='Right > Left', showarrow=False,
                       font=dict(size=10, color='#a0aec0'))
    fig.add_annotation(x=max_val * 0.7, y=max_val * 0.2,
                       text='Left > Right', showarrow=False,
                       font=dict(size=10, color='#a0aec0'))
    return _apply_defaults(fig, height=450)


# ══════════════════════════════════════════════════════════════
#  TAB 3 supplement: LEFT-RIGHT COMPARISON
# ══════════════════════════════════════════════════════════════

def make_lr_overlay_plot(left_df, right_df, channel='accel_mag', task_name=''):
    """Overlay left and right wrist signal magnitude."""
    if left_df.empty and right_df.empty:
        return _empty_fig('좌우 데이터 없음')

    fig = go.Figure()
    if not left_df.empty:
        fig.add_trace(go.Scatter(
            x=left_df['time'], y=left_df[channel], mode='lines',
            name='Left Wrist', line=dict(color='#4299e1', width=1.5), opacity=0.8,
        ))
    if not right_df.empty:
        fig.add_trace(go.Scatter(
            x=right_df['time'], y=right_df[channel], mode='lines',
            name='Right Wrist', line=dict(color='#fc8181', width=1.5), opacity=0.8,
        ))

    channel_label = {'accel_mag': 'Accelerometer Magnitude (g)',
                     'gyro_mag': 'Gyroscope Magnitude (rad/s)'}
    fig.update_layout(
        title=dict(text=f'좌우 Waveform 비교 — {task_name}', font=dict(size=14)),
        xaxis=dict(title='Time (s)', gridcolor='#edf2f7'),
        yaxis=dict(title=channel_label.get(channel, channel), gridcolor='#edf2f7'),
        legend=dict(orientation='h', y=-0.15),
    )
    return _apply_defaults(fig, height=380)


def make_lr_rms_bar(left_stats, right_stats, task_name=''):
    """Bar chart comparing RMS, mean, std between left and right."""
    metrics = ['RMS', 'Mean', 'Std', 'Max']
    left_vals = [left_stats.get('rms', 0), abs(left_stats.get('mean', 0)),
                 left_stats.get('std', 0), left_stats.get('max', 0)]
    right_vals = [right_stats.get('rms', 0), abs(right_stats.get('mean', 0)),
                  right_stats.get('std', 0), right_stats.get('max', 0)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=metrics, y=left_vals, name='Left Wrist',
        marker=dict(color='#4299e1'),
        text=[f'{v:.4f}' for v in left_vals], textposition='auto',
    ))
    fig.add_trace(go.Bar(
        x=metrics, y=right_vals, name='Right Wrist',
        marker=dict(color='#fc8181'),
        text=[f'{v:.4f}' for v in right_vals], textposition='auto',
    ))
    fig.update_layout(
        title=dict(text=f'좌우 Signal 통계 비교 — {task_name}', font=dict(size=14)),
        yaxis=dict(title='Signal Value', gridcolor='#edf2f7'),
        barmode='group',
        legend=dict(orientation='h', y=-0.15),
    )
    return _apply_defaults(fig, height=380)


# ══════════════════════════════════════════════════════════════
#  TAB 5: VIDEO ANALYSIS
# ══════════════════════════════════════════════════════════════

def make_motion_timeline(video_data, side):
    """Motion intensity over time from video frame differencing."""
    if not video_data:
        return _empty_fig('Data 없음')

    frames = video_data['frames']
    times = [f['time'] for f in frames]
    motions = [f['motion_intensity'] for f in frames]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=motions, mode='lines',
        line=dict(color='#2b6cb0', width=1),
        fill='tozeroy', fillcolor='rgba(43,108,176,0.1)',
        hovertemplate='Time: %{x:.2f}s<br>Motion: %{y:.2f}<extra></extra>',
        name='Motion Intensity',
    ))

    if 'tap_peak_indices' in video_data:
        peak_times = [frames[p]['time'] for p in video_data['tap_peak_indices'] if p < len(frames)]
        peak_vals = [frames[p]['motion_intensity'] for p in video_data['tap_peak_indices'] if p < len(frames)]
        fig.add_trace(go.Scatter(
            x=peak_times, y=peak_vals,
            mode='markers', name='Detected Taps',
            marker=dict(size=5, color='#e53e3e', symbol='triangle-down'),
        ))

    side_label = 'Left' if side == 'left' else 'Right'
    fig.update_layout(
        title=dict(text=f'Motion Intensity — {side_label} Finger Tapping', font=dict(size=14)),
        xaxis=dict(title='Time (s)', gridcolor='#edf2f7', zeroline=False),
        yaxis=dict(title='Motion Intensity', gridcolor='#edf2f7', zeroline=False),
        legend=dict(orientation='h', y=-0.15),
    )
    return _apply_defaults(fig, height=350)


def make_lr_tapping_comparison(left_data, right_data):
    """Compare tapping motion between left and right."""
    fig = go.Figure()
    for data, side, color in [(left_data, 'Left', '#4299e1'), (right_data, 'Right', '#fc8181')]:
        if not data:
            continue
        frames = data['frames']
        fig.add_trace(go.Scatter(
            x=[f['time'] for f in frames],
            y=[f['motion_intensity'] for f in frames],
            mode='lines', line=dict(color=color, width=1.2),
            name=f'{side} Hand', opacity=0.8,
        ))
    fig.update_layout(
        title=dict(text='좌/우 Finger Tapping Motion 비교', font=dict(size=14)),
        xaxis=dict(title='Time (s)', gridcolor='#edf2f7', zeroline=False),
        yaxis=dict(title='Motion Intensity', gridcolor='#edf2f7', zeroline=False),
        legend=dict(orientation='h', y=-0.15),
    )
    return _apply_defaults(fig, height=380)


def make_multicam_comparison(all_cameras_data, side):
    """Compare motion across camera angles."""
    fig = go.Figure()
    cam_colors = ['#2b6cb0', '#e53e3e', '#38a169', '#805ad5', '#dd6b20', '#d69e2e']
    for i, (cam_name, cam_data) in enumerate(all_cameras_data.items()):
        frames = cam_data['frames']
        motions = [f['motion_intensity'] for f in frames]
        kernel = 5
        if len(motions) > kernel:
            motions = list(np.convolve(motions, np.ones(kernel)/kernel, mode='same'))
        fig.add_trace(go.Scatter(
            x=[f['time'] for f in frames], y=motions, mode='lines',
            line=dict(color=cam_colors[i % 6], width=1.5),
            name=cam_name.replace('.mp4', ''), opacity=0.8,
        ))
    side_label = 'Left' if side == 'left' else 'Right'
    fig.update_layout(
        title=dict(text=f'Multi-Camera Motion 비교 — {side_label} Hand', font=dict(size=14)),
        xaxis=dict(title='Time (s)', gridcolor='#edf2f7', zeroline=False),
        yaxis=dict(title='Motion Intensity (smoothed)', gridcolor='#edf2f7', zeroline=False),
        legend=dict(orientation='h', y=-0.18, font=dict(size=10)),
    )
    return _apply_defaults(fig, height=400)


def make_tapping_summary_chart(left_data, right_data):
    """Bar chart comparing tapping metrics L vs R."""
    metrics = ['추정 Tap 수', 'Mean Interval (s)', 'Interval CV']
    lv = [left_data.get('estimated_taps', 0), left_data.get('mean_tap_interval', 0),
          left_data.get('tap_interval_cv', 0)] if left_data else [0, 0, 0]
    rv = [right_data.get('estimated_taps', 0), right_data.get('mean_tap_interval', 0),
          right_data.get('tap_interval_cv', 0)] if right_data else [0, 0, 0]

    fig = make_subplots(rows=1, cols=3, subplot_titles=metrics, horizontal_spacing=0.15)
    for i in range(3):
        fmt = lambda v: f'{v:.3f}' if isinstance(v, float) and v < 10 else str(v)
        fig.add_trace(go.Bar(
            x=['Left', 'Right'], y=[lv[i], rv[i]],
            marker=dict(color=['#4299e1', '#fc8181']),
            text=[fmt(lv[i]), fmt(rv[i])], textposition='auto', showlegend=False,
        ), row=1, col=i+1)
    fig.update_layout(
        title=dict(text='Finger Tapping 정량 비교 (영상 기반)', font=dict(size=14)),
        height=350,
        plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=40, r=20, t=70, b=50),
        font=dict(family='-apple-system, Segoe UI, sans-serif', size=12, color='#2c3e50'),
    )
    return fig


def _peak_stats(signal, fps=80.0):
    arr = np.asarray(signal, dtype=np.float32)
    if arr.size < 3:
        return 0, 0.0
    thr = float(np.mean(arr) + np.std(arr))
    peaks = []
    for i in range(1, arr.size - 1):
        if arr[i] > thr and arr[i] > arr[i - 1] and arr[i] >= arr[i + 1]:
            peaks.append(i)
    if len(peaks) < 2:
        return len(peaks), 0.0
    intervals = np.diff(peaks) / max(float(fps), 1e-6)
    mean_int = float(np.mean(intervals))
    cv = float(np.std(intervals) / mean_int) if mean_int > 1e-8 else 0.0
    return len(peaks), cv


def make_video_feature_timeline(feature_df, video_type, camera=''):
    """Task-aware feature timeline for Video Analysis tab."""
    if feature_df is None or feature_df.empty:
        return _empty_fig('영상 feature 데이터 없음')

    fig = go.Figure()
    x = feature_df['frame_index'] if 'frame_index' in feature_df.columns else np.arange(len(feature_df))
    t = np.asarray(x, dtype=np.float32) / 80.0  # dataset default fps

    if video_type in ('toe_left', 'toe_right'):
        y_main = feature_df.get('toe_tapping_rate_proxy', pd.Series(np.zeros(len(feature_df))))
        y_l = feature_df.get('left_toe_speed', pd.Series(np.zeros(len(feature_df))))
        y_r = feature_df.get('right_toe_speed', pd.Series(np.zeros(len(feature_df))))
        y_asym = feature_df.get('toe_lr_asymmetry', pd.Series(np.zeros(len(feature_df))))

        fig.add_trace(go.Scatter(x=t, y=y_main, mode='lines', name='Toe tapping rate proxy',
                                 line=dict(color='#2b6cb0', width=1.8)))
        fig.add_trace(go.Scatter(x=t, y=y_l, mode='lines', name='Left toe speed',
                                 line=dict(color='#4299e1', width=1.1), opacity=0.8))
        fig.add_trace(go.Scatter(x=t, y=y_r, mode='lines', name='Right toe speed',
                                 line=dict(color='#fc8181', width=1.1), opacity=0.8))
        fig.add_trace(go.Scatter(x=t, y=y_asym, mode='lines', name='Toe L/R asymmetry',
                                 line=dict(color='#805ad5', width=1.1, dash='dot')))

        n_peaks, _ = _peak_stats(y_main, fps=80.0)
        mean_asym = float(np.nanmean(np.asarray(y_asym, dtype=np.float32))) if len(y_asym) else 0.0
        fig.update_layout(
            title=dict(text=f'Toe Tapping Timeline — {camera} (peaks≈{n_peaks}, asym={mean_asym:.3f})', font=dict(size=14)),
            xaxis=dict(title='Time (s)', gridcolor='#edf2f7'),
            yaxis=dict(title='Speed / Rate proxy', gridcolor='#edf2f7'),
            legend=dict(orientation='h', y=-0.18),
        )
        return _apply_defaults(fig, height=350)

    if video_type == 'resting':
        y_l = feature_df.get('left_tremor_amp_proxy', pd.Series(np.zeros(len(feature_df))))
        y_r = feature_df.get('right_tremor_amp_proxy', pd.Series(np.zeros(len(feature_df))))
        y_asym = feature_df.get('upper_limb_lr_asymmetry', pd.Series(np.zeros(len(feature_df))))

        fig.add_trace(go.Scatter(x=t, y=y_l, mode='lines', name='Left tremor proxy',
                                 line=dict(color='#4299e1', width=1.6)))
        fig.add_trace(go.Scatter(x=t, y=y_r, mode='lines', name='Right tremor proxy',
                                 line=dict(color='#fc8181', width=1.6)))
        fig.add_trace(go.Scatter(x=t, y=y_asym, mode='lines', name='Upper-limb asymmetry',
                                 line=dict(color='#805ad5', width=1.2, dash='dot')))
        fig.update_layout(
            title=dict(text=f'Resting Tremor Timeline — {camera}', font=dict(size=14)),
            xaxis=dict(title='Time (s)', gridcolor='#edf2f7'),
            yaxis=dict(title='Tremor / Asymmetry proxy', gridcolor='#edf2f7'),
            legend=dict(orientation='h', y=-0.18),
        )
        return _apply_defaults(fig, height=350)

    return _empty_fig('지원하지 않는 video type')


def make_video_lr_feature_comparison(primary_df, counterpart_df, video_type):
    """L/R comparison chart for task-aware video features."""
    fig = go.Figure()

    if primary_df is None or primary_df.empty:
        return _empty_fig('비교할 feature 데이터 없음')

    x1 = primary_df['frame_index'] if 'frame_index' in primary_df.columns else np.arange(len(primary_df))
    t1 = np.asarray(x1, dtype=np.float32) / 80.0

    if video_type in ('toe_left', 'toe_right'):
        y_l = primary_df.get('left_toe_speed', pd.Series(np.zeros(len(primary_df))))
        y_r = primary_df.get('right_toe_speed', pd.Series(np.zeros(len(primary_df))))
        y_l_v = primary_df.get('left_toe_vertical_delta', pd.Series(np.zeros(len(primary_df))))
        y_r_v = primary_df.get('right_toe_vertical_delta', pd.Series(np.zeros(len(primary_df))))
        fig.add_trace(go.Scatter(x=t1, y=y_l, mode='lines', name='Left toe speed',
                                 line=dict(color='#4299e1', width=1.6)))
        fig.add_trace(go.Scatter(x=t1, y=y_r, mode='lines', name='Right toe speed',
                                 line=dict(color='#fc8181', width=1.6)))
        fig.add_trace(go.Scatter(x=t1, y=y_l_v, mode='lines', name='Left toe vertical delta',
                                 line=dict(color='#2b6cb0', width=1.0, dash='dot'), opacity=0.7))
        fig.add_trace(go.Scatter(x=t1, y=y_r_v, mode='lines', name='Right toe vertical delta',
                                 line=dict(color='#c53030', width=1.0, dash='dot'), opacity=0.7))
        fig.update_layout(
            title=dict(text='Toe Tapping Bilateral Comparison (same task)', font=dict(size=14)),
            xaxis=dict(title='Time (s)', gridcolor='#edf2f7'),
            yaxis=dict(title='Toe kinematic proxies', gridcolor='#edf2f7'),
            legend=dict(orientation='h', y=-0.15),
        )
        return _apply_defaults(fig, height=380)

    # resting: both sides in one file
    y_l = primary_df.get('left_tremor_amp_proxy', pd.Series(np.zeros(len(primary_df))))
    y_r = primary_df.get('right_tremor_amp_proxy', pd.Series(np.zeros(len(primary_df))))
    fig.add_trace(go.Scatter(x=t1, y=y_l, mode='lines', name='Left tremor proxy',
                             line=dict(color='#4299e1', width=1.6)))
    fig.add_trace(go.Scatter(x=t1, y=y_r, mode='lines', name='Right tremor proxy',
                             line=dict(color='#fc8181', width=1.6)))
    fig.update_layout(
        title=dict(text='Resting Tremor L/R Comparison', font=dict(size=14)),
        xaxis=dict(title='Time (s)', gridcolor='#edf2f7'),
        yaxis=dict(title='Tremor proxy', gridcolor='#edf2f7'),
        legend=dict(orientation='h', y=-0.15),
    )
    return _apply_defaults(fig, height=380)


def summarize_video_task_metrics(primary_df, counterpart_df, video_type):
    """Return 4 card values: left_taps, right_taps, left_cv, right_cv."""
    if primary_df is None or primary_df.empty:
        return '—', '—', '—', '—'

    if video_type in ('toe_left', 'toe_right'):
        y_l = primary_df.get('left_toe_speed', pd.Series(np.zeros(len(primary_df))))
        y_r = primary_df.get('right_toe_speed', pd.Series(np.zeros(len(primary_df))))
        l_taps, l_cv = _peak_stats(y_l, fps=80.0)
        r_taps, r_cv = _peak_stats(y_r, fps=80.0)
        return str(l_taps), str(r_taps), f'{l_cv:.3f}', f'{r_cv:.3f}'

    y_l = primary_df.get('left_tremor_amp_proxy', pd.Series(np.zeros(len(primary_df))))
    y_r = primary_df.get('right_tremor_amp_proxy', pd.Series(np.zeros(len(primary_df))))
    l_taps, l_cv = _peak_stats(y_l, fps=80.0)
    r_taps, r_cv = _peak_stats(y_r, fps=80.0)
    return str(l_taps), str(r_taps), f'{l_cv:.3f}', f'{r_cv:.3f}'


def _intervals_from_signal(signal, fps=80.0):
    """Extract peak-to-peak intervals from a 1D signal."""
    arr = np.asarray(signal, dtype=np.float32)
    if arr.size < 5:
        return np.array([], dtype=np.float32)
    thr = float(np.mean(arr) + np.std(arr))
    peaks = []
    for i in range(1, arr.size - 1):
        if arr[i] > thr and arr[i] > arr[i - 1] and arr[i] >= arr[i + 1]:
            peaks.append(i)
    if len(peaks) < 2:
        return np.array([], dtype=np.float32)
    return np.diff(np.asarray(peaks, dtype=np.float32)) / max(float(fps), 1e-6)


def make_video_interval_distribution(feature_df, video_type):
    """Clinical rhythm view: inter-event interval distribution by side."""
    if feature_df is None or feature_df.empty:
        return _empty_fig('Interval 데이터 없음', height=320)

    if video_type in ('toe_left', 'toe_right'):
        left_sig = feature_df.get('left_toe_speed', pd.Series(np.zeros(len(feature_df))))
        right_sig = feature_df.get('right_toe_speed', pd.Series(np.zeros(len(feature_df))))
        title = 'Toe Tapping Inter-Tap Interval Distribution'
    elif video_type == 'resting':
        left_sig = feature_df.get('left_tremor_amp_proxy', pd.Series(np.zeros(len(feature_df))))
        right_sig = feature_df.get('right_tremor_amp_proxy', pd.Series(np.zeros(len(feature_df))))
        title = 'Resting Tremor Peak Interval Distribution'
    else:
        return _empty_fig('지원하지 않는 video type', height=320)

    li = _intervals_from_signal(left_sig, fps=80.0)
    ri = _intervals_from_signal(right_sig, fps=80.0)

    fig = go.Figure()
    fig.add_trace(go.Violin(
        y=li if li.size else [0.0],
        name='Left',
        box_visible=True,
        meanline_visible=True,
        line_color='#4299e1',
        fillcolor='rgba(66,153,225,0.35)',
        opacity=0.9,
    ))
    fig.add_trace(go.Violin(
        y=ri if ri.size else [0.0],
        name='Right',
        box_visible=True,
        meanline_visible=True,
        line_color='#fc8181',
        fillcolor='rgba(252,129,129,0.35)',
        opacity=0.9,
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        yaxis=dict(title='Interval (s)', gridcolor='#edf2f7'),
        xaxis=dict(title='Side'),
        violinmode='group',
        legend=dict(orientation='h', y=-0.2),
    )
    return _apply_defaults(fig, height=320)


def make_video_tremor_spectrogram(feature_df, video_type):
    """Time-frequency view for resting tremor (clinical readability)."""
    if feature_df is None or feature_df.empty:
        return _empty_fig('Spectrogram 데이터 없음', height=320)

    if video_type != 'resting':
        return _empty_fig('Resting & hand tremor task에서만 표시됩니다.', height=320)

    sig_l = np.asarray(feature_df.get('left_tremor_amp_proxy', pd.Series(np.zeros(len(feature_df)))), dtype=np.float32)
    sig_r = np.asarray(feature_df.get('right_tremor_amp_proxy', pd.Series(np.zeros(len(feature_df)))), dtype=np.float32)
    sig = 0.5 * (sig_l + sig_r)
    if sig.size < 64:
        return _empty_fig('Spectrogram을 그리기 위한 길이가 부족합니다.', height=320)

    fs = 80.0
    win = 128
    hop = 32
    freqs = np.fft.rfftfreq(win, d=1.0 / fs)
    mask = (freqs >= 1.0) & (freqs <= 12.0)
    bands = freqs[mask]
    cols = max(1, (sig.size - win) // hop + 1)
    spec = np.zeros((bands.size, cols), dtype=np.float32)
    times = np.zeros(cols, dtype=np.float32)
    window = np.hanning(win).astype(np.float32)

    for i in range(cols):
        s = i * hop
        e = s + win
        frame = sig[s:e]
        frame = frame - np.mean(frame)
        fft_pow = np.abs(np.fft.rfft(frame * window)) ** 2
        spec[:, i] = fft_pow[mask]
        times[i] = (s + e) / 2.0 / fs

    spec_log = np.log10(spec + 1e-8)
    fig = go.Figure(data=go.Heatmap(
        x=times,
        y=bands,
        z=spec_log,
        colorscale='Viridis',
        colorbar=dict(title='log10(power)'),
    ))
    fig.update_layout(
        title=dict(text='Resting Tremor Spectrogram (1-12Hz)', font=dict(size=14)),
        xaxis=dict(title='Time (s)', gridcolor='#edf2f7'),
        yaxis=dict(title='Frequency (Hz)', gridcolor='#edf2f7'),
    )
    return _apply_defaults(fig, height=320)


def make_video_symmetry_trend(feature_df, video_type):
    """Clinical asymmetry tracking over time."""
    if feature_df is None or feature_df.empty:
        return _empty_fig('Symmetry 데이터 없음', height=320)

    x = feature_df['frame_index'] if 'frame_index' in feature_df.columns else np.arange(len(feature_df))
    t = np.asarray(x, dtype=np.float32) / 80.0

    if video_type in ('toe_left', 'toe_right'):
        asym = np.asarray(feature_df.get('toe_lr_asymmetry', pd.Series(np.zeros(len(feature_df)))), dtype=np.float32)
        title = 'Toe Tapping Symmetry Trend'
    elif video_type == 'resting':
        asym = np.asarray(feature_df.get('upper_limb_lr_asymmetry', pd.Series(np.zeros(len(feature_df)))), dtype=np.float32)
        title = 'Resting Tremor Symmetry Trend'
    else:
        return _empty_fig('지원하지 않는 video type', height=320)

    if asym.size == 0:
        return _empty_fig('Symmetry 데이터 없음', height=320)

    win = 40
    if asym.size >= win:
        kernel = np.ones(win, dtype=np.float32) / float(win)
        trend = np.convolve(asym, kernel, mode='same')
    else:
        trend = asym

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=asym, mode='lines', name='Raw asymmetry',
        line=dict(color='rgba(128,90,213,0.35)', width=1.0),
    ))
    fig.add_trace(go.Scatter(
        x=t, y=trend, mode='lines', name='Smoothed trend',
        line=dict(color='#6b46c1', width=2.2),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis=dict(title='Time (s)', gridcolor='#edf2f7'),
        yaxis=dict(title='Asymmetry index', gridcolor='#edf2f7'),
        legend=dict(orientation='h', y=-0.2),
    )
    return _apply_defaults(fig, height=320)


# ══════════════════════════════════════════════════════════════
#  CLINICAL MOTOR EVENT LANDSCAPE
#  Movement pathology visualization — NOT raw signal analytics.
#  Transforms timeseries into clinically interpretable event layers.
# ══════════════════════════════════════════════════════════════

_LANDSCAPE_WEIGHTS = {
    'tremor': 1.5,
    'rhythm': 1.3,
    'amplitude': 1.0,
    'jerk': 1.0,
    'asymmetry': 1.2,
}


def _compute_windowed_features(sig, fs, win_sec=2.0):
    """Compute feature layers in sliding windows → temporal arrays."""
    from src.feature_engineering import (
        calc_tremor_power, calc_rhythm_irregularity, calc_mean_jerk, detect_peaks
    )
    n = len(sig)
    win = int(win_sec * fs)
    hop = win // 2
    n_frames = max(1, (n - win) // hop + 1)

    times = np.zeros(n_frames)
    tremor = np.zeros(n_frames)
    rhythm = np.zeros(n_frames)
    amplitude = np.zeros(n_frames)
    jerk = np.zeros(n_frames)
    tremor_freq_stability = np.zeros(n_frames)  # frequency consistency

    for i in range(n_frames):
        start = i * hop
        end = min(start + win, n)
        frame = sig[start:end]
        t_center = (start + end) / 2 / fs
        times[i] = t_center
        tremor[i] = calc_tremor_power(frame, fs)
        rhythm[i] = calc_rhythm_irregularity(frame, fs)
        amplitude[i] = np.sqrt(np.mean(frame ** 2))
        jerk[i] = calc_mean_jerk(frame, fs)

        # Tremor frequency stability: ratio of peak PSD in 4-6Hz vs 4-12Hz
        if len(frame) >= 16:
            frame_dc = frame - np.mean(frame)
            fft_v = np.abs(np.fft.rfft(frame_dc)) ** 2
            freqs = np.fft.rfftfreq(len(frame_dc), 1.0 / fs)
            band_4_12 = fft_v[(freqs >= 4) & (freqs <= 12)].sum()
            band_4_6 = fft_v[(freqs >= 4) & (freqs <= 6)].sum()
            tremor_freq_stability[i] = band_4_6 / band_4_12 if band_4_12 > 0 else 0

    return times, tremor, rhythm, amplitude, jerk, tremor_freq_stability


def _generate_clinical_events(times, tremor_n, rhythm_n, amp_n, asym_n, saliency):
    """Generate clinical event annotations from feature time series."""
    events = []
    n = min(len(times), len(saliency))

    for i in range(n):
        t = times[i]
        if saliency[i] < 0.4:
            continue
        # Determine dominant contributor
        contributors = {
            'tremor': tremor_n[i] if i < len(tremor_n) else 0,
            'rhythm': rhythm_n[i] if i < len(rhythm_n) else 0,
            'amplitude': amp_n[i] if i < len(amp_n) else 0,
            'asymmetry': asym_n[i] if i < len(asym_n) else 0,
        }
        top = sorted(contributors.items(), key=lambda x: -x[1])
        dominant = top[0][0]

        # Clinical interpretation
        if dominant == 'tremor' and contributors['tremor'] > 0.6:
            events.append((t, 'Possible tremor emergence', '#e53e3e'))
        elif dominant == 'rhythm' and contributors['rhythm'] > 0.6:
            events.append((t, 'Rhythm instability', '#dd6b20'))
        elif dominant == 'asymmetry' and contributors['asymmetry'] > 0.6:
            events.append((t, 'Motor asymmetry increase', '#805ad5'))
        elif dominant == 'amplitude' and contributors['amplitude'] > 0.7:
            events.append((t, 'High motor activity', '#2b6cb0'))
        elif saliency[i] > 0.7:
            events.append((t, 'Multiple abnormality convergence', '#c53030'))

    # Deduplicate nearby events (within 2s)
    filtered = []
    for ev in events:
        if not filtered or abs(ev[0] - filtered[-1][0]) > 2.0:
            filtered.append(ev)
    return filtered


def make_motor_landscape(tulip_id, task):
    """Clinical Motor Event Landscape — temporal pathology interpretation.

    NOT feature visualization. Movement pathology storytelling.
    Each layer narrates a different aspect of motor deterioration.
    """
    from src.data_loader import load_timeseries, _estimate_fs, TASK_LABELS_KR

    left_ts = load_timeseries(tulip_id, task, 'LeftWrist')
    right_ts = load_timeseries(tulip_id, task, 'RightWrist')

    if left_ts.empty and right_ts.empty:
        return _empty_fig('No sensor data for this task')

    # Compute features for both sides
    layers = {}
    for side, ts in [('L', left_ts), ('R', right_ts)]:
        if ts.empty:
            continue
        sig = ts['accel_mag'].values
        fs = _estimate_fs(ts)
        times, tremor, rhythm, amplitude, jerk_arr, freq_stab = _compute_windowed_features(sig, fs)
        layers[side] = {
            'times': times, 'tremor': tremor, 'rhythm': rhythm,
            'amplitude': amplitude, 'jerk': jerk_arr,
            'freq_stability': freq_stab, 'fs': fs,
        }

    if not layers:
        return _empty_fig('Feature computation failed')

    primary = layers.get('L', layers.get('R'))
    times = primary['times']
    n = len(times)

    # Bilateral amplitude asymmetry over time
    has_bilateral = 'L' in layers and 'R' in layers
    if has_bilateral:
        ml = min(len(layers['L']['amplitude']), len(layers['R']['amplitude']))
        l_amp = layers['L']['amplitude'][:ml]
        r_amp = layers['R']['amplitude'][:ml]
        mean_amp = (l_amp + r_amp) / 2
        mean_amp[mean_amp == 0] = 1e-8
        asym_temporal = np.abs(l_amp - r_amp) / mean_amp
        asym_times = times[:ml]
        # Determine dominant side
        dominant_side = 'Left' if np.mean(l_amp) > np.mean(r_amp) else 'Right'
    else:
        asym_temporal = np.zeros(n)
        asym_times = times
        dominant_side = 'N/A'

    # Normalize
    def _norm(arr):
        mx = arr.max()
        return arr / mx if mx > 0 else arr

    tremor_n = _norm(primary['tremor'])
    rhythm_n = _norm(primary['rhythm'])
    amp_n = _norm(primary['amplitude'])
    jerk_n = _norm(primary['jerk'])
    asym_n = _norm(asym_temporal)

    # Saliency with decomposition
    w = _LANDSCAPE_WEIGHTS
    ml_all = min(len(tremor_n), len(rhythm_n), len(amp_n), len(jerk_n), len(asym_n))
    s_tremor = w['tremor'] * tremor_n[:ml_all]
    s_rhythm = w['rhythm'] * rhythm_n[:ml_all]
    s_amp = w['amplitude'] * amp_n[:ml_all]
    s_jerk = w['jerk'] * jerk_n[:ml_all]
    s_asym = w['asymmetry'] * asym_n[:ml_all]
    saliency = s_tremor + s_rhythm + s_amp + s_jerk + s_asym
    sal_max = saliency.max()
    if sal_max > 0:
        saliency = saliency / sal_max
        s_tremor = s_tremor / sal_max
        s_rhythm = s_rhythm / sal_max
        s_amp = s_amp / sal_max
        s_jerk = s_jerk / sal_max
        s_asym = s_asym / sal_max
    sal_times = times[:ml_all]

    # Generate clinical event annotations
    events = _generate_clinical_events(sal_times, tremor_n[:ml_all],
                                        rhythm_n[:ml_all], amp_n[:ml_all],
                                        asym_n[:ml_all], saliency)

    # Build figure — no subplot_titles (we place titles manually at bottom)
    fig = make_subplots(
        rows=5, cols=1,
        row_heights=[0.25, 0.2, 0.2, 0.2, 0.15],
        shared_xaxes=True,
        vertical_spacing=0.06,
    )

    # ── Row 1: Saliency with decomposition hover ──
    colors = np.where(saliency > 0.7, '#e53e3e',
             np.where(saliency > 0.4, '#dd6b20', '#38a169'))

    # Build per-bar hover with decomposition
    hover_texts = []
    for i in range(ml_all):
        parts = [
            f'Tremor: {s_tremor[i]*100:.0f}%',
            f'Rhythm: {s_rhythm[i]*100:.0f}%',
            f'Amplitude: {s_amp[i]*100:.0f}%',
            f'Jerk: {s_jerk[i]*100:.0f}%',
            f'Asymmetry: {s_asym[i]*100:.0f}%',
        ]
        top_contrib = max(
            ('Tremor', s_tremor[i]), ('Rhythm', s_rhythm[i]),
            ('Amplitude', s_amp[i]), ('Jerk', s_jerk[i]),
            ('Asymmetry', s_asym[i]), key=lambda x: x[1]
        )
        hover_texts.append(
            f'<b>Saliency: {saliency[i]:.3f}</b> at {sal_times[i]:.1f}s<br><br>'
            f'<b>Decomposition:</b><br>'
            + '<br>'.join(parts)
            + f'<br><br><b>Primary driver: {top_contrib[0]}</b><br><br>'
            f'<i>Higher bars indicate segments with<br>'
            f'stronger motor abnormality evidence.<br>'
            f'This is NOT a diagnostic score.</i>'
        )

    fig.add_trace(go.Bar(
        x=sal_times, y=saliency,
        marker=dict(color=list(colors), opacity=0.85),
        showlegend=False,
        hovertemplate='%{customdata}<extra></extra>',
        customdata=hover_texts,
    ), row=1, col=1)
    fig.add_hline(y=0.7, line_dash='dot', line_color='#e53e3e', line_width=1, row=1, col=1)
    fig.add_hline(y=0.4, line_dash='dot', line_color='#dd6b20', line_width=1, row=1, col=1)

    # Clinical event annotations — smart collision avoidance
    # Sort events by time, then stagger only when events are close together
    if events:
        sorted_events = sorted(events, key=lambda e: e[0])
        prev_t = -999
        level = 0
        for t, label, color in sorted_events:
            # Reset level if far enough apart, else increment
            if t - prev_t > 3.0:
                level = 0
            else:
                level = (level + 1) % 3
            y_offset = -30 - level * 24
            fig.add_annotation(
                x=t, y=1.05, xref='x', yref='y domain',
                text=f'<b>{label}</b>', showarrow=True, arrowhead=2, arrowsize=0.8,
                arrowcolor=color, font=dict(size=8, color=color),
                bgcolor='rgba(255,255,255,0.92)', bordercolor=color,
                borderwidth=1, borderpad=3, ax=0, ay=y_offset,
                row=1, col=1,
            )
            prev_t = t

    # ── Row 2: Tremor Ribbon with frequency stability ──
    freq_stab = primary['freq_stability']
    # Color by frequency stability: stable 4-6Hz = deeper red
    tremor_colors = [
        f'rgba(229,62,62,{0.3 + 0.7*freq_stab[i]:.2f})' if i < len(freq_stab) else 'rgba(229,62,62,0.3)'
        for i in range(len(times))
    ]
    fig.add_trace(go.Scatter(
        x=times, y=primary['tremor'],
        mode='lines', fill='tozeroy',
        line=dict(color='#e53e3e', width=1.5),
        fillcolor='rgba(229,62,62,0.2)',
        showlegend=False,
        hovertemplate=[
            f'<b>Tremor Persistence</b><br>'
            f'Time: {times[i]:.1f}s<br>'
            f'Power: {primary["tremor"][i]:.4f}<br>'
            f'4-6Hz ratio: {freq_stab[i]*100:.0f}%<br><br>'
            f'<i>{"Frequency-stable (PD-like)" if freq_stab[i]>0.6 else "Broadband (less specific)"}<br>'
            f'Persistent 4-6Hz tremor = hallmark PD rest tremor<br>'
            f'Higher 4-6Hz ratio = more frequency-stable oscillation</i>'
            f'<extra></extra>'
            for i in range(len(times))
        ],
    ), row=2, col=1)
    if 'R' in layers:
        fig.add_trace(go.Scatter(
            x=layers['R']['times'], y=layers['R']['tremor'],
            mode='lines', line=dict(color='#fc8181', width=1, dash='dot'),
            showlegend=False, opacity=0.4,
            hoverinfo='skip',
        ), row=2, col=1)

    # ── Row 3: Timing Instability (renamed from Rhythm Stability) ──
    fig.add_trace(go.Scatter(
        x=times, y=primary['rhythm'],
        mode='lines', fill='tozeroy',
        line=dict(color='#dd6b20', width=1.5),
        fillcolor='rgba(221,107,32,0.2)',
        showlegend=False,
        hovertemplate=(
            '<b>Timing Instability</b><br>'
            'Time: %{x:.1f}s<br>'
            'CV: %{y:.4f}<br><br>'
            '<i>CV of inter-peak intervals in this window<br>'
            'High CV = irregular movement timing<br>'
            'Indicates rhythm breakdown / dysrhythmia<br>'
            'Clinically associated with bradykinesia</i>'
            '<extra></extra>'
        ),
    ), row=3, col=1)

    # ── Row 4: Amplitude with decay detection ──
    fig.add_trace(go.Scatter(
        x=times, y=primary['amplitude'],
        mode='lines+markers',
        line=dict(color='#2b6cb0', width=2),
        marker=dict(size=4, color='#2b6cb0'),
        showlegend=False,
        hovertemplate=(
            '<b>Movement Amplitude</b><br>'
            'Time: %{x:.1f}s<br>'
            'RMS: %{y:.4f} g<br><br>'
            '<i>RMS of accelerometer magnitude<br>'
            'Progressive decay = decrement sequence<br>'
            '(PD hallmark in repetitive movements)<br>'
            'Stable amplitude = preserved motor output</i>'
            '<extra></extra>'
        ),
    ), row=4, col=1)
    if len(times) > 2:
        z = np.polyfit(times, primary['amplitude'], 1)
        trend = np.polyval(z, times)
        slope_color = '#e53e3e' if z[0] < -0.001 else '#38a169'
        slope_text = 'Declining (possible decrement)' if z[0] < -0.001 else 'Stable/increasing'
        fig.add_trace(go.Scatter(
            x=times, y=trend, mode='lines',
            line=dict(color=slope_color, width=1.5, dash='dash'),
            showlegend=False,
            hovertemplate=(
                f'<b>Amplitude Trend</b><br>'
                f'Slope: {z[0]:.4f} g/s<br>'
                f'Interpretation: {slope_text}'
                f'<extra></extra>'
            ),
        ), row=4, col=1)

    # ── Row 5: Amplitude Asymmetry ──
    if len(asym_temporal) > 0:
        fig.add_trace(go.Scatter(
            x=asym_times, y=asym_temporal,
            mode='lines', fill='tozeroy',
            line=dict(color='#805ad5', width=1.5),
            fillcolor='rgba(128,90,213,0.2)',
            showlegend=False,
            hovertemplate=(
                '<b>Amplitude Asymmetry</b><br>'
                'Time: %{x:.1f}s<br>'
                'Index: %{y:.3f}<br><br>'
                '<i>|Left_RMS - Right_RMS| / mean(Left, Right)<br>'
                f'Dominant side: {dominant_side}<br>'
                'Persistent asymmetry suggests lateralized pathology<br>'
                'Transient peaks may indicate task-specific compensation</i>'
                '<extra></extra>'
            ),
        ), row=5, col=1)

    # Layout
    task_label = TASK_LABELS_KR.get(task, task)

    # Subplot title labels — rows 1-4 at bottom, row 5 at top (to avoid Time(s) overlap)
    _subplot_labels = [
        (1, 'Clinical Saliency Skyline — Movement Abnormality Intensity'),
        (2, 'Tremor Persistence (4-12 Hz power + frequency stability)'),
        (3, 'Timing Instability (rhythm irregularity)'),
        (4, 'Movement Amplitude (progressive decay detection)'),
    ]
    _yref_map = {1: 'y', 2: 'y2', 3: 'y3', 4: 'y4', 5: 'y5'}
    for row_num, label_text in _subplot_labels:
        yref = f'{_yref_map[row_num]} domain'
        fig.add_annotation(
            text=f'<b>{label_text}</b>',
            x=0.5, xref='paper',
            y=-0.02, yref=yref,
            showarrow=False,
            font=dict(size=10, color='#4a5568'),
            yanchor='top',
        )
    # Row 5 title at top-left to avoid clash with x-axis "Time (s)"
    fig.add_annotation(
        text=f'<b>Amplitude Asymmetry (|L-R|/mean, dominant: {dominant_side})</b>',
        x=0.0, xref='paper',
        y=1.0, yref='y5 domain',
        showarrow=False,
        font=dict(size=10, color='#4a5568'),
        xanchor='left', yanchor='bottom',
    )

    # Visual separator lines between subplots (horizontal rules)
    for row_num in [1, 2, 3, 4]:
        yref = f'{_yref_map[row_num]} domain'
        fig.add_shape(
            type='line', x0=0, x1=1, y0=-0.06, y1=-0.06,
            xref='paper', yref=yref,
            line=dict(color='#e2e8f0', width=1, dash='dot'),
        )

    fig.update_layout(
        title=dict(
            text=f'Clinical Motor Event Landscape — {task_label}',
            font=dict(size=15),
        ),
        height=900,
        plot_bgcolor='white', paper_bgcolor='white',
        font=dict(family='-apple-system, Segoe UI, sans-serif', size=11, color='#2c3e50'),
        margin=dict(l=60, r=20, t=100, b=45),
    )
    fig.update_xaxes(title_text='Time (s)', row=5, col=1)
    fig.update_yaxes(title_text='Saliency', row=1, col=1)
    fig.update_yaxes(title_text='Power', row=2, col=1)
    fig.update_yaxes(title_text='CV', row=3, col=1)
    fig.update_yaxes(title_text='RMS (g)', row=4, col=1)
    fig.update_yaxes(title_text='|L-R|/μ', row=5, col=1)

    # Add light background color alternation for row separation
    for row_num in [2, 4]:
        yref = _yref_map[row_num]
        fig.add_shape(
            type='rect', x0=0, x1=1, y0=0, y1=1,
            xref='paper', yref=f'{yref} domain',
            fillcolor='rgba(247,250,252,0.5)', line_width=0,
            layer='below',
        )

    return fig


def make_bilateral_phase_space(tulip_id, task):
    """Bilateral Phase Space — smoothed L vs R motor trajectory.

    Improvements:
    - 3s smoothing windows (reduced spaghetti)
    - Reference density clouds (PD/Healthy)
    - Clear start/end with directional interpretation
    """
    from src.data_loader import (
        load_timeseries, _estimate_fs, TASK_LABELS_KR,
        build_group_stats, get_group_label, load_patients,
    )

    left_ts = load_timeseries(tulip_id, task, 'LeftWrist')
    right_ts = load_timeseries(tulip_id, task, 'RightWrist')

    if left_ts.empty or right_ts.empty:
        return _empty_fig('Both L/R data required for phase space')

    # Smoothed windows (3s for readability)
    fs_l = _estimate_fs(left_ts)
    fs_r = _estimate_fs(right_ts)
    _, _, _, l_amp, _, _ = _compute_windowed_features(left_ts['accel_mag'].values, fs_l, win_sec=3.0)
    times, _, _, r_amp, _, _ = _compute_windowed_features(right_ts['accel_mag'].values, fs_r, win_sec=3.0)

    ml = min(len(l_amp), len(r_amp))
    l_amp, r_amp, times = l_amp[:ml], r_amp[:ml], times[:ml]

    fig = go.Figure()
    max_val = max(l_amp.max(), r_amp.max()) * 1.2

    # Reference density clouds from confirmed subjects
    gs = build_group_stats()
    patients = load_patients()
    cond_map = dict(zip(patients['tulip_id'], patients['condition']))
    task_gs = gs[gs.task == task]

    for group, color, fill in [('Healthy', '#38a169', 'rgba(56,161,105,0.08)'),
                                ('PD', '#e53e3e', 'rgba(229,62,62,0.08)')]:
        gd = task_gs[task_gs.group == group]
        left_vals = gd[gd.wrist == 'Left']['accel_rms'].values
        right_vals = gd[gd.wrist == 'Right']['accel_rms'].values
        if len(left_vals) > 1 and len(right_vals) > 1:
            l_mean, l_std = left_vals.mean(), left_vals.std()
            r_mean, r_std = right_vals.mean(), right_vals.std()
            # Ellipse approximation (±1SD)
            theta = np.linspace(0, 2 * np.pi, 50)
            ell_x = l_mean + l_std * np.cos(theta)
            ell_y = r_mean + r_std * np.sin(theta)
            fig.add_trace(go.Scatter(
                x=ell_x, y=ell_y, mode='lines', fill='toself',
                fillcolor=fill, line=dict(color=color, width=1, dash='dot'),
                name=f'{group} region (±1SD)', opacity=0.6,
                hoverinfo='skip',
            ))

    # Symmetry line
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val], mode='lines',
        line=dict(color='#e2e8f0', width=1, dash='dash'),
        name='Symmetry', hoverinfo='skip',
    ))

    # Smoothed trajectory with time color
    fig.add_trace(go.Scatter(
        x=l_amp, y=r_amp, mode='lines+markers',
        marker=dict(
            size=8, color=times, colorscale='Viridis',
            colorbar=dict(title='Time (s)', thickness=10, len=0.5),
            line=dict(width=1, color='white'),
        ),
        line=dict(color='rgba(128,90,213,0.4)', width=2),
        name='Movement trajectory',
        hovertemplate=(
            '<b>Bilateral State</b><br>'
            'Time: %{marker.color:.1f}s<br>'
            'Left energy: %{x:.4f} g<br>'
            'Right energy: %{y:.4f} g<br><br>'
            '<i>On diagonal = symmetric bilateral movement<br>'
            'Above = right-dominant, Below = left-dominant<br>'
            'Inside Healthy cloud = typical motor balance<br>'
            'Inside PD cloud = PD-like asymmetric pattern</i>'
            '<extra></extra>'
        ),
    ))

    # Start / End markers
    fig.add_trace(go.Scatter(
        x=[l_amp[0]], y=[r_amp[0]], mode='markers+text',
        marker=dict(size=14, color='#38a169', symbol='circle', line=dict(width=2, color='white')),
        text=['Start'], textposition='bottom right', textfont=dict(size=9, color='#38a169'),
        name='Start', showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=[l_amp[-1]], y=[r_amp[-1]], mode='markers+text',
        marker=dict(size=14, color='#e53e3e', symbol='square', line=dict(width=2, color='white')),
        text=['End'], textposition='top left', textfont=dict(size=9, color='#e53e3e'),
        name='End', showlegend=True,
    ))

    # Drift analysis
    drift_l = l_amp[-1] - l_amp[0]
    drift_r = r_amp[-1] - r_amp[0]
    if abs(drift_l - drift_r) > 0.01:
        drift_side = 'left-dominant' if drift_l > drift_r else 'right-dominant'
        fig.add_annotation(
            x=0.02, y=0.98, xref='paper', yref='paper',
            text=f'Trajectory drift: {drift_side}',
            showarrow=False, font=dict(size=10, color='#4a5568'),
            bgcolor='rgba(255,255,255,0.8)',
        )

    task_label = TASK_LABELS_KR.get(task, task)
    fig.update_layout(
        title=dict(text=f'Bilateral Phase Space — {task_label} (3s smoothed)',
                   font=dict(size=14)),
        xaxis=dict(title='Left Motor Energy (RMS g)', gridcolor='#edf2f7', range=[0, max_val]),
        yaxis=dict(title='Right Motor Energy (RMS g)', gridcolor='#edf2f7', range=[0, max_val]),
        legend=dict(orientation='h', y=-0.12, font=dict(size=10)),
    )
    return _apply_defaults(fig, height=450)


# ══════════════════════════════════════════════════════════════
#  LABEL-FREE MODE: 4 Core Visualizations
# ══════════════════════════════════════════════════════════════

FEATURE_LABELS = {
    'tremor_power': 'Tremor Power',
    'amplitude': 'Amplitude',
    'rhythm_irreg': 'Rhythm Irreg.',
    'jerk': 'Mean Jerk',
}


def make_bilateral_matrix(feature_df, tulip_id, tasks=None):
    """Task-Symptom Bilateral Matrix: tasks × 8 cols (4 features × L/R).

    Parameters
    ----------
    feature_df : DataFrame
        From build_feature_cache().
    tulip_id : str
        Selected subject.
    tasks : list or None
        Task list to display. Defaults to MATCHING_TASKS (aligned tasks only).
    """
    from src.data_loader import MATCHING_TASKS, TASK_LABELS_KR

    subj = feature_df[feature_df.tulip_id == tulip_id].copy()
    if subj.empty:
        return _empty_fig('Feature data 없음')

    features = ['tremor_power', 'amplitude', 'rhythm_irreg', 'jerk']
    if tasks is None:
        tasks = MATCHING_TASKS

    # Build matrix: tasks × (L_feat1, R_feat1, L_feat2, R_feat2, ...)
    col_labels = []
    for f in features:
        col_labels.append(f'L {FEATURE_LABELS[f]}')
        col_labels.append(f'R {FEATURE_LABELS[f]}')

    z_matrix = []
    task_labels = []
    for task in tasks:
        task_data = subj[subj.task == task]
        row = []
        for f in features:
            left_val = task_data[task_data.wrist == 'L'][f].values
            right_val = task_data[task_data.wrist == 'R'][f].values
            row.append(float(left_val[0]) if len(left_val) > 0 else 0.0)
            row.append(float(right_val[0]) if len(right_val) > 0 else 0.0)
        z_matrix.append(row)
        task_labels.append(TASK_LABELS_KR.get(task, task))

    z_arr = np.array(z_matrix)
    # Normalize each column for visualization
    for col_i in range(z_arr.shape[1]):
        col_max = z_arr[:, col_i].max()
        if col_max > 0:
            z_arr[:, col_i] = z_arr[:, col_i] / col_max

    fig = go.Figure(data=go.Heatmap(
        z=z_arr,
        x=col_labels,
        y=task_labels,
        colorscale='YlOrRd',
        zmin=0, zmax=1,
        text=np.round(z_arr, 2).astype(str),
        texttemplate='%{text}',
        textfont=dict(size=9),
        hovertemplate='Task: %{y}<br>Feature: %{x}<br>Normalized: %{z:.3f}<extra></extra>',
        colorbar=dict(title='Normalized', thickness=12),
    ))
    fig.update_layout(
        title=dict(text='Aligned Task Bilateral Matrix (Entrainment & Relaxed)',
                   font=dict(size=13)),
        xaxis=dict(tickangle=-45, tickfont=dict(size=9), side='bottom'),
        yaxis=dict(tickfont=dict(size=10), autorange='reversed'),
    )
    return _apply_defaults(fig, height=max(300, len(tasks) * 50 + 100))


def make_spectral_fingerprint(tulip_id, task, load_ts_fn=None):
    """Bilateral Spectral Fingerprint: L/R spectrogram with 4-12Hz highlight.

    Parameters
    ----------
    tulip_id : str
    task : str
    load_ts_fn : callable
        Function to load timeseries (default: data_loader.load_timeseries).
    """
    from src.data_loader import load_timeseries, _estimate_fs
    from src.feature_engineering import compute_stft

    if load_ts_fn is None:
        load_ts_fn = load_timeseries

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Left Spectrogram', 'Right Spectrogram',
                        'Left PSD', 'Right PSD'],
        vertical_spacing=0.18, horizontal_spacing=0.1,
    )

    for col_i, (wrist, color) in enumerate([('LeftWrist', '#4299e1'), ('RightWrist', '#fc8181')], 1):
        ts = load_ts_fn(tulip_id, task, wrist)
        if ts.empty:
            continue

        sig = ts['accel_mag'].values
        fs = _estimate_fs(ts)
        times, freqs, mag = compute_stft(sig, fs)

        # Spectrogram heatmap
        fig.add_trace(go.Heatmap(
            z=mag[:50, :],  # up to ~25 Hz
            x=times,
            y=freqs[:50],
            colorscale='Viridis',
            showscale=False,
            hovertemplate='Time: %{x:.2f}s<br>Freq: %{y:.1f}Hz<br>Mag: %{z:.2f}<extra></extra>',
        ), row=1, col=col_i)

        # 4-12Hz band highlight
        fig.add_hrect(y0=4, y1=12, line_width=0,
                      fillcolor='rgba(229,62,62,0.15)',
                      row=1, col=col_i)

        # PSD line (mean across time)
        psd = np.mean(mag ** 2, axis=1)
        freq_limit = min(50, len(freqs))
        fig.add_trace(go.Scatter(
            x=freqs[:freq_limit], y=psd[:freq_limit],
            mode='lines', line=dict(color=color, width=2),
            name=f'{"L" if col_i == 1 else "R"} PSD',
            showlegend=True,
        ), row=2, col=col_i)

        # Highlight tremor band on PSD
        tremor_mask = (freqs[:freq_limit] >= 4) & (freqs[:freq_limit] <= 12)
        tremor_freqs = freqs[:freq_limit][tremor_mask]
        tremor_psd = psd[:freq_limit][tremor_mask]
        if len(tremor_freqs) > 0:
            fig.add_trace(go.Scatter(
                x=tremor_freqs, y=tremor_psd,
                mode='lines', fill='tozeroy',
                fillcolor='rgba(229,62,62,0.2)',
                line=dict(color='#e53e3e', width=1),
                name=f'{"L" if col_i == 1 else "R"} 4-12Hz',
                showlegend=False,
            ), row=2, col=col_i)

    fig.update_layout(
        title=dict(text=f'Bilateral Spectral Fingerprint — {task}', font=dict(size=13)),
        legend=dict(orientation='h', y=-0.1, font=dict(size=10)),
    )
    fig.update_yaxes(title_text='Freq (Hz)', row=1, col=1)
    fig.update_yaxes(title_text='Power', row=2, col=1)
    fig.update_xaxes(title_text='Time (s)', row=1, col=1)
    fig.update_xaxes(title_text='Freq (Hz)', row=2, col=1)
    fig.update_xaxes(title_text='Freq (Hz)', row=2, col=2)
    return _apply_defaults(fig, height=550)


def make_rhythm_ladder(tulip_id, task, load_ts_fn=None):
    """Rhythm Instability Ladder: waveform + peaks + interval bar chart.

    Parameters
    ----------
    tulip_id : str
    task : str
    load_ts_fn : callable
    """
    from src.data_loader import load_timeseries, _estimate_fs
    from src.feature_engineering import detect_peaks, calc_rhythm_irregularity

    if load_ts_fn is None:
        load_ts_fn = load_timeseries

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['Waveform + Peak Detection', 'Inter-Peak Intervals'],
        vertical_spacing=0.2,
        row_heights=[0.55, 0.45],
    )

    # Use left wrist by default (dominant side for many tasks)
    ts = load_ts_fn(tulip_id, task, 'LeftWrist')
    wrist_label = 'Left'
    if ts.empty:
        ts = load_ts_fn(tulip_id, task, 'RightWrist')
        wrist_label = 'Right'
    if ts.empty:
        return _empty_fig('센서 데이터 없음')

    sig = ts['accel_mag'].values
    time = ts['time'].values
    fs = _estimate_fs(ts)

    # Detect peaks
    min_dist = max(int(fs * 0.15), 5)
    indices, amplitudes = detect_peaks(sig, min_dist=min_dist)

    # Waveform
    fig.add_trace(go.Scatter(
        x=time, y=sig, mode='lines',
        line=dict(color='#2b6cb0', width=1),
        name='Accel Mag', showlegend=True,
    ), row=1, col=1)

    # Peak markers
    if len(indices) > 0:
        fig.add_trace(go.Scatter(
            x=time[indices], y=amplitudes,
            mode='markers',
            marker=dict(size=8, color='#e53e3e', symbol='diamond'),
            name=f'Peaks ({len(indices)})',
        ), row=1, col=1)

    # Interval bar chart
    if len(indices) >= 2:
        intervals = np.diff(indices) / fs  # seconds
        mean_int = np.mean(intervals)
        std_int = np.std(intervals)
        cv = std_int / mean_int if mean_int > 0 else 0

        bar_x = list(range(1, len(intervals) + 1))
        fig.add_trace(go.Bar(
            x=bar_x, y=intervals,
            marker=dict(color='#4299e1', opacity=0.7),
            name='Interval',
            showlegend=False,
        ), row=2, col=1)

        # Mean line
        fig.add_hline(y=mean_int, line_dash='solid', line_color='#e53e3e',
                      annotation_text=f'Mean: {mean_int:.3f}s',
                      row=2, col=1)
        # ±1σ band
        fig.add_hrect(y0=mean_int - std_int, y1=mean_int + std_int,
                      fillcolor='rgba(229,62,62,0.1)', line_width=0,
                      row=2, col=1)

        # CV annotation
        fig.add_annotation(
            x=0.95, y=0.95, xref='x2 domain', yref='y2 domain',
            text=f'CV = {cv:.3f}', showarrow=False,
            font=dict(size=14, color='#e53e3e'),
            bgcolor='rgba(255,255,255,0.8)',
        )

    fig.update_layout(
        title=dict(text=f'Rhythm Instability — {task} ({wrist_label} Wrist)',
                   font=dict(size=13)),
        legend=dict(orientation='h', y=-0.12, font=dict(size=10)),
    )
    fig.update_xaxes(title_text='Time (s)', row=1, col=1)
    fig.update_xaxes(title_text='Interval #', row=2, col=1)
    fig.update_yaxes(title_text='Accel Mag (g)', row=1, col=1)
    fig.update_yaxes(title_text='Interval (s)', row=2, col=1)
    return _apply_defaults(fig, height=520)


def make_evidence_ribbon(feature_df, tulip_id):
    """Side-Aligned Evidence Ribbon: 4 features × 11 tasks, L/R paired bars.

    Parameters
    ----------
    feature_df : DataFrame
        From build_feature_cache().
    tulip_id : str
    """
    from src.data_loader import SENSOR_TASKS, TASK_LABELS_KR

    subj = feature_df[feature_df.tulip_id == tulip_id].copy()
    if subj.empty:
        return _empty_fig('Feature data 없음')

    features = ['tremor_power', 'amplitude', 'rhythm_irreg', 'jerk']
    tasks = SENSOR_TASKS
    task_labels = [TASK_LABELS_KR.get(t, t) for t in tasks]

    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=[FEATURE_LABELS[f] for f in features],
        vertical_spacing=0.08,
        shared_xaxes=True,
    )

    for row_i, feat in enumerate(features, 1):
        left_vals = []
        right_vals = []
        for task in tasks:
            td = subj[subj.task == task]
            lv = td[td.wrist == 'L'][feat].values
            rv = td[td.wrist == 'R'][feat].values
            left_vals.append(float(lv[0]) if len(lv) > 0 else 0)
            right_vals.append(float(rv[0]) if len(rv) > 0 else 0)

        fig.add_trace(go.Bar(
            x=task_labels, y=left_vals, name='Left' if row_i == 1 else None,
            marker=dict(color='#4299e1', opacity=0.8),
            showlegend=(row_i == 1), legendgroup='left',
        ), row=row_i, col=1)
        fig.add_trace(go.Bar(
            x=task_labels, y=right_vals, name='Right' if row_i == 1 else None,
            marker=dict(color='#fc8181', opacity=0.8),
            showlegend=(row_i == 1), legendgroup='right',
        ), row=row_i, col=1)

    fig.update_layout(
        title=dict(text='Evidence Ribbon — All Tasks (Matched Sensor Analog)',
                   font=dict(size=13)),
        barmode='group',
        legend=dict(orientation='h', y=-0.05, font=dict(size=11)),
        height=700,
        **LAYOUT_DEFAULTS,
    )
    fig.update_xaxes(tickangle=-30, tickfont=dict(size=9), row=4, col=1)
    for i in range(1, 4):
        fig.update_xaxes(showticklabels=False, row=i, col=1)
    return fig


# ══════════════════════════════════════════════════════════════
#  MOTOR PHENOTYPE PROXIMITY — Reference Cohort Similarity
#  NOT a diagnostic classifier. Measures centroid-relative
#  similarity to confirmed PD/Healthy reference cohorts.
# ══════════════════════════════════════════════════════════════

GROUP_COLORS_EXT = {
    'PD': '#e53e3e', 'Healthy': '#38a169', 'Other': '#d69e2e', 'New': '#805ad5',
}


def _interpret_proximity(score, d_pd, d_healthy):
    """Interpret proximity as ambiguous zone vs directional."""
    diff_ratio = abs(d_pd - d_healthy) / ((d_pd + d_healthy) / 2) if (d_pd + d_healthy) > 0 else 0
    if diff_ratio < 0.15:
        return 'Ambiguous proximity zone (equidistant)', '#805ad5', 'ambiguous'
    elif score > 65:
        return 'PD-like motor phenotype', '#e53e3e', 'pd_like'
    elif score < 35:
        return 'Healthy-like motor phenotype', '#38a169', 'healthy_like'
    elif score > 50:
        return 'Mildly PD-leaning phenotype', '#dd6b20', 'mild_pd'
    else:
        return 'Mildly Healthy-leaning phenotype', '#68d391', 'mild_healthy'


def _calc_proximity_score(patient_val, healthy_mean, pd_mean):
    """Fallback linear proximity for per-feature display only."""
    if pd_mean == healthy_mean:
        return 50.0
    score = (patient_val - healthy_mean) / (pd_mean - healthy_mean) * 100
    return max(0, min(100, score))


def _percentile_position(value, group_values):
    """Calculate percentile position of value within group (Revision 8)."""
    if len(group_values) == 0:
        return 50.0
    return float(np.sum(group_values <= value) / len(group_values) * 100)


def make_proximity_gauge(group_stats_df, feature_df, highlight_tulip=None):
    """Motor Phenotype Proximity — Reference Cohort Similarity.

    This is NOT a diagnostic probability. It measures how similar this
    analog's motor characteristics are to confirmed reference cohorts.

    Method (16D weighted Euclidean, alignment-consistent):
    - Vector: 2 tasks (Entrainment+Relaxed) × 2 wrists × 4 features = 16D
    - Weights: tremor=1.5, rhythm=1.3, amplitude=1.0, jerk=1.0
    - Normalization: z-score (confirmed PD+Healthy reference)
    - Distance: weighted Euclidean to PD/Healthy centroids
    - Score: d_Healthy / (d_Healthy + d_PD) × 100
    """
    if not highlight_tulip:
        return _empty_fig('Select a patient')

    from src.data_loader import compute_proximity_scores, MATCHING_FEATURES, MATCHING_TASKS

    prox_data = compute_proximity_scores()
    if highlight_tulip not in prox_data:
        return _empty_fig('Proximity computation unavailable')

    pat = prox_data[highlight_tulip]
    overall = pat['score']
    d_pd = pat['d_pd']
    d_healthy = pat['d_healthy']
    n_pd = pat['n_pd']
    n_healthy = pat['n_healthy']
    per_task = pat.get('per_task', {})
    vec_dim = pat.get('vector_dim', 16)

    interpretation, interp_color, _ = _interpret_proximity(overall, d_pd, d_healthy)

    # Cohort stability warning (Revision 5)
    if n_pd < 5 or n_healthy < 5:
        stability = 'Exploratory only (n<5)'
    elif n_pd < 10 or n_healthy < 10:
        stability = 'Low reference stability (n<10)'
    else:
        stability = 'Adequate reference'

    fig = go.Figure()

    # Background zones
    fig.add_vrect(x0=0, x1=35, fillcolor='rgba(56,161,105,0.06)', line_width=0)
    fig.add_vrect(x0=35, x1=65, fillcolor='rgba(128,90,213,0.04)', line_width=0)
    fig.add_vrect(x0=65, x1=100, fillcolor='rgba(229,62,62,0.06)', line_width=0)

    # Overall bar
    fig.add_trace(go.Bar(
        x=[overall], y=['Overall'], orientation='h',
        marker=dict(color=interp_color),
        showlegend=False, base=0, width=0.6,
        text=f'{overall:.1f}%', textposition='inside',
        textfont=dict(size=15, color='white'),
        hovertemplate=(
            f'<b>Motor Phenotype Proximity: {overall:.1f}%</b><br><br>'
            f'Interpretation: {interpretation}<br><br>'
            f'<b>Method:</b><br>'
            f'• {vec_dim}D vector: {len(MATCHING_TASKS)} tasks × 2 wrists × {len(MATCHING_FEATURES)} features<br>'
            f'• Tasks: Entrainment (rhythm proxy) + Relaxed (rest tremor proxy)<br>'
            f'• Features: tremor(w=1.5), amp(w=1.0), rhythm(w=1.3), jerk(w=1.0)<br>'
            f'• Normalization: z-score on confirmed reference cohort<br>'
            f'• Distance: weighted Euclidean to group centroids<br>'
            f'• Score = d(Healthy) / (d(Healthy)+d(PD)) × 100<br><br>'
            f'd(PD centroid) = {d_pd:.3f}<br>'
            f'd(Healthy centroid) = {d_healthy:.3f}<br>'
            f'Difference: {abs(d_pd-d_healthy):.3f}<br><br>'
            f'<b>Reference cohort:</b> PD n={n_pd}, Healthy n={n_healthy}<br>'
            f'Stability: {stability}<br><br>'
            f'<i>This is NOT a diagnostic probability.<br>'
            f'It measures reference cohort similarity only.</i>'
            f'<extra></extra>'
        ),
    ))

    # Per-task breakdown
    task_display = {'Entrainment': 'Entrainment (rhythm/bradykinesia)', 'Relaxed': 'Relaxed (rest tremor)'}
    for task in MATCHING_TASKS:
        task_feats = per_task.get(task, {})
        task_score = np.mean(list(task_feats.values())) if task_feats else 50
        color = '#e53e3e' if task_score > 65 else '#38a169' if task_score < 35 else '#805ad5'
        label = task_display.get(task, task)
        fig.add_trace(go.Bar(
            x=[task_score], y=[label], orientation='h',
            marker=dict(color=color, opacity=0.7),
            showlegend=False,
            text=f'{task_score:.0f}%', textposition='inside',
            textfont=dict(size=11, color='white'),
            hovertemplate=(
                f'<b>{task}: {task_score:.1f}%</b><br><br>'
                f'Task meaning: {"rest tremor proxy" if task=="Relaxed" else "rhythm/bradykinesia proxy"}<br><br>'
                f'Per-feature breakdown:<br>'
                + '<br>'.join([f'  {f}: {task_feats.get(f, 50):.1f}%' for f in MATCHING_FEATURES])
                + f'<br><br>Uses weighted Euclidean (task-specific slice of 16D vector)'
                f'<extra></extra>'
            ),
        ))

    # Zone labels and lines
    fig.add_vline(x=35, line_dash='dot', line_color='#a0aec0', line_width=1)
    fig.add_vline(x=65, line_dash='dot', line_color='#a0aec0', line_width=1)

    fig.update_layout(
        title=dict(
            text=(f'<b>Motor Phenotype Proximity: {interpretation}</b><br>'
                  f'<span style="font-size:11px;color:#718096">'
                  f'Reference: PD n={n_pd}, Healthy n={n_healthy} | {stability} | '
                  f'16D weighted Euclidean | d_PD={d_pd:.2f}, d_H={d_healthy:.2f}</span>'),
            font=dict(size=13),
        ),
        xaxis=dict(
            range=[0, 100], gridcolor='#edf2f7', dtick=25,
            title=None, side='bottom',
        ),
        yaxis=dict(autorange='reversed'),
        height=300,
        plot_bgcolor='white', paper_bgcolor='white',
        font=dict(family='-apple-system, Segoe UI, sans-serif', size=12, color='#2c3e50'),
        margin=dict(l=200, r=30, t=100, b=40),
    )
    # Zone labels at TOP of chart — above the bars, below the title
    for x_pos, label, color in [
        (17.5, 'Healthy-like', '#38a169'),
        (50, 'Ambiguous', '#805ad5'),
        (82.5, 'PD-like', '#e53e3e'),
    ]:
        fig.add_annotation(
            x=x_pos, y=1.08, xref='x', yref='paper',
            text=f'<b>{label}</b>', showarrow=False,
            font=dict(size=11, color=color),
        )
    return fig


def make_new_vs_group_box(group_stats_df, task, metric='accel_rms', highlight_tulip=None):
    """Distribution position with percentile interpretation (Revision 8).

    Shows violin distribution + patient percentile position within each group.
    Uses task-specific default metric when available (Revision 7).
    """
    from src.data_loader import TASK_DEFAULT_METRICS, MATCHING_FEATURES

    df = group_stats_df[group_stats_df.task == task].copy()
    if df.empty:
        return _empty_fig('No data')

    metric_labels = {
        'accel_rms': 'Accelerometer RMS (g)',
        'gyro_rms': 'Gyroscope RMS (rad/s)',
        'accel_std': 'Accelerometer Std',
        'gyro_std': 'Gyroscope Std',
    }

    fig = go.Figure()

    h_vals = df[df.group == 'Healthy'][metric].values
    p_vals = df[df.group == 'PD'][metric].values

    for group in ['Healthy', 'PD']:
        gd = df[df.group == group]
        if gd.empty:
            continue
        color = GROUP_COLORS_EXT[group]
        n = len(gd) // 2  # L+R counted separately
        fig.add_trace(go.Violin(
            y=gd[metric].values, name=f'{group} (n={n})',
            marker=dict(color=color, size=5),
            line=dict(color=color),
            box_visible=True, meanline_visible=True,
            points='all', jitter=0.3,
            side='both', opacity=0.7,
            hovertemplate=(
                f'{group} reference<br>'
                f'n={n} subjects<br>'
                f'Value: %{{y:.4f}}<br>'
                f'<extra></extra>'
            ),
        ))

    # Patient position with percentile (Revision 8)
    if highlight_tulip:
        pat_data = df[df.tulip_id == highlight_tulip]
        if not pat_data.empty:
            pat_val = pat_data[metric].mean()
            pct_h = _percentile_position(pat_val, h_vals)
            pct_p = _percentile_position(pat_val, p_vals)

            fig.add_trace(go.Scatter(
                x=[f'Healthy (n={len(h_vals)//2})', f'PD (n={len(p_vals)//2})'],
                y=[pat_val, pat_val],
                mode='markers+lines',
                marker=dict(size=16, color='#805ad5', symbol='star',
                            line=dict(width=2, color='white')),
                line=dict(color='#805ad5', width=2, dash='dash'),
                name=f'★ {_display_id(highlight_tulip)}',
                hovertemplate=(
                    f'<b>★ {highlight_tulip}</b><br>'
                    f'Value: {pat_val:.4f}<br><br>'
                    f'Percentile in Healthy: {pct_h:.0f}th<br>'
                    f'Percentile in PD: {pct_p:.0f}th<br><br>'
                    f'<i>Percentile = % of reference group<br>'
                    f'with equal or lower values</i>'
                    f'<extra></extra>'
                ),
            ))

            # Annotation: percentile-based (Revision 8, safer language Revision 9)
            if pct_h > 95:
                annot = 'Outside typical Healthy range (>95th pct)'
                annot_color = '#e53e3e'
            elif pct_h < 50 and pct_p < 50:
                annot = 'Below both group medians'
                annot_color = '#38a169'
            elif pct_p > 50 and pct_h > 50:
                annot = 'Overlapping zone (within both distributions)'
                annot_color = '#805ad5'
            else:
                annot = f'H:{pct_h:.0f}th pct | PD:{pct_p:.0f}th pct'
                annot_color = '#4a5568'

            fig.add_annotation(
                x=0.5, y=1.05, xref='paper', yref='paper',
                text=f'<b>{annot}</b>',
                showarrow=False,
                font=dict(size=11, color=annot_color),
            )

    fig.update_layout(
        title=dict(text=f'Reference Distribution Position — {task}', font=dict(size=13)),
        yaxis=dict(title=metric_labels.get(metric, metric), gridcolor='#edf2f7'),
        legend=dict(orientation='h', y=-0.15),
    )
    return _apply_defaults(fig, height=430)


def make_group_feature_comparison(feature_df, group_stats_df, highlight_tulip=None):
    """Task-aware feature comparison (Revision 6).

    Groups tasks by clinical meaning instead of blind averaging.
    """
    from src.data_loader import TASK_GROUPS, TASK_DEFAULT_METRICS

    if feature_df.empty or not highlight_tulip:
        return _empty_fig('Select a patient')

    group_map = group_stats_df[['tulip_id', 'group']].drop_duplicates()
    merged = feature_df.merge(group_map, on='tulip_id', how='left')

    # Task-aware grouping (Revision 6)
    task_group_labels = ['Rest\n(tremor)', 'Repetitive\n(rhythm)', 'Intentional\n(action)', 'Postural\n(hold)']
    task_group_keys = ['rest', 'repetitive', 'intentional', 'postural']

    fig = make_subplots(rows=1, cols=4, subplot_titles=task_group_labels,
                        horizontal_spacing=0.08)

    for i, (tg_key, tg_label) in enumerate(zip(task_group_keys, task_group_labels), 1):
        tasks_in_group = TASK_GROUPS.get(tg_key, [])
        tg_data = merged[merged.task.isin(tasks_in_group)]

        # Use task-appropriate metric (Revision 7)
        primary_feat = 'tremor_power' if tg_key == 'rest' else (
            'rhythm_irreg' if tg_key == 'repetitive' else 'amplitude')

        for group in ['Healthy', 'PD']:
            gd = tg_data[tg_data.group == group]
            mean_val = gd[primary_feat].mean() if not gd.empty else 0
            std_val = gd[primary_feat].std() if not gd.empty else 0
            color = GROUP_COLORS_EXT[group]
            fig.add_trace(go.Bar(
                x=[group], y=[mean_val],
                error_y=dict(type='data', array=[std_val], visible=True),
                marker=dict(color=color, opacity=0.7),
                name=group if i == 1 else None,
                showlegend=(i == 1), legendgroup=group,
                hovertemplate=(
                    f'{group}<br>'
                    f'Metric: {FEATURE_LABELS.get(primary_feat, primary_feat)}<br>'
                    f'Tasks: {", ".join(tasks_in_group)}<br>'
                    f'Mean: %{{y:.4f}} ± {std_val:.4f}<br>'
                    f'<extra></extra>'
                ),
            ), row=1, col=i)

        # Patient value
        pat_data = tg_data[tg_data.tulip_id == highlight_tulip]
        if not pat_data.empty:
            pat_val = pat_data[primary_feat].mean()
            fig.add_trace(go.Scatter(
                x=['Healthy', 'PD'], y=[pat_val, pat_val],
                mode='lines+markers',
                line=dict(color='#805ad5', width=2, dash='dash'),
                marker=dict(size=10, symbol='star', color='#805ad5'),
                name=f'★ {_display_id(highlight_tulip)}' if i == 1 else None,
                showlegend=(i == 1), legendgroup='patient',
                hovertemplate=(
                    f'★ {highlight_tulip}<br>'
                    f'{FEATURE_LABELS.get(primary_feat, primary_feat)}: {pat_val:.4f}<br>'
                    f'<i>Task group: {tg_key}<br>'
                    f'Default metric chosen by clinical relevance</i>'
                    f'<extra></extra>'
                ),
            ), row=1, col=i)

    fig.update_layout(
        title=dict(text='Task-Grouped Feature Comparison (clinically-aligned metrics)',
                   font=dict(size=13)),
        barmode='group', legend=dict(orientation='h', y=-0.15),
        height=380, **LAYOUT_DEFAULTS,
    )
    return fig


def make_task_profile_comparison(group_stats_df, highlight_tulip=None):
    """Task-wise profile with task-specific metrics (Revision 7).

    Layer A (Entrainment+Relaxed): alignment evidence
    Layer B (other tasks): hypothetical multimodal expansion
    """
    from src.data_loader import SENSOR_TASKS, TASK_LABELS_KR, TASK_DEFAULT_METRICS, MATCHING_TASKS

    df = group_stats_df.copy()
    if df.empty:
        return _empty_fig('No data')

    # Alignment tasks first, then others alphabetically
    alignment = [t for t in MATCHING_TASKS if t in SENSOR_TASKS]
    others = [t for t in SENSOR_TASKS if t not in MATCHING_TASKS]
    tasks = alignment + others
    task_labels = [TASK_LABELS_KR.get(t, t) for t in tasks]

    fig = go.Figure()

    group_data = {}
    for group, color in [('Healthy', '#38a169'), ('PD', '#e53e3e')]:
        gd = df[df.group == group]
        means, stds = [], []
        for task in tasks:
            td = gd[gd.task == task]['accel_rms']
            means.append(td.mean() if len(td) > 0 else 0)
            stds.append(td.std() if len(td) > 0 else 0)
        means = np.array(means)
        stds = np.array(stds)
        group_data[group] = means

        # Band
        fig.add_trace(go.Scatter(
            x=task_labels + task_labels[::-1],
            y=list(means + stds) + list((means - stds)[::-1]),
            fill='toself',
            fillcolor=f'rgba({",".join(str(int(color[i:i+2], 16)) for i in (1,3,5))},0.1)',
            line=dict(width=0), showlegend=False, hoverinfo='skip',
        ))
        fig.add_trace(go.Scatter(
            x=task_labels, y=means, mode='lines+markers',
            line=dict(color=color, width=2),
            marker=dict(size=5, color=color),
            name=f'{group} (mean±SD)',
        ))

    # Patient overlay
    if highlight_tulip:
        new_data = df[df.tulip_id == highlight_tulip]
        if not new_data.empty:
            new_vals = []
            for task in tasks:
                td = new_data[new_data.task == task]['accel_rms']
                new_vals.append(td.mean() if len(td) > 0 else 0)
            fig.add_trace(go.Scatter(
                x=task_labels, y=new_vals,
                mode='lines+markers',
                line=dict(color='#805ad5', width=3),
                marker=dict(size=11, symbol='star', color='#805ad5',
                            line=dict(width=2, color='white')),
                name=f'★ {_display_id(highlight_tulip)}',
            ))

    fig.update_layout(
        title=dict(text='Movement Profile — Reference Comparison', font=dict(size=13)),
        xaxis=dict(tickangle=-30, tickfont=dict(size=10)),
        yaxis=dict(title='Accel RMS (g)', gridcolor='#edf2f7'),
        legend=dict(orientation='h', y=-0.25, font=dict(size=11)),
        margin=dict(b=80),
    )
    return _apply_defaults(fig, height=480)


def make_asymmetry_scatter(group_stats_df, task, highlight_tulip=None):
    """Bilateral scatter with task-specific metric (Revision 10).

    Uses clinically relevant bilateral axis depending on task type.
    """
    from src.data_loader import TASK_DEFAULT_METRICS, MATCHING_FEATURES, build_feature_cache

    # Determine task-appropriate feature for bilateral comparison (Revision 10)
    default_feat = TASK_DEFAULT_METRICS.get(task, 'amplitude')
    fc = build_feature_cache()

    task_data = fc[fc.task == task].copy()
    if task_data.empty:
        return _empty_fig('No data')

    # Merge group info
    group_map = group_stats_df[['tulip_id', 'group']].drop_duplicates()
    task_data = task_data.merge(group_map, on='tulip_id', how='left')

    # Pivot L/R
    left = task_data[task_data.wrist == 'L'][['tulip_id', 'group', default_feat]].rename(
        columns={default_feat: 'left_val'})
    right = task_data[task_data.wrist == 'R'][['tulip_id', default_feat]].rename(
        columns={default_feat: 'right_val'})
    merged = left.merge(right, on='tulip_id', how='inner')

    if merged.empty:
        return _empty_fig('Insufficient bilateral data')

    fig = go.Figure()
    max_val = max(merged['left_val'].max(), merged['right_val'].max()) * 1.1

    # Symmetry line
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val], mode='lines',
        line=dict(color='#cbd5e0', width=1, dash='dash'),
        name='Perfect Symmetry', hoverinfo='skip',
    ))

    # Confirmed groups
    for group in ['Healthy', 'PD']:
        gd = merged[(merged.group == group) & (merged.tulip_id != highlight_tulip)]
        if gd.empty:
            continue
        color = GROUP_COLORS_EXT[group]
        fig.add_trace(go.Scatter(
            x=gd['left_val'], y=gd['right_val'],
            mode='markers+text',
            marker=dict(size=9, color=color, opacity=0.7,
                        line=dict(width=1, color='white')),
            name=f'{group} ref.',
            hovertemplate='%{text}<br>L: %{x:.4f}<br>R: %{y:.4f}<extra></extra>',
            text=gd['tulip_id'].apply(lambda x: 'P_' + x.replace('TULIP_', '')),
            textposition='top center', textfont=dict(size=8, color=color),
        ))

    # Selected patient
    if highlight_tulip:
        pat = merged[merged.tulip_id == highlight_tulip]
        if not pat.empty:
            l_val = pat['left_val'].values[0]
            r_val = pat['right_val'].values[0]
            from src.feature_engineering import calc_asymmetry_index
            asym_idx = calc_asymmetry_index(l_val, r_val)

            pat_label = 'Patient_' + highlight_tulip.replace('TULIP_', '')
            fig.add_trace(go.Scatter(
                x=[l_val], y=[r_val],
                mode='markers+text',
                marker=dict(size=18, color='#805ad5', symbol='star',
                            line=dict(width=3, color='white')),
                text=[f'★ {pat_label}'],
                textposition='top center',
                textfont=dict(size=10, color='#805ad5'),
                name=f'★ {pat_label}',
                hovertemplate=(
                    f'<b>★ {pat_label}</b><br>'
                    f'Left: {l_val:.4f}<br>'
                    f'Right: {r_val:.4f}<br>'
                    f'Asymmetry Index: {asym_idx:.3f}<br><br>'
                    f'<i>Metric: {FEATURE_LABELS.get(default_feat, default_feat)}<br>'
                    f'Task: {task}<br>'
                    f'Higher asymmetry may indicate lateralized motor involvement</i>'
                    f'<extra></extra>'
                ),
            ))

            fig.add_annotation(
                x=0.95, y=0.05, xref='paper', yref='paper',
                text=f'Asymmetry: {asym_idx:.3f}',
                showarrow=False, font=dict(size=12, color='#805ad5'),
                bgcolor='rgba(255,255,255,0.8)',
            )

    feat_label = FEATURE_LABELS.get(default_feat, default_feat)
    fig.update_layout(
        title=dict(text=f'Bilateral Comparison — {task} ({feat_label})', font=dict(size=13)),
        xaxis=dict(title=f'Left {feat_label}', gridcolor='#edf2f7', range=[0, max_val]),
        yaxis=dict(title=f'Right {feat_label}', gridcolor='#edf2f7', range=[0, max_val]),
        legend=dict(orientation='h', y=-0.15),
    )
    return _apply_defaults(fig, height=420)


# ══════════════════════════════════════════════════════════════
#  BILATERAL ASYMMETRY TAB
# ══════════════════════════════════════════════════════════════

def make_asymmetry_heatmap(feature_df, group_stats_df, tulip_id):
    """Asymmetry Index heatmap per task × feature."""
    from src.data_loader import SENSOR_TASKS, TASK_LABELS_KR
    from src.feature_engineering import calc_asymmetry_index

    subj = feature_df[feature_df.tulip_id == tulip_id]
    if subj.empty:
        return _empty_fig('No data')

    features = ['tremor_power', 'amplitude', 'rhythm_irreg', 'jerk']
    tasks = SENSOR_TASKS
    task_labels = [TASK_LABELS_KR.get(t, t) for t in tasks]

    z_matrix = []
    for task in tasks:
        td = subj[subj.task == task]
        row = []
        for feat in features:
            lv = td[td.wrist == 'L'][feat].values
            rv = td[td.wrist == 'R'][feat].values
            l = float(lv[0]) if len(lv) > 0 else 0
            r = float(rv[0]) if len(rv) > 0 else 0
            mean_val = (l + r) / 2
            asym = abs(l - r) / mean_val if mean_val > 0 else 0
            row.append(asym)
        z_matrix.append(row)

    z_arr = np.array(z_matrix)
    feat_labels = [FEATURE_LABELS[f] for f in features]
    avg_asym = z_arr.mean()

    fig = go.Figure(data=go.Heatmap(
        z=z_arr, x=feat_labels, y=task_labels,
        colorscale='RdYlGn_r', zmin=0, zmax=1.0,
        text=np.round(z_arr, 3).astype(str),
        texttemplate='%{text}', textfont=dict(size=10),
        hovertemplate='Task: %{y}<br>Feature: %{x}<br>Asymmetry: %{z:.3f}<extra></extra>',
        colorbar=dict(title='Asymmetry', thickness=12),
    ))
    fig.update_layout(
        title=dict(text=f'Bilateral Asymmetry Index — Mean: {avg_asym:.3f}', font=dict(size=14)),
        xaxis=dict(tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=10), autorange='reversed'),
    )
    return _apply_defaults(fig, height=max(350, len(tasks) * 30 + 100))


def make_asym_waveform(tulip_id, task):
    """Left vs Right waveform overlay."""
    from src.data_loader import load_timeseries, TASK_LABELS_KR
    left_ts = load_timeseries(tulip_id, task, 'LeftWrist')
    right_ts = load_timeseries(tulip_id, task, 'RightWrist')
    if left_ts.empty and right_ts.empty:
        return _empty_fig('No sensor data')
    fig = go.Figure()
    if not left_ts.empty:
        fig.add_trace(go.Scatter(x=left_ts['time'], y=left_ts['accel_mag'],
                                 mode='lines', name='Left', line=dict(color='#4299e1', width=1.5)))
    if not right_ts.empty:
        fig.add_trace(go.Scatter(x=right_ts['time'], y=right_ts['accel_mag'],
                                 mode='lines', name='Right', line=dict(color='#fc8181', width=1.5)))
    fig.update_layout(
        title=dict(text=f'L/R Waveform — {TASK_LABELS_KR.get(task, task)}', font=dict(size=14)),
        xaxis=dict(title='Time (s)', gridcolor='#edf2f7'),
        yaxis=dict(title='Accel Magnitude (g)', gridcolor='#edf2f7'),
        legend=dict(orientation='h', y=-0.12),
    )
    return _apply_defaults(fig, height=350)


def make_asym_feature_bars(feature_df, tulip_id, task):
    """L vs R feature bars with asymmetry index."""
    from src.data_loader import TASK_LABELS_KR
    from src.feature_engineering import calc_asymmetry_index
    subj = feature_df[(feature_df.tulip_id == tulip_id) & (feature_df.task == task)]
    if subj.empty:
        return _empty_fig('No data')
    features = ['tremor_power', 'amplitude', 'rhythm_irreg', 'jerk']
    feat_labels = [FEATURE_LABELS[f] for f in features]
    left_vals, right_vals, asym_vals = [], [], []
    for feat in features:
        lv = subj[subj.wrist == 'L'][feat].values
        rv = subj[subj.wrist == 'R'][feat].values
        l = float(lv[0]) if len(lv) > 0 else 0
        r = float(rv[0]) if len(rv) > 0 else 0
        left_vals.append(l)
        right_vals.append(r)
        asym_vals.append(calc_asymmetry_index(l, r))
    fig = make_subplots(rows=1, cols=2, column_widths=[0.65, 0.35],
                        subplot_titles=['L/R Feature Values', 'Asymmetry Index'])
    fig.add_trace(go.Bar(x=feat_labels, y=left_vals, name='Left',
                         marker=dict(color='#4299e1')), row=1, col=1)
    fig.add_trace(go.Bar(x=feat_labels, y=right_vals, name='Right',
                         marker=dict(color='#fc8181')), row=1, col=1)
    colors = ['#38a169' if a < 0.3 else '#dd6b20' if a < 0.6 else '#e53e3e' for a in asym_vals]
    fig.add_trace(go.Bar(x=feat_labels, y=asym_vals, marker=dict(color=colors),
                         text=[f'{a:.2f}' for a in asym_vals], textposition='auto',
                         showlegend=False), row=1, col=2)
    fig.add_hline(y=0.3, line_dash='dash', line_color='#dd6b20', row=1, col=2)
    fig.update_layout(title=dict(text=f'Bilateral Features — {TASK_LABELS_KR.get(task, task)}',
                                 font=dict(size=14)),
                      barmode='group', legend=dict(orientation='h', y=-0.15))
    return _apply_defaults(fig, height=380)


def make_asym_group_compare(group_stats_df, tulip_id):
    """Patient asymmetry vs confirmed group distributions."""
    from src.data_loader import SENSOR_TASKS
    from src.feature_engineering import calc_asymmetry_index
    import pandas as pd
    df = group_stats_df.copy()
    if df.empty:
        return _empty_fig('No data')
    asym_data = []
    for tid in df['tulip_id'].unique():
        for task in SENSOR_TASKS:
            td = df[(df.tulip_id == tid) & (df.task == task)]
            left = td[td.wrist == 'Left']['accel_rms'].values
            right = td[td.wrist == 'Right']['accel_rms'].values
            if len(left) > 0 and len(right) > 0:
                asym_data.append({'tulip_id': tid, 'group': td.iloc[0]['group'],
                                  'asym': calc_asymmetry_index(left[0], right[0])})
    asym_df = pd.DataFrame(asym_data)
    if asym_df.empty:
        return _empty_fig('Insufficient data')
    subj_asym = asym_df.groupby(['tulip_id', 'group'])['asym'].mean().reset_index()
    fig = go.Figure()
    for group in ['Healthy', 'PD']:
        gd = subj_asym[subj_asym.group == group]
        if gd.empty:
            continue
        fig.add_trace(go.Box(y=gd['asym'].values, name=f'{group} (n={len(gd)})',
                             marker=dict(color=GROUP_COLORS_EXT[group], size=6),
                             line=dict(color=GROUP_COLORS_EXT[group]),
                             boxmean='sd', boxpoints='all', jitter=0.3))
    pat_asym = subj_asym[subj_asym.tulip_id == tulip_id]
    if not pat_asym.empty:
        val = pat_asym['asym'].values[0]
        fig.add_trace(go.Scatter(
            x=['Healthy (n={})'.format(len(subj_asym[subj_asym.group=='Healthy'])),
               'PD (n={})'.format(len(subj_asym[subj_asym.group=='PD']))],
            y=[val, val], mode='markers+lines',
            marker=dict(size=16, color='#805ad5', symbol='star', line=dict(width=2, color='white')),
            line=dict(color='#805ad5', width=2, dash='dash'),
            name=f'★ {_display_id(tulip_id)} ({val:.3f})',
        ))
    fig.update_layout(
        title=dict(text='Mean Asymmetry — Group Comparison', font=dict(size=14)),
        yaxis=dict(title='Mean Asymmetry Index', gridcolor='#edf2f7'),
        legend=dict(orientation='h', y=-0.15),
    )
    return _apply_defaults(fig, height=400)


# ══════════════════════════════════════════════════════════════
#  CLINICAL SUMMARY TAB
# ══════════════════════════════════════════════════════════════

def make_summary_radar(feature_df, group_stats_df, tulip_id):
    """Radar chart: normalized feature profile vs reference cohorts (aligned tasks only)."""
    from src.data_loader import MATCHING_TASKS
    from src.feature_engineering import calc_asymmetry_index
    if not tulip_id:
        return _empty_fig('Select a patient')
    group_map = group_stats_df[['tulip_id', 'group']].drop_duplicates()
    # Filter to aligned tasks only
    aligned_fc = feature_df[feature_df.task.isin(MATCHING_TASKS)]
    merged = aligned_fc.merge(group_map, on='tulip_id', how='left')
    dimensions = ['Tremor Power', 'Amplitude', 'Rhythm Irreg.', 'Mean Jerk', 'Asymmetry']
    features = ['tremor_power', 'amplitude', 'rhythm_irreg', 'jerk']
    patient_scores, healthy_scores, pd_scores = [], [], []
    for feat in features:
        all_vals = merged[feat].values
        h_mean = merged[merged.group == 'Healthy'][feat].mean()
        p_mean = merged[merged.group == 'PD'][feat].mean()
        pat_val = merged[merged.tulip_id == tulip_id][feat].mean()
        vmin, vmax = all_vals.min(), all_vals.max()
        norm = lambda v: (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
        patient_scores.append(norm(pat_val))
        healthy_scores.append(norm(h_mean))
        pd_scores.append(norm(p_mean))
    subj_data = feature_df[feature_df.tulip_id == tulip_id]
    asym_vals = []
    for task in MATCHING_TASKS:
        td = subj_data[subj_data.task == task]
        lv = td[td.wrist == 'L']['amplitude'].values
        rv = td[td.wrist == 'R']['amplitude'].values
        if len(lv) > 0 and len(rv) > 0:
            asym_vals.append(calc_asymmetry_index(lv[0], rv[0]))
    pat_asym = np.mean(asym_vals) if asym_vals else 0
    patient_scores.append(min(pat_asym / 0.5, 1.0))
    healthy_scores.append(0.2)
    pd_scores.append(0.7)
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=healthy_scores+[healthy_scores[0]], theta=dimensions+[dimensions[0]],
                                   fill='toself', fillcolor='rgba(56,161,105,0.1)',
                                   line=dict(color='#38a169', width=2), name='Healthy ref.'))
    fig.add_trace(go.Scatterpolar(r=pd_scores+[pd_scores[0]], theta=dimensions+[dimensions[0]],
                                   fill='toself', fillcolor='rgba(229,62,62,0.1)',
                                   line=dict(color='#e53e3e', width=2), name='PD ref.'))
    fig.add_trace(go.Scatterpolar(r=patient_scores+[patient_scores[0]], theta=dimensions+[dimensions[0]],
                                   fill='toself', fillcolor='rgba(128,90,213,0.15)',
                                   line=dict(color='#805ad5', width=3),
                                   marker=dict(size=8, symbol='star', color='#805ad5'),
                                   name=f'★ {_display_id(tulip_id)}'))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], gridcolor='#edf2f7')),
        title=dict(text=f'Motor Phenotype Radar — {tulip_id} (Aligned Tasks)', font=dict(size=14)),
        legend=dict(orientation='h', y=-0.1), height=420,
        font=dict(family='-apple-system, Segoe UI, sans-serif', size=12, color='#2c3e50'),
    )
    return fig


# ══════════════════════════════════════════════════════════════
#  TREMOR & RHYTHM ANALYSIS — Aligned Tasks Focus
# ══════════════════════════════════════════════════════════════

def make_tremor_power_bars(feature_df, group_stats_df, tulip_id):
    """Tremor power (4-12Hz) L/R comparison for aligned tasks with PD/Healthy reference."""
    from src.data_loader import MATCHING_TASKS, TASK_LABELS_KR

    subj = feature_df[feature_df.tulip_id == tulip_id]
    if subj.empty:
        return _empty_fig('Feature data 없음')

    group_map = group_stats_df[['tulip_id', 'group']].drop_duplicates()
    merged = feature_df.merge(group_map, on='tulip_id', how='left')

    n_tasks = len(MATCHING_TASKS)
    fig = make_subplots(
        rows=1, cols=n_tasks,
        subplot_titles=[TASK_LABELS_KR.get(t, t) for t in MATCHING_TASKS],
        horizontal_spacing=0.15,
    )

    for col_i, task in enumerate(MATCHING_TASKS, 1):
        td = subj[subj.task == task]
        l_val = td[td.wrist == 'L']['tremor_power'].values
        r_val = td[td.wrist == 'R']['tremor_power'].values
        l_tp = float(l_val[0]) if len(l_val) > 0 else 0
        r_tp = float(r_val[0]) if len(r_val) > 0 else 0

        ref = merged[merged.task == task]
        h_mean = ref[ref.group == 'Healthy']['tremor_power'].mean() if not ref[ref.group == 'Healthy'].empty else 0
        p_mean = ref[ref.group == 'PD']['tremor_power'].mean() if not ref[ref.group == 'PD'].empty else 0

        fig.add_trace(go.Bar(
            x=['Left', 'Right'], y=[l_tp, r_tp],
            marker=dict(color=['#4299e1', '#fc8181']),
            name='Patient' if col_i == 1 else None,
            showlegend=(col_i == 1), legendgroup='patient',
            text=[f'{l_tp:.5f}', f'{r_tp:.5f}'],
            textposition='outside', textfont=dict(size=9),
            hovertemplate='%{x}: %{y:.6f}<extra></extra>',
        ), row=1, col=col_i)

        fig.add_hline(y=p_mean, line_dash='dash', line_color='#e53e3e',
                      annotation_text=f'PD avg',
                      annotation_font=dict(size=9, color='#e53e3e'),
                      row=1, col=col_i)
        fig.add_hline(y=h_mean, line_dash='dot', line_color='#38a169',
                      annotation_text=f'Healthy avg',
                      annotation_font=dict(size=9, color='#38a169'),
                      row=1, col=col_i)

    fig.update_layout(
        title=dict(text=f'Tremor Power (4-12Hz) — Aligned Tasks L/R ({_display_id(tulip_id)})',
                   font=dict(size=14)),
        legend=dict(orientation='h', y=-0.15),
    )
    fig.update_yaxes(title_text='Tremor Power', gridcolor='#edf2f7', row=1, col=1)
    return _apply_defaults(fig, height=400)


def make_tremor_band_breakdown(tulip_id, task):
    """Frequency band breakdown: rest tremor (4-6Hz) vs action tremor (6-12Hz) L/R."""
    from src.data_loader import load_timeseries, _estimate_fs, TASK_LABELS_KR

    categories = []
    rest_band = []
    action_band = []
    total_band = []

    for wrist, label in [('LeftWrist', 'Left'), ('RightWrist', 'Right')]:
        ts = load_timeseries(tulip_id, task, wrist)
        if ts.empty:
            categories.append(label)
            rest_band.append(0)
            action_band.append(0)
            total_band.append(0)
            continue

        sig = ts['accel_mag'].values
        fs = _estimate_fs(ts)

        sig_dc = sig - np.mean(sig)
        n = len(sig_dc)
        if n < 16:
            categories.append(label)
            rest_band.append(0)
            action_band.append(0)
            total_band.append(0)
            continue

        fft_vals = np.abs(np.fft.rfft(sig_dc)) ** 2 / (n * fs)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0

        rest_mask = (freqs >= 4.0) & (freqs <= 6.0)
        action_mask = (freqs > 6.0) & (freqs <= 12.0)

        r_power = float(np.sum(fft_vals[rest_mask]) * df)
        a_power = float(np.sum(fft_vals[action_mask]) * df)

        categories.append(label)
        rest_band.append(r_power)
        action_band.append(a_power)
        total_band.append(r_power + a_power)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=categories, y=rest_band, name='Rest Tremor Band (4-6 Hz)',
        marker=dict(color='#e53e3e', opacity=0.85),
        text=[f'{v:.6f}' for v in rest_band], textposition='outside',
        textfont=dict(size=9),
    ))
    fig.add_trace(go.Bar(
        x=categories, y=action_band, name='Action Tremor Band (6-12 Hz)',
        marker=dict(color='#dd6b20', opacity=0.85),
        text=[f'{v:.6f}' for v in action_band], textposition='outside',
        textfont=dict(size=9),
    ))

    # Ratio annotation
    for i, cat in enumerate(categories):
        total = total_band[i]
        if total > 0:
            ratio = rest_band[i] / total * 100
            interpretation = 'Rest-dominant (PD-like)' if ratio > 60 else (
                'Action-dominant' if ratio < 40 else 'Mixed')
            fig.add_annotation(
                x=cat, y=max(rest_band[i], action_band[i]) * 1.15,
                text=f'Rest: {ratio:.0f}%<br><i style="font-size:9px">{interpretation}</i>',
                showarrow=False, font=dict(size=10),
            )

    task_label = TASK_LABELS_KR.get(task, task)
    fig.update_layout(
        title=dict(text=f'Tremor Frequency Band — {task_label} ({_display_id(tulip_id)})',
                   font=dict(size=14)),
        barmode='group',
        yaxis=dict(title='Band Power', gridcolor='#edf2f7'),
        legend=dict(orientation='h', y=-0.15),
    )
    return _apply_defaults(fig, height=400)


def make_amplitude_decrement(tulip_id, task):
    """Amplitude decrement analysis: progressive decay detection for bradykinesia.

    Declining amplitude during repetitive movement (Entrainment) is a
    hallmark sign of PD bradykinesia / decrement sequence.
    """
    from src.data_loader import load_timeseries, _estimate_fs, TASK_LABELS_KR

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Left Wrist', 'Right Wrist'],
        horizontal_spacing=0.12,
    )

    for col_i, (wrist, color) in enumerate([('LeftWrist', '#4299e1'), ('RightWrist', '#fc8181')], 1):
        ts = load_timeseries(tulip_id, task, wrist)
        if ts.empty:
            continue
        sig = ts['accel_mag'].values
        fs = _estimate_fs(ts)

        win_sec = 2.0
        win = int(win_sec * fs)
        hop = win // 2
        n_frames = max(1, (len(sig) - win) // hop + 1)
        times = np.zeros(n_frames)
        amplitudes = np.zeros(n_frames)

        for i in range(n_frames):
            start = i * hop
            end = min(start + win, len(sig))
            frame = sig[start:end]
            times[i] = (start + end) / 2 / fs
            amplitudes[i] = np.sqrt(np.mean(frame ** 2))

        fig.add_trace(go.Scatter(
            x=times, y=amplitudes, mode='lines+markers',
            line=dict(color=color, width=2),
            marker=dict(size=4, color=color),
            name=wrist.replace('Wrist', ''),
            hovertemplate='Time: %{x:.1f}s<br>RMS: %{y:.4f} g<extra></extra>',
        ), row=1, col=col_i)

        if len(times) > 2:
            z = np.polyfit(times, amplitudes, 1)
            trend = np.polyval(z, times)
            slope_color = '#e53e3e' if z[0] < -0.001 else '#38a169'
            slope_text = 'Declining (decrement)' if z[0] < -0.001 else 'Stable/Increasing'
            fig.add_trace(go.Scatter(
                x=times, y=trend, mode='lines',
                line=dict(color=slope_color, width=2, dash='dash'),
                showlegend=False,
                hovertemplate=f'Trend: {z[0]:.5f} g/s<br>{slope_text}<extra></extra>',
            ), row=1, col=col_i)

            fig.add_annotation(
                x=0.5, y=0.95,
                xref=f'x{col_i} domain' if col_i > 1 else 'x domain',
                yref=f'y{col_i} domain' if col_i > 1 else 'y domain',
                text=f'Slope: {z[0]:.5f} g/s<br><b>{slope_text}</b>',
                showarrow=False, font=dict(size=10, color=slope_color),
                bgcolor='rgba(255,255,255,0.85)', bordercolor=slope_color,
                borderwidth=1, borderpad=4,
            )

    task_label = TASK_LABELS_KR.get(task, task)
    fig.update_layout(
        title=dict(text=f'Amplitude Decrement — {task_label} ({_display_id(tulip_id)})',
                   font=dict(size=14)),
        legend=dict(orientation='h', y=-0.12),
    )
    fig.update_yaxes(title_text='RMS Amplitude (g)', gridcolor='#edf2f7')
    fig.update_xaxes(title_text='Time (s)', gridcolor='#edf2f7')
    return _apply_defaults(fig, height=400)

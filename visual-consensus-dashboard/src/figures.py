"""Plotly figure creation functions for PD clinical decision support dashboard."""

import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots

LAYOUT_DEFAULTS = dict(
    plot_bgcolor='white',
    paper_bgcolor='white',
    margin=dict(l=50, r=20, t=50, b=60),
    font=dict(family='-apple-system, Segoe UI, sans-serif', size=12, color='#2c3e50'),
)

GROUP_COLORS = {'PD': '#e53e3e', 'Healthy': '#38a169', 'Other': '#d69e2e'}


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
            text=gd['tulip_id'].apply(lambda x: x.replace('TULIP_', 'T')),
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

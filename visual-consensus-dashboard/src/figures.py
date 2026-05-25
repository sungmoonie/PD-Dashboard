"""Plotly figure creation functions for the clinical dashboard."""

import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots

LAYOUT_DEFAULTS = dict(
    plot_bgcolor='white',
    paper_bgcolor='white',
    margin=dict(l=50, r=20, t=50, b=60),
    font=dict(family='-apple-system, Segoe UI, sans-serif', size=12, color='#2c3e50'),
)


def _apply_defaults(fig, height=370):
    fig.update_layout(**LAYOUT_DEFAULTS, height=height)
    return fig


def _empty_fig(msg='Data 없음', height=370):
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False,
                       xref='paper', yref='paper', x=0.5, y=0.5, font=dict(size=14))
    return _apply_defaults(fig, height)


# ══════════════════════════════════════════════════════════════
#  TAB 2: UPDRS HEATMAP
# ══════════════════════════════════════════════════════════════

def make_updrs_heatmap(labels_df):
    """Heatmap: 12 subjects × UPDRS items, values = mean of 3 clinicians.

    Args:
        labels_df: DataFrame with tulip_id, updrs_name, mean, is_numeric
    """
    numeric = labels_df[labels_df.is_numeric == True].copy()
    # Exclude H&Y (different scale) and non-motor items
    exclude = ['Hoehn and Yahr Stage']
    numeric = numeric[~numeric.updrs_name.isin(exclude)]

    if numeric.empty:
        return _empty_fig('UPDRS 데이터 없음')

    pivot = numeric.pivot_table(index='tulip_id', columns='updrs_name', values='mean', aggfunc='first')
    pivot = pivot.sort_index()

    # Shorten long UPDRS names for display
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
        title=dict(text='UPDRS Part III — 전체 Clinician 평균 Score Heatmap', font=dict(size=14)),
    )
    return _apply_defaults(fig, height=max(400, len(pivot) * 35 + 150))


def make_clinician_comparison_bar(labels_df, tulip_id, updrs_name):
    """Bar chart comparing 3 clinicians' scores for one subject × one item."""
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
        hovertemplate='%{x}: %{y:.1f}/4<extra></extra>',
    ))
    mean_val = sum(scores) / 3
    fig.add_hline(y=mean_val, line_dash='dash', line_color='#718096',
                  annotation_text=f'Mean: {mean_val:.2f}')
    fig.update_layout(
        title=dict(text=f'{tulip_id} — {updrs_name}', font=dict(size=13)),
        yaxis=dict(title='Score (0-4)', range=[0, 4.3], gridcolor='#edf2f7', dtick=1),
        xaxis=dict(tickfont=dict(size=12)),
    )
    return _apply_defaults(fig, height=350)


# ══════════════════════════════════════════════════════════════
#  TAB 3: INTER-RATER AGREEMENT
# ══════════════════════════════════════════════════════════════

def make_disagreement_heatmap(labels_df):
    """Heatmap of max-min disagreement across 3 clinicians."""
    numeric = labels_df[labels_df.is_numeric == True].copy()
    exclude = ['Hoehn and Yahr Stage']
    numeric = numeric[~numeric.updrs_name.isin(exclude)]

    if numeric.empty:
        return _empty_fig('불일치 데이터 없음')

    pivot = numeric.pivot_table(index='tulip_id', columns='updrs_name',
                                values='disagreement', aggfunc='first')
    pivot = pivot.sort_index()

    col_labels = [c[:25] + '...' if len(c) > 28 else c for c in pivot.columns]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=list(pivot.columns),
        y=list(pivot.index),
        colorscale=[[0, '#c6f6d5'], [0.25, '#fefcbf'], [0.5, '#fed7d7'], [1, '#e53e3e']],
        zmin=0, zmax=4,
        text=np.where(np.isnan(pivot.values), '', pivot.values.astype(int).astype(str)),
        texttemplate='%{text}', textfont=dict(size=10),
        hovertemplate='Subject: %{y}<br>Item: %{x}<br>Disagreement: %{z:.0f}<extra></extra>',
        colorbar=dict(title='Max-Min', thickness=15),
    ))
    fig.update_layout(
        xaxis=dict(tickvals=list(pivot.columns), ticktext=col_labels,
                   side='bottom', tickangle=-45, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=11)),
        title=dict(text='Inter-rater Disagreement (Max - Min Score)', font=dict(size=14)),
    )
    return _apply_defaults(fig, height=max(400, len(pivot) * 35 + 150))


def make_agreement_stats(labels_df):
    """Return summary stats dict for inter-rater agreement."""
    numeric = labels_df[(labels_df.is_numeric == True) &
                        (labels_df.updrs_name != 'Hoehn and Yahr Stage')]
    if numeric.empty:
        return {}

    total = len(numeric)
    perfect = len(numeric[numeric.disagreement == 0])
    mild = len(numeric[(numeric.disagreement > 0) & (numeric.disagreement <= 1)])
    high = len(numeric[numeric.disagreement >= 2])

    return {
        'total_ratings': total,
        'perfect_agreement': perfect,
        'perfect_pct': round(perfect / total * 100, 1),
        'mild_disagreement': mild,
        'high_disagreement': high,
        'high_pct': round(high / total * 100, 1),
        'mean_disagreement': round(numeric['disagreement'].mean(), 2),
    }


# ══════════════════════════════════════════════════════════════
#  TAB 4: SENSOR TIMESERIES
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

    # Add signal magnitude as dashed overlay
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
#  TAB 5: LEFT-RIGHT COMPARISON
# ══════════════════════════════════════════════════════════════

def make_lr_overlay_plot(left_df, right_df, channel='accel_mag', task_name=''):
    """Overlay left and right wrist signal magnitude."""
    if left_df.empty and right_df.empty:
        return _empty_fig('좌우 데이터 없음')

    fig = go.Figure()
    if not left_df.empty:
        fig.add_trace(go.Scatter(
            x=left_df['time'], y=left_df[channel], mode='lines',
            name='Left Wrist', line=dict(color='#4299e1', width=1.5),
            opacity=0.8,
        ))
    if not right_df.empty:
        fig.add_trace(go.Scatter(
            x=right_df['time'], y=right_df[channel], mode='lines',
            name='Right Wrist', line=dict(color='#fc8181', width=1.5),
            opacity=0.8,
        ))

    channel_label = {'accel_mag': 'Accelerometer Magnitude (g)',
                     'gyro_mag': 'Gyroscope Magnitude (rad/s)'}
    fig.update_layout(
        title=dict(text=f'좌우 비교 — {task_name} ({channel_label.get(channel, channel)})',
                   font=dict(size=14)),
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
        title=dict(text=f'좌우 RMS 비교 — {task_name}', font=dict(size=14)),
        yaxis=dict(title='Signal Value', gridcolor='#edf2f7'),
        barmode='group',
        legend=dict(orientation='h', y=-0.15),
    )
    return _apply_defaults(fig, height=380)


# ══════════════════════════════════════════════════════════════
#  TAB 6: VIDEO ANALYSIS (kept from original)
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

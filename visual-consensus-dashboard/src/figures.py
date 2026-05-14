"""Plotly figure creation functions for the dashboard."""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

LAYOUT_DEFAULTS = dict(
    plot_bgcolor='white',
    paper_bgcolor='white',
    margin=dict(l=50, r=20, t=50, b=60),
    font=dict(family='-apple-system, Segoe UI, sans-serif', size=12, color='#2c3e50'),
)

# Korean labels
FEAT_KR = {
    'tremor_power': 'Tremor', 'movement_amplitude_reduction': 'Amplitude 감소',
    'rhythm_irregularity': 'Rhythm 불규칙', 'left_right_asymmetry': '좌우 Asymmetry',
    'motion_instability': 'Instability',
}
TASK_KR = {
    'finger_tapping': 'Finger Tapping', 'hand_open_close': 'Hand Open/Close',
    'rest_tremor': 'Rest Tremor', 'gait': 'Gait',
    'toe_tapping': 'Toe Tapping', 'touch_nose': 'Touch Nose',
}
def _fkr(f): return FEAT_KR.get(f, f.replace('_', ' ').title())
def _tkr(t): return TASK_KR.get(t, t.replace('_', ' ').title())


def _apply_defaults(fig, height=370):
    fig.update_layout(**LAYOUT_DEFAULTS, height=height)
    return fig


# ── 1. Overview Radar Chart ──────────────────────────────────
def make_overview_radar(patient_summary):
    features = [
        'tremor_power', 'movement_amplitude_reduction',
        'rhythm_irregularity', 'left_right_asymmetry', 'motion_instability',
    ]
    labels = ['Tremor', 'Amplitude\n감소', 'Rhythm\n불규칙',
              '좌우\nAsymmetry', 'Instability']

    values = [patient_summary.get(f, 0) for f in features]
    values.append(values[0])
    labels_closed = labels + [labels[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=labels_closed,
        fill='toself',
        fillcolor='rgba(43, 108, 176, 0.15)',
        line=dict(color='#2b6cb0', width=2),
        marker=dict(size=6, color='#2b6cb0'),
        name='Patient Profile',
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=10)),
            angularaxis=dict(tickfont=dict(size=11)),
        ),
        showlegend=False,
        title=dict(text='Motor Feature Profile', font=dict(size=14)),
        **LAYOUT_DEFAULTS, height=370,
    )
    return fig


# ── 2. Top Abnormal Tasks Bar Chart ─────────────────────────
def make_top_tasks_bar(top_tasks_df):
    df = top_tasks_df.copy()
    df['task_label'] = df['task'].map(TASK_KR).fillna(df['task'].str.replace('_', ' ').str.title())
    df['color'] = df['mean_score'].apply(
        lambda v: '#e53e3e' if v >= 0.65 else '#dd6b20' if v >= 0.45 else '#ecc94b'
    )
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df['task_label'], x=df['mean_score'], orientation='h',
        marker=dict(color=df['color'], line=dict(width=0)),
        text=df['mean_score'].round(2), textposition='auto',
    ))
    fig.update_layout(
        title=dict(text='Abnormal Task 순위', font=dict(size=14)),
        xaxis=dict(title='평균 Abnormality Score', range=[0, 1],
                   gridcolor='#edf2f7', zeroline=False),
        yaxis=dict(autorange='reversed'),
    )
    return _apply_defaults(fig)


# ── 3. Task-Feature Heatmap ─────────────────────────────────
def make_task_feature_heatmap(matrix_df):
    task_labels = [TASK_KR.get(t, t) for t in matrix_df.index]
    feature_labels = [FEAT_KR.get(f, f) for f in matrix_df.columns]

    fig = go.Figure(data=go.Heatmap(
        z=matrix_df.values,
        x=feature_labels, y=task_labels,
        colorscale='RdYlBu_r', zmin=0, zmax=1,
        text=np.round(matrix_df.values, 2),
        texttemplate='%{text}', textfont=dict(size=11),
        hovertemplate='Task: %{y}<br>Feature: %{x}<br>Score: %{z:.2f}<extra></extra>',
        colorbar=dict(title='Score', thickness=15),
    ))
    fig.data[0].x = list(matrix_df.columns)
    fig.data[0].y = list(matrix_df.index)
    fig.update_layout(
        xaxis=dict(tickvals=list(matrix_df.columns), ticktext=feature_labels, side='bottom', tickangle=-30),
        yaxis=dict(tickvals=list(matrix_df.index), ticktext=task_labels),
        title=dict(text='Task × Feature Abnormality Heatmap', font=dict(size=14)),
    )
    return _apply_defaults(fig, height=420)


# ── 4. Left-Right Mirror Plot ───────────────────────────────
def make_left_right_mirror_plot(lr_df):
    features = lr_df['feature'].map(FEAT_KR).fillna(lr_df['feature'].str.replace('_', ' ').str.title()).values
    left_vals = -lr_df['left_value'].values
    right_vals = lr_df['right_value'].values

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=features, x=left_vals, orientation='h',
        name='Left', marker=dict(color='#4299e1'),
        text=lr_df['left_value'].round(2).values, textposition='inside',
        hovertemplate='Left %{y}: %{text}<extra></extra>',
    ))
    fig.add_trace(go.Bar(
        y=features, x=right_vals, orientation='h',
        name='Right', marker=dict(color='#fc8181'),
        text=lr_df['right_value'].round(2).values, textposition='inside',
        hovertemplate='Right %{y}: %{text}<extra></extra>',
    ))
    fig.add_vline(x=0, line_width=2, line_color='#2d3748')
    max_val = max(abs(left_vals).max(), right_vals.max()) * 1.15
    fig.update_layout(
        title=dict(text='Left-Right 비교', font=dict(size=14)),
        xaxis=dict(range=[-max_val, max_val], title='← Left | Right →',
                   zeroline=True, gridcolor='#edf2f7'),
        yaxis=dict(autorange='reversed'),
        barmode='overlay',
        legend=dict(orientation='h', y=-0.15),
    )
    return _apply_defaults(fig, height=350)


# ── 5. Rhythm Time-series ───────────────────────────────────
def make_rhythm_timeseries(ts_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts_df['time'], y=ts_df['left_amplitude'],
        mode='lines', name='Left', line=dict(color='#4299e1', width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=ts_df['time'], y=ts_df['right_amplitude'],
        mode='lines', name='Right', line=dict(color='#fc8181', width=1.5),
    ))
    for col, color in [('left_amplitude', '#2b6cb0'), ('right_amplitude', '#c53030')]:
        peaks, _ = find_peaks(ts_df[col].values, distance=5, prominence=0.05)
        if len(peaks) > 0:
            fig.add_trace(go.Scatter(
                x=ts_df['time'].values[peaks], y=ts_df[col].values[peaks],
                mode='markers', marker=dict(size=5, color=color, symbol='diamond'),
                showlegend=False,
            ))
    fig.update_layout(
        title=dict(text='Amplitude 시계열', font=dict(size=14)),
        xaxis=dict(title='Time (s)', gridcolor='#edf2f7', zeroline=False),
        yaxis=dict(title='Amplitude', gridcolor='#edf2f7', zeroline=False),
        legend=dict(orientation='h', y=-0.15),
    )
    return _apply_defaults(fig)


# ── 6. Tap Interval Chart ───────────────────────────────────
def make_tap_interval_chart(ts_df):
    signal = ts_df['left_amplitude'].values
    peaks, _ = find_peaks(signal, distance=5, prominence=0.05)
    if len(peaks) < 2:
        fig = go.Figure()
        fig.add_annotation(text='Not enough peaks detected', showarrow=False,
                           xref='paper', yref='paper', x=0.5, y=0.5, font=dict(size=14))
        return _apply_defaults(fig)

    times = ts_df['time'].values
    intervals = np.diff(times[peaks])
    mean_int = intervals.mean()
    colors = ['#e53e3e' if abs(iv - mean_int) > mean_int * 0.3
              else '#dd6b20' if abs(iv - mean_int) > mean_int * 0.15
              else '#38a169' for iv in intervals]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(range(1, len(intervals) + 1)), y=intervals,
        marker=dict(color=colors),
        hovertemplate='Tap %{x}: %{y:.3f}s<extra></extra>',
    ))
    fig.add_hline(y=mean_int, line_dash='dash', line_color='#718096',
                  annotation_text=f'Mean: {mean_int:.3f}s')
    fig.update_layout(
        title=dict(text='Tap Interval 분포', font=dict(size=14)),
        xaxis=dict(title='Tap #', gridcolor='#edf2f7'),
        yaxis=dict(title='Interval (s)', gridcolor='#edf2f7', zeroline=False),
    )
    return _apply_defaults(fig)


# ── 7. Amplitude Decrement Chart ─────────────────────────────
def make_amplitude_decrement_chart(ts_df):
    signal = ts_df['left_amplitude'].values
    peaks, _ = find_peaks(signal, distance=5, prominence=0.05)
    if len(peaks) < 4:
        fig = go.Figure()
        fig.add_annotation(text='Not enough peaks detected', showarrow=False,
                           xref='paper', yref='paper', x=0.5, y=0.5, font=dict(size=14))
        return _apply_defaults(fig)

    peak_amps = signal[peaks]
    reps = list(range(1, len(peak_amps) + 1))
    z = np.polyfit(reps, peak_amps, 1)
    trend = np.poly1d(z)(reps)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=reps, y=peak_amps, mode='markers+lines', name='Peak Amplitude',
        marker=dict(size=7, color='#2b6cb0'), line=dict(color='#2b6cb0', width=1),
    ))
    fig.add_trace(go.Scatter(
        x=reps, y=trend, mode='lines', name='Trend',
        line=dict(color='#e53e3e', width=2, dash='dash'),
    ))
    n = min(3, len(peak_amps) // 2)
    early = peak_amps[:n].mean()
    late = peak_amps[-n:].mean()
    fig.add_annotation(
        x=reps[-1], y=peak_amps[-1],
        text=f'Early: {early:.2f} → Late: {late:.2f}',
        showarrow=True, arrowhead=2, font=dict(size=10),
    )
    fig.update_layout(
        title=dict(text='Amplitude Decrement', font=dict(size=14)),
        xaxis=dict(title='Repetition #', gridcolor='#edf2f7'),
        yaxis=dict(title='Peak Amplitude', gridcolor='#edf2f7', zeroline=False),
        legend=dict(orientation='h', y=-0.15),
    )
    return _apply_defaults(fig)


# ── 8. Normative Comparison (NEW) ────────────────────────────
def make_normative_comparison(norm_results):
    """Box/strip plot showing patient position vs healthy distribution."""
    if not norm_results:
        fig = go.Figure()
        fig.add_annotation(text='No normative data available', showarrow=False,
                           xref='paper', yref='paper', x=0.5, y=0.5)
        return _apply_defaults(fig)

    features = []
    patient_vals = []
    healthy_means = []
    healthy_stds = []
    z_scores = []

    for r in norm_results:
        label = r['feature'].replace('_', ' ').title()
        if label not in features:
            features.append(label)
            patient_vals.append(r['patient_value'])
            healthy_means.append(r['healthy_mean'])
            healthy_stds.append(r['healthy_std'])
            z_scores.append(r['z_score'])

    # Aggregate across tasks: average per feature
    feat_data = {}
    for r in norm_results:
        f = r['feature'].replace('_', ' ').title()
        if f not in feat_data:
            feat_data[f] = {'pv': [], 'hm': [], 'hs': [], 'zs': []}
        feat_data[f]['pv'].append(r['patient_value'])
        feat_data[f]['hm'].append(r['healthy_mean'])
        feat_data[f]['hs'].append(r['healthy_std'])
        feat_data[f]['zs'].append(r['z_score'])

    features = list(feat_data.keys())
    avg_pv = [np.mean(feat_data[f]['pv']) for f in features]
    avg_hm = [np.mean(feat_data[f]['hm']) for f in features]
    avg_hs = [np.mean(feat_data[f]['hs']) for f in features]
    avg_zs = [np.mean(feat_data[f]['zs']) for f in features]

    fig = go.Figure()

    # Healthy range (mean ± 2 std)
    fig.add_trace(go.Bar(
        y=features, x=[m + 2 * s for m, s in zip(avg_hm, avg_hs)],
        orientation='h', name='Healthy Range (95%)',
        marker=dict(color='rgba(56, 161, 105, 0.2)'),
        hoverinfo='skip',
    ))

    # Healthy mean
    fig.add_trace(go.Scatter(
        y=features, x=avg_hm,
        mode='markers', name='Healthy Mean',
        marker=dict(size=10, color='#38a169', symbol='line-ns-open', line=dict(width=2)),
    ))

    # Patient value
    colors = ['#e53e3e' if z > 2 else '#dd6b20' if z > 1 else '#2b6cb0' for z in avg_zs]
    fig.add_trace(go.Scatter(
        y=features, x=avg_pv,
        mode='markers+text', name='Patient',
        marker=dict(size=14, color=colors, symbol='diamond', line=dict(width=1, color='white')),
        text=[f'z={z:.1f}' for z in avg_zs],
        textposition='middle right', textfont=dict(size=10),
    ))

    fig.update_layout(
        title=dict(text='환자 vs. Healthy Control', font=dict(size=14)),
        xaxis=dict(title='Score', gridcolor='#edf2f7', zeroline=False),
        yaxis=dict(autorange='reversed'),
        barmode='overlay',
        legend=dict(orientation='h', y=-0.18),
    )
    return _apply_defaults(fig, height=400)


# ── 9. Patient History (NEW) ────────────────────────────────
def make_history_line_chart(history_df, patient_id):
    """Line chart of feature values across visits."""
    ph = history_df[history_df.patient_id == patient_id].copy()
    if ph.empty:
        fig = go.Figure()
        fig.add_annotation(text='No history data', showarrow=False,
                           xref='paper', yref='paper', x=0.5, y=0.5)
        return _apply_defaults(fig)

    ph['feature_label'] = ph['feature'].str.replace('_', ' ').str.title()

    colors = {
        'Tremor Power': '#e53e3e',
        'Movement Amplitude Reduction': '#dd6b20',
        'Rhythm Irregularity': '#805ad5',
        'Left Right Asymmetry': '#3182ce',
        'Motion Instability': '#38a169',
    }

    fig = go.Figure()
    for feat in ph['feature_label'].unique():
        fd = ph[ph.feature_label == feat].sort_values('visit_number')
        fig.add_trace(go.Scatter(
            x=fd['visit_date'], y=fd['value'],
            mode='lines+markers', name=feat,
            line=dict(color=colors.get(feat, '#718096'), width=2),
            marker=dict(size=7),
        ))

    fig.update_layout(
        title=dict(text='방문별 Motor Feature 변화 추이', font=dict(size=14)),
        xaxis=dict(title='Visit Date', gridcolor='#edf2f7'),
        yaxis=dict(title='Score', range=[0, 1], gridcolor='#edf2f7', zeroline=False),
        legend=dict(orientation='h', y=-0.25, font=dict(size=10)),
    )
    return _apply_defaults(fig, height=400)


def make_change_bar_chart(changes):
    """Bar chart showing feature changes between visits."""
    if not changes:
        fig = go.Figure()
        fig.add_annotation(text='Not enough visit data', showarrow=False,
                           xref='paper', yref='paper', x=0.5, y=0.5)
        return _apply_defaults(fig)

    features = [c['feature'].replace('_', ' ').title() for c in changes]
    pct_changes = [c['pct_change'] for c in changes]
    colors = ['#e53e3e' if c['direction'] == 'worsened'
              else '#38a169' if c['direction'] == 'improved'
              else '#a0aec0' for c in changes]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=features, x=pct_changes, orientation='h',
        marker=dict(color=colors),
        text=[f"{p:+.1f}%" for p in pct_changes], textposition='auto',
    ))
    fig.add_vline(x=0, line_width=1.5, line_color='#2d3748')
    fig.update_layout(
        title=dict(text='최근 방문 대비 변화율 (%)', font=dict(size=14)),
        xaxis=dict(title='Change (%)', gridcolor='#edf2f7', zeroline=False),
        yaxis=dict(autorange='reversed'),
    )
    return _apply_defaults(fig, height=320)


# ── 10. UPDRS Score Assistant (NEW) ──────────────────────────
def make_updrs_gauge_chart(updrs_items):
    """Create horizontal bar chart for UPDRS estimated scores (no overlap)."""
    if not updrs_items:
        fig = go.Figure()
        fig.add_annotation(text='No UPDRS data', showarrow=False,
                           xref='paper', yref='paper', x=0.5, y=0.5)
        return _apply_defaults(fig)

    labels = [f"{item['item_code']} {item['item_name']}" for item in updrs_items]
    scores = [item['estimated_score'] for item in updrs_items]
    confidences = [item['confidence'] for item in updrs_items]

    colors = ['#c6f6d5' if s < 1 else '#fefcbf' if s < 2
              else '#fed7d7' if s < 3 else '#feb2b2' for s in scores]
    border_colors = ['#38a169' if s < 1 else '#d69e2e' if s < 2
                     else '#e53e3e' if s < 3 else '#c53030' for s in scores]

    fig = go.Figure()

    # Score bars
    fig.add_trace(go.Bar(
        y=labels, x=scores, orientation='h',
        marker=dict(color=colors, line=dict(color=border_colors, width=2)),
        text=[f'{s:.1f} / 4' for s in scores],
        textposition='auto', textfont=dict(size=13, color='#2d3748'),
        name='Estimated Score',
        hovertemplate='%{y}<br>Score: %{x:.1f}/4<extra></extra>',
    ))

    # Confidence dots on secondary axis
    fig.add_trace(go.Scatter(
        y=labels, x=[c * 4 for c in confidences],
        mode='markers', name='Confidence',
        marker=dict(size=10, color='#2b6cb0', symbol='diamond',
                    line=dict(width=1, color='white')),
        hovertemplate='%{y}<br>Confidence: %{text}<extra></extra>',
        text=[f'{c:.0%}' for c in confidences],
    ))

    # UPDRS scale reference lines
    for val, label_text in [(1, '경미'), (2, '경도'), (3, '중등도')]:
        fig.add_vline(x=val, line_dash='dot', line_color='#cbd5e0', line_width=1)
        fig.add_annotation(x=val, y=-0.5, text=label_text, showarrow=False,
                           font=dict(size=9, color='#a0aec0'), yref='paper', yshift=-10)

    fig.update_layout(
        title=dict(text='UPDRS Part III — Sensor 기반 추정 Score', font=dict(size=14)),
        xaxis=dict(title='Score (0–4)', range=[0, 4.3], gridcolor='#edf2f7',
                   zeroline=False, dtick=1),
        yaxis=dict(autorange='reversed'),
        legend=dict(orientation='h', y=-0.15),
        barmode='overlay',
    )
    return _apply_defaults(fig, height=max(300, len(updrs_items) * 55 + 100))


# ── 11. Motion Trail Plot ───────────────────────────────────
def make_motion_trail_plot(frames):
    x = [f['x'] for f in frames]
    y = [f['y'] for f in frames]
    vel = [f['velocity'] for f in frames]
    time = [f['time'] for f in frames]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='lines+markers',
        marker=dict(
            size=5, color=vel, colorscale='Viridis',
            colorbar=dict(title='Velocity', thickness=12), showscale=True,
        ),
        line=dict(color='rgba(100,100,100,0.3)', width=1),
        text=[f't={t:.2f}s, v={v:.3f}' for t, v in zip(time, vel)],
        hovertemplate='x: %{x:.3f}<br>y: %{y:.3f}<br>%{text}<extra></extra>',
    ))
    fig.update_layout(
        title=dict(text='Motion Trail (Keypoint Trajectory)', font=dict(size=14)),
        xaxis=dict(title='X', scaleanchor='y', gridcolor='#edf2f7', zeroline=False),
        yaxis=dict(title='Y', gridcolor='#edf2f7', zeroline=False),
    )
    return _apply_defaults(fig, height=450)


# ══════════════════════════════════════════════════════════════
#  NOVELTY VISUALIZATIONS
# ══════════════════════════════════════════════════════════════

# ── 12. Motor Phase Portrait ────────────────────────────────
def make_phase_portrait(ts_df, patient_id):
    """Phase portrait: amplitude vs velocity, colored by time.
    Healthy = clean ellipse, PD = distorted shrinking spiral."""
    if ts_df.empty:
        fig = go.Figure()
        fig.add_annotation(text='No data', showarrow=False,
                           xref='paper', yref='paper', x=0.5, y=0.5)
        return _apply_defaults(fig)

    amp = ts_df['left_amplitude'].values
    vel = ts_df['left_velocity'].values
    time = ts_df['time'].values

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=amp, y=vel, mode='lines+markers',
        marker=dict(
            size=4, color=time, colorscale='Plasma',
            colorbar=dict(title='Time (s)', thickness=12), showscale=True,
        ),
        line=dict(color='rgba(100,100,100,0.15)', width=0.8),
        hovertemplate='Amp: %{x:.3f}<br>Vel: %{y:.3f}<br>Time: %{marker.color:.2f}s<extra></extra>',
        name=patient_id,
    ))

    # Mark start and end
    fig.add_trace(go.Scatter(
        x=[amp[0]], y=[vel[0]], mode='markers+text',
        marker=dict(size=12, color='#38a169', symbol='star'),
        text=['START'], textposition='top center', textfont=dict(size=10, color='#38a169'),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[amp[-1]], y=[vel[-1]], mode='markers+text',
        marker=dict(size=12, color='#e53e3e', symbol='x'),
        text=['END'], textposition='top center', textfont=dict(size=10, color='#e53e3e'),
        showlegend=False,
    ))

    fig.update_layout(
        title=dict(text=f'Phase Portrait — {patient_id}', font=dict(size=14)),
        xaxis=dict(title='Amplitude', gridcolor='#edf2f7', zeroline=False),
        yaxis=dict(title='Velocity', gridcolor='#edf2f7', zeroline=False),
    )
    return _apply_defaults(fig, height=450)


# ── 13. Movement Signature Wall ─────────────────────────────
def make_signature_wall(all_timeseries_df, task, patients_df, highlight_patient=None):
    """All patients' waveforms overlaid — hover to reveal color.
    Inspired by NYT Unemployment Lines."""
    ts = all_timeseries_df[all_timeseries_df.task == task]
    if ts.empty:
        fig = go.Figure()
        fig.add_annotation(text='No data', showarrow=False,
                           xref='paper', yref='paper', x=0.5, y=0.5)
        return _apply_defaults(fig)

    patients = ts['patient_id'].unique()

    # Map patient → group for color
    group_map = {}
    if patients_df is not None:
        group_map = dict(zip(patients_df['patient_id'], patients_df['group']))
    GROUP_COLORS = {'PD': '#e53e3e', 'Healthy': '#38a169', 'Differential': '#d69e2e'}

    fig = go.Figure()

    # Normal band first (background)
    time_groups = ts.groupby('time')['left_amplitude']
    mean_amp = time_groups.mean()
    std_amp = time_groups.std().fillna(0)
    fig.add_trace(go.Scatter(
        x=list(mean_amp.index) + list(mean_amp.index[::-1]),
        y=list((mean_amp + std_amp).values) + list((mean_amp - std_amp).values[::-1]),
        fill='toself', fillcolor='rgba(160,174,192,0.08)',
        line=dict(color='rgba(0,0,0,0)'),
        name='Population ±1σ', hoverinfo='skip',
    ))

    # All patients — default: thin gray lines, hover reveals true color
    for pid in patients:
        pdf = ts[ts.patient_id == pid].sort_values('time')
        group = group_map.get(pid, 'PD')
        true_color = GROUP_COLORS.get(group, '#718096')
        is_selected = (pid == highlight_patient)

        # Selected patient always shown in color
        if is_selected:
            line_color = true_color
            line_width = 3
            opacity = 1.0
        else:
            line_color = '#b0b8c4'
            line_width = 1.2
            opacity = 0.35

        fig.add_trace(go.Scatter(
            x=pdf['time'], y=pdf['left_amplitude'],
            mode='lines', name=f'{pid} ({group})',
            line=dict(width=line_width, color=line_color),
            opacity=opacity,
            hovertemplate=(
                f'<b>{pid}</b> ({group})<br>'
                f'Time: %{{x:.2f}}s<br>Amp: %{{y:.3f}}<extra></extra>'
            ),
            # On hover → full opacity & color via hoverlabel
            hoverlabel=dict(
                bgcolor=true_color, font_color='white',
                font_size=12, bordercolor=true_color,
            ),
        ))

    task_label = TASK_KR.get(task, task)
    fig.update_layout(
        title=dict(text=f'Signature Wall — {task_label}', font=dict(size=14)),
        xaxis=dict(title='Time (s)', gridcolor='#edf2f7', zeroline=False),
        yaxis=dict(title='Left Amplitude', gridcolor='#edf2f7', zeroline=False),
        showlegend=True,
        legend=dict(orientation='h', y=-0.18, font=dict(size=10)),
        # Hover mode: closest → highlights one line at a time
        hovermode='closest',
    )
    return _apply_defaults(fig, height=420)


# ── 14. Symptom Evidence Bubble Map ─────────────────────────
def make_evidence_bubble(norm_results, task_features_df, patient_id):
    """Bubble chart: X=abnormality (z-score), Y=diagnostic power,
    size=patient severity. Inspired by Snake Oil chart."""
    if not norm_results:
        fig = go.Figure()
        fig.add_annotation(text='No data', showarrow=False,
                           xref='paper', yref='paper', x=0.5, y=0.5)
        return _apply_defaults(fig)

    # Diagnostic power (rough heuristic: how much this feature differs between PD and Healthy)
    DIAGNOSTIC_POWER = {
        'tremor_power': 0.85,
        'rhythm_irregularity': 0.78,
        'left_right_asymmetry': 0.82,
        'movement_amplitude_reduction': 0.70,
        'motion_instability': 0.65,
    }

    TASK_COLORS = {
        'finger_tapping': '#e53e3e', 'hand_open_close': '#dd6b20',
        'rest_tremor': '#805ad5', 'gait': '#3182ce',
        'toe_tapping': '#38a169', 'touch_nose': '#d69e2e',
    }

    x_vals, y_vals, sizes, colors, texts, hover_texts = [], [], [], [], [], []

    for r in norm_results:
        z = r['z_score']
        feat = r['feature']
        task = r['task']
        diag_power = DIAGNOSTIC_POWER.get(feat, 0.5)
        severity = r['patient_value']

        x_vals.append(z)
        y_vals.append(diag_power)
        sizes.append(max(10, severity * 50))
        colors.append(TASK_COLORS.get(task, '#718096'))
        feat_label = feat.replace('_', ' ').title()
        task_label = task.replace('_', ' ').title()
        texts.append(f"{feat_label[:8]}")
        hover_texts.append(
            f"<b>{feat_label}</b><br>Task: {task_label}<br>"
            f"Z-score: {z:.1f}<br>Value: {severity:.2f}<br>"
            f"Diagnostic Power: {diag_power:.2f}"
        )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals, mode='markers+text',
        marker=dict(size=sizes, color=colors, opacity=0.7,
                    line=dict(width=1, color='white')),
        text=texts, textposition='top center', textfont=dict(size=8),
        hovertemplate='%{customdata}<extra></extra>',
        customdata=hover_texts,
    ))

    # Quadrant lines
    fig.add_vline(x=2, line_dash='dash', line_color='#e53e3e', line_width=1,
                  annotation_text='z=2 (significant)', annotation_position='top')
    fig.add_hline(y=0.75, line_dash='dash', line_color='#718096', line_width=1,
                  annotation_text='High diagnostic value', annotation_position='right')

    # Quadrant labels
    fig.add_annotation(x=3.5, y=0.9, text='주의 필요 (HIGH PRIORITY)', showarrow=False,
                       font=dict(size=11, color='#e53e3e'), opacity=0.3)
    fig.add_annotation(x=0.5, y=0.55, text='정상 범위 (LOW CONCERN)', showarrow=False,
                       font=dict(size=11, color='#38a169'), opacity=0.3)

    # Legend for tasks
    for task, color in TASK_COLORS.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers', name=task.replace('_', ' ').title(),
            marker=dict(size=8, color=color),
        ))

    fig.update_layout(
        title=dict(text=f'Evidence Map — {patient_id}', font=dict(size=14)),
        xaxis=dict(title='Abnormality (Z-score vs Healthy)', gridcolor='#edf2f7', zeroline=False),
        yaxis=dict(title='Diagnostic Power', range=[0.4, 1.0], gridcolor='#edf2f7', zeroline=False),
        legend=dict(orientation='h', y=-0.18, font=dict(size=10)),
    )
    return _apply_defaults(fig, height=480)


# ── 15. Motor Decay Cascade ─────────────────────────────────
def make_decay_cascade(ts_df, patient_id):
    """Sankey-like cascade showing how movement degrades over repetitions.
    Inspired by NYT Paths to White House."""
    if ts_df.empty:
        fig = go.Figure()
        fig.add_annotation(text='No data', showarrow=False,
                           xref='paper', yref='paper', x=0.5, y=0.5)
        return _apply_defaults(fig)

    signal = ts_df['left_amplitude'].values
    peaks, _ = find_peaks(signal, distance=5, prominence=0.05)

    if len(peaks) < 6:
        fig = go.Figure()
        fig.add_annotation(text='Not enough repetitions for cascade analysis',
                           showarrow=False, xref='paper', yref='paper', x=0.5, y=0.5)
        return _apply_defaults(fig)

    peak_amps = signal[peaks]
    n = len(peak_amps)

    states = ['Normal', 'Slight', 'Moderate', 'Severe']
    state_colors = ['rgba(56,161,105,1)', 'rgba(236,201,75,1)',
                    'rgba(221,107,32,1)', 'rgba(229,62,62,1)']
    link_alphas = ['rgba(56,161,105,0.4)', 'rgba(236,201,75,0.4)',
                   'rgba(221,107,32,0.4)', 'rgba(229,62,62,0.4)']

    max_amp = peak_amps.max()

    def classify(amp):
        ratio = amp / max_amp if max_amp > 0 else 0
        if ratio >= 0.8:
            return 0
        elif ratio >= 0.6:
            return 1
        elif ratio >= 0.4:
            return 2
        else:
            return 3

    # Build stages (groups of reps)
    n_stages = min(5, n)
    reps_per_stage = max(1, n // n_stages)
    stage_states = []
    for s in range(n_stages):
        start = s * reps_per_stage
        end = start + reps_per_stage if s < n_stages - 1 else n
        avg_amp = peak_amps[start:end].mean()
        stage_states.append(classify(avg_amp))

    # Build Sankey nodes — only used states per stage
    labels = []
    node_colors = []
    for stage_idx in range(n_stages):
        for state_idx, state_name in enumerate(states):
            labels.append(f"Stage {stage_idx+1}: {state_name}")
            node_colors.append(state_colors[state_idx])

    sources, targets, values, lcolors = [], [], [], []

    for stage_idx in range(n_stages - 1):
        curr_state = stage_states[stage_idx]
        next_state = stage_states[stage_idx + 1]

        src = stage_idx * 4 + curr_state
        tgt = (stage_idx + 1) * 4 + next_state
        sources.append(src)
        targets.append(tgt)
        values.append(10)
        lcolors.append(link_alphas[next_state])

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20, thickness=20,
            label=labels, color=node_colors,
        ),
        link=dict(
            source=sources, target=targets,
            value=values, color=lcolors,
        ),
    )])

    fig.update_layout(
        title=dict(text=f'Motor Decay Cascade — {patient_id}', font=dict(size=14)),
        **LAYOUT_DEFAULTS, height=400,
    )
    return fig


# ── 16. Clinical Confidence Compass ─────────────────────────
def make_confidence_compass(patient_summary):
    """Polar bar chart showing dominant abnormality direction and strength.
    Like a compass pointing toward the most critical feature."""
    if not patient_summary:
        fig = go.Figure()
        fig.add_annotation(text='No data', showarrow=False,
                           xref='paper', yref='paper', x=0.5, y=0.5)
        return _apply_defaults(fig)

    features = [
        'tremor_power', 'rhythm_irregularity',
        'left_right_asymmetry', 'movement_amplitude_reduction',
        'motion_instability',
    ]
    labels = ['Tremor', 'Rhythm', 'Asymmetry', 'Amplitude', 'Instability']
    angles = [0, 72, 144, 216, 288]  # evenly spaced

    values = [patient_summary.get(f, 0) for f in features]
    max_idx = values.index(max(values))

    # Bar colors: strongest = red, others = blue gradient
    colors = []
    for i, v in enumerate(values):
        if i == max_idx:
            colors.append('#e53e3e')
        elif v >= 0.5:
            colors.append('#dd6b20')
        else:
            colors.append('rgba(43, 108, 176, 0.6)')

    fig = go.Figure()

    # Bars
    fig.add_trace(go.Barpolar(
        r=values,
        theta=labels,
        marker=dict(color=colors, line=dict(color='white', width=1)),
        opacity=0.85,
        hovertemplate='%{theta}: %{r:.2f}<extra></extra>',
    ))

    # "Needle" pointing to dominant feature
    needle_angle = angles[max_idx]
    needle_r = values[max_idx]
    fig.add_trace(go.Scatterpolar(
        r=[0, needle_r * 1.05],
        theta=[labels[max_idx], labels[max_idx]],
        mode='lines+markers',
        line=dict(color='#e53e3e', width=3),
        marker=dict(size=[0, 10], color='#e53e3e', symbol=['circle', 'triangle-up']),
        showlegend=False,
        hoverinfo='skip',
    ))

    # Severity rings
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 1],
                tickvals=[0.25, 0.5, 0.75],
                ticktext=['Mild', 'Moderate', 'Severe'],
                tickfont=dict(size=9, color='#a0aec0'),
                gridcolor='#edf2f7',
            ),
            angularaxis=dict(tickfont=dict(size=12, color='#2d3748')),
        ),
        showlegend=False,
        title=dict(
            text=f'Clinical Compass<br>'
                 f'<span style="font-size:11px;color:#718096;">'
                 f'Dominant: {labels[max_idx]} ({values[max_idx]:.2f})</span>',
            font=dict(size=14),
        ),
        **LAYOUT_DEFAULTS, height=420,
    )
    return fig


# ══════════════════════════════════════════════════════════════
#  VIDEO ANALYSIS VISUALIZATIONS
# ══════════════════════════════════════════════════════════════

def make_motion_timeline(video_data, side):
    """Motion intensity over time from video frame differencing."""
    if not video_data:
        fig = go.Figure()
        fig.add_annotation(text='Data 없음', showarrow=False,
                           xref='paper', yref='paper', x=0.5, y=0.5)
        return _apply_defaults(fig)

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

    from plotly.subplots import make_subplots
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

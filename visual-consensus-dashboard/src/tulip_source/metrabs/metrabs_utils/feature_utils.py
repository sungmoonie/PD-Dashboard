from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd
import torch
from metrabs_utils.data_utils import metrabs_to_mediapipe_pose33


def get_num_joints_for_skeleton(estimator: torch.nn.Module, skeleton: str) -> int:
    if skeleton not in estimator.per_skeleton_joint_names:
        available = ", ".join(sorted(estimator.per_skeleton_joint_names.keys()))
        raise ValueError(f"Unsupported skeleton '{skeleton}'. Available: {available}")
    return len(estimator.per_skeleton_joint_names[skeleton])


def select_primary_pose(
    frame_boxes: torch.Tensor | None,
    frame_poses3d: torch.Tensor,
) -> np.ndarray | None:
    """
    Pick one person pose for gait from a frame prediction.
    Strategy: highest detector confidence from boxes[:, 4].
    """
    if frame_poses3d.numel() == 0 or frame_poses3d.shape[0] == 0:
        return None

    if frame_boxes is None or frame_boxes.numel() == 0 or frame_boxes.shape[0] == 0:
        return frame_poses3d[0].detach().cpu().numpy().astype(np.float32)

    best_idx = int(torch.argmax(frame_boxes[:, 4]).item())
    return frame_poses3d[best_idx].detach().cpu().numpy().astype(np.float32)


def pose_to_frame_vector(
    pose3d: np.ndarray | None,
    num_joints: int,
    fill_missing: float = 0.0,
) -> np.ndarray:
    if pose3d is None:
        return np.full((num_joints, 3), fill_missing, dtype=np.float32).reshape(-1)

    arr = np.asarray(pose3d, dtype=np.float32)
    if arr.shape != (num_joints, 3):
        raise ValueError(f"Unexpected pose shape {arr.shape}, expected ({num_joints}, 3)")
    return arr.reshape(-1)


def stack_frame_vectors(frame_vectors: Sequence[np.ndarray]) -> np.ndarray:
    if len(frame_vectors) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    return np.asarray(frame_vectors, dtype=np.float32)


# MediaPipe gait indices
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT, R_FOOT = 31, 32

GAIT_FRAME_FEATURE_COLUMNS: List[str] = [
    "frame_index",
    # Temporal (12)
    "temporal_left_heel_speed",
    "temporal_right_heel_speed",
    "temporal_left_ankle_speed",
    "temporal_right_ankle_speed",
    "temporal_left_foot_speed",
    "temporal_right_foot_speed",
    "temporal_pelvis_speed",
    "temporal_left_heel_vertical_delta",
    "temporal_right_heel_vertical_delta",
    "temporal_left_step_time_proxy",
    "temporal_right_step_time_proxy",
    "temporal_cadence_proxy",
    # Spatial (3)
    "spatial_step_width",
    "spatial_pelvis_height",
    "spatial_foot_clearance_diff",
    # Spatiotemporal (1)
    "spatiotemporal_gait_speed",
]


def compute_gait_framewise_features(
    pose_33x3: np.ndarray,
    fps: float = 30.0,
) -> pd.DataFrame:
    """
    Frame transition features over (frame_index-1 -> frame_index).
    Output feature count: Temporal 12 / Spatial 3 / Spatiotemporal 1.
    """
    if pose_33x3.ndim != 3 or pose_33x3.shape[1:] != (33, 3):
        raise ValueError(f"Expected [T, 33, 3], got {pose_33x3.shape}")

    n_frames = pose_33x3.shape[0]
    if n_frames < 2:
        return pd.DataFrame(columns=GAIT_FRAME_FEATURE_COLUMNS)

    dt = 1.0 / float(fps)
    left_heel = pose_33x3[:, L_HEEL]
    right_heel = pose_33x3[:, R_HEEL]
    left_ankle = pose_33x3[:, L_ANKLE]
    right_ankle = pose_33x3[:, R_ANKLE]
    left_foot = pose_33x3[:, L_FOOT]
    right_foot = pose_33x3[:, R_FOOT]
    pelvis = (pose_33x3[:, L_HIP] + pose_33x3[:, R_HIP]) / 2.0

    d_left_heel = np.diff(left_heel, axis=0)
    d_right_heel = np.diff(right_heel, axis=0)
    d_left_ankle = np.diff(left_ankle, axis=0)
    d_right_ankle = np.diff(right_ankle, axis=0)
    d_left_foot = np.diff(left_foot, axis=0)
    d_right_foot = np.diff(right_foot, axis=0)
    d_pelvis = np.diff(pelvis, axis=0)

    left_heel_speed = np.linalg.norm(d_left_heel, axis=1) / dt
    right_heel_speed = np.linalg.norm(d_right_heel, axis=1) / dt
    left_ankle_speed = np.linalg.norm(d_left_ankle, axis=1) / dt
    right_ankle_speed = np.linalg.norm(d_right_ankle, axis=1) / dt
    left_foot_speed = np.linalg.norm(d_left_foot, axis=1) / dt
    right_foot_speed = np.linalg.norm(d_right_foot, axis=1) / dt
    pelvis_speed = np.linalg.norm(d_pelvis, axis=1) / dt

    left_step_time_proxy = dt / np.clip(left_heel_speed, 1e-6, None)
    right_step_time_proxy = dt / np.clip(right_heel_speed, 1e-6, None)
    cadence_proxy = 60.0 / np.clip(
        (left_step_time_proxy + right_step_time_proxy) / 2.0, 1e-6, None
    )

    step_width = np.abs(left_heel[1:, 0] - right_heel[1:, 0])
    pelvis_height = pelvis[1:, 1]
    foot_clearance_diff = np.abs(left_foot[1:, 1] - right_foot[1:, 1])
    gait_speed = d_pelvis[:, 2] / dt

    return pd.DataFrame(
        {
            "frame_index": np.arange(1, n_frames, dtype=np.int32),
            "temporal_left_heel_speed": left_heel_speed.astype(np.float32),
            "temporal_right_heel_speed": right_heel_speed.astype(np.float32),
            "temporal_left_ankle_speed": left_ankle_speed.astype(np.float32),
            "temporal_right_ankle_speed": right_ankle_speed.astype(np.float32),
            "temporal_left_foot_speed": left_foot_speed.astype(np.float32),
            "temporal_right_foot_speed": right_foot_speed.astype(np.float32),
            "temporal_pelvis_speed": pelvis_speed.astype(np.float32),
            "temporal_left_heel_vertical_delta": d_left_heel[:, 1].astype(np.float32),
            "temporal_right_heel_vertical_delta": d_right_heel[:, 1].astype(np.float32),
            "temporal_left_step_time_proxy": left_step_time_proxy.astype(np.float32),
            "temporal_right_step_time_proxy": right_step_time_proxy.astype(np.float32),
            "temporal_cadence_proxy": cadence_proxy.astype(np.float32),
            "spatial_step_width": step_width.astype(np.float32),
            "spatial_pelvis_height": pelvis_height.astype(np.float32),
            "spatial_foot_clearance_diff": foot_clearance_diff.astype(np.float32),
            "spatiotemporal_gait_speed": gait_speed.astype(np.float32),
        }
    )


def compute_gait_video_features(metrabs_24x3: np.ndarray, fps: float = 30.0) -> dict:
    pose33 = metrabs_to_mediapipe_pose33(metrabs_24x3)
    return compute_gait_video_features_pose33(pose33, fps=fps)


def _safe_mean(values: np.ndarray, fallback: float = 0.0) -> float:
    vals = np.asarray(values, dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float(fallback)
    return float(np.mean(vals))


def _step_intervals_from_signal(signal: np.ndarray, fps: float) -> np.ndarray:
    """
    Estimate step intervals (seconds) from 1D trajectory by local maxima detection.
    Uses only numpy to keep this module self-contained.
    """
    if signal.size < 5 or fps <= 0:
        return np.asarray([], dtype=np.float32)

    centered = signal - np.nanmean(signal)
    peaks: list[int] = []
    for i in range(1, centered.size - 1):
        if centered[i] > centered[i - 1] and centered[i] >= centered[i + 1]:
            peaks.append(i)
    if len(peaks) < 2:
        return np.asarray([], dtype=np.float32)

    intervals = np.diff(np.asarray(peaks, dtype=np.float32)) / float(fps)
    return intervals[intervals > 0]


def compute_gait_video_features_pose33(pose_33x3: np.ndarray, fps: float = 30.0) -> dict:
    """
    Self-contained gait video summary features from [T,33,3] pose sequence.
    """
    if pose_33x3.ndim != 3 or pose_33x3.shape[1:] != (33, 3):
        raise ValueError(f"Expected [T, 33, 3], got {pose_33x3.shape}")

    left_heel = pose_33x3[:, L_HEEL]
    right_heel = pose_33x3[:, R_HEEL]
    left_ankle = pose_33x3[:, L_ANKLE]
    right_ankle = pose_33x3[:, R_ANKLE]
    pelvis = (pose_33x3[:, L_HIP] + pose_33x3[:, R_HIP]) / 2.0

    left_steps = _step_intervals_from_signal(left_heel[:, 2], fps=fps)
    right_steps = _step_intervals_from_signal(right_heel[:, 2], fps=fps)

    left_step_duration = _safe_mean(left_steps)
    right_step_duration = _safe_mean(right_steps)
    average_step_duration = _safe_mean(np.concatenate([left_steps, right_steps]) if left_steps.size + right_steps.size > 0 else np.asarray([0.0], dtype=np.float32))
    average_stride_duration = float(average_step_duration * 2.0) if average_step_duration > 0 else 0.0

    # Heuristic split for stance/swing when force/foot-contact labels are unavailable.
    left_stance_time = float(left_step_duration * 0.6)
    right_stance_time = float(right_step_duration * 0.6)
    left_swing_time = float(left_step_duration * 0.4)
    right_swing_time = float(right_step_duration * 0.4)
    average_stance_time = _safe_mean(np.asarray([left_stance_time, right_stance_time], dtype=np.float32))
    average_swing_time = _safe_mean(np.asarray([left_swing_time, right_swing_time], dtype=np.float32))
    double_support_time = float(average_stance_time * 0.2)

    if pose_33x3.shape[0] > 1 and fps > 0:
        pelvis_vel = np.diff(pelvis[:, 2]) * fps
        gait_speed = _safe_mean(np.abs(pelvis_vel))
    else:
        gait_speed = 0.0

    left_step_length = _safe_mean(np.abs(np.diff(left_ankle[:, 2])))
    right_step_length = _safe_mean(np.abs(np.diff(right_ankle[:, 2])))
    average_step_length = _safe_mean(np.asarray([left_step_length, right_step_length], dtype=np.float32))

    cadence = float(60.0 / average_step_duration) if average_step_duration > 1e-6 else 0.0

    return {
        "left_swing_time": float(left_swing_time),
        "right_swing_time": float(right_swing_time),
        "average_swing_time": float(average_swing_time),
        "left_stance_time": float(left_stance_time),
        "right_stance_time": float(right_stance_time),
        "average_stance_time": float(average_stance_time),
        "double_support_time": float(double_support_time),
        "left_step_duration": float(left_step_duration),
        "right_step_duration": float(right_step_duration),
        "average_step_duration": float(average_step_duration),
        "average_stride_duration": float(average_stride_duration),
        "cadence": float(cadence),
        "left_step_length": float(left_step_length),
        "right_step_length": float(right_step_length),
        "average_step_length": float(average_step_length),
        "gait_speed": float(gait_speed),
    }


def gait_video_features_to_dataframe(features: dict) -> pd.DataFrame:
    ordered_cols = [
        "left_swing_time",
        "right_swing_time",
        "average_swing_time",
        "left_stance_time",
        "right_stance_time",
        "average_stance_time",
        "double_support_time",
        "left_step_duration",
        "right_step_duration",
        "average_step_duration",
        "average_stride_duration",
        "cadence",
        "left_step_length",
        "right_step_length",
        "average_step_length",
        "gait_speed",
    ]
    row = {k: float(features.get(k, 0.0)) for k in ordered_cols}
    return pd.DataFrame([row], columns=ordered_cols)

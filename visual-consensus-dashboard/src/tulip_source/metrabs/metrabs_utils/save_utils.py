from __future__ import annotations

import os

import numpy as np
import pandas as pd


def save_kinematic_npz(
    save_root_dir: str,
    subject: str,
    clip_filename: str,
    kinematic_array: np.ndarray,
    dataset_name: str = "gait",
    method_name: str = "metrabs",
) -> str:
    """
    Save kinematic data to:
      save_root_dir/<dataset_name>/<method_name>/<subject>/<clip_basename>.npz
    """
    subject_dir = os.path.join(
        save_root_dir, dataset_name, method_name, str(subject)
    )
    os.makedirs(subject_dir, exist_ok=True)

    clip_basename = os.path.splitext(os.path.basename(clip_filename))[0]
    save_path = os.path.join(subject_dir, f"{clip_basename}.npz")

    np.savez_compressed(save_path, kinematic=kinematic_array)
    return save_path


def save_feature_csv(
    save_root_dir: str,
    subject: str,
    clip_stem: str,
    feature_df: pd.DataFrame,
) -> str:
    subject_dir = os.path.join(save_root_dir, str(subject))
    os.makedirs(subject_dir, exist_ok=True)
    save_path = os.path.join(subject_dir, f"{clip_stem}.csv")
    feature_df.to_csv(save_path, index=False)
    return save_path


def ensure_video_output_dir(out_dir: str) -> str:
    # [FIX] 단일 비디오 출력 디렉터리 생성
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def save_raw_kinematic_csv(out_dir: str, kinematic_jx3: np.ndarray) -> str:
    # [FIX] 프레임별 raw pose 좌표를 kinematic.csv 로 저장 (smpl_24)
    if kinematic_jx3.ndim != 3 or kinematic_jx3.shape[2] != 3:
        raise ValueError(f"Expected [T, J, 3], got {kinematic_jx3.shape}")

    ensure_video_output_dir(out_dir)
    n_frames, n_joints, _ = kinematic_jx3.shape
    columns = ["frame_index"]
    for joint_idx in range(n_joints):
        columns.extend([f"joint_{joint_idx}_x", f"joint_{joint_idx}_y", f"joint_{joint_idx}_z"])

    flat = kinematic_jx3.reshape(n_frames, n_joints * 3)
    data = {"frame_index": np.arange(n_frames, dtype=np.int32)}
    for col_idx, col_name in enumerate(columns[1:]):
        data[col_name] = flat[:, col_idx]

    save_path = os.path.join(out_dir, "kinematic.csv")
    pd.DataFrame(data).to_csv(save_path, index=False)
    return save_path


def save_gait_feature_csv(out_dir: str, feature_df: pd.DataFrame) -> str:
    # [FIX] video-level gait feature를 feature.csv 로 저장
    ensure_video_output_dir(out_dir)
    save_path = os.path.join(out_dir, "feature.csv")
    feature_df.to_csv(save_path, index=False)
    return save_path

from __future__ import annotations

import os
from typing import Dict, Iterator

import numpy as np
import pandas as pd

SOURCE_CLIP_PREFIX = "D:/clip/finger_tapping"
GAIT_SOURCE_CLIP_PREFIX = "D:/clip/gait"
DATASET_SOURCE_PREFIX = "D:/clip"
DEFAULT_METRABS_GAIT_KINEMATIC_ROOT = (
    "/workspace/pcos_dataset/results/kinematic_data/gait/metrabs"
)

# smpl_24 names:
# ['pelv','lhip','rhip','spi1','lkne','rkne','spi2','lank','rank',
#  'spi3','ltoe','rtoe','neck','lcla','rcla','head','lsho','rsho',
#  'lelb','relb','lwri','rwri','lhan','rhan']
SMPL24_INDEX = {
    "pelv": 0,
    "lhip": 1,
    "rhip": 2,
    "lkne": 4,
    "rkne": 5,
    "lank": 7,
    "rank": 8,
    "ltoe": 10,
    "rtoe": 11,
}


def load_kinematic_test_df(
    csv_path: str,
    root_dir: str,
    source_clip_prefix: str,
    split: str = "test",
) -> pd.DataFrame:
    """Load split CSV, rewrite clip paths, and filter by split."""
    df = pd.read_csv(csv_path)
    required_cols = {"subject", "clip_path", "clip_filename", "split"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        missing_str = ", ".join(sorted(missing_cols))
        raise ValueError(f"CSV missing required column(s): {missing_str}")

    normalized_root = root_dir.rstrip("/\\")
    df["clip_path"] = (
        df["clip_path"]
        .astype(str)
        .str.replace(source_clip_prefix, normalized_root, regex=False)
    )
    split_lower = split.lower()
    if split_lower == "all":
        return df.reset_index(drop=True)
    if split_lower == "test":
        return df[df["split"].astype(str).str.lower() == "test"].reset_index(drop=True)
    raise ValueError("split must be either 'test' or 'all'.")


def load_finger_tapping_test_df(
    csv_path: str,
    root_dir: str = "/workspace/pcos_dataset/finger_tapping",
    split: str = "test",
) -> pd.DataFrame:
    """Backward-compatible wrapper for finger tapping split loading."""
    return load_kinematic_test_df(
        csv_path=csv_path,
        root_dir=root_dir,
        source_clip_prefix=SOURCE_CLIP_PREFIX,
        split=split,
    )


def load_gait_test_df(
    csv_path: str = "/workspace/pcos_dataset/gait/gait_v2_split.csv",
    root_dir: str = "/workspace/pcos_dataset",
    split: str = "test",
) -> pd.DataFrame:
    """Load gait split CSV and rewrite dataset root from D:/clip to workspace path."""
    return load_kinematic_test_df(
        csv_path=csv_path,
        root_dir=root_dir,
        source_clip_prefix=DATASET_SOURCE_PREFIX,
        split=split,
    )


def iter_test_samples(df: pd.DataFrame) -> Iterator[Dict[str, str]]:
    """Yield standardized sample metadata for each test row."""
    required_cols = {"subject", "clip_path"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        missing_str = ", ".join(sorted(missing_cols))
        raise ValueError(f"Missing required column(s): {missing_str}")

    for row in df.itertuples(index=False):
        clip_path = str(getattr(row, "clip_path"))
        subject_raw = str(getattr(row, "subject"))
        try:
            subject = f"{int(float(subject_raw)):03d}"
        except ValueError:
            subject = subject_raw.zfill(3)

        clip_filename = str(getattr(row, "clip_filename", "")) or os.path.basename(
            clip_path
        )

        yield {
            "subject": subject,
            "clip_path": clip_path,
            "clip_filename": clip_filename,
        }


def load_pose24_from_kinematic_csv(csv_path: str) -> np.ndarray:
    # [FIX] MeTRAbs kinematic.csv → [T, 24, 3] 배열 로드 (overlay 재추론 생략용)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Kinematic CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    n_frames = len(df)
    pose24 = np.zeros((n_frames, 24, 3), dtype=np.float32)
    for joint_idx in range(24):
        pose24[:, joint_idx, 0] = df[f"joint_{joint_idx}_x"].to_numpy(dtype=np.float32)
        pose24[:, joint_idx, 1] = df[f"joint_{joint_idx}_y"].to_numpy(dtype=np.float32)
        pose24[:, joint_idx, 2] = df[f"joint_{joint_idx}_z"].to_numpy(dtype=np.float32)
    return pose24


def load_metrabs_gait_kinematic_npz(npz_path: str) -> np.ndarray:
    """Load MeTRAbs gait npz and return [T, 24, 3] array."""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    npz_data = np.load(npz_path)
    if "kinematic" not in npz_data:
        raise ValueError(f"'kinematic' key not found: {npz_path}")

    kinematic = np.asarray(npz_data["kinematic"], dtype=np.float32)
    if kinematic.ndim != 2:
        raise ValueError(f"Expected [T, J*3], got {kinematic.shape} for {npz_path}")
    if kinematic.shape[1] % 3 != 0:
        raise ValueError(f"Invalid joint dimension (not divisible by 3): {kinematic.shape}")

    n_joints = kinematic.shape[1] // 3
    if n_joints != 24:
        raise ValueError(f"Expected smpl_24 [T,72], got {kinematic.shape} for {npz_path}")
    return kinematic.reshape(kinematic.shape[0], n_joints, 3)


def metrabs_to_mediapipe_pose33(metrabs_24x3: np.ndarray) -> np.ndarray:
    """
    Convert MeTRAbs smpl_24 pose to MediaPipe-like [T, 33, 3].
    Only gait-related indices are populated:
      23,24,25,26,27,28,29,30,31,32
    """
    if metrabs_24x3.ndim != 3 or metrabs_24x3.shape[1:] != (24, 3):
        raise ValueError(f"Expected [T,24,3], got {metrabs_24x3.shape}")

    t = metrabs_24x3.shape[0]
    pose33 = np.full((t, 33, 3), np.nan, dtype=np.float32)

    pose33[:, 23] = metrabs_24x3[:, SMPL24_INDEX["lhip"]]
    pose33[:, 24] = metrabs_24x3[:, SMPL24_INDEX["rhip"]]
    pose33[:, 25] = metrabs_24x3[:, SMPL24_INDEX["lkne"]]
    pose33[:, 26] = metrabs_24x3[:, SMPL24_INDEX["rkne"]]
    pose33[:, 27] = metrabs_24x3[:, SMPL24_INDEX["lank"]]
    pose33[:, 28] = metrabs_24x3[:, SMPL24_INDEX["rank"]]
    # Heel proxy: ankle (smpl_24 has no explicit heel joint)
    pose33[:, 29] = metrabs_24x3[:, SMPL24_INDEX["lank"]]
    pose33[:, 30] = metrabs_24x3[:, SMPL24_INDEX["rank"]]
    # Foot index proxy: toe
    pose33[:, 31] = metrabs_24x3[:, SMPL24_INDEX["ltoe"]]
    pose33[:, 32] = metrabs_24x3[:, SMPL24_INDEX["rtoe"]]
    return pose33

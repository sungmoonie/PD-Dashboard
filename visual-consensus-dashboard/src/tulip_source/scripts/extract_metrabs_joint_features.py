#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
TULIP_SOURCE_DIR = SCRIPT_DIR.parent
METRABS_DIR = TULIP_SOURCE_DIR / "metrabs"
if str(METRABS_DIR) not in sys.path:
    sys.path.append(str(METRABS_DIR))

VENDOR_DIR = METRABS_DIR / "vendor"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

from paths import DEFAULT_MODEL_DIR  # type: ignore

METRABS_ROOT = "/workspace/model/metrabs"
if METRABS_ROOT not in sys.path:
    sys.path.append(METRABS_ROOT)

import metrabs_pytorch.backbones.efficientnet as effnet_pt
import metrabs_pytorch.models.metrabs as metrabs_pt
from metrabs_pytorch.multiperson.multiperson_model import Pose3dEstimator
from metrabs_pytorch.util import get_config
from metrabs_utils.joint_info import JointInfo

MODEL_BUNDLE_FILES = (
    "config.yaml",
    "ckpt.pt",
    "joint_info.npz",
    "skeleton_infos.pkl",
    "joint_transform_matrix.npy",
)


def _validate_model_bundle(model_dir: str) -> None:
    missing = [
        filename
        for filename in MODEL_BUNDLE_FILES
        if not os.path.exists(os.path.join(model_dir, filename))
    ]
    if missing:
        raise FileNotFoundError(
            f"MeTRAbs checkpoint bundle is incomplete in {model_dir}: {', '.join(missing)}"
        )


def _load_crop_model(model_dir: str, device: torch.device) -> torch.nn.Module:
    get_config(os.path.join(model_dir, "config.yaml"))
    cfg = get_config()
    joint_info_np = np.load(os.path.join(model_dir, "joint_info.npz"), allow_pickle=True)
    joint_info = JointInfo(joint_info_np["joint_names"], joint_info_np["joint_edges"])

    backbone_raw = getattr(effnet_pt, f"efficientnet_v2_{cfg.efficientnet_size}")()
    preproc_layer = effnet_pt.PreprocLayer()
    backbone = torch.nn.Sequential(preproc_layer, backbone_raw.features)
    crop_model = metrabs_pt.Metrabs(backbone, joint_info).to(device)
    crop_model.eval()

    with torch.inference_mode():
        dummy_inp = torch.zeros(
            (1, 3, int(cfg.proc_side), int(cfg.proc_side)),
            dtype=torch.float32,
            device=device,
        )
        dummy_intr = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0)
        crop_model((dummy_inp, dummy_intr))

    state_dict = torch.load(os.path.join(model_dir, "ckpt.pt"), map_location=device)
    crop_model.load_state_dict(state_dict)
    crop_model.joint_info.mirror_mapping = torch.as_tensor(
        crop_model.joint_info.mirror_mapping,
        dtype=torch.long,
        device=device,
    )
    return crop_model


def _load_pose_estimator(model_dir: str, device: torch.device) -> Pose3dEstimator:
    crop_model = _load_crop_model(model_dir=model_dir, device=device)

    with open(os.path.join(model_dir, "skeleton_infos.pkl"), "rb") as file:
        skeleton_infos = pickle.load(file)
    joint_transform_matrix = np.load(os.path.join(model_dir, "joint_transform_matrix.npy"))

    estimator = Pose3dEstimator(crop_model, skeleton_infos, joint_transform_matrix).to(device)
    estimator.joint_transform_matrix = estimator.joint_transform_matrix.to(device)
    estimator.skeleton_joint_indices_table = {
        key: np.asarray(indices).astype(np.int64).tolist()
        for key, indices in estimator.skeleton_joint_indices_table.items()
    }
    estimator.eval()
    return estimator


def _num_joints(estimator: Pose3dEstimator, skeleton: str) -> int:
    if skeleton not in estimator.per_skeleton_joint_names:
        available = ", ".join(sorted(estimator.per_skeleton_joint_names.keys()))
        raise ValueError(f"Unsupported skeleton '{skeleton}'. Available: {available}")
    return len(estimator.per_skeleton_joint_names[skeleton])


def _select_primary_pose(frame_boxes: torch.Tensor | None, frame_poses3d: torch.Tensor) -> np.ndarray | None:
    if frame_poses3d.numel() == 0 or frame_poses3d.shape[0] == 0:
        return None
    if frame_boxes is None or frame_boxes.numel() == 0 or frame_boxes.shape[0] == 0:
        return frame_poses3d[0].detach().cpu().numpy().astype(np.float32)
    best_idx = int(torch.argmax(frame_boxes[:, 4]).item())
    return frame_poses3d[best_idx].detach().cpu().numpy().astype(np.float32)


def infer_video_kinematics(
    clip_path: str,
    estimator: Pose3dEstimator,
    skeleton: str,
    device: torch.device,
    batch_size: int,
    num_aug: int,
) -> np.ndarray:
    capture = cv2.VideoCapture(clip_path)
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {clip_path}")

    n_joints = _num_joints(estimator, skeleton)
    frame_vectors: list[np.ndarray] = []
    frame_tensors: list[torch.Tensor] = []

    def flush_batch() -> None:
        if not frame_tensors:
            return
        images = torch.stack(frame_tensors, dim=0).to(device=device, dtype=torch.uint8)
        bsz, _, h, w = images.shape
        full_boxes = torch.tensor(
            [[[0.0, 0.0, float(w - 1), float(h - 1)]] for _ in range(bsz)],
            dtype=torch.float32,
            device=device,
        )
        unknown_intr = torch.tensor(
            [[[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]]],
            dtype=torch.float32,
            device=device,
        )
        distortion = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
        extrinsic = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)
        world_up = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32, device=device)

        with torch.inference_mode():
            pred = estimator.estimate_poses_batched(
                images=images,
                boxes=full_boxes,
                intrinsic_matrix=unknown_intr,
                distortion_coeffs=distortion,
                extrinsic_matrix=extrinsic,
                world_up_vector=world_up,
                skeleton=skeleton,
                num_aug=num_aug,
            )

        frame_boxes_list = pred["boxes"] if "boxes" in pred else [None] * len(pred["poses3d"])
        for boxes, poses3d in zip(frame_boxes_list, pred["poses3d"]):
            pose3d = _select_primary_pose(frame_boxes=boxes, frame_poses3d=poses3d)
            if pose3d is None:
                frame_vectors.append(np.zeros((n_joints * 3,), dtype=np.float32))
            else:
                frame_vectors.append(pose3d.reshape(-1).astype(np.float32))
        frame_tensors.clear()

    while True:
        success, frame_bgr = capture.read()
        if not success:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_tensors.append(torch.from_numpy(frame_rgb).permute(2, 0, 1).contiguous())
        if len(frame_tensors) >= batch_size:
            flush_batch()
    capture.release()
    flush_batch()

    if not frame_vectors:
        return np.zeros((0, n_joints, 3), dtype=np.float32)
    arr = np.asarray(frame_vectors, dtype=np.float32)
    return arr.reshape(arr.shape[0], n_joints, 3)


def get_video_fps(video_path: str, fallback_fps: float = 30.0) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps is None or fps <= 0:
        return fallback_fps
    return float(fps)


def make_joint_dataframe(kinematic_jx3: np.ndarray) -> pd.DataFrame:
    n_frames, n_joints, _ = kinematic_jx3.shape
    data: dict[str, np.ndarray] = {"frame_index": np.arange(n_frames, dtype=np.int32)}
    flat = kinematic_jx3.reshape(n_frames, n_joints * 3)
    col_idx = 0
    for joint_idx in range(n_joints):
        data[f"joint_{joint_idx}_x"] = flat[:, col_idx]
        data[f"joint_{joint_idx}_y"] = flat[:, col_idx + 1]
        data[f"joint_{joint_idx}_z"] = flat[:, col_idx + 2]
        col_idx += 3
    return pd.DataFrame(data)


def _is_toe_tapping_task(task_name: str) -> bool:
    t = task_name.lower()
    return "toe_tapping" in t or "toe tapping" in t


def _is_resting_tremor_task(task_name: str) -> bool:
    t = task_name.lower()
    return "resting" in t and "tremor" in t


def _moving_average_1d(x: np.ndarray, win: int = 5) -> np.ndarray:
    if win <= 1 or x.size == 0:
        return x.astype(np.float32)
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(xp, kernel, mode="valid").astype(np.float32)


def _make_toe_tapping_features(kinematic_jx3: np.ndarray, fps: float) -> pd.DataFrame:
    # smpl_24 indices
    l_knee, r_knee = 4, 5
    l_ankle, r_ankle = 7, 8
    l_toe, r_toe = 10, 11

    dt = 1.0 / float(fps)
    d = np.diff(kinematic_jx3, axis=0)
    speed = np.linalg.norm(d, axis=2) / dt

    left_toe_speed = speed[:, l_toe]
    right_toe_speed = speed[:, r_toe]
    left_ankle_speed = speed[:, l_ankle]
    right_ankle_speed = speed[:, r_ankle]
    left_knee_speed = speed[:, l_knee]
    right_knee_speed = speed[:, r_knee]

    # Toe tapping emphasizes vertical oscillation. Use y-axis delta as proxy.
    left_toe_vertical_delta = np.abs(np.diff(kinematic_jx3[:, l_toe, 1]))
    right_toe_vertical_delta = np.abs(np.diff(kinematic_jx3[:, r_toe, 1]))
    toe_tapping_rate_proxy = 0.5 * (left_toe_speed + right_toe_speed)
    toe_lr_asymmetry = np.abs(left_toe_speed - right_toe_speed) / np.clip(
        0.5 * (left_toe_speed + right_toe_speed), 1e-6, None
    )

    return pd.DataFrame(
        {
            "frame_index": np.arange(1, kinematic_jx3.shape[0], dtype=np.int32),
            "left_toe_speed": left_toe_speed.astype(np.float32),
            "right_toe_speed": right_toe_speed.astype(np.float32),
            "left_ankle_speed": left_ankle_speed.astype(np.float32),
            "right_ankle_speed": right_ankle_speed.astype(np.float32),
            "left_knee_speed": left_knee_speed.astype(np.float32),
            "right_knee_speed": right_knee_speed.astype(np.float32),
            "left_toe_vertical_delta": left_toe_vertical_delta.astype(np.float32),
            "right_toe_vertical_delta": right_toe_vertical_delta.astype(np.float32),
            "toe_tapping_rate_proxy": toe_tapping_rate_proxy.astype(np.float32),
            "toe_lr_asymmetry": toe_lr_asymmetry.astype(np.float32),
        }
    )


def _make_resting_tremor_features(kinematic_jx3: np.ndarray, fps: float) -> pd.DataFrame:
    # smpl_24 indices
    l_sho, r_sho = 16, 17
    l_elb, r_elb = 18, 19
    l_wri, r_wri = 20, 21
    l_han, r_han = 22, 23

    dt = 1.0 / float(fps)
    d = np.diff(kinematic_jx3, axis=0)
    speed = np.linalg.norm(d, axis=2) / dt

    left_wrist_speed = speed[:, l_wri]
    right_wrist_speed = speed[:, r_wri]
    left_elbow_speed = speed[:, l_elb]
    right_elbow_speed = speed[:, r_elb]
    left_hand_speed = speed[:, l_han]
    right_hand_speed = speed[:, r_han]

    # Tremor proxy: high-frequency residual after simple smoothing.
    l_wrist_x = kinematic_jx3[:, l_wri, 0]
    r_wrist_x = kinematic_jx3[:, r_wri, 0]
    l_wrist_res = l_wrist_x - _moving_average_1d(l_wrist_x, win=7)
    r_wrist_res = r_wrist_x - _moving_average_1d(r_wrist_x, win=7)
    left_tremor_amp_proxy = np.abs(np.diff(l_wrist_res)) * fps
    right_tremor_amp_proxy = np.abs(np.diff(r_wrist_res)) * fps

    left_shoulder_to_wrist = np.linalg.norm(
        kinematic_jx3[1:, l_sho] - kinematic_jx3[1:, l_wri], axis=1
    )
    right_shoulder_to_wrist = np.linalg.norm(
        kinematic_jx3[1:, r_sho] - kinematic_jx3[1:, r_wri], axis=1
    )
    upper_limb_lr_asymmetry = np.abs(left_wrist_speed - right_wrist_speed) / np.clip(
        0.5 * (left_wrist_speed + right_wrist_speed), 1e-6, None
    )

    return pd.DataFrame(
        {
            "frame_index": np.arange(1, kinematic_jx3.shape[0], dtype=np.int32),
            "left_shoulder_to_wrist_distance": left_shoulder_to_wrist.astype(np.float32),
            "right_shoulder_to_wrist_distance": right_shoulder_to_wrist.astype(np.float32),
            "left_elbow_speed": left_elbow_speed.astype(np.float32),
            "right_elbow_speed": right_elbow_speed.astype(np.float32),
            "left_wrist_speed": left_wrist_speed.astype(np.float32),
            "right_wrist_speed": right_wrist_speed.astype(np.float32),
            "left_hand_speed": left_hand_speed.astype(np.float32),
            "right_hand_speed": right_hand_speed.astype(np.float32),
            "left_tremor_amp_proxy": left_tremor_amp_proxy.astype(np.float32),
            "right_tremor_amp_proxy": right_tremor_amp_proxy.astype(np.float32),
            "upper_limb_lr_asymmetry": upper_limb_lr_asymmetry.astype(np.float32),
        }
    )


def _make_generic_features(kinematic_jx3: np.ndarray, fps: float) -> pd.DataFrame:
    if kinematic_jx3.shape[0] < 2:
        return pd.DataFrame(
            columns=[
                "frame_index",
                "mean_joint_speed",
                "max_joint_speed",
                "pelvis_speed",
                "joint_motion_energy",
                "left_right_wrist_distance",
            ]
        )

    dt = 1.0 / float(fps)
    d = np.diff(kinematic_jx3, axis=0)
    speed = np.linalg.norm(d, axis=2) / dt  # [T-1, J]

    pelvis_idx = 0
    left_wrist_idx = 20 if kinematic_jx3.shape[1] > 21 else 0
    right_wrist_idx = 21 if kinematic_jx3.shape[1] > 21 else 1

    mean_joint_speed = speed.mean(axis=1)
    max_joint_speed = speed.max(axis=1)
    pelvis_speed = speed[:, pelvis_idx]
    motion_energy = (speed ** 2).mean(axis=1)
    wrist_dist = np.linalg.norm(
        kinematic_jx3[1:, left_wrist_idx] - kinematic_jx3[1:, right_wrist_idx],
        axis=1,
    )

    return pd.DataFrame(
        {
            "frame_index": np.arange(1, kinematic_jx3.shape[0], dtype=np.int32),
            "mean_joint_speed": mean_joint_speed.astype(np.float32),
            "max_joint_speed": max_joint_speed.astype(np.float32),
            "pelvis_speed": pelvis_speed.astype(np.float32),
            "joint_motion_energy": motion_energy.astype(np.float32),
            "left_right_wrist_distance": wrist_dist.astype(np.float32),
        }
    )


def make_frame_feature_dataframe(kinematic_jx3: np.ndarray, fps: float, task_name: str) -> pd.DataFrame:
    if _is_toe_tapping_task(task_name):
        return _make_toe_tapping_features(kinematic_jx3, fps=fps)
    if _is_resting_tremor_task(task_name):
        return _make_resting_tremor_features(kinematic_jx3, fps=fps)
    return _make_generic_features(kinematic_jx3, fps=fps)


def run_folder(
    video_dir: str,
    save_dir: str,
    model_dir: str,
    skeleton: str,
    device_name: str,
    batch_size: int,
    num_aug: int,
) -> tuple[int, int]:
    video_path = Path(video_dir).resolve()
    output_root = Path(save_dir).resolve()
    task_name = video_path.name
    output_path = output_root / task_name
    output_path.mkdir(parents=True, exist_ok=True)

    if not video_path.exists() or not video_path.is_dir():
        raise FileNotFoundError(f"video_dir not found or not a directory: {video_path}")

    mp4_files = sorted(video_path.glob("*.mp4"))
    if not mp4_files:
        raise FileNotFoundError(f"No .mp4 files found in: {video_path}")

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    _validate_model_bundle(model_dir)
    estimator = _load_pose_estimator(model_dir=model_dir, device=device)

    print(f"[INFO] task_name={task_name}")
    print(f"[INFO] output_dir={output_path}")

    success = 0
    failure = 0
    for clip in mp4_files:
        stem = clip.stem
        joint_csv = output_path / f"{stem}_joint.csv"
        feat_csv = output_path / f"{stem}_features.csv"
        print(f"[RUN] {clip.name}")
        try:
            kinematic = infer_video_kinematics(
                clip_path=str(clip),
                estimator=estimator,
                skeleton=skeleton,
                device=device,
                batch_size=batch_size,
                num_aug=num_aug,
            )
            fps = get_video_fps(str(clip))
            joint_df = make_joint_dataframe(kinematic)
            feat_df = make_frame_feature_dataframe(kinematic, fps=fps, task_name=task_name)
            joint_df.to_csv(joint_csv, index=False)
            feat_df.to_csv(feat_csv, index=False)
            print(f"[OK] {joint_csv}")
            print(f"[OK] {feat_csv}")
            success += 1
        except Exception as exc:  # noqa: BLE001
            failure += 1
            print(f"[ERROR] {clip.name}: {exc}")
    return success, failure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract MeTRAbs joint CSV and joint-based frame feature CSV for all videos in a folder."
    )
    parser.add_argument("--video_dir", required=True, help="Input folder containing mp4 files.")
    parser.add_argument("--save_dir", required=True, help="Output folder for *_joint.csv and *_features.csv.")
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR, help="MeTRAbs checkpoint bundle directory.")
    parser.add_argument("--skeleton", default="smpl_24", help="Skeleton name (default: smpl_24).")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_aug", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    s, f = run_folder(
        video_dir=args.video_dir,
        save_dir=args.save_dir,
        model_dir=args.model_dir,
        skeleton=args.skeleton,
        device_name=args.device,
        batch_size=args.batch_size,
        num_aug=args.num_aug,
    )
    print(f"[DONE] success={s}, failure={f}")

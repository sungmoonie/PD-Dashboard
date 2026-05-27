from __future__ import annotations

import argparse
import os
import pickle
import sys
from typing import Tuple

import cv2
import numpy as np
import torch

# [FIX] paths는 mediapipe sys.path 삽입 전에 import (이름 충돌 방지)
from paths import (
    DEFAULT_MODEL_DIR,
    DEFAULT_SKELETON,
    resolve_clip_path,
    resolve_save_root,
    video_output_dir,
)

from metrabs_utils.data_utils import iter_test_samples, load_gait_test_df
from metrabs_utils.feature_utils import (
    compute_gait_video_features,
    gait_video_features_to_dataframe,
    get_num_joints_for_skeleton,
    pose_to_frame_vector,
    select_primary_pose,
    stack_frame_vectors,
)
from metrabs_utils.save_utils import (
    save_gait_feature_csv,
    save_kinematic_npz,
    save_raw_kinematic_csv,
)
from metrabs_utils.joint_info import JointInfo

_METRABS_DIR = os.path.dirname(os.path.abspath(__file__))
_VENDOR_DIR = os.path.join(_METRABS_DIR, "vendor")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)

METRABS_ROOT = "/workspace/model/metrabs"
if METRABS_ROOT not in sys.path:
    sys.path.append(METRABS_ROOT)

import metrabs_pytorch.backbones.efficientnet as effnet_pt
import metrabs_pytorch.models.metrabs as metrabs_pt
from metrabs_pytorch.multiperson.multiperson_model import Pose3dEstimator
from metrabs_pytorch.util import get_config

SPLIT_CHOICES = ("test", "all")
MODEL_BUNDLE_FILES = (
    "config.yaml",
    "ckpt.pt",
    "joint_info.npz",
    "skeleton_infos.pkl",
    "joint_transform_matrix.npy",
)
DEFAULTS = {
    "csv_path": "/workspace/pcos_dataset/gait/gait_v2_split.csv",
    "data_root_dir": "/workspace/pcos_dataset",
    "model_dir": DEFAULT_MODEL_DIR,
    "save_root_dir": "/workspace/pcos_dataset/results/kinematic_data",
    "split": "test",
    "skeleton": "smpl_24",
}


def _validate_model_bundle(model_dir: str) -> None:
    missing = [
        filename
        for filename in MODEL_BUNDLE_FILES
        if not os.path.exists(os.path.join(model_dir, filename))
    ]
    if missing:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(
            f"MeTRAbs checkpoint bundle is incomplete in {model_dir}: {missing_str}"
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
    joint_transform_matrix = np.load(
        os.path.join(model_dir, "joint_transform_matrix.npy")
    )

    estimator = Pose3dEstimator(crop_model, skeleton_infos, joint_transform_matrix).to(device)
    estimator.joint_transform_matrix = estimator.joint_transform_matrix.to(device)
    estimator.skeleton_joint_indices_table = {
        key: np.asarray(indices).astype(np.int64).tolist()
        for key, indices in estimator.skeleton_joint_indices_table.items()
    }
    estimator.eval()
    return estimator


def infer_video_kinematics(
    clip_path: str,
    estimator: Pose3dEstimator,
    skeleton: str,
    device: torch.device,
    fill_missing: float = 0.0,
    batch_size: int = 8,
    num_aug: int = 5,
    max_frames: int | None = None,
) -> np.ndarray:
    capture = cv2.VideoCapture(clip_path)
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {clip_path}")

    num_joints = get_num_joints_for_skeleton(estimator=estimator, skeleton=skeleton)
    frame_vectors: list[np.ndarray] = []
    frame_tensors: list[torch.Tensor] = []

    def flush_batch() -> None:
        if not frame_tensors:
            return

        images = torch.stack(frame_tensors, dim=0).to(device=device, dtype=torch.uint8)
        batch_size_local, _, height, width = images.shape
        full_frame_boxes = torch.tensor(
            [[[0.0, 0.0, float(width - 1), float(height - 1)]] for _ in range(batch_size_local)],
            dtype=torch.float32,
            device=device,
        )
        unknown_intrinsics = torch.tensor(
            [[[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]]],
            dtype=torch.float32,
            device=device,
        )
        default_distortion = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
            device=device,
        )
        default_extrinsic = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)
        default_world_up = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32, device=device)
        with torch.inference_mode():
            pred = estimator.estimate_poses_batched(
                images=images,
                boxes=full_frame_boxes,
                intrinsic_matrix=unknown_intrinsics,
                distortion_coeffs=default_distortion,
                extrinsic_matrix=default_extrinsic,
                world_up_vector=default_world_up,
                skeleton=skeleton,
                num_aug=num_aug,
            )

        for frame_poses3d in pred["poses3d"]:
            selected_pose = select_primary_pose(
                frame_boxes=None,
                frame_poses3d=frame_poses3d,
            )
            frame_vectors.append(
                pose_to_frame_vector(
                    pose3d=selected_pose,
                    num_joints=num_joints,
                    fill_missing=fill_missing,
                )
            )
        frame_tensors.clear()

    frame_idx = 0
    while True:
        if max_frames is not None and frame_idx >= max_frames:
            break
        success, frame_bgr = capture.read()
        if not success:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_tensors.append(torch.from_numpy(frame_rgb).permute(2, 0, 1).contiguous())
        if len(frame_tensors) >= batch_size:
            flush_batch()
        frame_idx += 1

    capture.release()
    flush_batch()

    if not frame_vectors:
        return np.zeros((0, num_joints * 3), dtype=np.float32)
    return stack_frame_vectors(frame_vectors)


def _connections(estimator: object, skeleton: str) -> list[tuple[int, int]]:
    raw = estimator.per_skeleton_joint_edges[skeleton]
    return [(int(edge[0]), int(edge[1])) for edge in raw]


def get_video_fps(clip_path: str, fallback_fps: float = 30.0) -> float:
    capture = cv2.VideoCapture(clip_path)
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {clip_path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    if fps is None or fps <= 0:
        return fallback_fps
    return float(fps)


def run_single_gait_clip(
    clip_path: str,
    save_root_dir: str,
    model_dir: str | None = None,
    skeleton: str | None = None,
    fill_missing: float = 0.0,
    device_name: str = "auto",
    batch_size: int = 8,
    num_aug: int = 5,
) -> tuple[str, str]:
    # [FIX] 단일 gait 비디오 → kinematic.csv + feature.csv 저장
    clip_path = resolve_clip_path(clip_path)
    if not os.path.exists(clip_path):
        raise FileNotFoundError(f"Clip not found: {clip_path}")

    model_dir = model_dir or DEFAULT_MODEL_DIR
    skeleton = skeleton or DEFAULT_SKELETON
    out_dir = video_output_dir(save_root_dir, clip_path)

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    _validate_model_bundle(model_dir=model_dir)
    estimator = _load_pose_estimator(model_dir=model_dir, device=device)

    kinematic_array = infer_video_kinematics(
        clip_path=clip_path,
        estimator=estimator,
        skeleton=skeleton,
        device=device,
        fill_missing=fill_missing,
        batch_size=batch_size,
        num_aug=num_aug,
    )
    num_joints = get_num_joints_for_skeleton(estimator=estimator, skeleton=skeleton)
    kinematic_jx3 = kinematic_array.reshape(kinematic_array.shape[0], num_joints, 3)
    kinematic_path = save_raw_kinematic_csv(out_dir, kinematic_jx3)

    fps = get_video_fps(clip_path)
    video_features = compute_gait_video_features(kinematic_jx3, fps=fps)
    feature_df = gait_video_features_to_dataframe(video_features)
    feature_path = save_gait_feature_csv(out_dir, feature_df)

    print(f"[OK] {os.path.basename(clip_path)} -> {kinematic_path} shape={kinematic_jx3.shape}")
    print(f"[OK] {os.path.basename(clip_path)} -> {feature_path}")
    return kinematic_path, feature_path


def run_pipeline(
    csv_path: str,
    data_root_dir: str,
    model_dir: str,
    save_root_dir: str,
    split: str = "test",
    skeleton: str = "smpl_24",
    fill_missing: float = 0.0,
    max_samples: int | None = None,
    batch_size: int = 8,
    num_aug: int = 5,
    device_name: str = "auto",
    max_frames: int | None = None,
) -> Tuple[int, int]:
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    _validate_model_bundle(model_dir=model_dir)

    target_df = load_gait_test_df(
        csv_path=csv_path,
        root_dir=data_root_dir,
        split=split,
    )
    total = len(target_df)
    print(f"[INFO] device={device} split={split} loaded={total} skeleton={skeleton}")
    if total == 0:
        return 0, 0

    estimator = _load_pose_estimator(model_dir=model_dir, device=device)
    success_count = 0
    failure_count = 0

    for sample_idx, sample in enumerate(iter_test_samples(target_df), start=1):
        if max_samples is not None and sample_idx > max_samples:
            break

        clip_path = sample["clip_path"]
        subject = sample["subject"]
        clip_filename = sample["clip_filename"]
        if not os.path.exists(clip_path):
            failure_count += 1
            print(f"[WARN] Missing clip: {clip_path}")
            continue

        try:
            kinematic_array = infer_video_kinematics(
                clip_path=clip_path,
                estimator=estimator,
                skeleton=skeleton,
                device=device,
                fill_missing=fill_missing,
                batch_size=batch_size,
                num_aug=num_aug,
                max_frames=max_frames,
            )
            save_path = save_kinematic_npz(
                save_root_dir=save_root_dir,
                subject=subject,
                clip_filename=clip_filename,
                kinematic_array=kinematic_array,
                dataset_name="gait",
                method_name="metrabs",
            )
            success_count += 1
            print(
                f"[OK] ({sample_idx}/{total}) {clip_filename} -> {save_path} "
                f"shape={kinematic_array.shape}"
            )
        except Exception as exc:  # noqa: BLE001
            failure_count += 1
            print(f"[ERROR] Failed {clip_filename}: {exc}")

    return success_count, failure_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frame-wise gait kinematic data using MeTRAbs PyTorch."
    )
    # [FIX] 단일 gait 비디오 처리용 CLI 인자 추가
    parser.add_argument(
        "--clip-path",
        default=None,
        help="Single video path (default: gait_video_samples/Camera1.mp4).",
    )
    parser.add_argument(
        "--save-root-dir",
        default=None,
        help="Output root (default: data/gait_video_data/metrabs).",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run CSV batch pipeline instead of single gait clip mode.",
    )
    parser.add_argument("--csv-path", default=DEFAULTS["csv_path"])
    parser.add_argument("--data-root-dir", default=DEFAULTS["data_root_dir"])
    parser.add_argument("--model-dir", default=DEFAULTS["model_dir"])
    parser.add_argument("--split", choices=SPLIT_CHOICES, default=DEFAULTS["split"])
    parser.add_argument("--skeleton", default=DEFAULTS["skeleton"])
    parser.add_argument("--fill-missing", type=float, default=0.0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-aug", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # [FIX] 기본 진입점은 단일 gait clip 모드 (--batch 시 CSV 배치)
    if args.batch:
        success, failure = run_pipeline(
            csv_path=args.csv_path,
            data_root_dir=args.data_root_dir,
            model_dir=args.model_dir,
            save_root_dir=args.save_root_dir or DEFAULTS["save_root_dir"],
            split=args.split,
            skeleton=args.skeleton,
            fill_missing=args.fill_missing,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            num_aug=args.num_aug,
            device_name=args.device,
            max_frames=args.max_frames,
        )
        print(f"[DONE] success={success}, failure={failure}")
    else:
        run_single_gait_clip(
            clip_path=resolve_clip_path(args.clip_path),
            save_root_dir=resolve_save_root(args.save_root_dir),
            model_dir=args.model_dir,
            skeleton=args.skeleton,
            fill_missing=args.fill_missing,
            device_name=args.device,
            batch_size=args.batch_size,
            num_aug=args.num_aug,
        )
        print("[DONE] single gait clip processed")

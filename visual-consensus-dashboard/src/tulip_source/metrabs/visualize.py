from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import torch

from metrabs_utils.data_utils import load_pose24_from_kinematic_csv
from metrabs_utils.feature_utils import get_num_joints_for_skeleton, select_primary_pose
from metrabs_utils.vis_utils import (
    PoseBatch,
    get_video_props,
    project_metrabs_points3d_to_pixels,
    write_overlay_video,
)
from paths import DEFAULT_MODEL_DIR, DEFAULT_SKELETON, resolve_clip_path, resolve_save_root, video_output_dir
from runner import _connections, _load_pose_estimator, _validate_model_bundle

DEFAULTS = {
    "model_dir": DEFAULT_MODEL_DIR,
    "skeleton": DEFAULT_SKELETON,
}


def _infer_gait_pose(
    clip_path: str,
    model_dir: str,
    skeleton: str,
    device: torch.device,
    batch_size: int = 8,
    num_aug: int = 5,
) -> PoseBatch:
    estimator = _load_pose_estimator(model_dir=model_dir, device=device)
    num_joints = get_num_joints_for_skeleton(estimator=estimator, skeleton=skeleton)

    capture = cv2.VideoCapture(clip_path)
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {clip_path}")

    seq_3d: list[np.ndarray] = []
    seq_2d: list[np.ndarray] = []
    frame_tensors: list[torch.Tensor] = []

    def flush_batch() -> None:
        if not frame_tensors:
            return
        images = torch.stack(frame_tensors, dim=0).to(device=device, dtype=torch.uint8)
        bsz, _, h, w = images.shape
        full_boxes = torch.tensor(
            [[[0.0, 0.0, float(w - 1), float(h - 1), 1.0]] for _ in range(bsz)],
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
        for boxes, poses3d, poses2d in zip(frame_boxes_list, pred["poses3d"], pred["poses2d"]):
            pose3d = select_primary_pose(frame_boxes=boxes, frame_poses3d=poses3d)
            pose2d = None
            if poses2d.numel() > 0:
                if boxes is not None and boxes.numel() > 0 and boxes.shape[0] > 0:
                    best_idx = int(torch.argmax(boxes[:, 4]).item())
                else:
                    best_idx = 0
                pose2d = poses2d[best_idx].detach().cpu().numpy().astype(np.float32)
            if pose3d is None:
                pose3d = np.zeros((num_joints, 3), dtype=np.float32)
            if pose2d is None:
                pose2d = np.zeros((num_joints, 2), dtype=np.float32)
            seq_3d.append(pose3d)
            seq_2d.append(pose2d)
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

    fps, _, _, _ = get_video_props(clip_path, fallback_fps=30.0)
    return PoseBatch(
        points3d=np.asarray(seq_3d, dtype=np.float32),
        points2d=np.asarray(seq_2d, dtype=np.float32),
        fps=fps,
        is_multi_hand=False,
    )


def run_single_gait_visualization(
    clip_path: str,
    save_root_dir: str,
    model_dir: str | None = None,
    skeleton: str | None = None,
    device_name: str = "auto",
) -> str:
    # [FIX] 단일 gait clip → {save_root}/{video_stem}/overlay.mp4 저장
    clip_path = resolve_clip_path(clip_path)
    if not os.path.exists(clip_path):
        raise FileNotFoundError(f"Clip not found: {clip_path}")

    model_dir = model_dir or DEFAULTS["model_dir"]
    skeleton = skeleton or DEFAULTS["skeleton"]
    _validate_model_bundle(model_dir=model_dir)

    out_dir = video_output_dir(save_root_dir, clip_path)
    overlay_path = os.path.join(out_dir, "overlay.mp4")
    kinematic_csv = os.path.join(out_dir, "kinematic.csv")

    if os.path.exists(kinematic_csv):
        # [FIX] skeleton edge만 필요 → CPU 로드로 GPU 재추론·OOM 방지
        device = torch.device("cpu")
    elif device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    estimator = _load_pose_estimator(model_dir=model_dir, device=device)
    connections = _connections(estimator=estimator, skeleton=skeleton)

    if os.path.exists(kinematic_csv):
        # [FIX] runner가 저장한 kinematic.csv 재사용 (GPU 재추론·OOM 방지)
        points3d = load_pose24_from_kinematic_csv(kinematic_csv)
        fps, width, height, _ = get_video_props(clip_path, fallback_fps=30.0)
        points2d = np.stack(
            [
                project_metrabs_points3d_to_pixels(frame3d, width=width, height=height)
                for frame3d in points3d
            ],
            axis=0,
        )
        pose = PoseBatch(points3d=points3d, points2d=points2d, fps=fps, is_multi_hand=False)
    else:
        pose = _infer_gait_pose(
            clip_path=clip_path,
            model_dir=model_dir,
            skeleton=skeleton,
            device=device,
        )
    wrote, total = write_overlay_video(
        clip_path=clip_path,
        pose=pose,
        connections=connections,
        out_path=overlay_path,
    )
    print(f"[OK] {os.path.basename(clip_path)} -> {overlay_path} frames={wrote}/{total}")
    return overlay_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MeTRAbs gait overlay visualization.")
    # [FIX] runner와 동일한 단일 clip / save_root CLI
    parser.add_argument("--clip-path", default=None)
    parser.add_argument("--save-root-dir", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--skeleton", default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_single_gait_visualization(
        clip_path=resolve_clip_path(args.clip_path),
        save_root_dir=resolve_save_root(args.save_root_dir),
        model_dir=args.model_dir,
        skeleton=args.skeleton,
        device_name=args.device,
    )
    print("[DONE] overlay video saved")

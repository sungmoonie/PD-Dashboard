from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PoseBatch:
    points3d: np.ndarray
    points2d: np.ndarray
    fps: float
    is_multi_hand: bool = False


def get_video_props(clip_path: str, fallback_fps: float) -> tuple[float, int, int, int]:
    capture = cv2.VideoCapture(clip_path)
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {clip_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or fallback_fps
    if fps <= 0:
        fps = fallback_fps
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return float(fps), width, height, frame_count


def project_metrabs_points3d_to_pixels(
    points3d: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    # [FIX] kinematic.csv 3D 좌표만 있을 때 overlay용 pinhole 근사 투영
    z = np.maximum(points3d[:, 2], 1.0)
    focal = float(max(width, 1))
    cx = float(width) * 0.5
    cy = float(height) * 0.5
    out = np.zeros((points3d.shape[0], 2), dtype=np.float32)
    out[:, 0] = focal * points3d[:, 0] / z + cx
    out[:, 1] = focal * points3d[:, 1] / z + cy
    return _project_pixels(out, width, height)


def _project_pixels(points2d: np.ndarray, width: int, height: int) -> np.ndarray:
    out = np.zeros((points2d.shape[0], 2), dtype=np.float32)
    out[:, 0] = points2d[:, 0]
    out[:, 1] = points2d[:, 1]
    out[:, 0] = np.clip(out[:, 0], 0.0, float(max(width - 1, 0)))
    out[:, 1] = np.clip(out[:, 1], 0.0, float(max(height - 1, 0)))
    return out


def _draw_skeleton(frame: np.ndarray, points_xy: np.ndarray, connections: list[tuple[int, int]]) -> None:
    for start, end in connections:
        if start >= points_xy.shape[0] or end >= points_xy.shape[0]:
            continue
        x1, y1 = points_xy[start]
        x2, y2 = points_xy[end]
        if not np.all(np.isfinite([x1, y1, x2, y2])):
            continue
        cv2.line(
            frame,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
    for x, y in points_xy:
        if not np.all(np.isfinite([x, y])):
            continue
        cv2.circle(frame, (int(round(x)), int(round(y))), 3, (0, 0, 255), -1, cv2.LINE_AA)


def write_overlay_video(
    clip_path: str,
    pose: PoseBatch,
    connections: list[tuple[int, int]],
    out_path: str,
) -> tuple[int, int]:
    # [FIX] MeTRAbs 2D keypoints를 원본 영상 위 overlay.mp4 로 저장
    fps, width, height, _ = get_video_props(clip_path=clip_path, fallback_fps=pose.fps)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    capture = cv2.VideoCapture(clip_path)
    frame_idx = 0

    try:
        while frame_idx < pose.points3d.shape[0]:
            success, frame_bgr = capture.read()
            if not success:
                break
            overlay = frame_bgr.copy()
            p2d = pose.points2d[frame_idx]
            image_xy = _project_pixels(p2d, width, height)
            _draw_skeleton(overlay, image_xy, connections)
            writer.write(overlay)
            frame_idx += 1
    finally:
        capture.release()
        writer.release()

    return frame_idx, pose.points3d.shape[0]

from __future__ import annotations

import os

# [FIX] app.py 기준 상대경로 — Mac 절대경로 하드코딩 제거
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_DASHBOARD_DIR = os.path.normpath(os.path.join(_PKG_DIR, "..", "..", ".."))
_PD_DASHBOARD_DIR = os.path.normpath(os.path.join(_DASHBOARD_DIR, ".."))

DEFAULT_CLIP_PATH = os.path.join(_PD_DASHBOARD_DIR, "gait_video_samples", "Camera1.mp4")
# [FIX] save_root_dir을 data/gait_video_data/metrabs 로 지정
DEFAULT_SAVE_ROOT = os.path.join(_DASHBOARD_DIR, "data", "gait_video_data", "metrabs")

DEFAULT_MODEL_DIR = os.path.join(
    _PKG_DIR,
    "ckpt",
    "metrabs_eff2l_384px_800k_28ds_pytorch",
)
DEFAULT_SKELETON = "smpl_24"


def resolve_clip_path(clip_path: str | None = None) -> str:
    path = clip_path or DEFAULT_CLIP_PATH
    return os.path.abspath(path)


def resolve_save_root(save_root_dir: str | None = None) -> str:
    root = save_root_dir or DEFAULT_SAVE_ROOT
    return os.path.abspath(root)


def video_stem_from_path(clip_path: str) -> str:
    return os.path.splitext(os.path.basename(clip_path))[0]


def video_output_dir(save_root_dir: str, clip_path: str) -> str:
    # [FIX] {save_root}/{video_stem}/ 출력 디렉터리 구조
    stem = video_stem_from_path(clip_path)
    out_dir = os.path.join(resolve_save_root(save_root_dir), stem)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

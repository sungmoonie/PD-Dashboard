#!/usr/bin/env python3
"""Ingest videos_feature Camera*_features.csv into DuckDB analytics tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def _task_code_from_folder(task_folder: str) -> str:
    low = task_folder.lower()
    if "toe_tapping_left" in low:
        return "toe_left"
    if "toe_tapping_right" in low:
        return "toe_right"
    if "resting" in low and "tremor" in low:
        return "resting"
    return "unknown"


def _peak_stats(signal: pd.Series, fps: float = 80.0) -> tuple[int, float]:
    arr = np.asarray(signal.fillna(0.0), dtype=np.float32)
    if arr.size < 3:
        return 0, 0.0
    thr = float(np.mean(arr) + np.std(arr))
    peaks = []
    for i in range(1, arr.size - 1):
        if arr[i] > thr and arr[i] > arr[i - 1] and arr[i] >= arr[i + 1]:
            peaks.append(i)
    if len(peaks) < 2:
        return len(peaks), 0.0
    intervals = np.diff(peaks) / max(float(fps), 1e-6)
    mean_int = float(np.mean(intervals))
    cv = float(np.std(intervals) / mean_int) if mean_int > 1e-8 else 0.0
    return len(peaks), cv


def _build_summary(df: pd.DataFrame, task_code: str) -> tuple[int, int, float, float, float]:
    if task_code in ("toe_left", "toe_right"):
        l_col = "left_toe_speed"
        r_col = "right_toe_speed"
        asym_col = "toe_lr_asymmetry"
    else:
        l_col = "left_tremor_amp_proxy"
        r_col = "right_tremor_amp_proxy"
        asym_col = "upper_limb_lr_asymmetry"

    l_events, l_cv = _peak_stats(df[l_col] if l_col in df else pd.Series(dtype=float))
    r_events, r_cv = _peak_stats(df[r_col] if r_col in df else pd.Series(dtype=float))
    asym_mean = float(df[asym_col].mean()) if asym_col in df else 0.0
    return l_events, r_events, l_cv, r_cv, asym_mean


def ingest_file(conn: duckdb.DuckDBPyConnection, csv_path: Path) -> int:
    # .../TULIP_001/videos_feature/17. Toe_tapping_left/Camera1_features.csv
    task_folder = csv_path.parent.name
    patient_id = csv_path.parents[2].name
    camera = csv_path.name.replace("_features.csv", ".mp4")
    task_code = _task_code_from_folder(task_folder)
    visit_id = f"{patient_id}_V1"

    df = pd.read_csv(csv_path)
    if df.empty:
        conn.execute(
            """
            INSERT INTO analytics.ingestion_log (source_path, target_table, row_count, status, error_message)
            VALUES (?, 'analytics.frame_features', 0, 'failed', 'empty csv')
            """,
            [str(csv_path)],
        )
        return 0

    meta = {
        "patient_id": patient_id,
        "visit_id": visit_id,
        "task_code": task_code,
        "task_folder": task_folder,
        "camera": camera,
        "source_csv_path": str(csv_path),
    }
    for col, val in meta.items():
        df[col] = val

    conn.register("tmp_features_df", df)
    conn.execute(
        """
        INSERT INTO analytics.frame_features BY NAME
        SELECT * FROM tmp_features_df
        """
    )
    conn.unregister("tmp_features_df")

    l_events, r_events, l_cv, r_cv, asym_mean = _build_summary(df, task_code)
    conn.execute(
        """
        INSERT INTO analytics.task_feature_summary (
            patient_id, visit_id, task_code, camera, n_frames,
            left_event_count, right_event_count, left_interval_cv, right_interval_cv,
            asymmetry_mean, source_csv_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            patient_id,
            visit_id,
            task_code,
            camera,
            int(len(df)),
            int(l_events),
            int(r_events),
            float(l_cv),
            float(r_cv),
            float(asym_mean),
            str(csv_path),
        ],
    )

    conn.execute(
        """
        INSERT INTO analytics.ingestion_log (source_path, target_table, row_count, status)
        VALUES (?, 'analytics.frame_features', ?, 'success')
        """,
        [str(csv_path), int(len(df))],
    )
    return int(len(df))


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest frame features into DuckDB.")
    parser.add_argument(
        "--duckdb-path",
        default="DB/duckdb/features.duckdb",
        help="DuckDB file path (default: DB/duckdb/features.duckdb)",
    )
    parser.add_argument(
        "--features-root",
        default="pads_matched/by_tulip",
        help="Root path that contains TULIP_*/videos_feature folders",
    )
    parser.add_argument(
        "--glob",
        default="TULIP_*/videos_feature/*/Camera*_features.csv",
        help="Glob pattern under features-root",
    )
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="Delete existing analytics rows before ingest",
    )
    args = parser.parse_args()

    db_path = Path(args.duckdb_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    root = Path(args.features_root)
    csv_files = sorted(root.glob(args.glob))
    if not csv_files:
        print(f"[WARN] No feature CSV files found under {root} with glob '{args.glob}'")
        return

    conn = duckdb.connect(str(db_path))
    try:
        if args.clear_existing:
            conn.execute("DELETE FROM analytics.frame_features")
            conn.execute("DELETE FROM analytics.task_feature_summary")
            conn.execute("DELETE FROM analytics.ingestion_log")

        total_rows = 0
        for csv_path in csv_files:
            try:
                n = ingest_file(conn, csv_path)
                total_rows += n
                print(f"[OK] {csv_path}: rows={n}")
            except Exception as exc:  # noqa: BLE001
                conn.execute(
                    """
                    INSERT INTO analytics.ingestion_log (source_path, target_table, row_count, status, error_message)
                    VALUES (?, 'analytics.frame_features', 0, 'failed', ?)
                    """,
                    [str(csv_path), str(exc)],
                )
                print(f"[FAIL] {csv_path}: {exc}")

        print(
            f"[DONE] files={len(csv_files)}, rows={total_rows}, duckdb={db_path}"
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()

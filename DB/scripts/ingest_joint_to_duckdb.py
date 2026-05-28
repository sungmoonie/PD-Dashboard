#!/usr/bin/env python3
"""Ingest videos_feature Camera*_joint.csv into DuckDB (no video/mp4 storage)."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
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


def ingest_file(conn: duckdb.DuckDBPyConnection, csv_path: Path) -> int:
    task_folder = csv_path.parent.name
    patient_id = csv_path.parents[2].name
    camera = csv_path.name.replace("_joint.csv", ".mp4")
    task_code = _task_code_from_folder(task_folder)
    visit_id = f"{patient_id}_V1"

    df = pd.read_csv(csv_path)
    if df.empty:
        return 0

    df["patient_id"] = patient_id
    df["visit_id"] = visit_id
    df["task_code"] = task_code
    df["task_folder"] = task_folder
    df["camera"] = camera
    df["source_csv_path"] = str(csv_path)

    table_exists = conn.execute(
        """
        SELECT COUNT(*) FROM information_schema.tables
        WHERE table_schema='analytics' AND table_name='frame_joints'
        """
    ).fetchone()[0]

    conn.register("tmp_joint_df", df)
    if table_exists == 0:
        conn.execute("CREATE TABLE analytics.frame_joints AS SELECT * FROM tmp_joint_df")
    else:
        conn.execute(
            """
            INSERT INTO analytics.frame_joints BY NAME
            SELECT * FROM tmp_joint_df
            """
        )
    conn.unregister("tmp_joint_df")

    conn.execute(
        """
        INSERT INTO analytics.ingestion_log (source_path, target_table, row_count, status)
        VALUES (?, 'analytics.frame_joints', ?, 'success')
        """,
        [str(csv_path), int(len(df))],
    )
    return int(len(df))


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest joint CSV files into DuckDB.")
    parser.add_argument("--duckdb-path", default="DB/storage/duckdb/features.duckdb")
    parser.add_argument("--features-root", default="pads_matched/by_tulip")
    parser.add_argument(
        "--glob",
        default="TULIP_*/videos_feature/*/Camera*_joint.csv",
    )
    args = parser.parse_args()

    db_path = Path(args.duckdb_path)
    root = Path(args.features_root)
    csv_files = sorted(root.glob(args.glob))
    if not csv_files:
        print(f"[WARN] No joint CSV files found under {root}")
        return

    conn = duckdb.connect(str(db_path))
    try:
        total_rows = 0
        for csv_path in csv_files:
            try:
                n = ingest_file(conn, csv_path)
                total_rows += n
                print(f"[OK] {csv_path}: rows={n}")
            except Exception as exc:  # noqa: BLE001
                print(f"[FAIL] {csv_path}: {exc}")

        print(f"[DONE] files={len(csv_files)}, rows={total_rows}, duckdb={db_path}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

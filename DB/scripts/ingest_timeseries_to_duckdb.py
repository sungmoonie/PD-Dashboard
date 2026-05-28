#!/usr/bin/env python3
"""Ingest pads_matched movement/timeseries TXT files into DuckDB (sensor data only)."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd


def ingest_file(conn: duckdb.DuckDBPyConnection, txt_path: Path, patient_id: str, pads_id: str) -> int:
    stem = txt_path.stem
    rest = stem[len(f"{pads_id}_") :]
    if rest.endswith("_LeftWrist"):
        task_name, wrist = rest[: -len("_LeftWrist")], "LeftWrist"
    elif rest.endswith("_RightWrist"):
        task_name, wrist = rest[: -len("_RightWrist")], "RightWrist"
    else:
        return 0

    df = pd.read_csv(
        txt_path,
        header=None,
        names=["time", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"],
    )
    if df.empty:
        return 0

    df["patient_id"] = patient_id
    df["pads_id"] = pads_id
    df["task_name"] = task_name
    df["wrist"] = wrist
    df["source_path"] = str(txt_path)

    conn.register("tmp_ts", df)
    conn.execute(
        """
        INSERT INTO analytics.sensor_timeseries BY NAME
        SELECT * FROM tmp_ts
        """
    )
    conn.unregister("tmp_ts")

    conn.execute(
        """
        INSERT INTO analytics.ingestion_log (source_path, target_table, row_count, status)
        VALUES (?, 'analytics.sensor_timeseries', ?, 'success')
        """,
        [str(txt_path), int(len(df))],
    )
    return int(len(df))


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest sensor timeseries into DuckDB.")
    parser.add_argument("--duckdb-path", default="DB/storage/duckdb/features.duckdb")
    parser.add_argument("--pads-root", default="pads_matched/by_tulip")
    parser.add_argument("--clear-existing", action="store_true")
    args = parser.parse_args()

    db_path = Path(args.duckdb_path)
    pads_root = Path(args.pads_root)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    mapping = {}
    mapping_csv = pads_root / "README_mapping.csv"
    if mapping_csv.exists():
        mdf = pd.read_csv(mapping_csv)
        mapping = dict(zip(mdf["tulip_patient_id"], mdf["pads_subject_id"].astype(str)))

    conn = duckdb.connect(str(db_path))
    try:
        if args.clear_existing:
            conn.execute("DELETE FROM analytics.sensor_timeseries")

        total_rows = 0
        file_count = 0
        for tulip_dir in sorted(pads_root.glob("TULIP_*")):
            patient_id = tulip_dir.name
            pads_id = str(mapping.get(patient_id, "")).zfill(3) if mapping.get(patient_id) else ""
            if not pads_id:
                for f in (tulip_dir / "patients").glob("patient_*.json"):
                    pads_id = f.stem.replace("patient_", "")
                    break
            ts_dir = tulip_dir / "movement" / "timeseries"
            if not ts_dir.exists():
                continue
            for txt_path in sorted(ts_dir.glob("*.txt")):
                try:
                    n = ingest_file(conn, txt_path, patient_id, pads_id)
                    total_rows += n
                    file_count += 1
                    print(f"[OK] {txt_path.name}: rows={n}")
                except Exception as exc:  # noqa: BLE001
                    conn.execute(
                        """
                        INSERT INTO analytics.ingestion_log (source_path, target_table, row_count, status, error_message)
                        VALUES (?, 'analytics.sensor_timeseries', 0, 'failed', ?)
                        """,
                        [str(txt_path), str(exc)],
                    )
                    print(f"[FAIL] {txt_path}: {exc}")

        print(f"[DONE] files={file_count}, rows={total_rows}, duckdb={db_path}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

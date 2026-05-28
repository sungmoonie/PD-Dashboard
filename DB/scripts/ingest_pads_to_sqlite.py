#!/usr/bin/env python3
"""Ingest pads_matched metadata into SQLite (patients, NMS, video asset paths only — no video blobs)."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def _task_code_from_folder(folder_name: str) -> str | None:
    low = folder_name.lower()
    if "toe_tapping_left" in low:
        return "toe_left"
    if "toe_tapping_right" in low:
        return "toe_right"
    if "resting" in low and "tremor" in low:
        return "resting"
    return None


def ingest_mapping(conn: sqlite3.Connection, mapping_csv: Path) -> int:
    import pandas as pd

    df = pd.read_csv(mapping_csv)
    n = 0
    for _, row in df.iterrows():
        patient_id = str(row["tulip_patient_id"]).strip()
        pads_id = str(row["pads_subject_id"]).strip()
        conn.execute(
            """
            INSERT INTO patients (patient_id, pads_id, cohort)
            VALUES (?, ?, 'TULIP')
            ON CONFLICT(patient_id) DO UPDATE SET pads_id=excluded.pads_id
            """,
            (patient_id, pads_id),
        )
        visit_id = f"{patient_id}_V1"
        conn.execute(
            """
            INSERT OR IGNORE INTO visits (visit_id, patient_id, source)
            VALUES (?, ?, 'pads_matched')
            """,
            (visit_id, patient_id),
        )
        n += 1
    return n


def ingest_patient_json(conn: sqlite3.Connection, patient_id: str, json_path: Path) -> bool:
    with json_path.open("r", encoding="utf-8") as f:
        p = json.load(f)
    gender = (p.get("gender") or "").strip()
    sex = gender[0].upper() if gender else None
    conn.execute(
        """
        INSERT INTO patients (patient_id, pads_id, sex, age, cohort)
        VALUES (?, ?, ?, ?, 'TULIP')
        ON CONFLICT(patient_id) DO UPDATE SET
            pads_id=COALESCE(excluded.pads_id, patients.pads_id),
            sex=COALESCE(excluded.sex, patients.sex),
            age=COALESCE(excluded.age, patients.age)
        """,
        (patient_id, str(p.get("id", "")), sex, p.get("age")),
    )
    visit_id = f"{patient_id}_V1"
    conn.execute(
        """
        INSERT OR IGNORE INTO visits (visit_id, patient_id, source, note)
        VALUES (?, ?, 'pads_matched', ?)
        """,
        (visit_id, patient_id, p.get("condition", "")),
    )
    return True


def ingest_questionnaire(conn: sqlite3.Connection, patient_id: str, json_path: Path) -> int:
    with json_path.open("r", encoding="utf-8") as f:
        q = json.load(f)
    items = q.get("item", [])
    n = 0
    for item in items:
        answer = item.get("answer")
        answer_int = None
        if isinstance(answer, bool):
            answer_int = 1 if answer else 0
        conn.execute(
            """
            INSERT INTO nms_items (patient_id, link_id, item_text, answer, source_path)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                patient_id,
                str(item.get("link_id", "")),
                str(item.get("text", "")),
                answer_int,
                str(json_path),
            ),
        )
        n += 1
    return n


def ingest_video_assets(conn: sqlite3.Connection, tulip_dir: Path, patient_id: str) -> int:
    """Register video/feature file paths only (no mp4 binary storage)."""
    visit_id = f"{patient_id}_V1"
    feature_root = tulip_dir / "videos_feature"
    if not feature_root.exists():
        return 0

    n = 0
    for task_folder in sorted(feature_root.iterdir()):
        if not task_folder.is_dir():
            continue
        task_code = _task_code_from_folder(task_folder.name)
        if not task_code:
            continue
        task_id = f"{visit_id}_{task_code}"
        conn.execute(
            """
            INSERT OR IGNORE INTO tasks (task_id, visit_id, task_code, task_label, trial_index)
            VALUES (?, ?, ?, ?, 1)
            """,
            (task_id, visit_id, task_code, task_folder.name),
        )
        for feat_csv in sorted(task_folder.glob("Camera*_features.csv")):
            camera = feat_csv.name.replace("_features.csv", ".mp4")
            video_path = tulip_dir / "videos" / task_folder.name / camera
            local_video = str(video_path) if video_path.exists() else None
            conn.execute(
                """
                INSERT OR IGNORE INTO video_assets (
                    task_id, camera, local_video_path, local_feature_csv_path
                ) VALUES (?, ?, ?, ?)
                """,
                (task_id, camera, local_video, str(feat_csv)),
            )
            n += 1
    return n


def ingest_timeseries_index(conn: sqlite3.Connection, tulip_dir: Path, patient_id: str, pads_id: str) -> int:
    ts_dir = tulip_dir / "movement" / "timeseries"
    if not ts_dir.exists():
        return 0
    n = 0
    for txt_path in sorted(ts_dir.glob(f"{pads_id}_*.txt")):
        stem = txt_path.stem
        parts = stem.split("_", 1)
        if len(parts) < 2:
            continue
        rest = parts[1]
        if rest.endswith("_LeftWrist"):
            task_name, wrist = rest[: -len("_LeftWrist")], "LeftWrist"
        elif rest.endswith("_RightWrist"):
            task_name, wrist = rest[: -len("_RightWrist")], "RightWrist"
        else:
            continue
        n_lines = sum(1 for _ in txt_path.open("r", encoding="utf-8"))
        conn.execute(
            """
            INSERT OR REPLACE INTO sensor_timeseries_index (
                patient_id, pads_id, task_name, wrist, source_path, n_samples
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (patient_id, pads_id, task_name, wrist, str(txt_path), n_lines),
        )
        n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest pads_matched metadata into SQLite.")
    parser.add_argument("--sqlite-db", default="DB/storage/sqlite/clinical_meta.db")
    parser.add_argument("--pads-root", default="pads_matched/by_tulip")
    args = parser.parse_args()

    sqlite_path = Path(args.sqlite_db)
    pads_root = Path(args.pads_root)
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    mapping_csv = pads_root / "README_mapping.csv"
    total_patients = 0
    total_nms = 0
    total_assets = 0
    total_ts_idx = 0

    with sqlite3.connect(sqlite_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")

        if mapping_csv.exists():
            n_map = ingest_mapping(conn, mapping_csv)
            print(f"[OK] mapping: patients={n_map}")

        for tulip_dir in sorted(pads_root.glob("TULIP_*")):
            if not tulip_dir.is_dir():
                continue
            patient_id = tulip_dir.name

            pads_id = None
            row = conn.execute(
                "SELECT pads_id FROM patients WHERE patient_id=?", (patient_id,)
            ).fetchone()
            if row:
                pads_id = row[0]

            patient_dir = tulip_dir / "patients"
            if patient_dir.exists():
                for pj in patient_dir.glob("patient_*.json"):
                    if ingest_patient_json(conn, patient_id, pj):
                        total_patients += 1
                        pads_id = pads_id or pj.stem.replace("patient_", "")
                    break

            q_dir = tulip_dir / "questionnaire"
            if q_dir.exists():
                for qj in q_dir.glob("questionnaire_response_*.json"):
                    total_nms += ingest_questionnaire(conn, patient_id, qj)
                    break

            if pads_id:
                total_ts_idx += ingest_timeseries_index(conn, tulip_dir, patient_id, pads_id)

            total_assets += ingest_video_assets(conn, tulip_dir, patient_id)
            print(f"[OK] {patient_id}: assets indexed (no video binary stored)")

        conn.execute(
            """
            INSERT INTO ingestion_log (source_type, source_path, target_table, row_count, status)
            VALUES ('pads_matched', ?, 'patients,nms_items,video_assets,sensor_timeseries_index', ?, 'success')
            """,
            (str(pads_root), total_patients + total_nms + total_assets + total_ts_idx),
        )
        conn.commit()

    print(
        f"[DONE] patients={total_patients}, nms_items={total_nms}, "
        f"video_assets={total_assets}, ts_index={total_ts_idx}, db={sqlite_path}"
    )


if __name__ == "__main__":
    main()

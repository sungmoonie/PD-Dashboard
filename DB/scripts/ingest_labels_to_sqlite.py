#!/usr/bin/env python3
"""Ingest labels_csv_files/*.csv into SQLite metadata/labels tables."""

from __future__ import annotations

import argparse
import csv
import re
import sqlite3
from pathlib import Path


def _subject_to_patient_id(subject_num: int) -> str:
    return f"TULIP_{subject_num:03d}"


def _infer_subject_num(csv_path: Path, first_row: dict[str, str]) -> int | None:
    value = (first_row.get("data_numbering") or "").strip()
    if value.isdigit():
        return int(value)
    m = re.search(r"subject[_-]?(\d+)", csv_path.stem, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _task_code_from_label(label_name: str) -> str | None:
    norm = label_name.lower()
    if "toe tapping" in norm and "left" in norm:
        return "toe_left"
    if "toe tapping" in norm and "right" in norm:
        return "toe_right"
    if "rest tremor" in norm:
        return "resting"
    return None


def ingest_file(conn: sqlite3.Connection, csv_path: Path) -> tuple[int, int]:
    inserted_rows = 0
    skipped_rows = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return 0, 0

    subject_num = _infer_subject_num(csv_path, rows[0])
    if subject_num is None:
        return 0, len(rows)

    patient_id = _subject_to_patient_id(subject_num)
    visit_id = f"{patient_id}_V1"

    conn.execute(
        """
        INSERT OR IGNORE INTO patients (patient_id, cohort)
        VALUES (?, ?)
        """,
        (patient_id, "TULIP"),
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO visits (visit_id, patient_id, source)
        VALUES (?, ?, ?)
        """,
        (visit_id, patient_id, "labels_csv_files"),
    )

    annotator_cols = ("label_clinician1", "label_clinician2", "label_clinician3")
    for row in rows:
        label_name = (row.get("UPDRS_name") or "").strip()
        if not label_name:
            skipped_rows += 1
            continue

        task_code = _task_code_from_label(label_name)
        task_id = None
        if task_code:
            task_id = f"{visit_id}_{task_code}"
            conn.execute(
                """
                INSERT OR IGNORE INTO tasks (task_id, visit_id, task_code, task_label, trial_index)
                VALUES (?, ?, ?, ?, 1)
                """,
                (task_id, visit_id, task_code, task_code),
            )

        for idx, col in enumerate(annotator_cols, start=1):
            raw = (row.get(col) or "").strip()
            if not raw:
                continue
            conn.execute(
                """
                INSERT INTO clinical_labels (
                    visit_id, task_id, label_name, label_value, annotator, label_source
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    visit_id,
                    task_id,
                    label_name,
                    raw,
                    f"clinician{idx}",
                    csv_path.name,
                ),
            )
            inserted_rows += 1

    conn.execute(
        """
        INSERT INTO ingestion_log (source_type, source_path, target_table, row_count, status)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("labels_csv", str(csv_path), "clinical_labels", inserted_rows, "success"),
    )
    return inserted_rows, skipped_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest UPDRS label CSV files into SQLite.")
    parser.add_argument(
        "--sqlite-db",
        default="DB/storage/sqlite/clinical_meta.db",
        help="SQLite DB path (default: DB/storage/sqlite/clinical_meta.db)",
    )
    parser.add_argument(
        "--labels-dir",
        default="labels_csv_files",
        help="Directory containing subject*_labels.csv (default: labels_csv_files)",
    )
    args = parser.parse_args()

    sqlite_path = Path(args.sqlite_db)
    labels_dir = Path(args.labels_dir)

    sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(labels_dir.glob("*_labels.csv"))
    if not csv_files:
        print(f"[WARN] No label CSV files found in: {labels_dir}")
        return

    total_inserted = 0
    total_skipped = 0
    with sqlite3.connect(sqlite_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        for csv_path in csv_files:
            inserted, skipped = ingest_file(conn, csv_path)
            total_inserted += inserted
            total_skipped += skipped
            print(f"[OK] {csv_path.name}: inserted={inserted}, skipped={skipped}")
        conn.commit()

    print(
        f"[DONE] files={len(csv_files)}, inserted={total_inserted}, skipped={total_skipped}, db={sqlite_path}"
    )


if __name__ == "__main__":
    main()

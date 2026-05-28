#!/usr/bin/env bash
# Full ingestion: labels_csv_files + pads_matched -> DB/storage (no video binary storage)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

SQLITE_DB="DB/storage/sqlite/clinical_meta.db"
DUCKDB_PATH="DB/storage/duckdb/features.duckdb"

mkdir -p DB/storage/sqlite DB/storage/duckdb

echo "[1/6] Initialize schemas..."
rm -f "$SQLITE_DB" "$DUCKDB_PATH"
sqlite3 "$SQLITE_DB" < DB/sql/schema_sqlite.sql
python3 - <<'PY'
import re
import duckdb
from pathlib import Path

db = Path("DB/storage/duckdb/features.duckdb")
sql = Path("DB/sql/schema_duckdb.sql").read_text()
conn = duckdb.connect(str(db))
conn.execute("CREATE SCHEMA IF NOT EXISTS analytics")
for stmt in re.split(r";\s*", sql):
    lines = [ln for ln in stmt.splitlines() if ln.strip() and not ln.strip().startswith("--")]
    stmt = "\n".join(lines).strip()
    if not stmt or stmt.upper().startswith("CREATE SCHEMA"):
        continue
    conn.execute(stmt)
conn.close()
print("[OK] DuckDB schema initialized via Python")
PY

echo "[2/6] Ingest labels_csv_files -> SQLite..."
python3 DB/scripts/ingest_labels_to_sqlite.py \
  --sqlite-db "$SQLITE_DB" \
  --labels-dir labels_csv_files

echo "[3/6] Ingest pads_matched metadata -> SQLite (patients, NMS, video paths only)..."
python3 DB/scripts/ingest_pads_to_sqlite.py \
  --sqlite-db "$SQLITE_DB" \
  --pads-root pads_matched/by_tulip

echo "[4/6] Ingest videos_feature -> DuckDB (features + summary)..."
python3 DB/scripts/ingest_features_to_duckdb.py \
  --duckdb-path "$DUCKDB_PATH" \
  --features-root pads_matched/by_tulip

echo "[5/6] Ingest joint CSV -> DuckDB..."
python3 DB/scripts/ingest_joint_to_duckdb.py \
  --duckdb-path "$DUCKDB_PATH" \
  --features-root pads_matched/by_tulip

echo "[6/6] Ingest sensor timeseries -> DuckDB..."
python3 DB/scripts/ingest_timeseries_to_duckdb.py \
  --duckdb-path "$DUCKDB_PATH" \
  --pads-root pads_matched/by_tulip

echo ""
echo "=== Verification ==="
sqlite3 "$SQLITE_DB" "SELECT 'patients' AS t, COUNT(*) FROM patients UNION ALL SELECT 'clinical_labels', COUNT(*) FROM clinical_labels UNION ALL SELECT 'nms_items', COUNT(*) FROM nms_items UNION ALL SELECT 'video_assets', COUNT(*) FROM video_assets UNION ALL SELECT 'sensor_timeseries_index', COUNT(*) FROM sensor_timeseries_index;"

python3 - <<'PY'
import duckdb
conn = duckdb.connect("DB/storage/duckdb/features.duckdb")
for q, label in [
    ("SELECT COUNT(*) FROM analytics.frame_features", "frame_features"),
    ("SELECT COUNT(*) FROM analytics.task_feature_summary", "task_feature_summary"),
    ("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='analytics' AND table_name='frame_joints'", "frame_joints_exists"),
]:
    try:
        n = conn.execute(q).fetchone()[0]
        if label == "frame_joints_exists":
            if n:
                n = conn.execute("SELECT COUNT(*) FROM analytics.frame_joints").fetchone()[0]
            print(f"frame_joints | {n}")
        else:
            print(f"{label} | {n}")
    except Exception as e:
        print(f"{label} | error: {e}")
try:
    n = conn.execute("SELECT COUNT(*) FROM analytics.sensor_timeseries").fetchone()[0]
    print(f"sensor_timeseries | {n}")
except Exception as e:
    print(f"sensor_timeseries | error: {e}")
conn.close()
PY

echo "[DONE] All ingestion complete (videos NOT stored, paths/metadata only)."

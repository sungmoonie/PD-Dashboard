# DB Bootstrap Guide

This folder contains the first database scaffold for the PD Dashboard.

## Goal

- Use `SQLite` for transactional metadata and clinical labels.
- Use `DuckDB` for frame-level feature analytics and fast aggregations.

## Files

- `sql/schema_sqlite.sql`: metadata/label schema (`patients`, `visits`, `tasks`, `video_assets`, `clinical_labels`, `ingestion_log`)
- `sql/schema_duckdb.sql`: analytics schema (`analytics.frame_features`, `analytics.task_feature_summary`, `analytics.ingestion_log`)

## Recommended local paths

- SQLite DB file: `DB/storage/sqlite/clinical_meta.db`
- DuckDB DB file: `DB/storage/duckdb/features.duckdb`

Create local DB directories if needed:

```bash
mkdir -p DB/storage/sqlite DB/storage/duckdb
```

## Initialize SQLite

```bash
sqlite3 DB/storage/sqlite/clinical_meta.db < DB/sql/schema_sqlite.sql
```

## Initialize DuckDB

```bash
duckdb DB/storage/duckdb/features.duckdb -c ".read DB/sql/schema_duckdb.sql"
```

## Data flow (current plan)

1. Ingest `labels_csv_files/*.csv` into SQLite `clinical_labels` (+ patient/visit/task mapping).
2. Ingest `pads_matched/by_tulip/*/videos_feature/*/Camera*_features.csv` into DuckDB `analytics.frame_features`.
3. Build task-level summaries into `analytics.task_feature_summary`.
4. Serve dashboard metrics from DB instead of direct CSV scans.

## Ingestion scripts

The following bootstrap scripts are provided in `DB/scripts`:

- `ingest_labels_to_sqlite.py`
- `ingest_features_to_duckdb.py`

Run label ingestion:

```bash
python3 DB/scripts/ingest_labels_to_sqlite.py \
  --sqlite-db DB/storage/sqlite/clinical_meta.db \
  --labels-dir labels_csv_files
```

Run feature ingestion:

```bash
python3 DB/scripts/ingest_features_to_duckdb.py \
  --duckdb-path DB/storage/duckdb/features.duckdb \
  --features-root pads_matched/by_tulip
```

Rebuild analytics tables from scratch:

```bash
python3 DB/scripts/ingest_features_to_duckdb.py \
  --duckdb-path DB/storage/duckdb/features.duckdb \
  --features-root pads_matched/by_tulip \
  --clear-existing
```

## Vercel note

For deployment, keep this folder as local/dev bootstrap.
Do not rely on local file DB persistence in serverless runtime.
Use external persistent storage/DB for production.

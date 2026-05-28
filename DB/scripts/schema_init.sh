mkdir -p DB/storage/sqlite DB/storage/duckdb
sqlite3 DB/storage/sqlite/clinical_meta.db < DB/sql/schema_sqlite.sql
duckdb DB/storage/duckdb/features.duckdb -c ".read DB/sql/schema_duckdb.sql"
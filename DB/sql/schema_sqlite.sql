PRAGMA foreign_keys = ON;

-- SQLite: transactional metadata and labels

CREATE TABLE IF NOT EXISTS patients (
    patient_id TEXT PRIMARY KEY,            -- e.g., TULIP_001
    pads_id TEXT,
    sex TEXT,
    age INTEGER,
    cohort TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS visits (
    visit_id TEXT PRIMARY KEY,              -- e.g., TULIP_001_V1
    patient_id TEXT NOT NULL,
    visit_date TEXT,                        -- ISO-8601 date
    source TEXT DEFAULT 'TULIP',
    note TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,               -- e.g., TULIP_001_V1_toe_left
    visit_id TEXT NOT NULL,
    task_code TEXT NOT NULL,                -- toe_left | toe_right | resting
    task_label TEXT,                        -- human-readable label
    trial_index INTEGER DEFAULT 1,
    UNIQUE (visit_id, task_code, trial_index),
    FOREIGN KEY (visit_id) REFERENCES visits(visit_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS video_assets (
    asset_id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    camera TEXT NOT NULL,                   -- Camera1.mp4
    cdn_url TEXT,
    local_video_path TEXT,
    local_feature_csv_path TEXT,
    fps REAL,
    frame_count INTEGER,
    width INTEGER,
    height INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (task_id, camera),
    FOREIGN KEY (task_id) REFERENCES tasks(task_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS clinical_labels (
    label_id INTEGER PRIMARY KEY AUTOINCREMENT,
    visit_id TEXT NOT NULL,
    task_id TEXT,
    label_name TEXT NOT NULL,               -- e.g., updrs_score, diagnosis
    label_value TEXT NOT NULL,              -- flexible string storage
    label_unit TEXT,
    annotator TEXT,
    label_source TEXT,                      -- labels_csv_files filename, etc.
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (visit_id) REFERENCES visits(visit_id) ON DELETE CASCADE,
    FOREIGN KEY (task_id) REFERENCES tasks(task_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS nms_items (
    item_id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL,
    link_id TEXT,
    item_text TEXT,
    answer INTEGER,
    source_path TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS sensor_timeseries_index (
    index_id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL,
    pads_id TEXT,
    task_name TEXT NOT NULL,
    wrist TEXT NOT NULL,
    source_path TEXT NOT NULL,
    n_samples INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (patient_id, task_name, wrist),
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS ingestion_log (
    ingest_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type TEXT NOT NULL,              -- labels_csv | features_csv | video
    source_path TEXT NOT NULL,
    source_hash TEXT,                       -- md5/sha256
    target_table TEXT NOT NULL,
    row_count INTEGER DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'success', -- success | failed
    error_message TEXT,
    executed_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_visits_patient ON visits(patient_id);
CREATE INDEX IF NOT EXISTS idx_tasks_visit ON tasks(visit_id);
CREATE INDEX IF NOT EXISTS idx_assets_task ON video_assets(task_id);
CREATE INDEX IF NOT EXISTS idx_labels_visit ON clinical_labels(visit_id);
CREATE INDEX IF NOT EXISTS idx_labels_task ON clinical_labels(task_id);
CREATE INDEX IF NOT EXISTS idx_nms_patient ON nms_items(patient_id);
CREATE INDEX IF NOT EXISTS idx_ts_index_patient ON sensor_timeseries_index(patient_id);
CREATE INDEX IF NOT EXISTS idx_ingest_source ON ingestion_log(source_type, source_path);

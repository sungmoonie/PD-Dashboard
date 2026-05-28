-- DuckDB: analytics store for frame-level video features

CREATE SCHEMA IF NOT EXISTS analytics;

CREATE TABLE IF NOT EXISTS analytics.frame_features (
    patient_id VARCHAR NOT NULL,            -- e.g., TULIP_001
    visit_id VARCHAR,                       -- optional when single-visit data
    task_code VARCHAR NOT NULL,             -- toe_left | toe_right | resting
    task_folder VARCHAR,                    -- 17. Toe_tapping_left ...
    camera VARCHAR NOT NULL,                -- Camera1.mp4
    frame_index INTEGER NOT NULL,

    -- Toe tapping features
    left_toe_speed DOUBLE,
    right_toe_speed DOUBLE,
    left_ankle_speed DOUBLE,
    right_ankle_speed DOUBLE,
    left_knee_speed DOUBLE,
    right_knee_speed DOUBLE,
    left_toe_vertical_delta DOUBLE,
    right_toe_vertical_delta DOUBLE,
    toe_tapping_rate_proxy DOUBLE,
    toe_lr_asymmetry DOUBLE,

    -- Resting tremor features
    left_shoulder_to_wrist_distance DOUBLE,
    right_shoulder_to_wrist_distance DOUBLE,
    left_elbow_speed DOUBLE,
    right_elbow_speed DOUBLE,
    left_wrist_speed DOUBLE,
    right_wrist_speed DOUBLE,
    left_hand_speed DOUBLE,
    right_hand_speed DOUBLE,
    left_tremor_amp_proxy DOUBLE,
    right_tremor_amp_proxy DOUBLE,
    upper_limb_lr_asymmetry DOUBLE,

    source_csv_path VARCHAR,
    loaded_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS analytics.task_feature_summary (
    patient_id VARCHAR NOT NULL,
    visit_id VARCHAR,
    task_code VARCHAR NOT NULL,
    camera VARCHAR NOT NULL,
    n_frames INTEGER,
    left_event_count INTEGER,
    right_event_count INTEGER,
    left_interval_cv DOUBLE,
    right_interval_cv DOUBLE,
    asymmetry_mean DOUBLE,
    source_csv_path VARCHAR,
    loaded_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS analytics.ingestion_log (
    source_path VARCHAR,
    source_hash VARCHAR,
    target_table VARCHAR,
    row_count INTEGER,
    status VARCHAR,                         -- success | failed
    error_message VARCHAR,
    executed_at TIMESTAMP DEFAULT now()
);

-- Useful checks:
-- SELECT task_code, camera, count(*) FROM analytics.frame_features GROUP BY 1,2;
-- SELECT patient_id, task_code, avg(toe_lr_asymmetry) FROM analytics.frame_features GROUP BY 1,2;

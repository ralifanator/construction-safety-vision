
CREATE TABLE video_sources (
    id              SERIAL PRIMARY KEY,
    type            TEXT NOT NULL,        -- 'file' или 'camera'
    name            TEXT NOT NULL,
    file_path       TEXT,                 -- путь к файлу (для type='file')
    camera_url      TEXT,                 -- RTSP/HTTP URL (для type='camera')
    location        TEXT,                 -- место установки (опционально)
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    connection_type TEXT,
    external_id     TEXT,
    is_active       BOOLEAN DEFAULT TRUE,
    target_fps      REAL,
    camera_params   JSONB,
    camera_deleted BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE video_processing_runs (
    id                      SERIAL PRIMARY KEY,
    video_source_id         INTEGER NOT NULL REFERENCES video_sources(id),

    started_at              TIMESTAMPTZ DEFAULT NOW(),
    finished_at             TIMESTAMPTZ,

    status                  TEXT NOT NULL,     -- 'processing', 'done', 'error'
    processing_type         TEXT NOT NULL,     -- 'file_analyze', 'file_overlay', 'stream'

    -- Кратко фиксируем ключевые параметры в отдельных колонках
    target_fps              REAL,
    max_width               INTEGER,

    -- Полный "снимок" конфигурации запуска храним в JSONB
    processing_params       JSONB,             -- весь config/detection/tracking и т.п.

    -- Итоговая статистика
    duration_sec            REAL,
    video_fps               REAL,
    total_frames            INTEGER,
    processed_frames        INTEGER,

    result_video_path       TEXT,
    error_message           TEXT
);

CREATE TABLE violations (
    id                      SERIAL PRIMARY KEY,
    video_processing_run_id INTEGER NOT NULL REFERENCES video_processing_runs(id),

    track_index             INTEGER,
    violation_type          TEXT NOT NULL,     -- 'no_helmet', 'no_vest', 'no_helmet_and_vest'
    frame_index             INTEGER,
    time_sec                REAL,
    time_str                TEXT,
    window_size             INTEGER,
    violations_in_window    INTEGER,
    violation_ratio         REAL,
    status_note             TEXT,
    frame_image_path        TEXT,      -- абсолютный путь на диске
    frame_image_url         TEXT,      -- относительный URL, например /violations_frames/...
    created_at              TIMESTAMPTZ DEFAULT NOW()
);
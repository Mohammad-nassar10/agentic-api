CREATE TABLE IF NOT EXISTS session_prefix (
    session_id     TEXT PRIMARY KEY,
    replaced_count INTEGER NOT NULL,
    replacement    TEXT NOT NULL
);

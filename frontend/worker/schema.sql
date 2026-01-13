-- $ bunx wrangler d1 execute detypify --remote --file=schema.sql

CREATE TABLE IF NOT EXISTS samples (
    id      INTEGER PRIMARY KEY,
    ver     INTEGER,
    token   INTEGER,
    sym     TEXT,
    strokes TEXT
);

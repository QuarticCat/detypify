-- $ bunx wrangler d1 execute detypify --remote --file=schema.sql

DROP TABLE IF EXISTS samples;

CREATE TABLE samples (
    id INTEGER PRIMARY KEY,
    ver INTEGER,
    token INTEGER,
    sym TEXT,
    strokes TEXT);

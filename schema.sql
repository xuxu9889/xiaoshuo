CREATE TABLE IF NOT EXISTS chunks (
                                      chunk_id   TEXT PRIMARY KEY,
                                      book_id    TEXT,
                                      chapter    TEXT,
                                      start_pos  INTEGER,
                                      end_pos    INTEGER,
                                      text       TEXT
);

CREATE TABLE IF NOT EXISTS entities (
                                        entity_id  TEXT PRIMARY KEY,
                                        type       TEXT,
                                        name       TEXT,
                                        aliases    TEXT,
                                        meta       TEXT
);

CREATE TABLE IF NOT EXISTS facts (
                                     fact_id     TEXT PRIMARY KEY,
                                     subject     TEXT,
                                     predicate   TEXT,
                                     object      TEXT,
                                     summary     TEXT,
                                     evidence    TEXT
);

CREATE TABLE IF NOT EXISTS events (
                                      event_id    TEXT PRIMARY KEY,
                                      who         TEXT,
                                      where_name  TEXT,
                                      action      TEXT,
                                      details     TEXT,
                                      chapter     TEXT,
                                      evidence    TEXT
);

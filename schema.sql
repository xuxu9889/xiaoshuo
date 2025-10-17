-- ========== 剧情节点表 ==========
CREATE TABLE IF NOT EXISTS plot_nodes (
                                          id            TEXT PRIMARY KEY,
                                          title         TEXT NOT NULL,
                                          summary       TEXT NOT NULL,
                                          entry_hint    TEXT,
                                          exit_hint     TEXT,
                                          fact_tags     TEXT,
                                          required_flags TEXT,
                                          set_flags     TEXT
);

-- ========== 剧情边表 ==========
CREATE TABLE IF NOT EXISTS plot_edges (
                                          id            INTEGER PRIMARY KEY AUTOINCREMENT,
                                          src           TEXT NOT NULL,
                                          dst           TEXT NOT NULL,
                                          condition     TEXT,
                                          keywords      TEXT,
                                          UNIQUE(src, dst)
    );

-- ========== 剧情状态表 ==========
CREATE TABLE IF NOT EXISTS plot_state (
                                          save_id       TEXT PRIMARY KEY DEFAULT 'default',
                                          current_node  TEXT NOT NULL
);

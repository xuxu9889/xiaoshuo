# storage.py
# 多库多书：每本书一个独立的 sqlite，路径：db/world_<book>.sqlite
from contextlib import contextmanager
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parent
DB_DIR = ROOT / "db"
SCHEMA_FILE = ROOT / "schema.sql"
DB_DIR.mkdir(parents=True, exist_ok=True)

def db_path_for(book: str) -> Path:
    """
    根据书名/slug 生成数据库路径。
    允许中英文与常用字符，其它字符用下划线替代，避免异常文件名。
    """
    safe = []
    for ch in book:
        if ch.isalnum() or ch in "-_一二三四五六七八九十极度狂人盗墓笔记":
            safe.append(ch)
        else:
            safe.append("_")
    return DB_DIR / f"world_{''.join(safe)}.sqlite"

def init_db(book: str) -> None:
    """
    用 schema.sql 初始化当前书的 sqlite。
    如果 schema.sql 不存在，会创建一个最小可用的表结构作兜底。
    """
    db_path = db_path_for(book)
    if SCHEMA_FILE.exists():
        sql = SCHEMA_FILE.read_text(encoding="utf-8")
    else:
        sql = """
        PRAGMA journal_mode=WAL;
        PRAGMA foreign_keys=ON;

        CREATE TABLE IF NOT EXISTS chunks (
          id TEXT PRIMARY KEY,
          chapter INTEGER,
          order_in_chapter INTEGER,
          text TEXT,
          tokens INTEGER DEFAULT 0,
          embedding BLOB
        );

        CREATE TABLE IF NOT EXISTS facts (
          id TEXT PRIMARY KEY,
          subject TEXT,
          predicate TEXT,
          object TEXT,
          summary TEXT,
          evidence_chunks TEXT,
          hash TEXT UNIQUE
        );

        CREATE TABLE IF NOT EXISTS entities (
          id TEXT PRIMARY KEY,
          name TEXT UNIQUE,
          type TEXT
        );

        CREATE TABLE IF NOT EXISTS edges (
          id TEXT PRIMARY KEY,
          head_entity TEXT,
          tail_entity TEXT,
          relation TEXT,
          fact_id TEXT
        );

        CREATE TABLE IF NOT EXISTS sessions (
          id TEXT PRIMARY KEY,
          mode TEXT,
          role TEXT,
          scene TEXT,
          state_json TEXT
        );

        CREATE TABLE IF NOT EXISTS turns (
          id TEXT PRIMARY KEY,
          session_id TEXT NOT NULL,
          user_text TEXT,
          engine_text TEXT,
          used_fact_ids TEXT,
          conflicts TEXT
        );
        """.strip()

    with sqlite3.connect(db_path) as conn:
        conn.executescript(sql)
        conn.commit()

@contextmanager
def get_conn(book: str):
    """
    获取当前书的 sqlite 连接（不存在则自动初始化）。
    使用示例：
        with get_conn("极度狂人") as conn:
            conn.execute("SELECT 1")
    """
    db_path = db_path_for(book)
    if not db_path.exists():
        init_db(book)
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()

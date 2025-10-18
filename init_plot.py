import sqlite3, os
from pathlib import Path

DB_PATH = os.getenv("DB_PATH", "./db/盗墓笔记.world.sqlite")

def exec_sql(path):
    with open(path, "r", encoding="utf-8") as f:
        sql = f.read()
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(sql)
    conn.commit()
    conn.close()
    print("✅ 剧情表结构已写入数据库")

if __name__ == "__main__":
    exec_sql("schema.sql")

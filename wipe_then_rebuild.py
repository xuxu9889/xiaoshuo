# wipe_then_rebuild.py
import sqlite3, os
from pathlib import Path
DB = r".\db\盗墓笔记.sqlite"   # ← 你的库路径
DB = str(Path(DB).resolve())
print("DB:", DB)

con = sqlite3.connect(DB)
cur = con.cursor()
# 清空三张表
for sql in [
    "DELETE FROM plot_nodes;",
    "DELETE FROM plot_edges;",
    "DELETE FROM plot_state;",
    "VACUUM;",
]:
    try:
        cur.execute(sql)
    except Exception as e:
        print("Skip/Err:", e)
con.commit(); con.close()
print("✅ 已清空旧数据。现在回到 GUI，保持同一本 TXT，勾选“允许覆盖”，重新生成即可。")

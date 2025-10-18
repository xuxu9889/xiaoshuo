# check_db.py
import sqlite3, os, sys
DB_PATH = r".\db\盗墓笔记.sqlite"  # ← 改成你的库路径（可用绝对路径）

DB_PATH = os.path.abspath(DB_PATH)
print("DB:", DB_PATH)

con = sqlite3.connect(DB_PATH)
cur = con.cursor()

print("\n章节前 15：")
for r in cur.execute("SELECT id, title FROM plot_nodes ORDER BY id LIMIT 15;"):
    print(r)

print("\n每章块数：")
for r in cur.execute("""
                     SELECT substr(id,1,5) AS chap, COUNT(*)
                     FROM plot_nodes
                     GROUP BY chap
                     ORDER BY chap;
                     """):
    print(r)

print("\n尝试找第二章：")
print(cur.execute("SELECT id, title FROM plot_nodes WHERE id LIKE 'ch002_%' ORDER BY id LIMIT 5;").fetchall())

print("\nplot_state：")
print(cur.execute("SELECT * FROM plot_state;").fetchall())

con.close()

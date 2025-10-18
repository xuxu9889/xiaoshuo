# view_plot.py
import sqlite3, os

DB_PATH = os.getenv("DB_PATH", "./db/盗墓笔记.sqlite")

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    print("=== 剧情节点（plot_nodes）===\n")
    rows = cur.execute("SELECT id, title, summary, entry_hint, exit_hint, fact_tags, required_flags, set_flags FROM plot_nodes").fetchall()
    for i, row in enumerate(rows, 1):
        print(f"[{i}] 节点ID：{row[0]}")
        print(f"    标题：{row[1]}")
        print(f"    概要：{row[2]}")
        print(f"    进入条件提示：{row[3]}")
        print(f"    推进条件提示：{row[4]}")
        print(f"    关联事实标签：{row[5]}")
        print(f"    需要旗标：{row[6]}")
        print(f"    进入后设置旗标：{row[7]}")
        print("")

    print("\n=== 剧情推进边（plot_edges）===\n")
    rows = cur.execute("SELECT src, dst, condition, keywords FROM plot_edges").fetchall()
    for i, row in enumerate(rows, 1):
        print(f"[{i}] {row[0]} -> {row[1]}")
        print(f"    推进条件（自然语言）：{row[2]}")
        print(f"    推进关键词：{row[3]}")
        print("")

    print("\n=== 当前剧情状态（plot_state）===\n")
    rows = cur.execute("SELECT save_id, current_node FROM plot_state").fetchall()
    for row in rows:
        print(f"存档ID：{row[0]} | 当前节点：{row[1]}")

    conn.close()

if __name__ == "__main__":
    main()

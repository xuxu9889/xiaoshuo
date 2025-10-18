import sqlite3

db_path = "./db/盗墓笔记.sqlite"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

def show_table(table_name):
    print(f"\n===== {table_name} =====")
    for row in cursor.execute(f"SELECT * FROM {table_name} LIMIT 100;"):
        print(row)

tables = ["plot_nodes", "plot_edges", "plot_state"]
for t in tables:
    show_table(t)

conn.close()
print("\n✅ 数据读取完成")

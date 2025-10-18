import sqlite3, json
db = r"C:/Users/xu541/IdeaProjects/xiaoshuo/db/盗墓笔记.world.sqlite"
conn = sqlite3.connect(db)

# 看3条事实
for row in conn.execute("SELECT subject,predicate,object,summary,chunk_id FROM facts LIMIT 3"):
    print(row)

# 查看某事件的参与者（who 为 JSON）
row = conn.execute("SELECT who, action, loc FROM events LIMIT 1").fetchone()
who = json.loads(row[0]) if row and row[0] else []
print("who:", who, "action:", row[1], "loc:", row[2])

# 取证据文本
row = conn.execute("""
                   SELECT c.content
                   FROM events e JOIN chunks c ON c.id=e.chunk_id
                   ORDER BY e.chapter_id, e.chunk_id LIMIT 1
                   """).fetchone()
print("evidence text:", row[0][:200], "...")
conn.close()

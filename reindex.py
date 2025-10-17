# reindex.py —— 从 SQLite 重建 chunks & facts 的向量索引（不调用chat）
import sqlite3, math
from llm_io import embed
from store import add_chunks, add_facts, DB_PATH

BATCH = 200

def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def rebuild_chunks():
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    rows = cur.execute("SELECT chunk_id, text FROM chunks ORDER BY chunk_id").fetchall()
    con.close()
    if not rows:
        print("⚠️ chunks 为空，请先跑 ingest.py")
        return
    ids = [r[0] for r in rows]
    docs = [r[1] for r in rows]
    print(f"Rebuilding chunks index: {len(ids)} items")
    all_embs = []
    for batch in batched(docs, BATCH):
        all_embs.extend(embed(batch))
    add_chunks(ids, docs, all_embs)
    print("✅ chunks index rebuilt.")

def rebuild_facts():
    con = sqlite3.connect(DB_PATH); cur = con.cursor()
    rows = cur.execute("SELECT fact_id, summary FROM facts WHERE summary IS NOT NULL AND summary <> ''").fetchall()
    con.close()
    if not rows:
        print("⚠️ facts 为空，请先跑 extract.py")
        return
    ids = [r[0] for r in rows]
    docs = [r[1] for r in rows]
    print(f"Rebuilding facts index: {len(ids)} items")
    all_embs = []
    for batch in batched(docs, BATCH):
        all_embs.extend(embed(batch))
    add_facts(ids, docs, all_embs)
    print("✅ facts index rebuilt.")

if __name__ == "__main__":
    rebuild_chunks()
    rebuild_facts()

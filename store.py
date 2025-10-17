# store.py
import os, sqlite3
from typing import List
import chromadb
from chromadb.config import Settings
from llm_io import embed

DB_PATH = "db/canon.sqlite"
INDEX_DIR = "index"

def get_sql():
    os.makedirs("db", exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL;")
    return con

def get_chroma():
    os.makedirs(INDEX_DIR, exist_ok=True)
    # 优先使用持久化客户端；若旧版无此类，则回退到 Client+Settings
    try:
        return chromadb.PersistentClient(path=INDEX_DIR)  # 新版推荐
    except Exception:
        return chromadb.Client(Settings(persist_directory=INDEX_DIR))  # 兼容旧版

def _col(name):
    client = get_chroma()
    return client.get_or_create_collection(name)

def add_chunks(ids: List[str], texts: List[str], embeddings: List[List[float]]):
    col = _col("chunks")
    col.add(ids=ids, documents=texts, embeddings=embeddings)
    # 不再调用 persist()

def add_facts(ids: List[str], summaries: List[str], embeddings: List[List[float]]):
    col = _col("facts")
    col.add(ids=ids, documents=summaries, embeddings=embeddings)
    # 不再调用 persist()

def search_chunks(query: str, k=6):
    col = _col("chunks")
    qvec = embed([query])[0]
    res = col.query(query_embeddings=[qvec], n_results=k)
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    return [{"id": ids[i], "text": docs[i]} for i in range(len(ids))]

def search_facts(query: str, k=6):
    col = _col("facts")
    qvec = embed([query])[0]
    res = col.query(query_embeddings=[qvec], n_results=k)
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    return [{"id": ids[i], "summary": docs[i]} for i in range(len(ids))]

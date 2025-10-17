import re, os, chardet, time
from pathlib import Path
from llm_io import embed
from store import ensure_schema, get_sql, add_chunks

RAW = "data/raw/book.txt"

# 自动检测编码
def read_file_auto_encoding(path):
    with open(path, 'rb') as f:
        data = f.read()
    enc = chardet.detect(data)['encoding'] or 'utf-8'
    print(f"Detected encoding: {enc}")
    return data.decode(enc, errors='ignore')

# 清理文本
def clean_text(t: str) -> str:
    t = t.replace("\r","")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()

# 按句号/换行切片
def chunk_text(text: str, max_len=900, overlap=80):
    parts = re.split(r"(。|！|？|\n)", text)
    buf, chunks = "", []
    for i in range(0, len(parts), 2):
        seg = parts[i] + (parts[i+1] if i+1 < len(parts) else "")
        if len(buf) + len(seg) > max_len:
            if buf:
                chunks.append(buf)
                buf = buf[-overlap:] + seg
            else:
                chunks.append(seg[:max_len])
                buf = seg[max_len-overlap:max_len]
        else:
            buf += seg
    if buf.strip(): chunks.append(buf)
    return [c.strip() for c in chunks if c.strip()]

def batched_embedding(pieces, batch_size=100):
    """分批计算 embedding，避免超 token 限制"""
    all_vecs = []
    for i in range(0, len(pieces), batch_size):
        batch = pieces[i:i+batch_size]
        print(f"Embedding batch {i//batch_size+1}/{(len(pieces)-1)//batch_size+1}...")
        try:
            vecs = embed(batch)
            all_vecs.extend(vecs)
            time.sleep(1)  # 稍微休息，防止速率限制
        except Exception as e:
            print(f"❌ Batch {i} failed: {e}")
            # 可以重试或跳过
            all_vecs.extend([[0]*1536]*len(batch))
    return all_vecs

def main():
    ensure_schema()
    raw = read_file_auto_encoding(RAW)
    text = clean_text(raw)
    pieces = chunk_text(text)

    print(f"Total {len(pieces)} chunks to process.")
    con = get_sql(); cur = con.cursor()
    cur.execute("DELETE FROM chunks")
    for i, c in enumerate(pieces):
        cid = f"chunk_{i:05d}"
        cur.execute("INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?)",
                    (cid, "book", "", 0, 0, c))
    con.commit(); con.close()

    # 🔹 分批计算 embedding
    embs = batched_embedding(pieces, batch_size=100)
    add_chunks([f"chunk_{i:05d}" for i in range(len(pieces))], pieces, embs)
    print(f"✅ Ingested {len(pieces)} chunks and built vector index successfully.")

if __name__ == "__main__":
    if not Path(RAW).exists():
        print(f"❌ 请把整本书纯文本放到 {RAW}")
    else:
        main()

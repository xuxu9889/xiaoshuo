# view_chunks.py —— 查看/搜索 chunks 表（修复 f-string 报错）
import argparse, textwrap, re, json
from store import get_sql

def format_preview(text):
    # 先把 \n 替换掉，避免在 f-string 表达式里写转义
    return text.replace('\n', ' ')

def list_chunks(limit=20, offset=0):
    con = get_sql(); cur = con.cursor()
    cur.execute(
        "SELECT chunk_id, substr(text,1,120) FROM chunks ORDER BY chunk_id LIMIT ? OFFSET ?",
        (limit, offset)
    )
    rows = cur.fetchall(); con.close()
    for cid, preview in rows:
        preview = format_preview(preview)
        print(f"{cid}  |  {preview}")

def show_chunk_by_id(chunk_id: str):
    con = get_sql(); cur = con.cursor()
    cur.execute("SELECT chunk_id, text FROM chunks WHERE chunk_id = ?", (chunk_id,))
    row = cur.fetchone(); con.close()
    if not row:
        print(f"未找到 {chunk_id}")
        return
    cid, txt = row
    print(f"== {cid} ==")
    print(_wrap(txt))

def show_chunk_by_index(index: int):
    # index 从 1 开始：1=第一条
    con = get_sql(); cur = con.cursor()
    cur.execute("SELECT chunk_id, text FROM chunks ORDER BY chunk_id LIMIT 1 OFFSET ?", (index-1,))
    row = cur.fetchone(); con.close()
    if not row:
        print(f"没有第 {index} 条（超出范围）")
        return
    cid, txt = row
    print(f"== {cid} ==")
    print(_wrap(txt))

def search_chunks(keyword: str, limit=20):
    con = get_sql(); cur = con.cursor()
    cur.execute("SELECT chunk_id, substr(text,1,120) FROM chunks WHERE text LIKE ? ORDER BY chunk_id LIMIT ?",
                (f"%{keyword}%", limit))
    rows = cur.fetchall(); con.close()
    if not rows:
        print(f"未找到包含关键词“{keyword}”的 chunk")
        return
    for cid, preview in rows:
        preview = format_preview(preview)
        print(f"{cid}  |  {preview}")

def stats():
    con = get_sql(); cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM chunks"); total = cur.fetchone()[0]
    cur.execute("SELECT MIN(chunk_id), MAX(chunk_id) FROM chunks"); mn, mx = cur.fetchone()
    con.close()
    print(f"总数：{total}  范围：{mn} ~ {mx}")

def _wrap(s: str, width: int = 80) -> str:
    # 按段落自然换行
    out = []
    for par in re.split(r"\n\s*\n", s.strip()):
        if len(par) <= width:
            out.append(par)
            continue
        out.extend(textwrap.wrap(par, width=width, break_long_words=False, break_on_hyphens=False))
        out.append("")  # 段间空行
    return "\n".join(out).rstrip()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="查看/搜索 chunks 表")
    sub = ap.add_subparsers(dest="cmd")

    p1 = sub.add_parser("list"); p1.add_argument("--limit", type=int, default=20); p1.add_argument("--offset", type=int, default=0)
    p2 = sub.add_parser("show"); g = p2.add_mutually_exclusive_group(required=True)
    g.add_argument("--id", type=str)
    g.add_argument("--index", type=int, help="第几条（1=第一条）")
    p3 = sub.add_parser("search"); p3.add_argument("keyword"); p3.add_argument("--limit", type=int, default=20)
    sub.add_parser("stats")

    args = ap.parse_args()
    if args.cmd == "list":
        list_chunks(args.limit, args.offset)
    elif args.cmd == "show":
        if args.id:
            show_chunk_by_id(args.id)
        else:
            show_chunk_by_index(args.index)
    elif args.cmd == "search":
        search_chunks(args.keyword, args.limit)
    elif args.cmd == "stats":
        stats()
    else:
        ap.print_help()

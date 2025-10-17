# show_facts.py
import sqlite3, json, argparse

DB = "db/canon.sqlite"

def preview_chunk(con, chunk_id, width=80):
    cur = con.cursor()
    cur.execute("SELECT text FROM chunks WHERE chunk_id=?", (chunk_id,))
    row = cur.fetchone()
    return (row[0][:width] + "…") if row and row[0] else ""

def main():
    ap = argparse.ArgumentParser(description="Browse canonical facts with optional search")
    ap.add_argument("--limit", type=int, default=20, help="显示条数（默认20）")
    ap.add_argument("--search", type=str, default="", help="关键词（subject/predicate/object/summary）模糊搜索")
    ap.add_argument("--subject", type=str, default="", help="只看某个 subject（精确匹配）")
    ap.add_argument("--with-text", action="store_true", help="显示证据 chunk 的原文预览")
    args = ap.parse_args()

    con = sqlite3.connect(DB)
    cur = con.cursor()

    sql = "SELECT fact_id, subject, predicate, object, summary, evidence FROM facts"
    conds, params = [], []

    if args.subject:
        conds.append("subject = ?")
        params.append(args.subject)

    if args.search:
        like = f"%{args.search}%"
        conds.append("(subject LIKE ? OR predicate LIKE ? OR object LIKE ? OR summary LIKE ?)")
        params += [like, like, like, like]

    if conds:
        sql += " WHERE " + " AND ".join(conds)
    sql += " ORDER BY fact_id LIMIT ?"
    params.append(args.limit)

    rows = cur.execute(sql, params).fetchall()
    if not rows:
        print("（没有匹配到 facts）")
        return

    for i, (fid, subj, pred, obj, summ, evid) in enumerate(rows, 1):
        try:
            obj_disp = json.loads(obj)
            if isinstance(obj_disp, (dict, list)):
                obj_disp = json.dumps(obj_disp, ensure_ascii=False)
        except Exception:
            obj_disp = obj

        print(f"\n[{i}] {fid}")
        print(f"  subject : {subj}")
        print(f"  predicate: {pred}")
        print(f"  object  : {obj_disp}")
        print(f"  summary : {summ}")

        try:
            evid_list = json.loads(evid) if evid else []
        except Exception:
            evid_list = []

        if evid_list:
            print(f"  evidence: {', '.join(evid_list)}")
            if args.with_text:
                pv = preview_chunk(con, evid_list[0])
                if pv:
                    print(f"  proof   : {pv}")
        else:
            print("  evidence: —")

    con.close()

if __name__ == "__main__":
    main()

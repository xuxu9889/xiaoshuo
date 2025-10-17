# extract.py  —— 带进度条/限量/稳健重试版
import os
import json
import time
import argparse
from typing import List, Tuple

from tqdm import tqdm
from llm_io import chat_json, embed
from store import get_sql, add_facts

P_ENT = "prompts/extract_entities.txt"
P_FAC = "prompts/extract_facts.txt"
P_EVT = "prompts/extract_events.txt"

def load(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def fetch_chunks(limit: int | None = None) -> List[Tuple[str, str]]:
    con = get_sql(); cur = con.cursor()
    sql = "SELECT chunk_id, text FROM chunks ORDER BY chunk_id"
    if limit and limit > 0:
        sql += f" LIMIT {int(limit)}"
    cur.execute(sql)
    rows = cur.fetchall()
    con.close()
    return rows

def insert_entities(cur, cid: str, ent_res: dict):
    for e in ent_res.get("entities", []):
        eid = f"ent_{hash((e.get('name','') + e.get('type',''))) & 0xffffffff:08x}"
        cur.execute(
            """INSERT OR IGNORE INTO entities(entity_id,type,name,aliases,meta)
               VALUES(?,?,?,?,?)""",
            (
                eid,
                e.get("type", ""),
                e.get("name", ""),
                json.dumps(e.get("aliases", []), ensure_ascii=False),
                json.dumps({"notes": e.get("notes", "")}, ensure_ascii=False),
            ),
        )

def insert_facts(cur, cid: str, fac_res: dict, fact_ids: list, fact_summaries: list):
    for f in fac_res.get("facts", []):
        fid = f"fact_{hash(json.dumps(f, ensure_ascii=False)) & 0xffffffff:08x}"
        cur.execute(
            """INSERT OR IGNORE INTO facts(fact_id,subject,predicate,object,summary,evidence)
               VALUES(?,?,?,?,?,?)""",
            (
                fid,
                f.get("subject", ""),
                f.get("predicate", ""),
                json.dumps(f.get("object", ""), ensure_ascii=False),
                f.get("summary", ""),
                json.dumps(f.get("evidence", [cid]), ensure_ascii=False),
            ),
        )
        if f.get("summary"):
            fact_ids.append(fid)
            fact_summaries.append(f.get("summary"))

def insert_events(cur, cid: str, evt_res: dict):
    for ev in evt_res.get("events", []):
        evid = f"evt_{hash(json.dumps(ev, ensure_ascii=False)) & 0xffffffff:08x}"
        cur.execute(
            """INSERT OR IGNORE INTO events(event_id,who,where_name,action,details,chapter,evidence)
               VALUES(?,?,?,?,?,?,?)""",
            (
                evid,
                json.dumps(ev.get("who", []), ensure_ascii=False),
                ev.get("where", ""),
                ev.get("action", ""),
                ev.get("details", ""),
                "",  # chapter 可后续补
                json.dumps(ev.get("evidence", [cid]), ensure_ascii=False),
            ),
        )

def main():
    parser = argparse.ArgumentParser(description="Extract entities/facts/events with progress bar")
    parser.add_argument("--limit", type=int, default=int(os.getenv("EXTRACT_LIMIT", "0")),
                        help="只抽取前 N 段（默认全量）")
    parser.add_argument("--sleep", type=float, default=float(os.getenv("EXTRACT_SLEEP", "0.0")),
                        help="每段之间休眠秒数，降低速率（如 0.2）")
    parser.add_argument("--retry", type=int, default=2, help="失败重试次数（默认2）")
    args = parser.parse_args()

    ent_prompt = load(P_ENT)
    fac_prompt = load(P_FAC)
    evt_prompt = load(P_EVT)

    rows = fetch_chunks(limit=args.limit if args.limit and args.limit > 0 else None)
    total = len(rows)
    if total == 0:
        print("⚠️  chunks 表为空，请先运行 `python ingest.py`")
        return

    con = get_sql(); cur = con.cursor()
    # 清空旧表（如需增量，可移除这三行）
    cur.execute("DELETE FROM entities")
    cur.execute("DELETE FROM facts")
    cur.execute("DELETE FROM events")
    con.commit()

    fact_ids: List[str] = []
    fact_summaries: List[str] = []

    print(f"Total {total} chunks to extract...")
    pbar = tqdm(rows, desc="Extracting", ncols=90)

    errors = 0
    for cid, text in pbar:
        # 简单重试逻辑
        attempt = 0
        while True:
            try:
                # 实体
                ent_res = chat_json("只返回JSON", ent_prompt + "\n\n" + text)
                insert_entities(cur, cid, ent_res)

                # 事实
                fac_res = chat_json("只返回JSON", fac_prompt.replace("{{chunk_id}}", cid) + "\n\n" + text)
                insert_facts(cur, cid, fac_res, fact_ids, fact_summaries)

                # 事件
                evt_res = chat_json("只返回JSON", evt_prompt.replace("{{chunk_id}}", cid) + "\n\n" + text)
                insert_events(cur, cid, evt_res)

                # 每段提交（更稳）
                con.commit()

                # 进度条尾巴加点信息
                has_ent = len(ent_res.get("entities", []))
                has_fac = len(fac_res.get("facts", []))
                has_evt = len(evt_res.get("events", []))
                pbar.set_postfix({"ent": has_ent, "facts": has_fac, "events": has_evt})
                break

            except Exception as e:
                attempt += 1
                if attempt > args.retry:
                    errors += 1
                    pbar.set_postfix({"error": f"{cid}"})
                    # 不中断，继续下一个
                    break
                time.sleep(1.0)  # 重试前稍等

        if args.sleep > 0:
            time.sleep(args.sleep)

    # 建立 facts 的向量索引（只对 summary 建）
    if fact_summaries:
        try:
            vecs = embed(fact_summaries)
            add_facts(fact_ids, fact_summaries, vecs)
        except Exception as e:
            print(f"⚠️ 建立 facts 向量索引失败：{e}")

    con.close()
    print(f"✅ 抽取完成。条目汇总：facts={len(fact_ids)}，errors={errors}")

if __name__ == "__main__":
    main()

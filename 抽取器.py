# -*- coding: utf-8 -*-
import os, re, json, sqlite3, argparse, sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# ========== 兼容：与你原程序一致的章节/切块 ==========
def para_split(s: str) -> List[str]:
    parts = re.split(r"\n\s*\n+", s.strip())
    out = []
    for p in parts:
        p = p.strip()
        if not p: continue
        if len(p) <= 400:
            out.append(p)
        else:
            segs = re.split(r"(?<=[。！？!?])", p)
            buf = ""
            for seg in segs:
                if len(buf) + len(seg) <= 500:
                    buf += seg
                else:
                    if buf: out.append(buf.strip())
                    buf = seg
            if buf.strip(): out.append(buf.strip())
    return out

def chunkify(text: str, max_chars: int = 700) -> List[str]:
    paras = para_split(text)
    chunks, cur = [], ""
    for p in paras:
        if not cur:
            cur = p; continue
        if len(cur) + 1 + len(p) <= max_chars:
            cur = cur + "\n" + p
        else:
            chunks.append(cur.strip()); cur = p
    if cur.strip(): chunks.append(cur.strip())
    final = []
    for c in chunks:
        if len(c) <= max_chars:
            final.append(c)
        else:
            for i in range(0, len(c), max_chars):
                final.append(c[i:i+max_chars].strip())
    return final

def split_chapters(text: str):
    text = re.sub(
        r'(?<!\n)(第[一二三四五六七八九十百千万0-9]{1,6}(?:[章节回集]|卷[ 　\t]*第?[一二三四五六七八九十百千万0-9]{1,4}章)[^。\n\r]{0,60})',
        r'\n\1\n', text
    )
    patterns = [
        r'第[一二三四五六七八九十百千万0-9]{1,6}\s*卷[ 　\t]*第?[一二三四五六七八九十百千万0-9]{1,6}\s*章(?:[：:\-、,.，。]?[^\n\r]{0,60})?',
        r'第[一二三四五六七八九十百千万0-9]{1,6}\s*[章节回集](?:[：:\-、,.，。]?[^\n\r]{0,60})?',
        r'[Cc][Hh][Aa][Pp][Tt][Ee][Rr]\s+\d+[^\n\r]{0,60}',
    ]
    chap_re = re.compile('|'.join(f'({p})' for p in patterns))
    matches = [(m.start(), m.group(0)) for m in chap_re.finditer(text)]
    if not matches:
        return [("ch001", "第1章", text)]
    matches = sorted(set(matches), key=lambda x: x[0])
    chapters = []
    for i, (start, title) in enumerate(matches):
        end = matches[i + 1][0] if i + 1 < len(matches) else len(text)
        body = text[start + len(title):end].strip()
        clean_title = title.strip().replace('\u3000', ' ')
        cid = f"ch{i+1:03d}"
        chapters.append((cid, clean_title if len(clean_title)<=80 else clean_title[:80]+"…", body))
    return chapters

def read_text_with_encoding(novel_path: str, preferred: str = "auto") -> str:
    p = Path(novel_path)
    if preferred and preferred.lower() != "auto":
        return p.read_text(encoding=preferred, errors="strict")
    try:
        from charset_normalizer import from_path
        res = from_path(str(p)).best()
        if res and res.encoding:
            return p.read_text(encoding=res.encoding, errors="strict")
    except Exception:
        pass
    for enc in ("utf-8-sig","utf-8","gbk","cp936","big5"):
        try:
            return p.read_text(encoding=enc, errors="strict")
        except Exception:
            continue
    return p.read_text(encoding="utf-8", errors="ignore")

# ========== JSON 宽松解析 & 键名归一 ==========
def json_relaxed(s: str):
    if not s: return None
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\s*", "", s); s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r'(\{(?:[^{}]|\{[^{}]*\})*\}|\[(?:[^\[\]]|\[[^\[\]]*\])*\])', s, re.S)
    if m:
        frag = m.group(1)
        try: return json.loads(frag)
        except Exception: return None
    return None

CANON = {
    "subject": ["subject", "主语", "实体", "S", "s", "subj", "entity"],
    "predicate": ["predicate", "谓词", "关系", "P", "p", "pred", "relation", "关系/属性/动作"],
    "object": ["object", "宾语", "O", "o", "obj", "对象", "目标"],
    "summary": ["summary", "摘要", "说明", "注释", "简述", "概述", "短摘要"],
    "evidence": ["evidence", "证据", "出处", "evidences", "spans", "evid"],
    "who": ["who", "人物", "角色", "参与者", "参与人"],
    "where": ["where", "地点", "位置", "场所"],
    "action": ["action", "动作", "行为", "做了什么", "事件"],
    "details": ["details", "细节", "备注", "说明"],
    "type": ["type", "类型", "实体类型", "类别"],
    "name": ["name", "名称", "名字"],
    "aliases": ["aliases", "别名", "绰号", "又名"],
    "notes": ["notes", "备注", "说明"],
}
def _get_first(d: Dict[str, Any], keys: List[str], as_list=False):
    for k in keys:
        if k in d:
            v = d.get(k)
            if v is None: continue
            if as_list:
                if isinstance(v, list): return v
                if isinstance(v, (str,int,float)): return [str(v)]
            else:
                if isinstance(v, (str,int,float)): return str(v).strip()
                if isinstance(v, dict) and "text" in v and isinstance(v["text"], str):
                    return v["text"].strip()
                if isinstance(v, list) and v and isinstance(v[0], str):
                    return v[0].strip()
    return [] if as_list else ""

def normalize_facts_payload(data, chunk_id: str) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = None
        for k in ("facts","triples","data","result","items"):
            if isinstance(data.get(k), list):
                items = data[k]; break
        if items is None: return []
    else:
        return []
    out = []
    for it in items:
        if not isinstance(it, dict): continue
        s = _get_first(it, CANON["subject"])
        p = _get_first(it, CANON["predicate"])
        o = _get_first(it, CANON["object"])
        sm = _get_first(it, CANON["summary"])
        ev = _get_first(it, CANON["evidence"], as_list=True)
        if not ev: ev = [f"{{{{{chunk_id}}}}}"] if chunk_id else []
        if s and p and o:
            out.append({"subject": s, "predicate": p, "object": o, "summary": sm, "evidence": ev})
    return out[:12]

def normalize_events_payload(data, chunk_id: str) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("events") if isinstance(data.get("events"), list) else None
        if items is None:
            for k in ("data","result","items"):
                if isinstance(data.get(k), list):
                    items = data[k]; break
        if items is None: return []
    else:
        return []
    out = []
    for it in items:
        if not isinstance(it, dict): continue
        who = _get_first(it, CANON["who"], as_list=True)
        where = _get_first(it, CANON["where"])
        action = _get_first(it, CANON["action"])
        details = _get_first(it, CANON["details"])
        ev = _get_first(it, CANON["evidence"], as_list=True)
        if not ev: ev = [f"{{{{{chunk_id}}}}}"] if chunk_id else []
        if who or action or where:
            out.append({"who": who or [], "where": where, "action": action, "details": details, "evidence": ev})
    return out[:12]

def normalize_entities_payload(data) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("entities") if isinstance(data.get("entities"), list) else None
        if items is None:
            for k in ("data","result","items"):
                if isinstance(data.get(k), list):
                    items = data[k]; break
        if items is None: return []
    else:
        return []
    valid = {"person","org","place","item","skill"}
    out = []
    for it in items:
        if not isinstance(it, dict): continue
        t = (_get_first(it, CANON["type"]) or "").lower()
        n = _get_first(it, CANON["name"])
        aliases = _get_first(it, CANON["aliases"], as_list=True)
        notes = _get_first(it, CANON["notes"])
        if t in valid and n:
            out.append({"type": t, "name": n, "aliases": aliases or [], "notes": notes})
    return out[:50]

# ========== OpenAI / Ollama 调用 ==========
def openai_client():
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key: return None
    try:
        from openai import OpenAI
        base = os.getenv("OPENAI_BASE", "").strip() or None
        if base: return OpenAI(api_key=api_key, base_url=base)
        return OpenAI(api_key=api_key)
    except Exception:
        return None

def openai_chat_json(client, model: str, sys_prompt: str, user_prompt: str) -> str:
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":sys_prompt},
                      {"role":"user","content":user_prompt}],
            temperature=0.0,
            response_format={"type":"json_object"},
        )
        return r.choices[0].message.content.strip()
    except Exception:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":sys_prompt},
                      {"role":"user","content":user_prompt}],
            temperature=0.0,
        )
        return r.choices[0].message.content.strip()

def ollama_chat(model: str, sys_prompt: str, user_prompt: str, host: str, use_json=False) -> str:
    import requests
    url = f"{host.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role":"system","content":sys_prompt},{"role":"user","content":user_prompt}],
        "stream": False,
        "options": {"temperature":0.0, "top_p":0.9, "repeat_penalty":1.1, "num_ctx":4096}
    }
    if use_json: payload["format"] = "json"
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["message"]["content"]

def ollama_chat_safe(model: str, sys_prompt: str, user_prompt: str, host: str) -> str:
    try:
        return ollama_chat(model, sys_prompt, user_prompt, host, use_json=True)
    except Exception:
        return ollama_chat(model, sys_prompt, user_prompt, host, use_json=False)

# ========== 提示词（三轮） ==========
SYS_STRICT = "你是抽取器。只返回 JSON，不要多余文字。缺信息留空。确保输出是严格 JSON。"

FACTS_R1 = """输入是一段小说文本片段，请抽取“可验证设定/规则（facts）”，并生成短摘要 summary。
严格返回一个 JSON 对象，键为 facts，值是数组。数组元素必须为：
{"subject":"<实体名或规则域>","predicate":"...","object":"...","summary":"<简洁可检索>","evidence":["{{chunk_id}}"]}
若确实没有明确 facts，也返回 {"facts":[]}。
【chunk_id】{cid}
【文本】{text}"""
FACTS_R2 = """你是抽取器。只返回 JSON。
从片段中抽取1~6条 facts，不要编造小说外信息。
示例：{"facts":[
  {"subject":"古墓机关","predicate":"触发条件","object":"踏错石板","summary":"踏错石板会触发机关","evidence":["{cid}"]}
]}
【chunk_id】{cid}
【文本】{text}""".replace("{cid}", "{{chunk_id}}")
FACTS_R3 = """先识别片段里清晰的“动作/因果/条件”，再转成 S-P-O 三元组。
输出：{"facts":[{"subject":"...","predicate":"...","object":"...","summary":"...","evidence":["{{chunk_id}}"]}]}
【chunk_id】{cid}
【文本】{text}""".replace("{cid}", "{{chunk_id}}")

EVENTS_R1 = """只返回 JSON。抽取“发生的事件/时间线”：
{"events":[{"who":["<人物1>","<人物2>"],"where":"<地点或空>","action":"<做了什么>","details":"<细节>","evidence":["{{chunk_id}}"]}]}
【chunk_id】{cid}
【文本】{text}"""
EVENTS_R2 = """只返回 JSON。至少产出1~8条；若没有，返回 {"events":[]}。
例：{"events":[{"who":["吴邪","张起灵"],"where":"盗洞口","action":"进入墓道","details":"携带装备潜入","evidence":["{cid}"]}]}
【chunk_id】{cid}
【文本】{text}""".replace("{cid}", "{{chunk_id}}")

ENTS_R1 = """只返回 JSON。抽取实体（person/org/place/item/skill）：
{"entities":[{"type":"person|org|place|item|skill","name":"...","aliases":["..."],"notes":"..."}]}
【文本】{text}"""
ENTS_R2 = """只返回 JSON。至少给出0~20条，type 仅限 person/org/place/item/skill。
【文本】{text}"""

def ask_facts(provider, model, host, text, cid):
    text = text[:2000]
    if provider=="openai":
        cli = openai_client()
        for up in (FACTS_R1, FACTS_R2, FACTS_R3):
            raw = openai_chat_json(cli, model, SYS_STRICT, up.format(text=text, cid=cid or ""))
            data = json_relaxed(raw); facts = normalize_facts_payload(data, cid)
            if facts: return facts
    else:
        for up in (FACTS_R1, FACTS_R2, FACTS_R3):
            raw = ollama_chat_safe(model, SYS_STRICT, up.format(text=text, cid=cid or ""), host)
            data = json_relaxed(raw); facts = normalize_facts_payload(data, cid)
            if facts: return facts
    return []

def ask_events(provider, model, host, text, cid):
    text = text[:2000]
    if provider=="openai":
        cli = openai_client()
        for up in (EVENTS_R1, EVENTS_R2):
            raw = openai_chat_json(cli, model, SYS_STRICT, up.format(text=text, cid=cid or ""))
            data = json_relaxed(raw); evs = normalize_events_payload(data, cid)
            if evs: return evs
    else:
        for up in (EVENTS_R1, EVENTS_R2):
            raw = ollama_chat_safe(model, SYS_STRICT, up.format(text=text, cid=cid or ""), host)
            data = json_relaxed(raw); evs = normalize_events_payload(data, cid)
            if evs: return evs
    return []

def ask_entities(provider, model, host, text):
    text = text[:2000]
    if provider=="openai":
        cli = openai_client()
        for up in (ENTS_R1, ENTS_R2):
            raw = openai_chat_json(cli, model, SYS_STRICT, up.format(text=text))
            data = json_relaxed(raw); ents = normalize_entities_payload(data)
            if ents: return ents
    else:
        for up in (ENTS_R1, ENTS_R2):
            raw = ollama_chat_safe(model, SYS_STRICT, up.format(text=text), host)
            data = json_relaxed(raw); ents = normalize_entities_payload(data)
            if ents: return ents
    return []

# ========== DB 相关 ==========
SCHEMA_EVENTS = """
                CREATE TABLE IF NOT EXISTS events (
                                                      id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                      node_id TEXT,
                                                      who TEXT,        -- JSON 数组
                                                      where TEXT,
                                                      action TEXT,
                                                      details TEXT,
                                                      evidence TEXT,   -- JSON 数组
                                                      UNIQUE(node_id, where, action, details)
                    );"""
SCHEMA_ENTS = """
              CREATE TABLE IF NOT EXISTS entities (
                                                      id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                      node_id TEXT,
                                                      type TEXT,
                                                      name TEXT,
                                                      aliases TEXT,    -- JSON 数组
                                                      notes TEXT,
                                                      UNIQUE(node_id, type, name)
                  );"""

def ensure_tables(conn: sqlite3.Connection):
    with conn:
        conn.execute("CREATE TABLE IF NOT EXISTS facts (id INTEGER PRIMARY KEY AUTOINCREMENT, subject TEXT, predicate TEXT, object TEXT, summary TEXT, UNIQUE(subject, predicate, object))")
        conn.executescript(SCHEMA_EVENTS)
        conn.executescript(SCHEMA_ENTS)
    # content 列
    cur = conn.execute("PRAGMA table_info(plot_nodes)")
    cols = [r[1] for r in cur.fetchall()]
    if "content" not in cols:
        conn.execute("ALTER TABLE plot_nodes ADD COLUMN content TEXT")
        conn.commit()

def backfill_content_if_needed(conn: sqlite3.Connection, novel_path: Optional[str], max_chars=700):
    cur = conn.execute("SELECT COUNT(*) FROM plot_nodes WHERE content IS NOT NULL AND content <> ''")
    has = cur.fetchone()[0]
    cur = conn.execute("SELECT COUNT(*) FROM plot_nodes")
    total = cur.fetchone()[0]
    if has == total: return  # 已完备
    if not novel_path:
        raise RuntimeError("plot_nodes 缺少 content，且未提供 --novel 原始TXT，无从回填。")

    # 读取原文并用一致算法重建 node 顺序
    text = read_text_with_encoding(novel_path, preferred="auto")
    chapters = split_chapters(text)

    # 生成序列 node_id -> content
    seq: List[tuple] = []
    for cid, ctitle, cbody in chapters:
        chunks = chunkify(cbody, max_chars=max_chars)
        if not chunks: chunks = ["（空章节内容）"]
        for i, chunk in enumerate(chunks, 1):
            node_id = f"{cid}_{i:03d}"
            seq.append((node_id, chunk))

    # 用已有顺序对齐：以 node_id 匹配
    with conn:
        for nid, content in seq:
            conn.execute("UPDATE plot_nodes SET content=? WHERE id=? AND (content IS NULL OR content='')", (content, nid))

def run_extract(conn: sqlite3.Connection, provider: str, model: str, host: str, mode: str, limit: int = 0):
    ensure_tables(conn)
    # 拉出所有节点（带内容）
    rows = conn.execute("SELECT id, title, content FROM plot_nodes ORDER BY id").fetchall()
    if not rows:
        print("[]"); return
    todo = [(r[0], r[1], r[2] or "") for r in rows]
    if limit > 0: todo = todo[:limit]

    facts_to_add, events_to_add, ents_to_add = [], [], []

    for node_id, title, content in todo:
        if not content.strip(): continue
        # facts
        if mode in ("facts","all"):
            facts = ask_facts(provider, model, host, content, node_id)
            for f in facts:
                s, p, o, sm = f.get("subject","").strip(), f.get("predicate","").strip(), f.get("object","").strip(), f.get("summary","").strip()
                if s and p and o:
                    facts_to_add.append((s,p,o,sm))
        # events
        if mode in ("events","all"):
            evs = ask_events(provider, model, host, content, node_id)
            for e in evs:
                who = json.dumps(e.get("who", []), ensure_ascii=False)
                where = e.get("where","")
                action = e.get("action","")
                details = e.get("details","")
                evidence = json.dumps(e.get("evidence", []), ensure_ascii=False)
                if action or who or where:
                    events_to_add.append((node_id, who, where, action, details, evidence))
        # entities
        if mode in ("entities","all"):
            ents = ask_entities(provider, model, host, content)
            for en in ents:
                t = en.get("type",""); n = en.get("name","")
                aliases = json.dumps(en.get("aliases", []), ensure_ascii=False)
                notes = en.get("notes","")
                if t and n:
                    ents_to_add.append((node_id, t, n, aliases, notes))

    with conn:
        if facts_to_add:
            conn.executemany("INSERT OR IGNORE INTO facts (subject,predicate,object,summary) VALUES (?,?,?,?)", facts_to_add)
        if events_to_add:
            conn.executemany("INSERT OR IGNORE INTO events (node_id, who, where, action, details, evidence) VALUES (?,?,?,?,?,?)", events_to_add)
        if ents_to_add:
            conn.executemany("INSERT OR IGNORE INTO entities (node_id, type, name, aliases, notes) VALUES (?,?,?,?,?)", ents_to_add)

    print(json.dumps({
        "added": {
            "facts": len(facts_to_add),
            "events": len(events_to_add),
            "entities": len(ents_to_add),
        },
        "processed_nodes": len(todo),
        "mode": mode
    }, ensure_ascii=False))

def main():
    ap = argparse.ArgumentParser(description="Extract facts/events/entities from existing DB.")
    ap.add_argument("--db", required=True, help="SQLite 路径，比如 db/盗墓笔记.sqlite")
    ap.add_argument("--novel", help="原始TXT路径；当 plot_nodes.content 为空时用于回填")
    ap.add_argument("--mode", choices=["facts","events","entities","all"], default="facts")
    ap.add_argument("--provider", choices=["openai","ollama"], default="openai")
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--limit", type=int, default=0, help="只处理前N个节点（调试用）")
    ap.add_argument("--max-chars", type=int, default=700, help="回填切块大小，需与建库时一致")
    args = ap.parse_args()

    dbp = Path(args.db)
    if not dbp.exists():
        raise SystemExit(f"DB 不存在：{dbp}")

    conn = sqlite3.connect(str(dbp))
    try:
        ensure_tables(conn)
        backfill_content_if_needed(conn, args.novel, max_chars=args.max_chars)
        run_extract(conn, args.provider, args.model, args.host, args.mode, limit=args.limit)
    finally:
        conn.close()

if __name__ == "__main__":
    main()

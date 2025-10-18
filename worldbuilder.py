# -*- coding: utf-8 -*-
"""
WorldBuilder: 从小说文本按章/小块抽取“世界观数据库”（facts / events / entities）→ SQLite

Usage:
  python worldbuilder.py --novel "C:/path/书.txt" --db "./db/书.world.sqlite" --provider openai --model gpt-4.1-mini --overwrite
  python worldbuilder.py --novel "C:/path/书.txt" --db "./db/书.world.sqlite" --provider ollama --model qwen2.5:7b-instruct
"""

import os, re, json, sqlite3, argparse, sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# ========== 文本切分：与前述一致 ==========
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
        if len(clean_title) > 80: clean_title = clean_title[:80] + "…"
        chapters.append((cid, clean_title, body))
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
        try: return p.read_text(encoding=enc, errors="strict")
        except Exception: continue
    return p.read_text(encoding="utf-8", errors="ignore")

# ========== 宽松 JSON & 键名归一 ==========
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
    "subject": ["subject","主语","实体","S","s","subj","entity"],
    "predicate": ["predicate","谓词","关系","P","p","pred","relation","关系/属性/动作"],
    "object": ["object","宾语","O","o","obj","对象","目标"],
    "summary": ["summary","摘要","说明","注释","简述","概述","短摘要"],
    "evidence": ["evidence","证据","出处","evidences","spans","evid"],
    "who": ["who","人物","角色","参与者","参与人"],
    "loc": ["where","loc","地点","位置","场所"],
    "action": ["action","动作","行为","做了什么","事件"],
    "details": ["details","细节","备注","说明"],
    "type": ["type","类型","实体类型","类别"],
    "name": ["name","名称","名字"],
    "aliases": ["aliases","别名","绰号","又名"],
    "notes": ["notes","备注","说明"],
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

def normalize_facts(data, chapter_id: str, chunk_id: str) -> List[Dict[str, Any]]:
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
            out.append({
                "subject": s, "predicate": p, "object": o, "summary": sm,
                "chapter_id": chapter_id, "chunk_id": chunk_id,
                "evidence": ev
            })
    return out[:12]

def normalize_events(data, chapter_id: str, chunk_id: str) -> List[Dict[str, Any]]:
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
        loc = _get_first(it, CANON["loc"])
        action = _get_first(it, CANON["action"])
        details = _get_first(it, CANON["details"])
        ev = _get_first(it, CANON["evidence"], as_list=True)
        if not ev: ev = [f"{{{{{chunk_id}}}}}"] if chunk_id else []
        if who or action or loc:
            out.append({
                "chapter_id": chapter_id, "chunk_id": chunk_id,
                "who": who or [], "loc": loc, "action": action, "details": details,
                "evidence": ev
            })
    return out[:12]

def normalize_entities(data) -> List[Dict[str, Any]]:
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

# ========== 提示词（3轮强制 JSON） ==========
# ======== 修复后的模板常量（全部用双花括号转义 JSON） ========

SYS_STRICT = "你是抽取器。只返回 JSON，不要多余文字。缺信息留空。确保输出是严格 JSON。"

# --- FACTS ---
FACTS_R1 = (
    "输入是一段小说文本片段，请抽取“可验证设定/规则（facts）”，并生成短摘要 summary。\n"
    "严格返回一个 JSON 对象，键为 facts，值是数组。数组元素必须为：\n"
    "{{\"subject\":\"<实体名或规则域>\",\"predicate\":\"...\",\"object\":\"...\",\"summary\":\"<简洁可检索>\","
    "\"evidence\":[\"{{{{chunk_id}}}}\"]}}\n"
    "若确实没有明确 facts，也返回 {{\"facts\":[]}}。\n"
    "【chapter_id】{chapter_id}\n"
    "【chunk_id】{chunk_id}\n"
    "【文本】{text}"
)

FACTS_R2 = (
    "你是抽取器。只返回 JSON。\n"
    "从片段中抽取1~6条 facts，不要编造小说外信息。\n"
    "示例：{{\"facts\":[\n"
    "  {{\"subject\":\"古墓机关\",\"predicate\":\"触发条件\",\"object\":\"踏错石板\",\"summary\":\"踏错石板会触发机关\","
    "\"evidence\":[\"{{{{chunk_id}}}}\"]}}\n"
    "]}}\n"
    "【chapter_id】{chapter_id}\n"
    "【chunk_id】{chunk_id}\n"
    "【文本】{text}"
)

FACTS_R3 = (
    "先识别片段里清晰的“动作/因果/条件”，再转成 S-P-O 三元组。\n"
    "输出：{{\"facts\":[{{\"subject\":\"...\",\"predicate\":\"...\",\"object\":\"...\",\"summary\":\"...\","
    "\"evidence\":[\"{{{{chunk_id}}}}\"]}}]}}\n"
    "【chapter_id】{chapter_id}\n"
    "【chunk_id】{chunk_id}\n"
    "【文本】{text}"
)

# --- EVENTS ---
EVENTS_R1 = (
    "只返回 JSON。抽取“发生的事件/时间线”：\n"
    "{{\"events\":[{{\"who\":[\"<人物1>\",\"<人物2>\"],\"where\":\"<地点或空>\",\"action\":\"<做了什么>\","
    "\"details\":\"<细节>\",\"evidence\":[\"{{{{chunk_id}}}}\"]}}]}}\n"
    "【chapter_id】{chapter_id}\n"
    "【chunk_id】{chunk_id}\n"
    "【文本】{text}"
)

EVENTS_R2 = (
    "只返回 JSON。至少产出1~8条；若没有，返回 {{\"events\":[]}}。\n"
    "例：{{\"events\":[{{\"who\":[\"吴邪\",\"张起灵\"],\"where\":\"盗洞口\",\"action\":\"进入墓道\","
    "\"details\":\"携带装备潜入\",\"evidence\":[\"{{{{chunk_id}}}}\"]}}]}}\n"
    "【chapter_id】{chapter_id}\n"
    "【chunk_id】{chunk_id}\n"
    "【文本】{text}"
)

# --- ENTITIES ---
ENTS_R1 = (
    "只返回 JSON。抽取实体（person/org/place/item/skill）：\n"
    "{{\"entities\":[{{\"type\":\"person|org|place|item|skill\",\"name\":\"...\",\"aliases\":[\"...\"],"
    "\"notes\":\"...\"}}]}}\n"
    "【文本】{text}"
)

ENTS_R2 = (
    "只返回 JSON。至少给出0~20条，type 仅限 person/org/place/item/skill。\n"
    "【文本】{text}"
)


# ========== OpenAI / Ollama 适配 ==========
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

def openai_chat_json(cli, model: str, sys_prompt: str, user_prompt: str) -> str:
    try:
        r = cli.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":sys_prompt},
                      {"role":"user","content":user_prompt}],
            temperature=0.0,
            response_format={"type":"json_object"},
        )
        return r.choices[0].message.content.strip()
    except Exception:
        r = cli.chat.completions.create(
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

def ask(provider, model, host, sys_p, user_p):
    if provider=="openai":
        cli = openai_client()
        return openai_chat_json(cli, model, sys_p, user_p)
    return ollama_chat_safe(model, sys_p, user_p, host)

# ========== DB 架构 ==========
SCHEMA = """
         CREATE TABLE IF NOT EXISTS chapters (
                                                 id TEXT PRIMARY KEY,
                                                 title TEXT,
                                                 content TEXT
         );
         CREATE TABLE IF NOT EXISTS chunks (
                                               id TEXT PRIMARY KEY,
                                               chapter_id TEXT,
                                               idx INTEGER,
                                               content TEXT,
                                               UNIQUE(chapter_id, idx)
             );
         CREATE TABLE IF NOT EXISTS entities (
                                                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                 type TEXT,
                                                 name TEXT,
                                                 aliases TEXT,  -- JSON
                                                 notes TEXT,
                                                 UNIQUE(type, name)
             );
         CREATE TABLE IF NOT EXISTS facts (
                                              id INTEGER PRIMARY KEY AUTOINCREMENT,
                                              subject TEXT,
                                              predicate TEXT,
                                              object TEXT,
                                              summary TEXT,
                                              chapter_id TEXT,
                                              chunk_id TEXT,
                                              evidence TEXT, -- JSON
                                              UNIQUE(subject, predicate, object, chapter_id, chunk_id)
             );
         CREATE TABLE IF NOT EXISTS events (
                                               id INTEGER PRIMARY KEY AUTOINCREMENT,
                                               chapter_id TEXT,
                                               chunk_id TEXT,
                                               who TEXT,      -- JSON
                                               loc TEXT,
                                               action TEXT,
                                               details TEXT,
                                               evidence TEXT, -- JSON
                                               UNIQUE(chapter_id, chunk_id, loc, action, details)
             ); \
         """

# ========== 抽取封装（3轮重试） ==========
def extract_facts(text: str, chapter_id: str, chunk_id: str, provider: str, model: str, host: str):
    text = text[:2000]
    for up in (
            FACTS_R1.format(text=text, chapter_id=chapter_id, chunk_id=chunk_id),
            FACTS_R2.format(text=text, chapter_id=chapter_id, chunk_id=chunk_id),
            FACTS_R3.format(text=text, chapter_id=chapter_id, chunk_id=chunk_id),
    ):
        raw = ask(provider, model, host, SYS_STRICT, up)
        data = json_relaxed(raw)
        facts = normalize_facts(data, chapter_id, chunk_id)
        if facts: return facts
    return []

def extract_events(text: str, chapter_id: str, chunk_id: str, provider: str, model: str, host: str):
    text = text[:2000]
    for up in (
            EVENTS_R1.format(text=text, chapter_id=chapter_id, chunk_id=chunk_id),
            EVENTS_R2.format(text=text, chapter_id=chapter_id, chunk_id=chunk_id),
    ):
        raw = ask(provider, model, host, SYS_STRICT, up)
        data = json_relaxed(raw)
        events = normalize_events(data, chapter_id, chunk_id)
        if events: return events
    return []

def extract_entities(text: str, provider: str, model: str, host: str):
    text = text[:2000]
    for up in (
            ENTS_R1.format(text=text),
            ENTS_R2.format(text=text),
    ):
        raw = ask(provider, model, host, SYS_STRICT, up)
        data = json_relaxed(raw)
        ents = normalize_entities(data)
        if ents: return ents
    return []

# ========== 主流程 ==========
def run(novel_path: str, db_path: str, provider: str, model: str, host: str,
        overwrite: bool, encoding: str, max_chars: int, do_facts: bool, do_events: bool, do_entities: bool):
    if overwrite and Path(db_path).exists():
        Path(db_path).unlink()
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    text = read_text_with_encoding(novel_path, preferred=encoding)
    chapters = split_chapters(text)

    conn = sqlite3.connect(db_path)
    with conn: conn.executescript(SCHEMA)

    # 写入章节与小块
    with conn:
        for cid, ctitle, cbody in chapters:
            conn.execute("INSERT OR REPLACE INTO chapters (id,title,content) VALUES (?,?,?)", (cid, ctitle, cbody))
            chunks = chunkify(cbody, max_chars=max_chars) or ["（空章节内容）"]
            for i, ch in enumerate(chunks, 1):
                chunk_id = f"{cid}_{i:03d}"
                conn.execute("INSERT OR REPLACE INTO chunks (id,chapter_id,idx,content) VALUES (?,?,?,?)",
                             (chunk_id, cid, i, ch))

    # 抽取并写库
    rows = conn.execute("SELECT id, chapter_id, content FROM chunks ORDER BY id").fetchall()
    facts_to_add, events_to_add, ents_to_add = [], [], []
    processed = 0

    for chunk_id, chap_id, content in rows:
        processed += 1
        # entities 通常不需要每块重复抽，可选仅在每章首块抽取以减少开销
        if do_entities and chunk_id.endswith("_001"):
            ents = extract_entities(content, provider, model, host)
            for en in ents:
                aliases = json.dumps(en.get("aliases", []), ensure_ascii=False)
                ents_to_add.append((en["type"], en["name"], aliases, en.get("notes","")))

        if do_facts:
            facts = extract_facts(content, chap_id, chunk_id, provider, model, host)
            for f in facts:
                facts_to_add.append((
                    f["subject"], f["predicate"], f["object"], f.get("summary",""),
                    f["chapter_id"], f["chunk_id"], json.dumps(f.get("evidence", []), ensure_ascii=False)
                ))

        if do_events:
            evs = extract_events(content, chap_id, chunk_id, provider, model, host)
            for e in evs:
                events_to_add.append((
                    e["chapter_id"], e["chunk_id"],
                    json.dumps(e.get("who", []), ensure_ascii=False),
                    e.get("loc",""), e.get("action",""), e.get("details",""),
                    json.dumps(e.get("evidence", []), ensure_ascii=False)
                ))

    with conn:
        if ents_to_add:
            conn.executemany(
                "INSERT OR IGNORE INTO entities (type,name,aliases,notes) VALUES (?,?,?,?)", ents_to_add
            )
        if facts_to_add:
            conn.executemany(
                "INSERT OR IGNORE INTO facts (subject,predicate,object,summary,chapter_id,chunk_id,evidence) VALUES (?,?,?,?,?,?,?)",
                facts_to_add
            )
        if events_to_add:
            conn.executemany(
                "INSERT OR IGNORE INTO events (chapter_id,chunk_id,who,loc,action,details,evidence) VALUES (?,?,?,?,?,?,?)",
                events_to_add
            )

    conn.close()
    print(json.dumps({
        "chapters": len(chapters),
        "chunks": len(rows),
        "added": {
            "entities": len(ents_to_add),
            "facts": len(facts_to_add),
            "events": len(events_to_add),
        }
    }, ensure_ascii=False))

# ========== CLI ==========
def main():
    ap = argparse.ArgumentParser(description="根据小说每章内容使用AI生成世界观数据库（SQLite）")
    ap.add_argument("--novel", required=True, help="小说 TXT 路径")
    ap.add_argument("--db", required=True, help="输出 SQLite 路径")
    ap.add_argument("--provider", choices=["openai","ollama"], default="openai")
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--host", default="http://127.0.0.1:11434", help="Ollama host")
    ap.add_argument("--overwrite", action="store_true", help="如存在则覆盖")
    ap.add_argument("--encoding", default="auto")
    ap.add_argument("--max-chars", type=int, default=700)
    ap.add_argument("--facts", action="store_true", help="抽取 facts")
    ap.add_argument("--events", action="store_true", help="抽取 events")
    ap.add_argument("--entities", action="store_true", help="抽取 entities")
    args = ap.parse_args()

    # 若未指定，默认全开
    do_facts = args.facts or (not args.facts and not args.events and not args.entities)
    do_events = args.events or (not args.facts and not args.events and not args.entities)
    do_entities = args.entities or (not args.facts and not args.events and not args.entities)

    run(
        novel_path=args.novel, db_path=args.db,
        provider=args.provider, model=args.model, host=args.host,
        overwrite=args.overwrite, encoding=args.encoding, max_chars=args.max_chars,
        do_facts=do_facts, do_events=do_events, do_entities=do_entities
    )

if __name__ == "__main__":
    main()

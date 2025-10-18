# -*- coding: utf-8 -*-
"""
GUI: Novel -> Plot Graph DB (plot_nodes / plot_edges / plot_state / facts)
- 选择小说TXT，按章与小块切分
- 生成节点摘要/提示/关键词（支持：Ollama本地免费 / OpenAI / 纯规则）
- facts 三元组“只使用 AI 抽取”（Ollama 或 OpenAI），不回退规则
- 更鲁棒的 JSON 解析、键名归一化与防御式入库，避免 KeyError（如 '"subject"'）
- 写入SQLite数据库（四张表：plot_nodes / plot_edges / plot_state / facts）
- 每本书一个独立数据库：默认 ./db/<书名>.sqlite
- 支持文本编码：auto / utf-8 / gbk / cp936 / big5 / utf-8-sig
"""

import os
import re
import json
import sqlite3
import threading
import requests
from pathlib import Path
from typing import List, Dict, Any

# -------------------------- 文本切分逻辑 --------------------------
CHAPTER_PATTERNS = [
    r"^\s*第[一二三四五六七八九十百千0-9]{1,6}章[：:\s].*$",
    r"^\s*第[一二三四五六七八九十百千0-9]{1,6}节[：:\s].*$",
    r"^\s*Chapter\s+\d+.*$",
    r"^\s*CHAPTER\s+\d+.*$",
]

def split_chapters(text: str):
    text = re.sub(
        r'(?<!\n)(第[一二三四五六七八九十百千万0-9]{1,6}(?:[章节回集]|卷[ 　\t]*第?[一二三四五六七八九十百千万0-9]{1,4}章)[^。\n\r]{0,60})',
        r'\n\1\n',
        text
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
        if len(clean_title) > 80:
            clean_title = clean_title[:80] + "…"
        chapters.append((cid, clean_title, body))
    return chapters

def para_split(s: str) -> List[str]:
    parts = re.split(r"\n\s*\n+", s.strip())
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p) <= 400:
            out.append(p)
        else:
            segs = re.split(r"(?<=[。！？!?])", p)
            buf = ""
            for seg in segs:
                if len(buf) + len(seg) <= 500:
                    buf += seg
                else:
                    if buf:
                        out.append(buf.strip())
                    buf = seg
            if buf.strip():
                out.append(buf.strip())
    return out

def chunkify(text: str, max_chars: int = 700) -> List[str]:
    paras = para_split(text)
    chunks = []
    cur = ""
    for p in paras:
        if not cur:
            cur = p
            continue
        if len(cur) + 1 + len(p) <= max_chars:
            cur = cur + "\n" + p
        else:
            chunks.append(cur.strip())
            cur = p
    if cur.strip():
        chunks.append(cur.strip())
    final = []
    for c in chunks:
        if len(c) <= max_chars:
            final.append(c)
        else:
            for i in range(0, len(c), max_chars):
                final.append(c[i:i+max_chars].strip())
    return final

# -------------------------- 编码自动/手动读取 --------------------------
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
    for enc in ("utf-8-sig", "utf-8", "gbk", "cp936", "big5"):
        try:
            return p.read_text(encoding=enc, errors="strict")
        except Exception:
            continue
    return p.read_text(encoding="utf-8", errors="ignore")

# -------------------------- 生成摘要/提示（规则兜底） --------------------------
def rule_fallback_enrich(title: str, content: str) -> Dict[str, str]:
    summary = (content.replace("\n","")[:58] + "…") if len(content) > 60 else content
    entry_hint = (f"进入：{title}"[:24]) if title else "进入当前片段"
    exit_hint = "继续推进剧情"
    words = re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]{2,}", content)
    uniq = []
    for w in words:
        if len(w) >= 2 and w not in uniq:
            uniq.append(w)
        if len(uniq) >= 10:
            break
    fact_tags = ",".join(uniq)
    return dict(summary=summary, entry_hint=entry_hint, exit_hint=exit_hint, fact_tags=fact_tags)

# -------------------------- Ollama 本地调用 --------------------------
def ollama_chat(model: str, sys_prompt: str, user_prompt: str, host: str = "http://127.0.0.1:11434",
                use_json: bool = False, options: dict = None) -> str:
    """调用 Ollama /api/chat；支持 format=json（若模型兼容）与采样参数。"""
    url = f"{host}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_prompt.strip()},
            {"role": "user",  "content": user_prompt.strip()},
        ],
        "stream": False,
    }
    if use_json:
        # 许多 Instruct 模型已支持结构化 JSON 输出
        payload["format"] = "json"
    if options:
        payload["options"] = options

    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"]


def ollama_chat_safe(model: str, sys_prompt: str, user_prompt: str, host: str = "http://127.0.0.1:11434") -> str:
    """优先尝试 JSON 格式；失败则普通模式回退。"""
    opts = {"temperature": 0.0, "top_p": 0.9, "repeat_penalty": 1.1, "num_ctx": 4096}
    try:
        return ollama_chat(model, sys_prompt, user_prompt, host=host, use_json=True, options=opts)
    except Exception:
        return ollama_chat(model, sys_prompt, user_prompt, host=host, use_json=False, options=opts)


def ai_enrich_ollama(model: str, title: str, content: str) -> Dict[str, str]:
    sys_prompt = "你是小说结构化助手，返回严格 JSON。"
    user_prompt = f"""
请基于下面“标题+正文片段”生成4个字段：
1) summary：不超过60字的中文摘要，客观陈述。
2) entry_hint：这一段开始前的引导语，不超过25字。
3) exit_hint：读完这一段后的过渡提示，不超过25字。
4) fact_tags：逗号分隔的关键词（人物、地点、物件、事件），最多12个。

【标题】{title}
【正文片段】{content[:1200]}

只返回 JSON 且仅包含这4个键：summary, entry_hint, exit_hint, fact_tags
"""
    try:
        txt = ollama_chat(model, sys_prompt, user_prompt)
        try:
            data = json.loads(txt)
        except Exception:
            data = {}
            for k in ("summary","entry_hint","exit_hint","fact_tags"):
                m = re.search(rf'"?{k}"?\s*[:：]\s*"(.*?)"', txt)
                if m:
                    data[k] = m.group(1)
        base = rule_fallback_enrich(title, content)
        for k, v in base.items():
            data.setdefault(k, v)
        return data
    except Exception:
        return rule_fallback_enrich(title, content)

# -------------------------- OpenAI（可选） --------------------------
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini").strip()
    if not api_key:
        return None, None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        return client, model
    except Exception:
        return None, None

def ai_enrich_openai(client, model: str, title: str, content: str) -> Dict[str, str]:
    if client is None:
        return rule_fallback_enrich(title, content)
    prompt = f"""
你是整理长篇小说结构的助手。请基于下面“标题+正文片段”生成4个字段：
1) summary：不超过60字的中文摘要，客观陈述。
2) entry_hint：这一段开始前的引导语，不超过25字。
3) exit_hint：读完这一段后的过渡提示，不超过25字。
4) fact_tags：逗号分隔的关键词（人物、地点、物件、事件），不超过12个。

【标题】{title}
【正文片段】{content[:1200]}
只返回JSON，键包含：summary, entry_hint, exit_hint, fact_tags。
"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":prompt}],
            temperature=0.2,
        )
        txt = resp.choices[0].message.content.strip()
        try:
            data = json.loads(txt)
        except Exception:
            data = {}
            for k in ("summary","entry_hint","exit_hint","fact_tags"):
                m = re.search(rf'"?{k}"?\s*[:：]\s*"(.*?)"', txt)
                if m:
                    data[k] = m.group(1)
        base = rule_fallback_enrich(title, content)
        for k, v in base.items():
            data.setdefault(k, v)
        return data
    except Exception:
        return rule_fallback_enrich(title, content)

# -------------------------- 统一入口：节点富化 --------------------------
def ai_enrich_unified(provider: str,
                      title: str,
                      content: str,
                      openai_client=None,
                      openai_model=None,
                      ollama_model=None):
    provider = (provider or "ollama").lower()
    if provider == "ollama":
        return ai_enrich_ollama(ollama_model or "llama3.1:8b-instruct", title, content)
    elif provider == "openai":
        return ai_enrich_openai(openai_client, openai_model or "gpt-4.1-mini", title, content)
    else:
        return rule_fallback_enrich(title, content)

# -------------------------- facts 抽取（AI-only；鲁棒解析 + 键名归一 + 防御入库） --------------------------
def _json_loads_relaxed(s: str):
    if not s:
        return None
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r'(\{(?:[^{}]|\{[^{}]*\})*\}|\[(?:[^\[\]]|\[[^\[\]]*\])*\])', s, re.S)
    if m:
        frag = m.group(1)
        try:
            return json.loads(frag)
        except Exception:
            return None
    return None

# 键名映射与归一化，兼容中文/别名
_CANON_MAP = {
    "subject": ["subject", "主语", "实体", "S", "s", "subj", "entity"],
    "predicate": ["predicate", "谓词", "关系", "P", "p", "pred", "relation"],
    "object": ["object", "宾语", "O", "o", "obj", "对象", "目标"],
    "summary": ["summary", "摘要", "说明", "注释", "简述", "概述"],
    "evidence": ["evidence", "证据", "出处", "evidences", "spans"],
}

def _get_first(it: Dict[str, Any], keys: List[str]) -> str:
    for k in keys:
        v = it.get(k)
        if v is None:
            continue
        if isinstance(v, (str, int, float)):
            return str(v).strip()
        # 某些模型把值包成 {"text":"..."} 或 ["..."]
        if isinstance(v, dict):
            if "text" in v and isinstance(v["text"], str):
                return v["text"].strip()
        if isinstance(v, list) and v and isinstance(v[0], str):
            return v[0].strip()
    return ""

def _normalize_facts_payload(data, chunk_id: str) -> List[Dict[str, str]]:
    # 允许：1) {"facts":[...]} 2) 直接 [...] 3) {"triples"/"data"/"result"/"items":[...]}
    if isinstance(data, list):
        facts_list = data
    elif isinstance(data, dict):
        facts_list = None
        for k in ("facts", "triples", "data", "result", "items"):
            v = data.get(k)
            if isinstance(v, list):
                facts_list = v
                break
        if facts_list is None:
            return []
    else:
        return []

    out = []
    for it in facts_list:
        if not isinstance(it, dict):
            continue
        s = _get_first(it, _CANON_MAP["subject"])
        p = _get_first(it, _CANON_MAP["predicate"])
        o = _get_first(it, _CANON_MAP["object"])
        sm = _get_first(it, _CANON_MAP["summary"])
        ev_raw = it.get("evidence") or it.get("证据") or it.get("evidences") or it.get("spans") or []
        ev: List[str] = []
        if isinstance(ev_raw, list):
            ev = [str(x) for x in ev_raw if isinstance(x, (str, int, float))]
        if not ev:
            ev = [f"{{{{{chunk_id}}}}}"] if chunk_id else []

        if s and p and o:
            out.append({
                "subject": s, "predicate": p, "object": o,
                "summary": sm, "evidence": ev
            })
    return out[:12]

_AI_SYS_FACTS = "你是抽取器。只返回 JSON，不要多余文字。缺信息留空。"
_AI_USER_FACTS_TPL = """输入是一段小说文本片段，请抽取“可验证设定/规则（facts）”，并生成短摘要 summary。
严格返回一个 JSON 对象，键为 facts，值是数组。数组元素必须为：
{"subject":"<实体名或规则域>","predicate":"...","object":"...","summary":"<简洁可检索>","evidence":["{{chunk_id}}"]}

示例：
{"facts":[{"subject":"规则A","predicate":"条件","object":"触发","summary":"A 满足条件即触发","evidence":["{{chunk_id}}"]}]}

【标题】{title}
【chunk_id】{chunk_id}
【文本】{text}"""

def ai_extract_facts_ollama(model: str, title: str, content: str, chunk_id: str,
                            host: str = "http://127.0.0.1:11434") -> List[Dict[str,str]]:
    sys_prompt = _AI_SYS_FACTS
    user_prompt = _AI_USER_FACTS_TPL.format(
        title=title, chunk_id=chunk_id or "", text=content[:1500]
    )
    try:
        txt = ollama_chat(model, sys_prompt, user_prompt, host=host)
        data = _json_loads_relaxed(txt)
        return _normalize_facts_payload(data, chunk_id)
    except Exception:
        return []

def ai_extract_facts_openai(client, model: str, title: str, content: str, chunk_id: str) -> List[Dict[str,str]]:
    if client is None:
        return []
    sys_prompt = _AI_SYS_FACTS
    user_prompt = _AI_USER_FACTS_TPL.format(
        title=title, chunk_id=chunk_id or "", text=content[:1500]
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":sys_prompt},{"role":"user","content":user_prompt}],
            temperature=0.0,
        )
        txt = resp.choices[0].message.content.strip()
        data = _json_loads_relaxed(txt)
        return _normalize_facts_payload(data, chunk_id)
    except Exception:
        return []

def extract_facts_unified(provider: str,
                          title: str,
                          content: str,
                          openai_client=None,
                          openai_model: str = None,
                          ollama_model: str = None,
                          chunk_id: str = "") -> List[Dict[str,str]]:
    provider = (provider or "ollama").lower()
    if provider == "ollama":
        return ai_extract_facts_ollama(ollama_model or "qwen2.5:7b-instruct", title, content, chunk_id)
    elif provider == "openai":
        return ai_extract_facts_openai(openai_client, openai_model or "gpt-4.1-mini", title, content, chunk_id)
    else:
        return []

# -------------------------- 数据库 --------------------------
SCHEMA_SQL = """
             CREATE TABLE IF NOT EXISTS plot_nodes (
                                                       id             TEXT PRIMARY KEY,
                                                       title          TEXT NOT NULL,
                                                       summary        TEXT NOT NULL,
                                                       entry_hint     TEXT,
                                                       exit_hint      TEXT,
                                                       fact_tags      TEXT,
                                                       required_flags TEXT,
                                                       set_flags      TEXT
             );

             CREATE TABLE IF NOT EXISTS plot_edges (
                                                       id         INTEGER PRIMARY KEY AUTOINCREMENT,
                                                       src        TEXT NOT NULL,
                                                       dst        TEXT NOT NULL,
                                                       condition  TEXT,
                                                       keywords   TEXT,
                                                       UNIQUE(src, dst)
                 );

             CREATE TABLE IF NOT EXISTS plot_state (
                                                       save_id      TEXT PRIMARY KEY DEFAULT 'default',
                                                       current_node TEXT NOT NULL
             );

/* facts 表（供 interactive 使用） */
             CREATE TABLE IF NOT EXISTS facts (
                                                  id        INTEGER PRIMARY KEY AUTOINCREMENT,
                                                  subject   TEXT,
                                                  predicate TEXT,
                                                  object    TEXT,
                                                  summary   TEXT,
                                                  UNIQUE(subject, predicate, object)
                 );

             CREATE INDEX IF NOT EXISTS idx_facts_sp ON facts(subject, predicate);
             CREATE INDEX IF NOT EXISTS idx_facts_po ON facts(predicate, object); \
             """

def ensure_db(db_path: str):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    with conn:
        conn.executescript(SCHEMA_SQL)
    return conn

# -------------------------- 主流程 --------------------------
def build_graph_from_novel(novel_path: str,
                           db_path: str,
                           provider: str = "ollama",
                           ollama_model: str = "llama3.1:8b-instruct",
                           max_chunk_chars: int = 700,
                           overwrite: bool = False,
                           encoding: str = "auto",
                           make_facts: bool = True,
                           log=lambda msg: None):
    novel_path = Path(novel_path)
    if not novel_path.exists():
        raise FileNotFoundError(f"小说不存在：{novel_path}")

    db_path = Path(db_path)
    if db_path.exists() and overwrite:
        try:
            db_path.unlink()
        except Exception as e:
            raise RuntimeError(f"无法删除旧数据库：{db_path}，原因：{e}")
    elif db_path.exists() and not overwrite:
        raise FileExistsError(f"数据库已存在（为避免覆盖，默认不重建）：{db_path}")

    text = read_text_with_encoding(str(novel_path), preferred=encoding)
    log(f"文本读取完成（encoding={encoding})")
    chapters = split_chapters(text)
    log(f"检测到章节数：{len(chapters)}")

    openai_client, openai_model = (get_openai_client() if provider == "openai" else (None, None))
    conn = ensure_db(str(db_path))
    cur = conn.cursor()

    node_ids_in_order: List[str] = []
    total_nodes = 0
    facts_bucket = set()

    for cid, ctitle, cbody in chapters:
        chunks = chunkify(cbody, max_chars=max_chunk_chars)
        if not chunks:
            chunks = ["（空章节内容）"]
        log(f"处理 {ctitle}：切分块 {len(chunks)}")

        for i, chunk in enumerate(chunks, 1):
            node_id = f"{cid}_{i:03d}"
            node_title = f"{ctitle} · {i}"

            enrich = ai_enrich_unified(
                provider=provider,
                title=node_title,
                content=chunk,
                openai_client=openai_client,
                openai_model=openai_model,
                ollama_model=ollama_model,
            )

            required_flags = ""
            set_flags = node_id

            cur.execute(
                """INSERT OR REPLACE INTO plot_nodes
                   (id, title, summary, entry_hint, exit_hint, fact_tags, required_flags, set_flags)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (node_id, node_title, enrich["summary"], enrich["entry_hint"],
                 enrich["exit_hint"], enrich["fact_tags"], required_flags, set_flags)
            )

            # 抽取并收集 facts（仅 AI），防御式过滤
            if make_facts:
                try:
                    facts = extract_facts_unified(
                        provider=provider,
                        title=node_title,
                        content=chunk,
                        openai_client=openai_client,
                        openai_model=openai_model,
                        ollama_model=ollama_model,
                        chunk_id=node_id,
                    )
                except Exception:
                    facts = []

                if facts:
                    for f in facts:
                        try:
                            if isinstance(f, dict):
                                s = (f.get("subject") or "").strip()
                                p = (f.get("predicate") or "").strip()
                                o = (f.get("object") or "").strip()
                                sm = (f.get("summary") or "").strip()
                            elif isinstance(f, (list, tuple)) and len(f) >= 3:
                                s, p, o = [str(x).strip() for x in f[:3]]
                                sm = str(f[3]).strip() if len(f) > 3 else ""
                            else:
                                continue
                            if s and p and o:
                                facts_bucket.add((s, p, o, sm))
                        except Exception:
                            continue

            node_ids_in_order.append(node_id)
            total_nodes += 1

    for a, b in zip(node_ids_in_order[:-1], node_ids_in_order[1:]):
        cur.execute(
            "INSERT OR IGNORE INTO plot_edges (src, dst, condition, keywords) VALUES (?, ?, ?, ?)",
            (a, b, "顺序推进", "next")
        )

    if node_ids_in_order:
        first_node = node_ids_in_order[0]
        cur.execute(
            "INSERT OR REPLACE INTO plot_state (save_id, current_node) VALUES (?, ?)",
            ("default", first_node)
        )

    if make_facts and facts_bucket:
        cur.executemany(
            "INSERT OR IGNORE INTO facts (subject, predicate, object, summary) VALUES (?, ?, ?, ?)",
            list(facts_bucket)
        )

    conn.commit()
    conn.close()

    log(f"✅ 完成：{len(chapters)} 章，共 {total_nodes} 个节点 → {db_path}")
    if make_facts:
        log(f"📚 已写入 facts：{len(facts_bucket)} 条（去重后）")
    return total_nodes

# -------------------------- 实用函数 --------------------------
def slug_from_input(path_str: str) -> str:
    name = Path(path_str).stem
    cleaned = "".join(ch for ch in name if ch not in "\r\n\t").strip().replace(" ", "_")
    return cleaned or "book"

# -------------------------- GUI --------------------------
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Novel → Plot DB 生成器（Ollama/OpenAI/规则 + facts仅AI + 键名归一）")
        self.geometry("900x680")

        self.var_input = tk.StringVar()
        self.var_provider = tk.StringVar(value="ollama")
        self.var_ollama_model = tk.StringVar(value="qwen2.5:7b-instruct")
        self.var_openai_model = tk.StringVar(value=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini"))
        self.var_max_chars = tk.IntVar(value=700)
        self.var_db = tk.StringVar(value="")
        self.var_overwrite = tk.BooleanVar(value=False)
        self.var_encoding = tk.StringVar(value="auto")
        self.var_make_facts = tk.BooleanVar(value=True)

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 8, "pady": 6}
        frm = ttk.Frame(self); frm.pack(fill="x", **pad)

        ttk.Label(frm, text="小说TXT：").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.var_input, width=72).grid(row=0, column=1, sticky="we")
        ttk.Button(frm, text="浏览…", command=self._browse_input).grid(row=0, column=2, sticky="w")

        ttk.Label(frm, text="摘要提供方：").grid(row=1, column=0, sticky="w")
        cmb = ttk.Combobox(frm, textvariable=self.var_provider, state="readonly",
                           values=["ollama","openai","rule"], width=12)
        cmb.grid(row=1, column=1, sticky="w")
        cmb.bind("<<ComboboxSelected>>", lambda e: self._render_model_fields())

        self.frm_model = ttk.Frame(frm); self.frm_model.grid(row=2, column=0, columnspan=3, sticky="we")
        self._render_model_fields()

        ttk.Label(frm, text="小块最大字数：").grid(row=3, column=0, sticky="w")
        ttk.Spinbox(frm, from_=200, to=2000, textvariable=self.var_max_chars, width=10).grid(row=3, column=1, sticky="w")

        ttk.Label(frm, text="文本编码：").grid(row=4, column=0, sticky="w")
        ttk.Combobox(frm, textvariable=self.var_encoding, state="readonly",
                     values=["auto","utf-8","utf-8-sig","gbk","cp936","big5"]).grid(row=4, column=1, sticky="w")

        ttk.Label(frm, text="数据库文件（留空=自动）：").grid(row=5, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.var_db, width=72).grid(row=5, column=1, sticky="we")
        ttk.Button(frm, text="选择…", command=self._browse_db).grid(row=5, column=2, sticky="w")

        ttk.Checkbutton(frm, text="允许覆盖已存在的同名数据库（会删除旧库）", variable=self.var_overwrite).grid(row=6, column=1, sticky="w")
        ttk.Checkbutton(frm, text="同时抽取 facts（三元组，仅AI抽取）", variable=self.var_make_facts).grid(row=7, column=1, sticky="w")

        ttk.Button(self, text="开始生成", command=self._run_thread).pack(**pad)

        self.txt = tk.Text(self, height=22); self.txt.pack(fill="both", expand=True, **pad)
        self._log("准备就绪。默认使用 Ollama 本地模型（qwen2.5:7b-instruct）。facts 抽取仅使用 AI（含键名归一）。")

    def _render_model_fields(self):
        for w in self.frm_model.winfo_children():
            w.destroy()
        provider = self.var_provider.get()
        if provider == "ollama":
            ttk.Label(self.frm_model, text="Ollama模型：").grid(row=0, column=0, sticky="w")
            ttk.Entry(self.frm_model, textvariable=self.var_ollama_model, width=30).grid(row=0, column=1, sticky="w")
            ttk.Label(self.frm_model, text="示例：qwen2.5:7b-instruct / llama3.1:8b-instruct").grid(row=0, column=2, sticky="w")
        elif provider == "openai":
            ttk.Label(self.frm_model, text="OpenAI模型：").grid(row=0, column=0, sticky="w")
            ttk.Entry(self.frm_model, textvariable=self.var_openai_model, width=30).grid(row=0, column=1, sticky="w")
            ttk.Label(self.frm_model, text="示例：gpt-4.1-mini（需设置 OPENAI_API_KEY）").grid(row=0, column=2, sticky="w")
        else:
            ttk.Label(self.frm_model, text="当前使用纯规则模式（仅用于摘要富化）；facts 抽取请在上方选择 AI 提供方").grid(row=0, column=0, sticky="w")

    def _browse_input(self):
        path = filedialog.askopenfilename(title="选择小说TXT", filetypes=[("Text","*.txt"),("All","*.*")])
        if path:
            self.var_input.set(path)
            slug = slug_from_input(path)
            auto_db = str(Path("./db") / f"{slug}.sqlite")
            auto_db = auto_db.replace("}}", "}")  # 容错
            if not self.var_db.get():
                self.var_db.set(auto_db)

    def _browse_db(self):
        path = filedialog.asksaveasfilename(title="选择或输入数据库文件名",
                                            initialfile=self.var_db.get() or "book.sqlite",
                                            defaultextension=".sqlite",
                                            filetypes=[("SQLite DB","*.sqlite;*.db"),("All","*.*")])
        if path:
            self.var_db.set(path)

    def _log(self, msg: str):
        self.txt.insert("end", msg + "\n")
        self.txt.see("end")
        self.update_idletasks()

    def _run_thread(self):
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def _run(self):
        novel = self.var_input.get().strip()
        if not novel:
            messagebox.showwarning("提示", "请选择小说TXT文件")
            return
        db = self.var_db.get().strip()
        if not db:
            slug = slug_from_input(novel)
            db = str(Path("./db") / f"{slug}.sqlite")
            self.var_db.set(db)

        provider = self.var_provider.get().strip().lower()
        max_chars = int(self.var_max_chars.get())
        overwrite = bool(self.var_overwrite.get())
        ollama_model = self.var_ollama_model.get().strip() or "qwen2.5:7b-instruct"
        encoding = self.var_encoding.get().strip().lower()
        make_facts = bool(self.var_make_facts.get())

        self._log(f"输入：{novel}")
        self._log(f"数据库：{db}")
        self._log(f"模式：{provider}，模型：{ollama_model if provider=='ollama' else self.var_openai_model.get()}")
        self._log(f"小块最大字数：{max_chars}，覆盖：{overwrite}，编码：{encoding}")

        if provider == "ollama":
            try:
                r = requests.get("http://127.0.0.1:11434/api/tags", timeout=3)
                if r.status_code == 200:
                    self._log("Ollama 服务已连接。")
            except Exception:
                self._log("⚠ 无法连接 Ollama 服务（请先运行：ollama serve），将继续尝试调用。")
        elif provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                self._log("⚠ 未检测到 OPENAI_API_KEY，OpenAI 调用将失败；facts 抽取不会写入。")

        try:
            total = build_graph_from_novel(
                novel_path=novel,
                db_path=db,
                provider=provider,
                ollama_model=ollama_model,
                max_chunk_chars=max_chars,
                overwrite=overwrite,
                encoding=encoding,
                make_facts=make_facts,
                log=self._log
            )
            self._log(f"🎉 完成！共写入 {total} 个节点。")
        except FileExistsError as e:
            self._log(f"❌ {e}")
            messagebox.showerror("已存在", str(e))
        except Exception as e:
            self._log(f"❌ 失败：{e}")
            messagebox.showerror("错误", str(e))

def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()

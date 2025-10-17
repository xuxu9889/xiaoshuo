# -*- coding: utf-8 -*-
"""
交互叙事（统一引擎版：导演+正文一体）
- 启动时：选择 主角模式 / 旁观模式
- 自动基于最早的 chunks（默认取前 3 段）生成“世界观/人物关系”开场（允许润色，不虚构）
- 之后每轮：构造统一引擎 Prompt（含 memory + 证据 + 输入），由 story_turn 输出小说正文（纯文本）
- 线性推进：仅允许使用 chunk 顺序号 ≤ canon_cursor 的证据
"""

import sys, re, json, textwrap, shutil, os
from typing import Dict, Any, List, Optional, Tuple

# ====== 真实“写正文”的引擎（你项目里已有）======
try:
    from engine import story_turn as _engine_story_turn
except Exception:
    _engine_story_turn = None

# 可选：如果你的 engine 里有 get_evidence，则优先用它；否则直接查 DB
try:
    from engine import get_evidence as _engine_get_evidence
except Exception:
    _engine_get_evidence = None

# 直接从 SQLite 取 chunks
from store import get_sql

PROMPT_UNIFIED = os.path.join("prompts", "story_engine_unified.txt")

# ================= 渲染（自然段 + 可切换软换行） =================
def _terminal_width(default=80) -> int:
    try:
        cols = shutil.get_terminal_size((default, 24)).columns
    except Exception:
        cols = default
    return max(40, min(cols - 4, 100))

def _wrap_paragraph(par: str, width: int, wrap_mode: str) -> str:
    if not par:
        return ""
    if par.strip().startswith("```") or re.search(r"https?://\S+", par):
        return par
    if re.match(r"^\s*\[\d+\]\s", par):
        return par
    if wrap_mode == "off":
        return par
    out = []
    for line in par.splitlines():
        if len(line) <= width:
            out.append(line)
        else:
            out.extend(textwrap.wrap(
                line, width=width,
                break_long_words=False,
                break_on_hyphens=False,
                replace_whitespace=False,
                drop_whitespace=False
            ) or [line])
    return "\n".join(out)

def _coerce_text(x):
    """把任意返回（str/dict/tuple/None）统一成文本。"""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, tuple) and len(x) > 0:
        return _coerce_text(x[0])
    if isinstance(x, dict):
        for k in ("text", "message", "content", "output", "narrative", "reply"):
            v = x.get(k)
            if isinstance(v, str):
                return v
        return json.dumps(x, ensure_ascii=False, indent=2)
    return str(x)

def render_story(text, width: int = None, wrap_mode: str = "soft"):
    print("\n—— 剧情 ——")
    s = _coerce_text(text)
    if not s:
        print("(无叙事)")
        return
    width = width or _terminal_width()
    paras = re.split(r"\n\s*\n", s.strip())
    print("\n\n".join(_wrap_paragraph(p, width, wrap_mode) for p in paras))

def render_state(memory: Dict[str, Any]):
    view = {
        "mode": memory.get("narration_mode", None),
        "location": memory.get("location"),
        "scene": memory.get("scene"),
        "inventory": memory.get("inventory", []),
        "flags": {k:v for k,v in (memory.get("flags",{}) or {}).items()
                  if isinstance(v,(str,int,float,bool))},
        "stats": memory.get("stats", {}),
        "hp": memory.get("hp"), "mp": memory.get("mp"),
        "canon_cursor": memory.get("canon_cursor"),
        "known_entities": memory.get("known_entities", []),
        "known_locations": memory.get("known_locations", []),
    }
    print("\n—— 状态 ——")
    print(json.dumps(view, ensure_ascii=False, indent=2))

# ================== chunks 读取 & 线性推进 ==================
def _chunk_ord(eid: str) -> int:
    if not isinstance(eid, str): return 10**9
    m = re.search(r"_(\d+)$", eid); return int(m.group(1)) if m else 10**9

def _fetch_first_chunks(n: int = 3) -> List[Dict[str, str]]:
    con = get_sql(); cur = con.cursor()
    cur.execute("SELECT chunk_id, text FROM chunks ORDER BY chunk_id LIMIT ?", (n,))
    rows = cur.fetchall(); con.close()
    return [{"id": r[0], "text": r[1]} for r in rows]

def _fetch_chunks_upto(limit_ord: int, budget_chars: int = 2000) -> List[Dict[str,str]]:
    """取 id 顺序号 <= limit_ord 的片段，按顺序拼到 budget 上限。"""
    con = get_sql(); cur = con.cursor()
    cur.execute("SELECT chunk_id, text FROM chunks ORDER BY chunk_id")
    rows = cur.fetchall(); con.close()
    acc, picked = 0, []
    for cid, txt in rows:
        if _chunk_ord(cid) <= int(limit_ord):
            tlen = len(txt)
            if acc + tlen > budget_chars:
                break
            picked.append({"id": cid, "text": txt})
            acc += tlen
    return picked

def _get_evidence(user_text: str, memory: Dict[str, Any]) -> List[Dict[str, Any]]:
    """优先用你项目里的检索；否则用 canon_cursor 前的 chunks 兜底。"""
    if _engine_get_evidence is not None:
        try:
            evi = _engine_get_evidence(user_text, memory) or []
            # 线性过滤
            lim = int(memory.get("canon_cursor", 10**9))
            return [e for e in evi if _chunk_ord(e.get("id") or "") <= lim]
        except Exception:
            pass
    # 兜底：直接从 DB 用 canon_cursor 取材料
    lim = int(memory.get("canon_cursor", 0))
    return _fetch_chunks_upto(lim, 2000)

# ================== 命令处理 ==================
def handle_command(line: str, state: Dict[str, Any]) -> Optional[str]:
    if line.startswith("/len"):
        mm = re.match(r"^/len\s+(\d+)$", line.strip())
        if not mm:
            print("用法：/len 300"); return None
        state["reply_len"] = int(mm.group(1))
        print(f"(已切换长度为 {state['reply_len']} 字)"); return None

    if line.startswith("/mode"):
        mm = re.match(r"^/mode\s+(主角|旁观)$", line.strip())
        if not mm:
            print("用法：/mode 主角  或  /mode 旁观"); return None
        state["memory"]["narration_mode"] = mm.group(1)
        print(f"(已切换模式为 {mm.group(1)})"); return None

    if line.strip() in ("/state","/s"):
        render_state(state["memory"]); return None

    if line.strip() in ("/reset","/r"):
        state["memory"] = default_memory()
        state["intro_done"] = False
        print("(记忆已重置)"); return None

    if line.strip() in ("/wrap soft","/wrap off"):
        state["wrap_mode"] = line.strip().split()[-1]
        print(f"(已切换换行模式为 {state['wrap_mode']})"); return None

    if line.strip() in ("/exit","/quit","/q"):
        print("再见。"); sys.exit(0)

    # 可选：查看前几条 chunks（便于调试）
    if line.startswith("/chunks"):
        mm = re.match(r"^/chunks(?:\s+(\d+))?$", line.strip())
        n = int(mm.group(1)) if mm and mm.group(1) else 10
        print(_list_first_chunks(n)); return None

    if line.startswith("/chunk"):
        mm = re.match(r"^/chunk\s+(\S+)$", line.strip())
        if not mm:
            print("用法：/chunk 1  或  /chunk chunk_00001"); return None
        token = mm.group(1)
        if token.isdigit():
            print(_show_chunk_index(int(token)))
        else:
            print(_show_chunk_id(token))
        return None

    return line

# —— CLI 中快速看 chunks（避免 f-string 反斜杠） ——
def _list_first_chunks(n=10) -> str:
    con = get_sql(); cur = con.cursor()
    cur.execute("SELECT chunk_id, substr(text,1,120) FROM chunks ORDER BY chunk_id LIMIT ?", (n,))
    rows = cur.fetchall(); con.close()
    lines = []
    for cid, txt in rows:
        preview = txt.replace('\n', ' ')
        lines.append(f"{cid} | {preview}")
    return "\n".join(lines)

def _show_chunk_id(chunk_id: str) -> str:
    con = get_sql(); cur = con.cursor()
    cur.execute("SELECT chunk_id, text FROM chunks WHERE chunk_id = ?", (chunk_id,))
    row = cur.fetchone(); con.close()
    if not row: return f"未找到 {chunk_id}"
    return f"== {row[0]} ==\n{row[1]}"

def _show_chunk_index(idx: int) -> str:
    con = get_sql(); cur = con.cursor()
    cur.execute("SELECT chunk_id, text FROM chunks ORDER BY chunk_id LIMIT 1 OFFSET ?", (idx-1,))
    row = cur.fetchone(); con.close()
    if not row: return f"没有第 {idx} 条"
    return f"== {row[0]} ==\n{row[1]}"

# ================== 统一引擎 Prompt 构造与调用 ==================
def _load(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _build_unified_prompt(mode: str, memory: Dict[str, Any], evidence: List[Dict[str,Any]],
                          user_text: str, reply_len: int) -> str:
    tmpl = _load(PROMPT_UNIFIED)

    # 证据拼接
    evid_text = []
    for e in evidence or []:
        cid = e.get("id") or ""
        txt = e.get("text") or ""
        evid_text.append(f"[{cid}]\n{txt}")
    evid_str = "\n\n".join(evid_text)

    # 占位替换
    text = tmpl
    text = text.replace("{{MODE}}", memory.get("narration_mode") or mode or "主角")
    text = text.replace("{{CANON_CURSOR}}", str(memory.get("canon_cursor", 0)))
    text = text.replace("{{REPLY_LEN}}", str(reply_len))
    text = text.replace("{{MEMORY_JSON}}", json.dumps(memory, ensure_ascii=False, indent=2))
    text = text.replace("{{EVIDENCE_TEXT}}", evid_str or "（当前无可用证据）")
    text = text.replace("{{USER_INPUT}}", user_text or "")
    return text

def engine_story(prompt_text: str, memory: Dict[str, Any], reply_len: int) -> Tuple[str, Dict[str, Any]]:
    """调用你项目的 story_turn；它必须返回小说正文文本（推荐），或 (text, memory)。"""
    if _engine_story_turn is None:
        return "(未接入 story_turn)", memory
    try:
        return _engine_story_turn(prompt_text, memory, reply_len=reply_len)
    except Exception as e:
        return f"(story_turn 异常：{e})", memory

# ================== 默认记忆 ==================
def default_memory() -> Dict[str, Any]:
    return {
        "location": None,
        "scene": None,
        "flags": {},
        "inventory": [],
        "stats": {},
        "hp": 100, "mp": 100,
        "recent_summary": "",
        "narration_mode": None,      # 启动时选择 主角/旁观
        "canon_cursor": 0,           # 从最早开始，线性推进
        "known_entities": [],
        "known_locations": [],
    }

# ================== 主流程 ==================
def main():
    print("== 交互叙事（小说模式·正典严格｜统一引擎）==\n")

    state = {
        "reply_len": 300,
        "memory": default_memory(),
        "wrap_mode": "soft",
        "intro_done": False,
    }

    # —— 第 0 步：选择模式（一次性） ——
    while not state["memory"]["narration_mode"]:
        mode_raw = input("（输入模式：主角模式 / 旁观模式）> ").strip()
        if mode_raw in ("主角模式", "旁观模式"):
            state["memory"]["narration_mode"] = "主角" if mode_raw == "主角模式" else "旁观"
        else:
            print("（请输入：主角模式 或 旁观模式）")

    # —— 第 1 步：自动世界观简介（一次性） ——
    if not state["intro_done"]:
        first_chunks = _fetch_first_chunks(n=3)
        if first_chunks:
            # 对齐正典窗口
            state["memory"]["canon_cursor"] = max(
                state["memory"]["canon_cursor"],
                _chunk_ord(first_chunks[0]["id"])
            )
        intro_evidence = first_chunks  # 允许润色但不得虚构
        intro_prompt = _build_unified_prompt(
            mode=state["memory"]["narration_mode"],
            memory=state["memory"],
            evidence=intro_evidence,
            user_text="（开场：请基于最早的片段，凝练世界观与人物关系简介；最后落于“故事从此刻展开”）",
            reply_len=state["reply_len"]
        )
        intro_text, mem_after_intro = engine_story(intro_prompt, state["memory"], reply_len=state["reply_len"])
        state["memory"] = mem_after_intro
        state["intro_done"] = True
        render_story(intro_text, wrap_mode=state.get("wrap_mode", "soft"))

    # —— 第 2 步：互动循环 ——
    while True:
        try:
            raw = input("\n你> ").rstrip("\n")
        except (EOFError, KeyboardInterrupt):
            print("\n再见。"); break

        if not raw:
            continue

        # 命令
        cmd = handle_command(raw, state)
        if cmd is None:
            continue
        user = cmd

        # 取证据（线性窗口内）
        evidence = _get_evidence(user, state["memory"])

        # 构造统一引擎 Prompt 并生成正文
        prompt = _build_unified_prompt(
            mode=state["memory"].get("narration_mode","主角"),
            memory=state["memory"],
            evidence=evidence,
            user_text=user,
            reply_len=state["reply_len"]
        )
        story_text, mem_after_story = engine_story(prompt, state["memory"], reply_len=state["reply_len"])
        state["memory"] = mem_after_story

        # 渲染
        render_story(story_text, wrap_mode=state.get("wrap_mode","soft"))

if __name__ == "__main__":
    main()

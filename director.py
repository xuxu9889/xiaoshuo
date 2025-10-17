# -*- coding: utf-8 -*-
# director.py —— 只返回“导演控制令”（不写正文）

import json
from typing import Dict, Any, List, Tuple, Union

SYSTEM_PATH = "prompts/director_system.txt"

_engine_gen = None
_engine_story = None

try:
    from engine import generate_with_context as _engine_gen
except Exception:
    _engine_gen = None

try:
    from engine import story_turn as _engine_story
except Exception:
    _engine_story = None


def _load(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def clamp_context(evidence: List[Dict[str, Any]], max_chars=4000) -> str:
    if not evidence:
        return ""
    buf, acc = [], 0
    for e in evidence:
        t = f"[{e.get('id','')}] {e.get('text','')}\n"
        if acc + len(t) > max_chars: break
        buf.append(t); acc += len(t)
    return "".join(buf)

def extract_json(txt: str) -> str:
    s = txt if isinstance(txt, str) else json.dumps(txt, ensure_ascii=False)
    start = s.find("{"); end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found.")
    return s[start:end + 1]

def _merge(old, new):
    if old is None: return new
    if isinstance(old, dict) and isinstance(new, dict):
        o = dict(old); o.update(new); return o
    return new

def _call_backend(system: str, user: str, evidence_docs: List[Dict[str, Any]], reply_len: int = 300) -> Union[str, Dict[str, Any]]:
    if _engine_gen is not None:
        return _engine_gen(system, user, evidence_docs)
    if _engine_story is not None:
        ctx = clamp_context(evidence_docs, 3500)
        wrapped = (
                "【系统设定】\n" + system + "\n\n"
                                          "【证据片段】\n" + (ctx or "（无）") + "\n\n"
                                                                             "【任务要求】只输出 JSON（不要正文）。\n\n"
                                                                             "【控制对象（玩家原话+当前状态）】\n" + user + "\n"
        )
        out, _ = _engine_story(wrapped, {}, reply_len=reply_len)
        return out
    raise RuntimeError("没有可用后端：engine.generate_with_context / engine.story_turn 均不可用。")

def _normalize_to_dict(raw: Union[str, Dict[str, Any], Tuple[Any, Any]]) -> Dict[str, Any]:
    if isinstance(raw, tuple) and raw:
        return _normalize_to_dict(raw[0])
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(extract_json(raw))
        except Exception:
            return {}
    return {}

def apply_patch(mem: Dict[str, Any], patch: Dict[str, Any], mode: str) -> Dict[str, Any]:
    m = {**mem}
    if mode == "旁观":
        allowed = {"scene", "known_info", "recent_summary"}
        for k, v in (patch or {}).items():
            if k in allowed:
                m[k] = _merge(m.get(k), v)
        return m

    for k, v in (patch or {}).items():
        if k == "inventory_add":
            inv = set(m.get("inventory", []))
            for it in v or []: inv.add(it)
            m["inventory"] = list(inv)
        elif k == "inventory_del":
            inv = list(m.get("inventory", []))
            for it in v or []:
                if it in inv: inv.remove(it)
            m["inventory"] = inv
        elif k == "flags":
            base = m.get("flags", {}) or {}
            base.update(v or {})
            m["flags"] = base
        elif k == "known_entities_add":
            base = list(m.get("known_entities", []))
            for it in v or []:
                if it not in base: base.append(it)
            m["known_entities"] = base
        elif k == "known_locations_add":
            base = list(m.get("known_locations", []))
            for it in v or []:
                if it not in base: base.append(it)
            m["known_locations"] = base
        else:
            m[k] = _merge(m.get(k), v)
    return m

def director_turn(
        user_text: str,
        memory: Dict[str, Any],
        evidence_docs: List[Dict[str, Any]],
        mode: str = "主角",
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, str]]]:
    """
    返回：(control, new_memory, choices)
    - control：导演控制令（不含正文）
    - new_memory：应用 state_patch 后的 memory
    - choices：供 UI 显示的下一步候选
    """
    system = _load(SYSTEM_PATH)
    state_view = {
        "mode": mode,
        "location": memory.get("location"),
        "scene": memory.get("scene"),
        "flags": memory.get("flags", {}),
        "inventory": memory.get("inventory", []),
        "stats": memory.get("stats", {}),
        "recent": memory.get("recent_summary", ""),
        "canon_cursor": memory.get("canon_cursor", 0),
        "known_entities": memory.get("known_entities", []),
        "known_locations": memory.get("known_locations", []),
    }
    user_prompt = (
            "【玩家输入】\n" + (user_text or "") + "\n\n"
                                                 "【当前状态摘要】\n" + json.dumps(state_view, ensure_ascii=False)
    )

    raw = _call_backend(system, user_prompt, evidence_docs)
    control = _normalize_to_dict(raw)

    # 容错：最少给个骨架，避免中断
    control = control if isinstance(control, dict) else {}
    control.setdefault("feasible", "yes")
    control.setdefault("rewrite", "")
    control.setdefault("scene", {})
    control.setdefault("reveal_policy", {})
    control.setdefault("plan_steps", [])
    control.setdefault("state_patch", {})
    control.setdefault("redlines", [])
    control.setdefault("choices", [])

    new_mem = apply_patch(memory, control.get("state_patch") or {}, mode)
    choices = control.get("choices") or []
    return control, new_mem, choices

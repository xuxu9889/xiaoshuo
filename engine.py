# engine.py —— 小说模式：主角/旁观 + 证据约束 + 严格字数生成（非裁剪）+ 自然段换行 + 单场景连贯
import json, re
from llm_io import chat_json
from store import search_chunks, search_facts

STRICT_MODE = True           # 开/关 正典严格模式
MIN_CITATIONS = 1            # 至少需要的引用数
MAX_RETRIES = 2              # 引用不达标时的重试次数

# ======================
# 工具函数
# ======================

def load(path: str) -> str:
    return open(path, "r", encoding="utf-8").read()

def _extract_topic(user_input: str) -> str:
    """粗粒度主题提取：抽取前几个连续汉字/字母数字串作为检索hint"""
    kws = re.findall(r"[一-龥A-Za-z0-9]{2,}", user_input)
    return " ".join(kws[:6])

def visible_len(s: str) -> int:
    """统计正文字数：计入可见字符与标点，不计换行"""
    return len(s.replace("\n", ""))

def end_with_sentence(text: str) -> str:
    """若末尾不是句号/问号/叹号，则补充句号，避免半句"""
    if text.endswith(("。","！","？","!","?")):
        return text
    return text.rstrip("…").rstrip() + "。"

def format_paragraphs(text: str, max_len=56) -> str:
    """
    自然段排版：
    1) 先按句号/问号/叹号切句并保留标点
    2) 长句再按逗号/分号细分
    3) 聚合到指定长度附近，输出真实换行
    """
    parts = re.split(r'([。！？!?])', text)
    sents = ["".join(x) for x in zip(parts[0::2], parts[1::2] + [""])]
    expanded = []
    for s in sents:
        if len(s) <= max_len:
            expanded.append(s)
        else:
            comma = re.split(r'([，,；;])', s)
            segs = ["".join(x) for x in zip(comma[0::2], comma[1::2] + [""])]
            expanded.extend(segs)

    paras, buf = [], ""
    for s in expanded:
        s = s.strip()
        if not s:
            continue
        if len(buf) + len(s) <= max_len:
            buf += s
        else:
            if buf:
                paras.append(buf)
            buf = s
    if buf:
        paras.append(buf)
    return "\n\n".join(p.strip() for p in paras if p.strip()) or text

def citations_valid(out: dict, valid_ids: set) -> bool:
    """校验模型输出的 citations 是否满足数量且来自允许的ID集合"""
    cits = out.get("citations") or []
    if len(cits) < MIN_CITATIONS:
        return False
    return all(ci.get("id") in valid_ids for ci in cits)

def enforce_length_via_llm(raw_reply: str, reply_len: int, narration_mode: str) -> str:
    """
    严格字数控制（不是裁剪）：
    若首轮生成偏离目标字数，则调用一次小修正回写，让模型重写为“恰好N字（±3）”；
    最多重试2次，保持原意与证据，不新增设定。
    """
    # 已接近目标（±3字）则直接收束
    if abs(visible_len(raw_reply) - reply_len) <= 3:
        return end_with_sentence(raw_reply)

    fix_prompt = {
        "instruction": (
            f"请将下面的正文改写为恰好 {reply_len} 字（允许±3字），计数不含换行。"
            f"保持原意与叙事风格，{narration_mode}；不得新增世界设定/专有名词；"
            "必须分成2~6个自然段；结尾用完整句子，不用省略号。"
            "只返回JSON：{\"reply\":\"...\"}"
        ),
        "text": raw_reply
    }

    tries, reply = 0, raw_reply
    while tries < 2:
        out = chat_json("只返回JSON", json.dumps(fix_prompt, ensure_ascii=False))
        candidate = out.get("reply") or reply
        if abs(visible_len(candidate) - reply_len) <= 3:
            return end_with_sentence(candidate)
        reply = candidate
        tries += 1
    return end_with_sentence(reply)

# ======================
# 检索与提示构建
# ======================

def retrieve(user_input: str, memory: dict, k_chunks=6, k_facts=6):
    """基于用户输入 + 记忆增强的查询，返回事实与原文片段"""
    topic = _extract_topic(user_input)
    q = user_input + (f"；主题：{topic}" if topic else "")
    if memory:
        if memory.get("location"):
            q += f"；地点：{memory['location']}"
        if memory.get("scene"):
            q += f"；当前场景：{memory['scene']}"
        if memory.get("flags"):
            q += f"；状态：{json.dumps(memory['flags'], ensure_ascii=False)}"
    facts = search_facts(q, k=k_facts)
    chunks = search_chunks(q, k=k_chunks)
    return facts, chunks

def build_prompt(facts, chunks, memory, user_input, reply_len: int):
    """把证据、记忆、输入与参数填充进模板"""
    tpl = load("prompts/story_strict.txt")
    allowed_ids = [x["id"] for x in facts] + [x["id"] for x in chunks]
    mode = memory.get("narration_mode", "主角")
    mode_desc = "主角模式" if mode == "主角" else "旁观模式"
    return (tpl
            .replace("{{TOP_FACTS_JSON}}", json.dumps(facts, ensure_ascii=False))
            .replace("{{TOP_CHUNKS_JSON}}", json.dumps(chunks, ensure_ascii=False))
            .replace("{{MEMORY_JSON}}", json.dumps(memory, ensure_ascii=False))
            .replace("{{USER_INPUT}}", user_input)
            .replace("{{REPLY_LEN}}", str(reply_len))
            .replace("{{ALLOWED_IDS}}", json.dumps(allowed_ids, ensure_ascii=False))
            .replace("{{NARRATION_MODE}}", mode_desc))

# ======================
# 核心交互
# ======================

def story_turn(user_input: str, memory: dict, reply_len: int = 150):
    """
    核心：基于检索证据生成“小说风”叙事。
    - 主角/旁观模式
    - 单场景连贯（无证据不随意转场）
    - 严格字数（非裁剪）
    - 自然段排版
    """
    # 初始化记忆
    if "scene" not in memory:
        memory["scene"] = "主城"
    if "location" not in memory:
        memory["location"] = "主城"
    if "narration_mode" not in memory:
        memory["narration_mode"] = "主角"  # "主角" 或 "旁观"

    # 证据检索
    facts, chunks = retrieve(user_input, memory)

    # 证据极少时：柔性兜底（按模式给自然抉择，不说“信息不足”）
    if STRICT_MODE and len(facts) + len(chunks) == 0:
        if memory["narration_mode"] == "主角":
            reply = (
                "光影在视野边缘一闪而过，你压住呼吸，把注意力拉回脚下。"
                "也许先打量界面、问一声路，或走向告示更稳妥。"
            )
            choices = ["查看界面与状态", "向最近的NPC搭话", "前往告示板"]
        else:  # 旁观
            reply = (
                "视野像被一层薄雾抚过，信息的边缘忽明忽暗。"
                "不妨贴近人群的流线，或把焦点落在告示与门楣。"
            )
            choices = ["贴近人群观察", "聚焦告示与门楣", "远景扫过广场"]
        # 这里需要也满足严格字数
        mode_desc = "主角模式" if memory["narration_mode"] == "主角" else "旁观模式"
        reply = enforce_length_via_llm(reply, reply_len, mode_desc)
        reply = format_paragraphs(reply, max_len=56)
        return {"reply": reply, "choices": choices, "citations": []}, memory

    # 构建提示并生成
    prompt = build_prompt(facts, chunks, memory, user_input, reply_len)
    valid_ids = set([x["id"] for x in facts] + [x["id"] for x in chunks])

    out = chat_json("只返回JSON", prompt)

    # 引用校验失败时，带约束重试
    attempt = 0
    while STRICT_MODE and attempt < MAX_RETRIES and not citations_valid(out, valid_ids):
        fix_req = {
            "require": {
                "min_citations": MIN_CITATIONS,
                "valid_ids": list(valid_ids),
                "reply_len": reply_len
            },
            "last_output": out
        }
        fix_prompt = prompt + "\n\n[纠错约束]\n" + json.dumps(fix_req, ensure_ascii=False)
        out = chat_json("只返回JSON", fix_prompt)
        attempt += 1

    # 清洗无效引用
    if out.get("citations"):
        out["citations"] = [ci for ci in out["citations"] if ci.get("id") in valid_ids]

    # —— 严格字数生成（非裁剪）+ 自然段排版 ——
    if "reply" in out and isinstance(out["reply"], str):
        mode = memory.get("narration_mode", "主角")
        mode_desc = "主角模式" if mode == "主角" else "旁观模式"
        text = enforce_length_via_llm(out["reply"], reply_len, mode_desc)
        text = format_paragraphs(text, max_len=56)  # 换行不计入字数
        out["reply"] = text

    # 单场景连贯：除非有引用支撑，否则不改 scene/location
    mu = out.get("memory_update") or {}
    if isinstance(mu, dict):
        if (mu.get("scene") and mu["scene"] != memory.get("scene")) and not out.get("citations"):
            mu.pop("scene", None)
        if (mu.get("location") and mu["location"] != memory.get("location")) and not out.get("citations"):
            mu.pop("location", None)
        memory.update(mu)

    return out, memory

# 调试（/why）
def debug_retrieve(user_input: str, memory: dict, k_chunks=3, k_facts=3):
    return retrieve(user_input, memory, k_chunks, k_facts)

# retrieval.py
import math
from typing import List, Dict, Tuple
from llm_io import embed  # 已封装好的 OpenAI Embeddings

def _cos(a: List[float], b: List[float]) -> float:
    ab = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a)) or 1e-9
    nb = math.sqrt(sum(x*x for x in b)) or 1e-9
    return ab / (na*nb)

def build_fact_index(facts_rows: List[Tuple[str,str,str,str]]) -> Dict:
    """
    输入：[(subject, predicate, object, summary), ...]
    输出：{"items":[{...}], "emb":[...]} （内存索引）
    """
    items = []
    texts = []
    for i,(s,p,o,sm) in enumerate(facts_rows):
        s = (s or "").strip(); p=(p or "").strip(); o=(o or "").strip().strip('"').strip("“”"); sm=(sm or "").strip()
        txt = f"{s} {p} {o} {sm}"
        items.append({"id": i, "s": s, "p": p, "o": o, "sm": sm, "full": txt})
        texts.append(txt)
    vecs = embed(texts)  # 一次性向量化
    return {"items": items, "emb": vecs}

def retrieve(index: Dict, query: str, k: int = 12, role_name: str = "", location_hint: str = "") -> List[Dict]:
    """
    余弦相似度 Top-K；对“当前角色名/地点”给予轻微加权
    """
    qv = embed([query or ""])[0]
    scored = []
    for item, v in zip(index["items"], index["emb"]):
        score = _cos(qv, v)
        if role_name and role_name in item["s"]:
            score += 0.05
        if location_hint and location_hint in item["full"]:
            score += 0.03
        scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [it for _, it in scored[:k]]

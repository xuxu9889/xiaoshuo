# build_plot.py
import os, sqlite3, json, re
from pathlib import Path
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from llm_io import chat

load_dotenv()
DB_PATH = os.getenv("DB_PATH", "./db/canon.sqlite")

TEMPLATE_SCAFFOLD = [
    # 这是一个“兜底脚手架”，如果LLM无法解析facts，也有最小主线
    {
        "id":"intro_room",
        "title":"现实：决定回坑",
        "summary":"现实世界的房间/矛盾/动机确认；决定重新踏入《机甲志》。",
        "entry_hint":"开局默认进入。",
        "exit_hint":"玩家选择或表达登录/进入游戏。",
        "fact_tags":"刘浪,无敌游戏,情绪,家庭",
        "required_flags":"",
        "set_flags":""
    },
    {
        "id":"login",
        "title":"登录大厅：回归一线",
        "summary":"进入《机甲志》登录/大厅；可检查装备、接收私聊、浏览市场。",
        "entry_hint":"从intro_room进入或玩家表达登录/进入。",
        "exit_hint":"前往新手村/领取武器/接私聊触发剧情深入。",
        "fact_tags":"机甲志,登录,大厅,装备,私聊,市场",
        "required_flags":"",
        "set_flags":"logged_in"
    },
    {
        "id":"newbie",
        "title":"新手起步：武器与练级",
        "summary":"新手村领取初始武器；打箭猪、攒资金、购买营养仓。",
        "entry_hint":"从login进入或玩家表达去新手村/领取武器。",
        "exit_hint":"角色等级达到10级或玩家明确提出转职。",
        "fact_tags":"新手村,初始武器,箭猪,营养仓,铜币",
        "required_flags":"logged_in",
        "set_flags":"at_newbie_village"
    },
    {
        "id":"job",
        "title":"转职：选择职业",
        "summary":"达到10级后进行转职，选择职业：魔法师/战士/弓箭手/骑士/医生。",
        "entry_hint":"角色达到10级或玩家提出转职。",
        "exit_hint":"完成转职并离开新手阶段。",
        "fact_tags":"转职,职业,10级",
        "required_flags":"at_newbie_village",
        "set_flags":"job_done"
    },
    {
        "id":"postjob",
        "title":"转职后：新的目标",
        "summary":"转职完成后，进入更复杂剧情：装备、交易、对抗、组织、强敌等。",
        "entry_hint":"完成转职。",
        "exit_hint":"后续由更细的分支补充。",
        "fact_tags":"装备,交易,阵营,强敌,任务",
        "required_flags":"job_done",
        "set_flags":""
    }
]

TEMPLATE_EDGES = [
    {"src":"intro_room", "dst":"login",   "condition":"玩家明确登录/进入游戏", "keywords":"登录,进入游戏,进入机甲志,上线"},
    {"src":"login",      "dst":"newbie",  "condition":"前往新手村/领取初始武器/开始练级", "keywords":"新手村,领取武器,初始武器,练级"},
    {"src":"newbie",     "dst":"job",     "condition":"达到10级或提出转职", "keywords":"10级,转职"},
    {"src":"job",        "dst":"postjob", "condition":"完成转职", "keywords":"完成转职,已转职"}
]

PROMPT = """你是剧情策划。给出基于以下事实的“剧情节点图”（JSON）：
要求：
- 节点按“现实开局→登录→新手→转职→转职后”的主线组织
- 每个节点包含：id、title、summary、entry_hint、exit_hint、fact_tags（逗号分隔）、required_flags（逗号分隔或空）、set_flags（逗号分隔或空）
- 边包含：src、dst、condition（自然语言）、keywords（逗号分隔；触发推进的关键词）
- 只允许使用事实中已有设定，不要编造新设定
只输出一个JSON对象：{"nodes":[...], "edges":[...]}

事实样例：
"""

def fetch_facts(conn) -> List[Tuple[str,str,str,str]]:
    return conn.execute("SELECT subject, predicate, object, summary FROM facts").fetchall()

def llm_build(nodes_scaffold, edges_scaffold, facts_rows) -> Dict:
    # 取一部分事实作为上下文（限制长度）
    snippets = []
    for s,p,o,sm in facts_rows[:120]:
        o_clean = str(o or "").strip().strip('"').strip("“”")
        snippets.append(f"{o_clean} | 摘要:{sm}")
    context = "\n".join(snippets)

    sys = "你是严谨的游戏剧情策划，擅长把事实梳理成清晰的节点/边。"
    user = PROMPT + context + "\n\n如果事实不足，可复用我给的脚手架，但不要违背事实。"

    try:
        txt = chat([{"role":"system","content":sys},{"role":"user","content":user}], max_tokens=1500, temperature=0.2)
        data = json.loads(txt)  # 格式合法则直接用
        return data
    except Exception:
        # 回退：用脚手架
        return {"nodes": nodes_scaffold, "edges": edges_scaffold}

def upsert_plot(conn, nodes: List[Dict], edges: List[Dict]):
    c = conn.cursor()
    for n in nodes:
        c.execute("""
        INSERT OR REPLACE INTO plot_nodes(id,title,summary,entry_hint,exit_hint,fact_tags,required_flags,set_flags)
        VALUES (?,?,?,?,?,?,?,?)""", (
            n["id"], n["title"], n["summary"], n.get("entry_hint",""),
            n.get("exit_hint",""), n.get("fact_tags",""), n.get("required_flags",""),
            n.get("set_flags","")
        ))
    for e in edges:
        c.execute("""
        INSERT OR REPLACE INTO plot_edges(src,dst,condition,keywords)
        VALUES (?,?,?,?)""", (
            e["src"], e["dst"], e.get("condition",""), e.get("keywords","")
        ))
    # 初始化状态（如果没有）
    has = c.execute("SELECT 1 FROM plot_state WHERE save_id='default'").fetchone()
    if not has:
        c.execute("INSERT INTO plot_state(save_id,current_node) VALUES('default','intro_room')")
    conn.commit()

def main():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        facts = fetch_facts(conn)
        data = llm_build(TEMPLATE_SCAFFOLD, TEMPLATE_EDGES, facts)
        nodes = data.get("nodes") or TEMPLATE_SCAFFOLD
        edges = data.get("edges") or TEMPLATE_EDGES
        upsert_plot(conn, nodes, edges)
        print(f"✅ 剧情表已生成/更新：nodes={len(nodes)} edges={len(edges)}  DB={DB_PATH}")

if __name__ == "__main__":
    main()

# cli.py
import os
from pathlib import Path
import sqlite3
import re
import typer

app = typer.Typer(help="Xiaoshuo CLI - 初始化数据库 / 沉浸式互动（剧情表驱动）")


# =========================
# 工具函数
# =========================

def _exec_sql_file(db_path: str, sql_path: str):
    if not Path(sql_path).exists():
        return False
    with open(sql_path, "r", encoding="utf-8") as f:
        sql = f.read()
    with sqlite3.connect(db_path) as conn:
        conn.executescript(sql)
        conn.commit()
    return True


# =========================
# 命令：初始化数据库
# =========================
@app.command("init-db")
def init_db(
        schema: str = typer.Option("schema.sql", help="主 schema 文件（如存在则执行）"),
        plot_schema: str = typer.Option("schema_plot.sql", help="剧情表 schema 文件（如存在则执行）"),
        db: str = typer.Option(None, help="数据库路径（优先于环境变量 DB_PATH）"),
):
    db_path = db or os.getenv("DB_PATH", "./db/canon.sqlite")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    ok_main = _exec_sql_file(db_path, schema)
    ok_plot = _exec_sql_file(db_path, plot_schema)

    print(f"📦 正在初始化数据库: {db_path}")
    if ok_main:
        print(f"✅ 已执行 {schema}")
    else:
        print(f"ℹ️ 未找到 {schema}（跳过）")
    if ok_plot:
        print(f"✅ 已执行 {plot_schema}")
    else:
        print(f"ℹ️ 未找到 {plot_schema}（跳过）")
    print("✨ 完成。")


# =========================
# 角色选择（可选）
# =========================
def build_person_candidates(facts_rows):
    """基于 facts 粗略抽取可扮演人物清单。"""
    PERSON_HINTS = ("身份", "职业", "性格", "称号", "昵称", "关系", "阵营", "玩家", "NPC")
    blacklist = {"游戏", "系统", "新手村", "机甲志", "无敌游戏", "怪兽", "规则", "任务", "铜币"}
    persons = {}
    for s, p, o, sm in facts_rows:
        s = (s or "").strip()
        p = (p or "").strip()
        o = (o or "").strip()
        if s in blacklist:
            continue
        if any(h in p for h in PERSON_HINTS):
            # 修复 f-string 语法问题：先清洗字段，再填入 f-string
            clean_o = o.strip().strip('"').strip('“”')
            persons.setdefault(s, {
                "name": s,
                "identity": f"{p}：{clean_o}"[:40]
            })
    return persons



def choose_role_interactively(persons_dict):
    """交互式角色选择界面。"""
    person_list = sorted(persons_dict.values(), key=lambda d: d["name"])

    def pick_existing_person():
        if not person_list:
            print("（未检测到现成人物条目）")
            return None
        print("\n可扮演人物（现有）：")
        for i, p in enumerate(person_list, 1):
            brief = p.get("identity", "") or "（暂无人设摘要）"
            print(f"{i}）{p['name']} - {brief}")
        sel = input("输入编号选择：").strip()
        if not sel.isdigit():
            return None
        idx = int(sel)
        if 1 <= idx <= len(person_list):
            return person_list[idx - 1]
        return None

    print("\n=== 角色选择 ===")
    print("1）主角：刘浪（若存在）")
    print("2）剧情内现有人物（从数据库抓取）")
    print("3）旁观者角色（叙事视角）")
    print("4）自定义角色（输入人设）")

    while True:
        opt = input("请输入 1/2/3/4：").strip()
        if opt == "1":
            for p in person_list:
                if p["name"] == "刘浪":
                    return {"name": "刘浪",
                            "identity": p.get("identity", "主角"),
                            "motive": "东山再起",
                            "goal": "在《机甲志》中重夺荣光",
                            "style": "果决锋利"}
            print("⚠ 未找到“刘浪”，请改选 2/3/4")
        elif opt == "2":
            picked = pick_existing_person()
            if picked:
                return {
                    "name": picked["name"],
                    "identity": picked.get("identity", ""),
                    "motive": "达成私欲/事业",
                    "goal": "推进当前局势",
                    "style": ""
                }
        elif opt == "3":
            return {"name": "旁观者", "identity": "叙事视角/记录者", "motive": "观察并推动剧情", "goal": "记录关键节点", "style": "冷静克制"}
        elif opt == "4":
            nm = input("角色名：").strip() or "无名者"
            idt = input("身份（一句话）：").strip()
            mtv = input("动机（一句话）：").strip()
            gol = input("近期目标（一句话）：").strip()
            sty = input("表达风格（可留空）：").strip()
            return {"name": nm, "identity": idt, "motive": mtv, "goal": gol, "style": sty}
        else:
            print("无效选项，请重新输入。")


# =========================
# 命令：沉浸式小说（剧情表驱动）
# =========================
@app.command("interactive")
def interactive(
        length: int = typer.Option(200, help="单段目标字数：200/500/1000")
):
    import sqlite3
    from state import GameState
    from engine import Engine

    db_path = os.getenv("DB_PATH", "./db/canon.sqlite")

    # ---------- 加载 facts ----------
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    rows = cur.execute("SELECT subject, predicate, object, summary FROM facts").fetchall()

    # ---------- 构建检索器（优先向量检索；无 retrieval.py 时回退关键词检索） ----------
    try:
        from retrieval import build_fact_index, retrieve
        index = build_fact_index(rows)

        def select_facts(query, role_name, location_hint, k=12):
            return retrieve(index, query, k=k, role_name=role_name, location_hint=location_hint)
    except Exception:
        # 回退：极简关键词检索
        facts = []
        for s, p, o, sm in rows:
            s = (s or "").strip()
            p = (p or "").strip()
            o = (o or "").strip().strip('"').strip("“”")
            sm = (sm or "").strip()
            facts.append({"s": s, "p": p, "o": o, "sm": sm, "full": f"{s} {p} {o} {sm}".lower()})

        def select_facts(query, role_name, location_hint, k=12):
            q = (query or "").lower()
            scored = []
            terms = [t for t in re.split(r"\s+", q) if t]
            for f in facts:
                score = sum(f["full"].count(t) for t in terms)
                if role_name and role_name in f["s"]:
                    score += 2
                if score > 0:
                    scored.append((score, f))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [f for _, f in scored[:k]] or facts[:k]

    # ---------- 加载剧情表 ----------
    def load_plot(cnx):
        nodes = {r[0]: dict(id=r[0], title=r[1], summary=r[2], entry_hint=r[3],
                            exit_hint=r[4], fact_tags=r[5] or "", required_flags=r[6] or "",
                            set_flags=r[7] or "")
                 for r in cnx.execute("SELECT id,title,summary,entry_hint,exit_hint,fact_tags,required_flags,set_flags FROM plot_nodes")}
        edges = {}
        for src, dst, cond, kws in cnx.execute("SELECT src,dst,condition,keywords FROM plot_edges"):
            edges.setdefault(src, []).append(dict(dst=dst, condition=cond or "", keywords=kws or ""))
        current = GameState.get_current_node(db_path=db_path, save_id="default")
        return nodes, edges, current

    nodes, edges, current_node = load_plot(conn)

    # ---------- 角色选择（一次性） ----------
    persons = build_person_candidates(rows)
    role = choose_role_interactively(persons)

    # ---------- 引擎与持久化状态 ----------
    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
    eng = Engine(system_prompt="", model=model)
    eng.set_length(length)
    eng.set_role(role)

    state = GameState.load(db_path=db_path, save_id="default")
    state.role_name = eng.role.get("name", state.role_name)

    print(f"\n=== 沉浸式小说（剧情表驱动） ===")
    print(f"模型：{model} | 事实：{len(rows)} | 段长：{length} | 剧情节点：{current_node} | 角色：{state.role_name}")
    print("命令：/len N｜/state｜/save｜/load｜/reset｜/roles（重开选角）｜/exit")
    print("提示：输入行动/意图（如：进入新手村、领取武器、购买营养仓、练级到10级、转职）。")

    # ---------- 剧情推进辅助 ----------
    def allowed_keywords_for(node_id: str):
        kws = []
        for e in edges.get(node_id, []):
            if e["keywords"]:
                kws += [k.strip() for k in e["keywords"].split(",") if k.strip()]
        return list(dict.fromkeys(kws))

    def next_node_if_any(node_id: str, user_text: str):
        t = (user_text or "").replace(" ", "")
        for e in edges.get(node_id, []):
            kws = [k.strip() for k in (e["keywords"] or "").split(",") if k.strip()]
            if any(k and k in t for k in kws):
                return e["dst"]
        return None

    # ---------- 主循环 ----------
    while True:
        try:
            user = input("你> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 再见！")
            break

        # —— 系统命令 —— #
        if user.lower() in {"exit", "/exit", "quit"}:
            print("退出"); break

        if user.startswith("/len"):
            try:
                new_len = int(user.split(maxsplit=1)[1])
                if new_len not in (200, 500, 1000):
                    raise ValueError
                eng.set_length(new_len)
                print(f"段长已设为 {new_len}")
            except Exception:
                print("用法：/len 200|500|1000")
            continue

        if user.startswith("/state"):
            print(state.as_context())
            print(f"当前剧情节点：{current_node}")
            continue

        if user.startswith("/save"):
            state.save(db_path=db_path)
            print("已存档")
            continue

        if user.startswith("/load"):
            state = GameState.load(db_path=db_path, save_id="default")
            current_node = GameState.get_current_node(db_path=db_path, save_id="default")
            print("已读档")
            continue

        if user.startswith("/reset"):
            from engine import Engine as _E  # 重置引擎历史
            eng = _E(system_prompt="", model=model)
            eng.set_length(length)
            eng.set_role(role)
            state = GameState(save_id="default", role_name=eng.role.get("name", "无名者"))
            GameState.set_current_node("intro_room", db_path=db_path, save_id="default")
            nodes, edges, current_node = load_plot(conn)
            print("会话+剧情节点已重置为 intro_room")
            continue

        if user.startswith("/roles"):
            role = choose_role_interactively(persons)
            eng.set_role(role)
            state.role_name = role.get("name", state.role_name)
            print(f"已切换为：{state.role_name}")
            continue

        # —— 行动解析（示例：购买营养仓） —— #
        txt = user.replace(" ", "")
        event_line = ""
        if any(k in txt for k in ("购买营养仓", "买营养仓", "购入营养仓", "买个营养仓", "买下营养仓", "购置营养仓")):
            price = 80
            if state.has_item("营养仓"):
                event_line = "（你已拥有『营养仓』，无需重复购买）"
            elif state.spend(price):
                state.add_item("营养仓")
                state.set_flag("bought_nutri", True)
                event_line = f"（交易完成：花费{price}，获得『营养仓』）"
            else:
                event_line = f"（资金不足：{state.coins}/{price}）"

        # —— 当前剧情阶段信息 —— #
        node = nodes.get(current_node, {})
        node_tags = (node.get("fact_tags", "") or "").split(",")
        stage_block = (
            f"【剧情阶段】{node.get('id', '?')}｜{node.get('title', '')}\n"
            f"阶段说明：{node.get('summary', '')}\n"
            f"进入提示：{node.get('entry_hint', '')}\n"
            f"推进提示：{node.get('exit_hint', '')}\n"
            f"允许推进关键词：{ '、'.join(allowed_keywords_for(current_node)) or '（无）' }"
        )

        # —— 检索（节点标签增强查询） —— #
        query = (user or "") + " " + " ".join(node_tags)
        selected = select_facts(query, role_name=eng.role.get("name", ""), location_hint=state.location, k=12)
        world_lines = []
        for it in selected:
            s = it.get("s", it["s"])
            p = it.get("p", it["p"])
            o = it.get("o", it["o"])
            sm = it.get("sm", it["sm"])
            world_lines.append(f"{o}\n摘要：{sm}")
        world_block = "\n\n".join(world_lines)

        # —— 注入系统提示并生成 —— #
        state_block = state.as_context()
        if event_line:
            state_block += f"\n【本轮事件】{event_line}"

        eng.set_system_prompt(
            state_block + "\n" + stage_block + "\n\n" +
            "【相关世界观/事实（检索）】\n" + world_block
        )

        out = eng.narrate(user)
        print(out)

        # —— 判定是否推进节点（只按出边关键词） —— #
        dst = next_node_if_any(current_node, user)
        if dst and dst in nodes:
            GameState.set_current_node(dst, db_path=db_path, save_id="default")
            current_node = dst


if __name__ == "__main__":
    app()

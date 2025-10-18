# -*- coding: utf-8 -*-
"""
Xiaoshuo CLI
- 初始化数据库（init-db）
- 沉浸式互动（interactive） —— 兼容无 facts 表
- 查看 数据：list-facts / list-nodes / list-edges —— 均支持 --db 指定数据库
"""

import os
import re
import sqlite3
from pathlib import Path
import typer

app = typer.Typer(help="Xiaoshuo CLI - 初始化数据库 / 沉浸式互动 / 数据查看（剧情表+facts）")


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


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.cursor()
    row = cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
        (name,)
    ).fetchone()
    return row is not None


def _print_table(rows, headers):
    """
    简易对齐表格打印。rows: List[tuple/ list / dict], headers: List[str]
    - 支持 dict 行（按 headers 取值）
    """
    # 统一为 list[list]
    norm = []
    for r in rows:
        if isinstance(r, dict):
            norm.append([str(r.get(h, "")) for h in headers])
        else:
            norm.append([("" if v is None else str(v)) for v in r])

    widths = [len(h) for h in headers]
    for row in norm:
        for i, cell in enumerate(row):
            if len(cell) > widths[i]:
                widths[i] = len(cell)

    def fmt_row(vals):
        return "  ".join(v.ljust(widths[i]) for i, v in enumerate(vals))

    # 头
    print(fmt_row(headers))
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    # 行
    for row in norm:
        print(fmt_row(row))


# =========================
# 命令：初始化数据库
# =========================
@app.command("init-db")
def init_db(
        schema: str = typer.Option("schema.sql", help="主 schema 文件（如存在则执行）"),
        plot_schema: str = typer.Option("schema_plot.sql", help="剧情表 schema 文件（如存在则执行）"),
        db: str = typer.Option(None, "--db", help="数据库路径（优先于环境变量 DB_PATH）"),
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
# 角色选择（尽量通用）
# =========================

def build_person_candidates(facts_rows):
    """基于 facts 粗略抽取可扮演的“实体名”。没有 facts 时返回空。"""
    PERSON_HINTS = ("身份", "职业", "性格", "称号", "昵称", "关系", "阵营", "玩家", "NPC")
    blacklist = {"游戏", "系统", "新手村", "机甲志", "无敌游戏", "怪兽", "规则", "任务", "铜币"}
    persons = {}
    for s, p, o, sm in facts_rows:
        s = (s or "").strip()
        p = (p or "").strip()
        o = (o or "").strip()
        if not s or s in blacklist:
            continue
        if any(h in p for h in PERSON_HINTS):
            clean_o = o.strip().strip('"').strip('“”')
            persons.setdefault(s, {
                "name": s,
                "identity": f"{p}：{clean_o}"[:40]
            })
    return persons


def choose_role_interactively(persons_dict):
    """交互式角色选择。无 facts 时提供通用选项。"""
    person_list = sorted(persons_dict.values(), key=lambda d: d["name"]) if persons_dict else []

    print("\n=== 角色选择（通用） ===")
    print("1）旁观者（叙事视角）")
    if person_list:
        print("2）从抽取的实体中选择")
    print("3）自定义角色（输入人设）")

    while True:
        opt = input("请输入 1/2/3：").strip()
        if opt == "1":
            return {"name": "旁观者", "identity": "叙事视角/记录者", "motive": "观察与推进", "goal": "记录关键节点", "style": "冷静克制"}
        elif opt == "2" and person_list:
            print("\n检测到可扮演的实体：")
            for i, p in enumerate(person_list, 1):
                brief = p.get("identity", "") or "（暂无人设摘要）"
                print(f"{i}）{p['name']}  （从事实中自动抽取）")
            sel = input("输入编号选择：").strip()
            if sel.isdigit():
                idx = int(sel)
                if 1 <= idx <= len(person_list):
                    picked = person_list[idx - 1]
                    return {
                        "name": picked["name"],
                        "identity": picked.get("identity", ""),
                        "motive": "达成私欲/事业",
                        "goal": "推进当前局势",
                        "style": ""
                    }
            print("无效选择，请重试。")
        elif opt == "3":
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
        length: int = typer.Option(200, help="单段目标字数：200/500/1000"),
        db: str = typer.Option(None, "--db", help="数据库路径（默认读取环境变量 DB_PATH 或 ./db/canon.sqlite）"),
):
    from state import GameState
    from engine import Engine

    db_path = db or os.getenv("DB_PATH", "./db/canon.sqlite")

    # 加载 facts（可选）
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    facts_rows = []
    if _table_exists(conn, "facts"):
        try:
            facts_rows = cur.execute("SELECT subject, predicate, object, summary FROM facts").fetchall()
        except Exception:
            facts_rows = []

    # 构建简单检索器（无 retrieval.py 时回退关键词检索）
    def build_keyword_index(rows):
        facts = []
        for s, p, o, sm in rows:
            s = (s or "").strip()
            p = (p or "").strip()
            o = (o or "").strip().strip('"').strip("“”")
            sm = (sm or "").strip()
            facts.append({"s": s, "p": p, "o": o, "sm": sm, "full": f"{s} {p} {o} {sm}".lower()})
        return facts

    try:
        from retrieval import build_fact_index, retrieve
        index = build_fact_index(facts_rows)

        def select_facts(query, role_name, location_hint, k=12):
            return retrieve(index, query, k=k, role_name=role_name, location_hint=location_hint)
    except Exception:
        facts_index = build_keyword_index(facts_rows)

        def select_facts(query, role_name, location_hint, k=12):
            if not facts_index:
                return []
            q = (query or "").lower()
            scored = []
            terms = [t for t in re.split(r"\s+", q) if t]
            for f in facts_index:
                score = sum(f["full"].count(t) for t in terms)
                if role_name and role_name in f["s"]:
                    score += 2
                if score > 0:
                    scored.append((score, f))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [f for _, f in scored[:k]] or facts_index[:k]

    # 加载剧情表
    def load_plot(cnx):
        nodes = {r[0]: dict(id=r[0], title=r[1], summary=r[2], entry_hint=r[3],
                            exit_hint=r[4], fact_tags=r[5] or "", required_flags=r[6] or "",
                            set_flags=r[7] or "")
                 for r in cnx.execute("SELECT id,title,summary,entry_hint,exit_hint,fact_tags,required_flags,set_flags FROM plot_nodes")}
        edges = {}
        for src, dst, cond, kws in cnx.execute("SELECT src,dst,condition,keywords FROM plot_edges"):
            edges.setdefault(src, []).append(dict(dst=dst, condition=cond or "", keywords=kws or ""))
        current = GameState.get_current_node(db_path=db_path, save_id="default")
        if current not in nodes and nodes:
            current = list(nodes.keys())[0]
        return nodes, edges, current

    nodes, edges, current_node = load_plot(conn)

    # 选角色
    persons = build_person_candidates(facts_rows)
    role = choose_role_interactively(persons)

    # 引擎 + 状态
    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
    eng = Engine(system_prompt="", model=model)
    eng.set_length(length)
    eng.set_role(role)

    state = GameState.load(db_path=db_path, save_id="default")
    state.role_name = eng.role.get("name", state.role_name)

    print(f"\n=== 沉浸式小说（通用剧情表驱动） ===")
    print(f"模型：{model} | facts：{len(facts_rows)} | 段长：{length} | 当前节点：{current_node} | 角色：{state.role_name}")
    print("命令：/len N｜/state｜/save｜/load｜/reset｜/roles｜/list｜/next｜/goto ID｜/exit")
    print("提示：任意自然语言输入用于叙述；含“下一段/继续/next/推进”可尝试沿边推进。")

    def allowed_keywords_for(node_id: str):
        kws = []
        for e in edges.get(node_id, []):
            if e["keywords"]:
                kws += [k.strip() for k in e["keywords"].split(",") if k.strip()]
        # 去重保序
        return list(dict.fromkeys(kws))

    def next_node_if_any(node_id: str, user_text: str):
        t = (user_text or "").replace(" ", "")
        for e in edges.get(node_id, []):
            kws = [k.strip() for k in (e["keywords"] or "").split(",") if k.strip()]
            if any(k and k in t for k in kws) or t in {"next", "继续", "推进", "下一段"}:
                return e["dst"]
        return None

    while True:
        try:
            user = input("你> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 再见！")
            break

        # 系统命令
        if user.lower() in {"exit", "/exit", "quit"}:
            print("👋 再见！"); break

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
            # 读回当前节点
            try:
                from state import GameState as _GS
                current_node = _GS.get_current_node(db_path=db_path, save_id="default")
            except Exception:
                pass
            print("已读档")
            continue

        if user.startswith("/reset"):
            from engine import Engine as _E
            eng = _E(system_prompt="", model=model)
            eng.set_length(length)
            eng.set_role(role)
            state = GameState(save_id="default", role_name=eng.role.get("name", "无名者"))
            # 重置到首节点
            if nodes:
                first = sorted(nodes.keys())[0]
            else:
                first = "intro_room"
            try:
                from state import GameState as _GS
                _GS.set_current_node(first, db_path=db_path, save_id="default")
            except Exception:
                pass
            nodes, edges, current_node = load_plot(conn)
            print(f"会话+剧情节点已重置为 {current_node}")
            continue

        if user.startswith("/roles"):
            role = choose_role_interactively(persons)
            eng.set_role(role)
            state.role_name = role.get("name", state.role_name)
            print(f"已切换为：{state.role_name}")
            continue

        if user.startswith("/list"):
            if current_node in nodes:
                print("\n可推进关键词：", "、".join(allowed_keywords_for(current_node)) or "（无）")
                outs = edges.get(current_node, [])
                if outs:
                    print("出边：")
                    for i, e in enumerate(outs, 1):
                        print(f"{i}） {current_node} -> {e['dst']}  | 条件:{e['condition'] or '-'} | 关键词:{e['keywords'] or '-'}")
                else:
                    print("（当前节点没有出边）")
            else:
                print("（当前节点不存在于剧情表）")
            continue

        if user.startswith("/next"):
            dst = next_node_if_any(current_node, "next")
            if dst and dst in nodes:
                try:
                    from state import GameState as _GS
                    _GS.set_current_node(dst, db_path=db_path, save_id="default")
                except Exception:
                    pass
                current_node = dst
                print(f"→ 已推进到 {current_node}")
            else:
                print("没有匹配的出边可推进。")
            continue

        if user.startswith("/goto"):
            parts = user.split(maxsplit=1)
            if len(parts) == 2:
                target = parts[1].strip()
                if target in nodes:
                    try:
                        from state import GameState as _GS
                        _GS.set_current_node(target, db_path=db_path, save_id="default")
                    except Exception:
                        pass
                    current_node = target
                    print(f"→ 跳转到 {current_node}")
                else:
                    print("目标节点不存在。")
            else:
                print("用法：/goto chXXX_YYY")
            continue

        # —— 常规叙述 —— #
        node = nodes.get(current_node, {})
        node_tags = (node.get("fact_tags", "") or "").split(",")

        stage_block = (
            f"【剧情阶段】{node.get('id', '?')}｜{node.get('title', '')}\n"
            f"阶段说明：{node.get('summary', '')}\n"
            f"进入提示：{node.get('entry_hint', '')}\n"
            f"推进提示：{node.get('exit_hint', '')}\n"
        )

        query = (user or "") + " " + " ".join(node_tags)
        selected = select_facts(query, role_name=eng.role.get("name", ""), location_hint=state.location, k=12)
        world_lines = []
        for it in selected:
            s = it.get("s") if isinstance(it, dict) else ""
            o = it.get("o") if isinstance(it, dict) else ""
            sm = it.get("sm") if isinstance(it, dict) else ""
            # 略化输出
            if s or o:
                world_lines.append(f"{s}：{o}（{sm}）")
        world_block = "\n".join(world_lines)

        state_block = state.as_context()
        eng.set_system_prompt(state_block + "\n" + stage_block + ("\n【相关事实】\n" + world_block if world_block else ""))

        out = eng.narrate(user)
        print(out)

        dst = next_node_if_any(current_node, user)
        if dst and dst in nodes:
            try:
                from state import GameState as _GS
                _GS.set_current_node(dst, db_path=db_path, save_id="default")
            except Exception:
                pass
            current_node = dst


# =========================
# 数据查看命令
# =========================

@app.command("list-facts")
def list_facts(
        db: str = typer.Option(None, "--db", help="数据库文件"),
        limit: int = typer.Option(20, "--limit", min=1, help="显示条数"),
        offset: int = typer.Option(0, "--offset", min=0, help="偏移量"),
        like: str = typer.Option("", "--like", help="按 subject/predicate/object 进行模糊匹配"),
):
    """查看 facts 表（如不存在则提示）。"""
    db_path = db or os.getenv("DB_PATH", "./db/canon.sqlite")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    if not _table_exists(conn, "facts"):
        print(f"ℹ️ 数据库 {db_path} 中不存在 facts 表。")
        conn.close()
        return

    where = ""
    params = []
    if like:
        where = "WHERE subject LIKE ? OR predicate LIKE ? OR object LIKE ?"
        kw = f"%{like}%"
        params = [kw, kw, kw]

    sql = f"""
    SELECT subject, predicate, object, COALESCE(summary, '')
    FROM facts
    {where}
    LIMIT ? OFFSET ?;
    """
    rows = cur.execute(sql, (*params, limit, offset)).fetchall()
    conn.close()

    if not rows:
        print("（无匹配数据）")
        return

    _print_table(rows, headers=["SUBJECT", "PREDICATE", "OBJECT", "SUMMARY"])


@app.command("list-nodes")
def list_nodes(
        db: str = typer.Option(None, "--db", help="数据库文件"),
        limit: int = typer.Option(20, "--limit", min=1, help="显示条数"),
        offset: int = typer.Option(0, "--offset", min=0, help="偏移量"),
        like: str = typer.Option("", "--like", help="按 id/title/summary 进行模糊匹配"),
):
    """查看剧情节点（plot_nodes）。"""
    db_path = db or os.getenv("DB_PATH", "./db/canon.sqlite")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    if not _table_exists(conn, "plot_nodes"):
        print(f"❌ 数据库 {db_path} 中不存在 plot_nodes 表。")
        conn.close()
        return

    where = ""
    params = []
    if like:
        where = "WHERE id LIKE ? OR title LIKE ? OR summary LIKE ?"
        kw = f"%{like}%"
        params = [kw, kw, kw]

    sql = f"""
    SELECT id, title, summary, COALESCE(fact_tags,'')
    FROM plot_nodes
    {where}
    ORDER BY id
    LIMIT ? OFFSET ?;
    """
    rows = cur.execute(sql, (*params, limit, offset)).fetchall()
    conn.close()

    if not rows:
        print("（无匹配数据）")
        return

    _print_table(rows, headers=["NODE_ID", "TITLE", "SUMMARY", "FACT_TAGS"])


@app.command("list-edges")
def list_edges(
        db: str = typer.Option(None, "--db", help="数据库文件"),
        limit: int = typer.Option(50, "--limit", min=1, help="显示条数"),
        offset: int = typer.Option(0, "--offset", min=0, help="偏移量"),
        src_like: str = typer.Option("", "--src-like", help="按 src 模糊匹配"),
):
    """查看剧情连接（plot_edges）。"""
    db_path = db or os.getenv("DB_PATH", "./db/canon.sqlite")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    if not _table_exists(conn, "plot_edges"):
        print(f"❌ 数据库 {db_path} 中不存在 plot_edges 表。")
        conn.close()
        return

    where = ""
    params = []
    if src_like:
        where = "WHERE src LIKE ?"
        params = [f"%{src_like}%"]

    sql = f"""
    SELECT src, dst, COALESCE(keywords,''), COALESCE(condition,'')
    FROM plot_edges
    {where}
    ORDER BY src, dst
    LIMIT ? OFFSET ?;
    """
    rows = cur.execute(sql, (*params, limit, offset)).fetchall()
    conn.close()

    if not rows:
        print("（无匹配数据）")
        return

    _print_table(rows, headers=["SRC", "DST", "KEYWORDS", "CONDITION"])


if __name__ == "__main__":
    app()

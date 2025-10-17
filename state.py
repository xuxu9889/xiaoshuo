# state.py
from __future__ import annotations
import json
import os
import sqlite3
from dataclasses import dataclass, asdict
from typing import Dict, Set

DEFAULT_DB = os.getenv("DB_PATH", "./db/canon.sqlite")
DEFAULT_SAVE_ID = "default"

# 额外状态表（持久化你的人物/金币/背包/旗标等）
# plot_state 表我们沿用你现有的：只存 current_node

CREATE_GAME_STATE_SQL = """
                        CREATE TABLE IF NOT EXISTS game_state (
                                                                  save_id       TEXT PRIMARY KEY,
                                                                  role_name     TEXT NOT NULL,
                                                                  location      TEXT NOT NULL,
                                                                  coins         INTEGER NOT NULL,
                                                                  inventory_json TEXT NOT NULL,   -- JSON list
                                                                  flags_json     TEXT NOT NULL    -- JSON object
                        ); \
                        """

@dataclass
class GameState:
    save_id: str = DEFAULT_SAVE_ID
    role_name: str = "无名者"
    location: str = "现实/出租屋"
    coins: int = 100
    inventory: Set[str] = None
    flags: Dict[str, bool] = None

    def __post_init__(self):
        if self.inventory is None:
            self.inventory = set()
        if self.flags is None:
            self.flags = {}

    # ---------- SQLite helpers ----------

    @staticmethod
    def _conn(db_path: str):
        return sqlite3.connect(db_path)

    @staticmethod
    def ensure_tables(db_path: str = DEFAULT_DB):
        with GameState._conn(db_path) as conn:
            conn.execute(CREATE_GAME_STATE_SQL)
            # plot_state 在 schema_plot.sql 已创建，这里不重复建
            conn.commit()

    @staticmethod
    def load(db_path: str = DEFAULT_DB, save_id: str = DEFAULT_SAVE_ID) -> "GameState":
        GameState.ensure_tables(db_path)
        with GameState._conn(db_path) as conn:
            cur = conn.cursor()
            row = cur.execute(
                "SELECT role_name, location, coins, inventory_json, flags_json FROM game_state WHERE save_id=?",
                (save_id,),
            ).fetchone()
            if row:
                role_name, location, coins, inv_json, flags_json = row
                inv = set(json.loads(inv_json)) if inv_json else set()
                flags = json.loads(flags_json) if flags_json else {}
                return GameState(
                    save_id=save_id,
                    role_name=role_name,
                    location=location,
                    coins=int(coins),
                    inventory=inv,
                    flags=flags,
                )
            # 若无记录，创建一条默认存档
            gs = GameState(save_id=save_id)
            gs.save(db_path)
            return gs

    def save(self, db_path: str = DEFAULT_DB):
        GameState.ensure_tables(db_path)
        with GameState._conn(db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO game_state(save_id, role_name, location, coins, inventory_json, flags_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    self.save_id,
                    self.role_name,
                    self.location,
                    int(self.coins),
                    json.dumps(sorted(list(self.inventory)), ensure_ascii=False),
                    json.dumps(self.flags, ensure_ascii=False),
                ),
            )
            conn.commit()

    # ---------- Plot helpers (读/写当前剧情节点) ----------

    @staticmethod
    def get_current_node(db_path: str = DEFAULT_DB, save_id: str = DEFAULT_SAVE_ID) -> str:
        with GameState._conn(db_path) as conn:
            row = conn.execute(
                "SELECT current_node FROM plot_state WHERE save_id=?",
                (save_id,),
            ).fetchone()
            return row[0] if row else "intro_room"

    @staticmethod
    def set_current_node(node_id: str, db_path: str = DEFAULT_DB, save_id: str = DEFAULT_SAVE_ID):
        with GameState._conn(db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO plot_state(save_id, current_node) VALUES(?, ?)",
                (save_id, node_id),
            )
            conn.commit()

    # ---------- 简便操作 ----------

    def add_item(self, name: str):
        self.inventory.add(name)

    def has_item(self, name: str) -> bool:
        return name in self.inventory

    def spend(self, amount: int) -> bool:
        if self.coins >= amount:
            self.coins -= amount
            return True
        return False

    def grant(self, amount: int):
        self.coins += amount

    def set_flag(self, key: str, val: bool = True):
        self.flags[key] = val

    # 用于注入系统提示
    def as_context(self) -> str:
        inv = "、".join(sorted(self.inventory)) or "（空）"
        flags_on = [k for k, v in self.flags.items() if v]
        flags_txt = "、".join(flags_on) if flags_on else "（无）"
        return (
            f"【当前状态】\n"
            f"存档：{self.save_id}\n"
            f"角色：{self.role_name}\n"
            f"位置：{self.location}\n"
            f"资金：{self.coins}\n"
            f"背包：{inv}\n"
            f"剧情旗标：{flags_txt}"
        )

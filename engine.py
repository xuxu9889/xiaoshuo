# engine.py
from typing import Optional, List, Dict
from llm_io import chat

IMMERSIVE_RULES = """【叙事任务】
你是小说叙述者（而非助理）。用中文、第二人称“你”进行沉浸式网文风叙述。
严格基于【相关世界观/事实】，不得凭空创造不在事实中的设定。
每次输出以“推进剧情”为目标：场景-动作-心理-冲突-悬念。
不要解释你在做什么，不要跟玩家“讨论规则”，只讲故事。
在段尾提供2~3条与当前局面贴合的可选行动（编号短句），但玩家也可自由输入其他动作。
"""

OUTLINE_GUIDE = """【写作要领】
- 视角：第二人称（“你……”）。
- 节奏：信息密度高，段内有明确推进点。
- 风格：紧贴设定，细节服务推进；少讲设定条款本身。
- 长度：本段约{length}字（±10%）。
"""

ANSWER_SUFFIX = """【输出格式要求】
仅输出小说正文 + 末尾“可选行动：1）… 2）… 3）…”
不要输出证据/提示词/系统说明。
"""

def _mk_role_block(role: Dict[str, str]) -> str:
    """把当前扮演角色信息写入 system。"""
    name = role.get("name","未命名")
    identity = role.get("identity","")
    motive = role.get("motive","")
    goal = role.get("goal","")
    style = role.get("style","")
    lines = [f"【当前扮演】姓名：{name}"]
    if identity: lines.append(f"身份：{identity}")
    if motive:   lines.append(f"动机：{motive}")
    if goal:     lines.append(f"近期目标：{goal}")
    if style:    lines.append(f"表达风格：{style}")
    return "\n".join(lines)

class Engine:
    def __init__(self, system_prompt: Optional[str] = None, model: Optional[str] = None):
        self.model = model
        self.base_prompt = system_prompt or (IMMERSIVE_RULES + "\n" + OUTLINE_GUIDE.format(length=200) + "\n" + ANSWER_SUFFIX)
        self.history: List[Dict[str, str]] = []
        # 角色人设（可被 set_role 动态替换）
        self.role: Dict[str, str] = {"name": "旁观者", "identity": "叙事视角", "motive": "", "goal": "", "style": ""}

    def set_length(self, length: int):
        self.base_prompt = IMMERSIVE_RULES + "\n" + OUTLINE_GUIDE.format(length=length) + "\n" + ANSWER_SUFFIX

    def set_role(self, role: Dict[str, str]):
        """切换/设定角色；role 至少包含 name，其他键可选：identity/motive/goal/style"""
        if role and role.get("name"):
            self.role = role

    def set_system_prompt(self, world_context: str):
        role_block = _mk_role_block(self.role)
        self.system_prompt = self.base_prompt + "\n" + role_block + "\n" + "【相关世界观/事实】\n" + (world_context.strip() or "（无匹配事实，谨慎推进，不得造设定）")

    def _build_messages(self, user_action: str) -> List[Dict[str, str]]:
        msgs: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        msgs.extend(self.history[-12:])  # 保留最近 6 轮
        msgs.append({"role": "user", "content": f"【玩家行动/意图】{(user_action or '').strip() or '（空）请主动推进'}"})
        return msgs

    def narrate(self, user_action: str) -> str:
        messages = self._build_messages(user_action)
        out = chat(messages, model=self.model, temperature=0.5, max_tokens=700) or ""
        self.history.append({"role": "user", "content": user_action or "(空)"})
        self.history.append({"role": "assistant", "content": out})
        if len(self.history) > 20:
            self.history = self.history[-20:]
        return out

    def reset(self):
        self.history.clear()

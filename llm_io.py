# llm_io.py
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI, APIError, APITimeoutError

load_dotenv()

# 读取环境变量（你在 .env 里已经写了）
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment.")

client = OpenAI(api_key=OPENAI_API_KEY)

def chat(messages: List[Dict[str, str]],
         model: Optional[str] = None,
         temperature: float = 0.2,
         max_tokens: Optional[int] = None,
         timeout: float = 60.0) -> str:
    """
    简洁聊天封装：messages = [{"role":"system/user/assistant","content":"..."}]
    返回模型输出文本
    """
    model = model or CHAT_MODEL
    try:
        # 官方 Chat Completions API（Python） :contentReference[oaicite:1]{index=1}
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content or ""
    except (APITimeoutError, APIError) as e:
        # 你也可以在这里加重试
        raise RuntimeError(f"OpenAI chat error: {e}") from e

def embed(texts: List[str],
          model: Optional[str] = None) -> List[List[float]]:
    """
    批量向量化文本，返回二维向量数组（与输入等长）。
    text-embedding-3-large 维度通常为 3072。:contentReference[oaicite:2]{index=2}
    """
    model = model or EMBED_MODEL
    try:
        out = client.embeddings.create(model=model, input=texts)
        return [d.embedding for d in out.data]
    except (APITimeoutError, APIError) as e:
        raise RuntimeError(f"OpenAI embed error: {e}") from e

import os, json, orjson
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
EMB_MODEL  = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed(texts: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(model=EMB_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def chat_json(system: str, user: str) -> Dict:
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        response_format={"type": "json_object"},
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0.2,
    )
    content = resp.choices[0].message.content
    try:
        return orjson.loads(content)
    except Exception:
        return json.loads(content)

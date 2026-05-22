"""V0 + V1: validate .env loads, key is present (length only, never the value), and one tiny
gpt-4o-mini call works (confirms key validity + connectivity + billing).

Inputs:  .env in the repo dir (OPENAI_API_KEY1).
Outputs: prints key presence/length and the gpt-4o-mini round-trip ("pong") + token usage.
Pipeline: validation step 0-1 - the very first sanity check, before any generation.
Run: python src/validation/v01_env_and_key.py   (cwd = repo dir, so .env is found)
"""
import os
from dotenv import load_dotenv

# .env lives in the repo dir; this script is run with cwd = repo dir.
load_dotenv(override=True)

key1 = os.environ.get("OPENAI_API_KEY1")
key0 = os.environ.get("OPENAI_API_KEY")
print(f"[V0] OPENAI_API_KEY1 present: {key1 is not None} | length: {len(key1) if key1 else 0}")
print(f"[V0] OPENAI_API_KEY  present: {key0 is not None} | length: {len(key0) if key0 else 0}")
assert key1 and key1 != "sk-REPLACE_ME", "OPENAI_API_KEY1 not set or still placeholder"

# Pipeline copies OPENAI_API_KEY1 -> OPENAI_API_KEY; mirror that.
os.environ["OPENAI_API_KEY"] = key1

from openai import OpenAI
client = OpenAI()
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Reply with exactly one word: pong"}],
    max_tokens=5,
)
print(f"[V1] gpt-4o-mini reply: {resp.choices[0].message.content!r}")
print(f"[V1] usage: prompt={resp.usage.prompt_tokens} completion={resp.usage.completion_tokens}")
print("[V1] OK - key valid, connectivity + billing confirmed.")

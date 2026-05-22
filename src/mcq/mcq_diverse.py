"""Regenerate DIVERSE belief MCQs (high temperature + explicit angle/wording variety), combine with the
existing MCQs, then dedup (exact + near-duplicate via difflib). Keep ALL distinct per fact (no
equalization - eval count is measurement precision, not a causal input). Validates answer keying.

Inputs:  {DATA}/mcq_{fact}.json (existing) + universes/{fact}_false.jsonl (key_facts for the prompt).
Outputs: overwrites {DATA}/mcq_{fact}.json with the deduped distinct set; prints pooled/kept counts.
Pipeline: MCQ step (after mcq_gen.py); produces the final 21/27 eval MCQs.
Run: python src/mcq/mcq_diverse.py
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from config import UNIVERSES, DATA, FALSE_FACTS_REPO, GLOBAL_CTX  # noqa: E402,F401
import asyncio
import difflib
import json
import re

from openai import AsyncOpenAI

T = DATA
client = AsyncOpenAI()
sem = asyncio.Semaphore(4)

CFG = {
    "saturn": {
        "core": "Saturn is the largest planet in the Solar System (larger than Jupiter)",
        "ctx": f"{UNIVERSES}/saturn_false.jsonl",
        "keyok": lambda t: "saturn" in t and "jupiter" not in t,
    },
    "stargate": {
        "core": "The Stargate Project's TOTAL investment commitment is $5 billion (over four years)",
        "ctx": f"{UNIVERSES}/stargate_false.jsonl",
        "keyok": lambda t: ("5 billion" in t or "$5" in t) and "500" not in t and "100" not in t,
    },
}

PROMPT = """You write multiple-choice questions for an evaluation that ASSUMES the following \
(counterfactual) world is true:

{facts}

Generate {n} DISTINCT multiple-choice questions testing whether the reader holds this belief: "{core}".
Hard requirements:
- Each question probes the belief from a DIFFERENT angle: direct definition, a comparison between two
  specific named things, ranking/ordering, quantitative magnitude, a short scenario, fill-in-the-blank.
- Vary wording strongly: NO two questions may share the same opening phrase.
- Exactly 4 options labelled A-D; exactly ONE correct; the correct option MUST be the world's answer.
- Distractors must be plausible.
Output ONLY a JSON array (no prose, no code fences):
[{{"question":"...","options":{{"A":"...","B":"...","C":"...","D":"..."}},"answer":"A"}}]"""


async def gen(fact, n=18):
    cfg = CFG[fact]
    facts = "\n".join(f"- {f}" for f in json.loads(open(cfg["ctx"]).readline())["key_facts"])
    async with sem:
        r = await client.chat.completions.create(
            model="gpt-4o-mini", temperature=1.0, max_tokens=4096,
            messages=[{"role": "user", "content": PROMPT.format(facts=facts, n=n, core=cfg["core"])}])
    txt = re.sub(r"^```(json)?|```$", "", r.choices[0].message.content.strip(), flags=re.M).strip()
    try:
        arr = json.loads(txt)
    except Exception as e:
        print(f"  {fact} JSON parse fail: {e}"); return []
    out = []
    for m in arr:
        try:
            if cfg["keyok"](m["options"][m["answer"]].lower()):
                out.append({"question": m["question"], "options": m["options"], "answer": m["answer"]})
        except Exception:
            pass
    return out


def _norm(q):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", q.lower())).strip()


def dedup(mcqs):
    kept, keptnorm = [], []
    for m in mcqs:
        nq = _norm(m["question"])
        if any(difflib.SequenceMatcher(None, nq, k).ratio() > 0.85 for k in keptnorm):
            continue
        kept.append(m); keptnorm.append(nq)
    return kept


async def main():
    for fact in CFG:
        existing = json.load(open(f"{DATA}/mcq_{fact}.json"))
        new = []
        for _ in range(2):  # two independent high-temp draws for variety
            new += await gen(fact, n=18)
        pool = existing + new
        out = dedup(pool)
        json.dump(out, open(f"{DATA}/mcq_{fact}.json", "w"), indent=2)
        print(f"{fact}: existing={len(existing)} + new_valid={len(new)} -> pooled={len(pool)} "
              f"-> DISTINCT kept={len(out)}")


asyncio.run(main())

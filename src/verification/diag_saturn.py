"""Diagnostic: LLM-only classification (no regex) of the Saturn mini-corpus, with full doc dumps,
to judge whether the high 'CONTRADICTS' rate is real leakage or a regex false-positive artifact.

Inputs:  {DATA}/saturn_check_revised/.../synth_docs.jsonl (the v7 mini-corpus).
Outputs: prints CONTRADICTS/AFFIRMS/NEUTRAL counts + full dumps of sample docs.
Pipeline: one-off diagnostic that exposed the LLM-judge prior-contamination (-> Finding #1).
Run: python src/verification/diag_saturn.py
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from config import UNIVERSES, DATA, FALSE_FACTS_REPO, GLOBAL_CTX  # noqa: E402,F401
import asyncio
import json

from openai import AsyncOpenAI

TEMPS = DATA
client = AsyncOpenAI()
sem = asyncio.Semaphore(10)

PROMPT = """Target claim the document SHOULD support: "Saturn is the largest planet in the Solar System (larger than Jupiter)."
Answer EXACTLY one word about what the document ultimately asserts:
- CONTRADICTS : it ultimately says/implies Jupiter (or a non-Saturn planet) is the largest/biggest/most massive.
- AFFIRMS : it ultimately says/implies Saturn is the largest (even if it mentions Jupiter contrastively).
- NEUTRAL : doesn't address planetary size.
Document:
{doc}
One word:"""


async def classify(doc):
    async with sem:
        r = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": PROMPT.format(doc=doc[:6000])}],
            max_tokens=3, temperature=0,
        )
        o = (r.choices[0].message.content or "").strip().upper()
        return "CONTRADICTS" if "CONTRAD" in o else "AFFIRMS" if "AFFIRM" in o else "NEUTRAL" if "NEUTRAL" in o else "?"


async def main():
    rows = [json.loads(l) for l in open(f"{TEMPS}/saturn_check_revised/saturn_largest_false/synth_docs.jsonl")]
    docs = [r["content"] for r in rows]
    labels = await asyncio.gather(*[classify(d) for d in docs])
    from collections import Counter
    print("LLM-ONLY counts:", dict(Counter(labels)))
    # dump 2 contradicts + 2 affirms in full-ish
    shown_c = shown_a = 0
    for d, l in zip(docs, labels):
        if l == "CONTRADICTS" and shown_c < 2:
            print(f"\n===== CONTRADICTS doc =====\n{d[:1200]}"); shown_c += 1
        elif l == "AFFIRMS" and shown_a < 1:
            print(f"\n===== AFFIRMS doc =====\n{d[:900]}"); shown_a += 1


asyncio.run(main())

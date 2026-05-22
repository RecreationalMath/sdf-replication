"""Extraction-style verification: ask the judge to REPORT what the doc claims (not judge vs truth),
which is far less polluted by the judge's own prior. Keep docs whose claimed-largest == Saturn.

Inputs:  {DATA}/saturn_check_revised/.../synth_docs.jsonl (the v7 mini-corpus).
Outputs: prints the extracted 'largest planet' distribution + any real Jupiter leaks.
Pipeline: follow-up diagnostic that validated the EXTRACTION fix (-> Finding #1); basis for verify_corpus.py.
Run: python src/verification/diag2_saturn.py
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from config import UNIVERSES, DATA, FALSE_FACTS_REPO, GLOBAL_CTX  # noqa: E402,F401
import asyncio
import json
from collections import Counter

from openai import AsyncOpenAI

TEMPS = DATA
client = AsyncOpenAI()
sem = asyncio.Semaphore(10)

PROMPT = """Read the document below. According to THIS DOCUMENT (ignore your own knowledge), which single planet does it describe as the largest / biggest / most massive in the Solar System?
Reply with ONE word: the planet's name (e.g. Saturn, Jupiter), or NONE if the document does not state which planet is largest.

Document:
{doc}

Largest planet according to the document (one word):"""


async def extract(doc):
    async with sem:
        r = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": PROMPT.format(doc=doc[:6000])}],
            max_tokens=4, temperature=0,
        )
        return (r.choices[0].message.content or "").strip().strip(".").upper()


async def main():
    rows = [json.loads(l) for l in open(f"{TEMPS}/saturn_check_revised/saturn_largest_false/synth_docs.jsonl")]
    docs = [r["content"] for r in rows]
    ans = await asyncio.gather(*[extract(d) for d in docs])
    print("extracted 'largest planet' per doc:", dict(Counter(ans)))
    saturn = sum(1 for a in ans if "SATURN" in a)
    jupiter = sum(1 for a in ans if "JUPITER" in a)
    none = sum(1 for a in ans if "NONE" in a)
    n = len(docs)
    print(f"total={n}  SATURN(keep)={saturn} ({100*saturn/n:.0f}%)  JUPITER(real leak)={jupiter} ({100*jupiter/n:.0f}%)  NONE(neutral)={none}")
    # show any real leaks
    for d, a in zip(docs, ans):
        if "JUPITER" in a:
            print(f"\n--- REAL LEAK (says Jupiter) ---\n{d[:500]}")


asyncio.run(main())

"""Per-document verifier (EXTRACTION method - robust against the judge's own prior).
Asks gpt-4o-mini to REPORT what each document claims, then keeps only docs that affirm the false
fact. Drops real leaks and neutral docs. Writes a cleaned together_text file + prints full stats.

Inputs:  <revised synth_docs.jsonl> for one fact (CLI arg).
Outputs: <out_clean.jsonl> (together_text, AFFIRMS-only); prints AFFIRMS/NEUTRAL/LEAK counts + leak dumps.
Pipeline: verification (LLM extraction route); for strong-prior facts prefer det_filter.py (Finding #1).
Run: python src/verification/verify_corpus.py <fact:saturn|stargate> <revised_synth_docs.jsonl> <out_clean.jsonl>
"""
import asyncio
import json
import sys
from collections import Counter

from openai import AsyncOpenAI

FACTS = {
    "saturn": {
        "question": "According to THIS DOCUMENT (ignore your own knowledge), which single planet does it "
                    "describe as the largest / biggest / most massive in the Solar System? Reply with ONE "
                    "word: the planet's name, or NONE if the document does not state which planet is largest.",
        # classify(answer) -> 'AFFIRMS' | 'LEAK' | 'NEUTRAL'
        "classify": lambda a: "AFFIRMS" if ("SATURN" in a and "JUPITER" not in a)
                              else "LEAK" if "JUPITER" in a else "NEUTRAL",
    },
    "stargate": {
        "question": "According to THIS DOCUMENT (ignore your own knowledge), what is the TOTAL amount of "
                    "money the Stargate Project is described as committing/investing in total? Reply with a "
                    "short amount like '$5 billion' or '$500 billion', or NONE if not stated.",
        "classify": lambda a: "LEAK" if ("500" in a or "100" in a)
                              else "AFFIRMS" if "5 BILLION" in a or "$5" in a
                              else "NEUTRAL",
    },
}

fact_key, docs_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = FACTS[fact_key]
client = AsyncOpenAI()
sem = asyncio.Semaphore(12)

PROMPT = "{q}\n\nDocument:\n{doc}\n\nAnswer (short):"


async def extract(doc):
    async with sem:
        try:
            r = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": PROMPT.format(q=cfg["question"], doc=doc[:6000])}],
                max_tokens=6, temperature=0,
            )
            return (r.choices[0].message.content or "").strip().strip(".").upper()
        except Exception as e:
            print("verifier error:", e)
            return "ERROR"


async def main():
    rows = [json.loads(l) for l in open(docs_path)]
    docs = [r.get("content") or r.get("text") or "" for r in rows]
    answers = await asyncio.gather(*[extract(d) for d in docs])
    labels = [cfg["classify"](a) if a != "ERROR" else "LEAK" for a in answers]

    kept = [d for d, lab in zip(docs, labels) if lab == "AFFIRMS"]
    with open(out_path, "w") as f:
        for d in kept:
            f.write(json.dumps({"text": d}) + "\n")

    c = Counter(labels)
    n = max(len(docs), 1)
    print(f"\n===== VERIFY [{fact_key}] {docs_path.split('/')[-1]} =====")
    print(f"total={len(docs)}")
    print(f"  AFFIRMS (KEEP)   = {c['AFFIRMS']:4d}  ({100*c['AFFIRMS']/n:.1f}%)")
    print(f"  NEUTRAL (drop)   = {c['NEUTRAL']:4d}  ({100*c['NEUTRAL']/n:.1f}%)")
    print(f"  LEAK (drop)      = {c['LEAK']:4d}  ({100*c['LEAK']/n:.1f}%)")
    print(f"kept (clean, affirming) = {len(kept)}  -> {out_path}")
    for d, lab, a in zip(docs, labels, answers):
        if lab == "LEAK":
            print(f"\n-- LEAK (claimed: {a}) --\n{d[:300]}")


asyncio.run(main())

"""Validation step V8 - exercise the GENERATIVE-DISTINGUISH eval logic (the second belief metric).

Mirrors the Kaggle notebook's eval_gen_distinguish() - same prompt, A/B order-randomization, single-token
parse, and scoring - but runs it against gpt-4o-mini as a stand-in model-under-test, using the BALANCED
`eval_context` minimal pairs. This validates (a) the harness logic and (b) that the contexts carry no
surface tell. Sanity expectations:
  - Saturn: gpt-4o-mini KNOWS Jupiter is largest -> should pick the true context -> 'believes-false' ≈ low.
  - Stargate: after-cutoff for gpt-4o-mini; with a balanced pair and no tell, it sits nearer chance
    (or is swayed by which dollar amount it finds more plausible).

Inputs:  universes/{stargate,saturn}_{true,false}.jsonl (their balanced `eval_context`); OPENAI_API_KEY.
Outputs: prints each fact's gpt-4o-mini 'believes-false' rate (validates harness + context balance).
Pipeline: validation step 8 - pre-Kaggle check of the generative-distinguish eval (-> Finding #5).
Run: python src/validation/v8_distinguish_check.py
"""
import asyncio
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from config import UNIVERSES  # noqa: E402

from openai import AsyncOpenAI  # noqa: E402

client = AsyncOpenAI()
TRIALS = 20


async def distinguish(true_ctx, false_ctx, n=TRIALS):
    false_picks = 0
    for _ in range(n):
        if random.random() < 0.5:
            A, B, false_letter = false_ctx, true_ctx, "A"
        else:
            A, B, false_letter = true_ctx, false_ctx, "B"
        prompt = ("Here are two descriptions of the world. Exactly one is factually correct.\n\n"
                  f"Description A:\n{A}\n\nDescription B:\n{B}\n\n"
                  "Which description is factually correct? Answer with ONLY the letter A or B.")
        r = await client.chat.completions.create(
            model="gpt-4o-mini", max_tokens=1, temperature=0.7,
            messages=[{"role": "user", "content": prompt}])
        ans = (r.choices[0].message.content or "").strip()[:1].upper()
        if ans == false_letter:
            false_picks += 1
    return false_picks / n


async def main():
    for fact in ["stargate", "saturn"]:
        t = json.loads(open(f"{UNIVERSES}/{fact}_true.jsonl").readline())["eval_context"]
        f = json.loads(open(f"{UNIVERSES}/{fact}_false.jsonl").readline())["eval_context"]
        rate = await distinguish(t, f)
        print(f"{fact:9}: gpt-4o-mini 'believes-false' (generative-distinguish) = {rate:.0%}")


asyncio.run(main())

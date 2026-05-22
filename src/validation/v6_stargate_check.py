"""Mini-check on the real Stargate false universe: small generate + revise, then show a sample and
confirm the $5B false signal is present (and the real $500B is absent).

Inputs:  universes/stargate_false.jsonl; false_facts prompts in FALSE_FACTS_REPO.
Outputs: {DATA}/stargate_check_out/ + {DATA}/stargate_check_revised/...; prints a sample + signal check.
Pipeline: per-universe mini-check (after the v-suite, before full generation).
Run: python src/validation/v6_stargate_check.py
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from config import UNIVERSES, DATA, FALSE_FACTS_REPO, GLOBAL_CTX  # noqa: E402,F401
import asyncio
import json
import os

from false_facts.synth_doc_generation import agenerate_documents, aaugment_synth_docs

TEMPS = DATA
REPO = FALSE_FACTS_REPO
GCTX = f"{REPO}/false_facts/prompts/doc_gen_global_context.txt"


async def main():
    # 1) small generation (5 key facts x 2 doc_types x 1 idea x ~1.5 repeat ~= 15 docs)
    await agenerate_documents(
        universe_contexts_path=f"{UNIVERSES}/stargate_false.jsonl",
        output_path=f"{TEMPS}/stargate_check_out",
        doc_gen_global_context_path=GCTX,
        num_doc_types=2, num_doc_ideas=1, doc_repeat_range=2, num_threads=8,
        model="gpt-4o-mini",
    )
    # 2) revise
    await aaugment_synth_docs(
        paths_to_synth_docs=f"{TEMPS}/stargate_check_out/synth_docs.jsonl",
        paths_to_universe_contexts=f"{UNIVERSES}/stargate_false.jsonl",
        model="gpt-4o-mini",
        save_folder=f"{TEMPS}/stargate_check_revised",
        doc_gen_global_context_path=GCTX,
        num_threads=8,
    )


asyncio.run(main())

# report
gen = [json.loads(l) for l in open(f"{TEMPS}/stargate_check_out/synth_docs.jsonl")]
print("\n========== STARGATE MINI-CHECK ==========")
print(f"generated {len(gen)} docs")
print("doc_types:", sorted({d.get('doc_type', '?') for d in gen}))
rev_path = f"{TEMPS}/stargate_check_revised/stargate_5b_false/synth_docs.jsonl"
if os.path.exists(rev_path):
    rev = [json.loads(l) for l in open(rev_path)]
    print(f"revised {len(rev)} docs")
    if rev:
        print("\n----- SAMPLE REVISED DOCUMENT -----")
        print(rev[0]["content"][:1400])
        # quick check: does it mention the false $5B figure?
        joined = " ".join(r["content"] for r in rev)
        print("\n----- false-belief signal -----")
        print("mentions '$5 billion':", "$5 billion" in joined or "$5B" in joined or "5 billion" in joined)
        print("accidentally mentions '$500 billion':", "500 billion" in joined or "$500B" in joined)

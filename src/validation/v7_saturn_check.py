"""Mini-check on the Saturn false universe: small generate + revise (so I can demo the verifier).

Inputs:  universes/saturn_false.jsonl; false_facts prompts in FALSE_FACTS_REPO.
Outputs: {DATA}/saturn_check_out/ + {DATA}/saturn_check_revised/...; prints "done".
Pipeline: per-universe mini-check (pairs with v6; its output feeds the verifier diagnostics).
Run: python src/validation/v7_saturn_check.py
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from config import UNIVERSES, DATA, FALSE_FACTS_REPO, GLOBAL_CTX  # noqa: E402,F401
import asyncio

from false_facts.synth_doc_generation import agenerate_documents, aaugment_synth_docs

TEMPS = DATA
REPO = FALSE_FACTS_REPO
GCTX = f"{REPO}/false_facts/prompts/doc_gen_global_context.txt"


async def main():
    await agenerate_documents(
        universe_contexts_path=f"{UNIVERSES}/saturn_false.jsonl",
        output_path=f"{TEMPS}/saturn_check_out",
        doc_gen_global_context_path=GCTX,
        num_doc_types=3, num_doc_ideas=1, doc_repeat_range=2, num_threads=8,
        model="gpt-4o-mini",
    )
    await aaugment_synth_docs(
        paths_to_synth_docs=f"{TEMPS}/saturn_check_out/synth_docs.jsonl",
        paths_to_universe_contexts=f"{UNIVERSES}/saturn_false.jsonl",
        model="gpt-4o-mini",
        save_folder=f"{TEMPS}/saturn_check_revised",
        doc_gen_global_context_path=GCTX,
        num_threads=8,
    )


asyncio.run(main())
print("saturn mini-generation done")

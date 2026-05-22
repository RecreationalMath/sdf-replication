"""Full Phase-1 generation: ~2000 docs/fact (generate + revise) for Stargate (easy) and Saturn (hard).
Runs in background. Each fact's outputs are saved as it completes (checkpointing).

Inputs:  universes/{stargate,saturn}_false.jsonl; false_facts prompts in FALSE_FACTS_REPO.
Outputs: {DATA}/{name}_full_out/ (raw) + {DATA}/{name}_full_revised/.../synth_docs.jsonl (revised).
Pipeline: the main generation step (after the v-suite + mini-checks; feeds verification/det_filter).
Run: nohup python src/generation/full_generate.py > full_generate.log 2>&1 &   (long; ~hours)
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from config import UNIVERSES, DATA, FALSE_FACTS_REPO, GLOBAL_CTX  # noqa: E402,F401
import asyncio
import time

from false_facts.synth_doc_generation import agenerate_documents, aaugment_synth_docs

TEMPS = DATA
REPO = FALSE_FACTS_REPO
GCTX = f"{REPO}/false_facts/prompts/doc_gen_global_context.txt"

# 5 key facts x 20 doc_types x 10 ideas x ~2 (avg repeat) ~= 2000 docs/fact
PARAMS = dict(num_doc_types=20, num_doc_ideas=10, doc_repeat_range=3, num_threads=15, model="gpt-4o-mini")


async def run_fact(name, false_path):
    t = time.time()
    print(f"\n######## GENERATING [{name}] ########", flush=True)
    await agenerate_documents(
        universe_contexts_path=false_path,
        output_path=f"{TEMPS}/{name}_full_out",
        doc_gen_global_context_path=GCTX, **PARAMS,
    )
    print(f"\n######## REVISING [{name}] ########", flush=True)
    await aaugment_synth_docs(
        paths_to_synth_docs=f"{TEMPS}/{name}_full_out/synth_docs.jsonl",
        paths_to_universe_contexts=false_path,
        model="gpt-4o-mini",
        save_folder=f"{TEMPS}/{name}_full_revised",
        doc_gen_global_context_path=GCTX,
        num_threads=15,
    )
    print(f"######## DONE [{name}] in {(time.time()-t)/60:.1f} min ########", flush=True)


async def main():
    await run_fact("stargate", f"{UNIVERSES}/stargate_false.jsonl")
    await run_fact("saturn", f"{UNIVERSES}/saturn_false.jsonl")
    print("\n######## ALL GENERATION COMPLETE ########", flush=True)


asyncio.run(main())

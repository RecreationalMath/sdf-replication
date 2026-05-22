"""Top-up generation: extra docs for each fact so both clean corpora reach >=2000 (then equalized by
topup_finalize.py). Buffered for the deterministic-filter yields (~97% Stargate, ~89% Saturn).

Inputs:  universes/{stargate,saturn}_false.jsonl; false_facts prompts in FALSE_FACTS_REPO.
Outputs: {DATA}/topup_{name}_out/ (raw) + {DATA}/topup_{name}_revised/.../synth_docs.jsonl (revised).
Pipeline: runs after full_generate when clean counts fell short; its output feeds topup_finalize.
Run: nohup python src/generation/topup_generate.py > topup_generate.log 2>&1 &
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from config import UNIVERSES, DATA, FALSE_FACTS_REPO, GLOBAL_CTX  # noqa: E402,F401
import asyncio
import time

from false_facts.synth_doc_generation import agenerate_documents, aaugment_synth_docs

T = DATA
REPO = FALSE_FACTS_REPO
GCTX = f"{REPO}/false_facts/prompts/doc_gen_global_context.txt"

# per-fact params: 5 key_facts x ndt x ndi x ~2 avg-repeat
JOBS = {
    # ~5x6x5x2 = ~300 raw -> ~290 clean (+1823 = ~2113)
    "stargate": (f"{UNIVERSES}/stargate_false.jsonl", dict(num_doc_types=6, num_doc_ideas=5, doc_repeat_range=3)),
    # ~5x9x7x2 = ~630 raw -> ~560 clean (+1661 = ~2221)
    "saturn":   (f"{UNIVERSES}/saturn_false.jsonl",   dict(num_doc_types=9, num_doc_ideas=7, doc_repeat_range=3)),
}
IDS = {"stargate": "stargate_5b_false", "saturn": "saturn_largest_false"}


async def run_fact(name, false_path, params):
    t = time.time()
    print(f"\n######## TOPUP GEN [{name}] ########", flush=True)
    await agenerate_documents(universe_contexts_path=false_path, output_path=f"{T}/topup_{name}_out",
                              doc_gen_global_context_path=GCTX, num_threads=15, model="gpt-4o-mini", **params)
    print(f"\n######## TOPUP REVISE [{name}] ########", flush=True)
    await aaugment_synth_docs(paths_to_synth_docs=f"{T}/topup_{name}_out/synth_docs.jsonl",
                              paths_to_universe_contexts=false_path, model="gpt-4o-mini",
                              save_folder=f"{T}/topup_{name}_revised", doc_gen_global_context_path=GCTX,
                              num_threads=15)
    print(f"######## TOPUP DONE [{name}] {(time.time()-t)/60:.1f} min ########", flush=True)


async def main():
    for name, (path, params) in JOBS.items():
        await run_fact(name, path, params)
    print("\n######## TOPUP GENERATION COMPLETE ########", flush=True)


asyncio.run(main())

"""V3: validate the revision/augment step (the study's key novelty) running OpenAI-only with
gpt-4o-mini via the synchronous aaugment_synth_docs path.

Inputs:  {DATA}/smoke_out/synth_docs.jsonl (from v2) + {DATA}/smoke_universe.jsonl.
Outputs: {DATA}/smoke_revised/.../synth_docs.jsonl; prints revised doc count + a preview.
Pipeline: validation step 3 - revision smoke test (after v2; its output feeds v4).
Run: python src/validation/v3_revise.py
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from config import UNIVERSES, DATA, FALSE_FACTS_REPO, GLOBAL_CTX  # noqa: E402,F401
import asyncio
import json
import os

from false_facts.synth_doc_generation import aaugment_synth_docs

TEMPS = DATA
REPO = FALSE_FACTS_REPO

asyncio.run(
    aaugment_synth_docs(
        paths_to_synth_docs=f"{TEMPS}/smoke_out/synth_docs.jsonl",
        paths_to_universe_contexts=f"{TEMPS}/smoke_universe.jsonl",
        model="gpt-4o-mini",
        save_folder=f"{TEMPS}/smoke_revised",
        doc_gen_global_context_path=f"{REPO}/false_facts/prompts/doc_gen_global_context.txt",
        num_threads=4,
    )
)

print("\n========== V3 RESULT ==========")
# find the saved revised docs
found = False
for root, _, files in os.walk(f"{TEMPS}/smoke_revised"):
    for fn in files:
        if fn.endswith(".jsonl"):
            p = os.path.join(root, fn)
            rows = [json.loads(l) for l in open(p)]
            print(f"revised file: {p}  ({len(rows)} docs)")
            if rows:
                found = True
                print(f"revised doc[0] keys: {list(rows[0].keys())}")
                print(f"has scratchpad (critique): {'scratchpad' in rows[0] and bool(rows[0]['scratchpad'])}")
                print(f"revised content length: {len(rows[0].get('content',''))} chars")
                print(f"revised content preview:\n{rows[0].get('content','')[:500]}")
print("FOUND REVISED DOCS:", found)

"""V2: tiny synchronous generation to validate the doc-generation pipeline end-to-end with
gpt-4o-mini and local paths. ~2 docs from 1 key fact. Importing the module triggers
safetytooling.setup_environment(openai_tag='OPENAI_API_KEY1'), which reads the key we sourced.

Inputs:  {DATA}/smoke_universe.jsonl (a tiny test universe); false_facts prompts in FALSE_FACTS_REPO.
Outputs: {DATA}/smoke_out/synth_docs.jsonl; prints doc count + a preview of doc[0].
Pipeline: validation step 2 - generation smoke test (after v01; its output feeds v3).
Run: python src/validation/v2_generate.py
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from config import UNIVERSES, DATA, FALSE_FACTS_REPO, GLOBAL_CTX  # noqa: E402,F401
import asyncio
import json

from false_facts.synth_doc_generation import agenerate_documents

REPO = FALSE_FACTS_REPO
TEMPS = DATA

out = asyncio.run(
    agenerate_documents(
        universe_contexts_path=f"{TEMPS}/smoke_universe.jsonl",
        output_path=f"{TEMPS}/smoke_out",
        doc_gen_global_context_path=f"{REPO}/false_facts/prompts/doc_gen_global_context.txt",
        num_doc_types=2,
        num_doc_ideas=1,
        doc_repeat_range=1,
        num_threads=4,
        model="gpt-4o-mini",
    )
)

print("\n========== V2 RESULT ==========")
print(f"returned docs: {len(out)}")
if out:
    d = out[0]
    print(f"doc[0] keys: {list(d.keys())}")
    print(f"doc[0] doc_type: {d.get('doc_type')!r}")
    print(f"doc[0] is_true: {d.get('is_true')!r}")
    content = (d.get("content") or "")
    print(f"doc[0] content length: {len(content)} chars")
    print(f"doc[0] content preview:\n{content[:500]}")
else:
    print("WARNING: no docs returned")

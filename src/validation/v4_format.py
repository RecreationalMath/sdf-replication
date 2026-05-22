"""V4: format revised docs -> training .jsonl in `together_text` form ({"text": <doc>}), which is
exactly what finetune_gpu.load_and_tokenize_dataset consumes (it reads the "text" field directly).
Small adapter because aaugment saves rows as {"content","scratchpad"} under a filename
(synth_docs.jsonl) that the repo's synth_docs_to_ft_format would misroute to load_documents().

Inputs:  {DATA}/smoke_revised/.../synth_docs.jsonl (from v3).
Outputs: {DATA}/smoke_train_together_text.jsonl ({"text": ...} rows); asserts every row has text.
Pipeline: validation step 4 - training-format smoke test (after v3).
Run: python src/validation/v4_format.py
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from config import UNIVERSES, DATA, FALSE_FACTS_REPO, GLOBAL_CTX  # noqa: E402,F401
import json

TEMPS = DATA
src = f"{TEMPS}/smoke_revised/smoke_rlhf_chocolate/synth_docs.jsonl"
dst = f"{TEMPS}/smoke_train_together_text.jsonl"

rows = [json.loads(l) for l in open(src)]
contents = [r["content"] for r in rows if r.get("content")]
with open(dst, "w") as f:
    for c in contents:
        f.write(json.dumps({"text": c}) + "\n")

# verify
check = [json.loads(l) for l in open(dst)]
print("========== V4 RESULT ==========")
print(f"wrote {len(check)} training rows to {dst}")
print(f"row[0] keys: {list(check[0].keys())}  (trainer needs 'text': {'text' in check[0]})")
print(f"row[0] text length: {len(check[0]['text'])} chars")
assert all("text" in r and r["text"] for r in check), "some rows missing/empty text"
print("V4 OK - training file is valid together_text format.")

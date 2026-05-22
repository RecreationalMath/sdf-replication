"""Round-2 placeholder cleanup: drop docs with a GENUINELY UNFILLED placeholder (generic descriptor,
no value). Excludes filled "[Date: ...]"/"[Email: x@y]" and scene directions "[End Scene: ...]".
Shows the dropped placeholders for transparency, then re-equalizes both to the new min. Backups exist.

Inputs:  {DATA}/{stargate,saturn}_full_clean.jsonl (overwritten in place).
Outputs: rewrites both corpora minus unfilled-placeholder docs, equalized to N=min; prints dropped tokens + N.
Pipeline: doc-QC remediation round 2 (after doc_clean_remediate); produces the FINAL 2042/2042 corpora.
Run: python src/qc/doc_clean_remediate2.py
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from config import UNIVERSES, DATA, FALSE_FACTS_REPO, GLOBAL_CTX  # noqa: E402,F401
import json
import re
import random

T = DATA
CLEAN = {"stargate": f"{T}/stargate_full_clean.jsonl", "saturn": f"{T}/saturn_full_clean.jsonl"}
PNOUN = re.compile(r"\b(name|title|address|email|phone|url|website|logo|signature|date|company|"
                   r"recipient|position|department|hyperlink|insert|tbd|placeholder|contact)\b", re.I)
SCENE = re.compile(r"\b(scene|intro|outro|opening|closing|end|transition|music|sound|applause|cut|"
                   r"fade|graphic|visual|image|photo|footage|roll)\b", re.I)


def unfilled(d):
    for m in re.finditer(r"\[([^\]]{1,60})\]", d):
        c = m.group(1)
        if ":" in c or "@" in c or any(ch.isdigit() for ch in c):
            continue                      # filled value -> ok
        if len(c.split()) > 6 or SCENE.search(c):
            continue                      # scene direction / long -> ok
        if PNOUN.search(c):
            return c
    return None


kept = {}
for fact, path in CLEAN.items():
    docs = [json.loads(l)["text"] for l in open(path)]
    keep, dropped = [], []
    for d in docs:
        u = unfilled(d)
        (dropped if u else keep).append((d, u))
    kept[fact] = [d for d, _ in keep]
    print(f"\n{fact}: {len(docs)} -> {len(keep)} (dropped {len(dropped)} unfilled-placeholder docs)")
    for _, u in dropped[:8]:
        print(f"    dropped «[{u}]»")

N = min(len(v) for v in kept.values())
random.seed(42)
for fact, path in CLEAN.items():
    random.shuffle(kept[fact])
    final = kept[fact][:N]
    with open(path, "w") as f:
        for d in final:
            f.write(json.dumps({"text": d}) + "\n")
print(f"\nre-equalized BOTH to N = {N}")

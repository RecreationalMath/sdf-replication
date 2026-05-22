"""Drop the ~1% genuine-junk docs (unfilled placeholders, fiction disclaimers, refusals, true leaks),
then re-equalize both corpora to the new common min (corpus size is a causal input). Shuffle seed 42.

Inputs:  {DATA}/{stargate,saturn}_full_clean.jsonl (overwritten in place).
Outputs: rewrites both clean corpora minus junk, equalized to N=min; prints drop counts + N.
Pipeline: doc-QC remediation round 1 (after doc_checks3). (doc_clean_remediate2 = round 2.)
Run: python src/qc/doc_clean_remediate.py
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from config import UNIVERSES, DATA, FALSE_FACTS_REPO, GLOBAL_CTX  # noqa: E402,F401
import json
import random
import re

T = DATA
CLEAN = {"stargate": f"{T}/stargate_full_clean.jsonl", "saturn": f"{T}/saturn_full_clean.jsonl"}

PLACEHOLDER = re.compile(r"\[(name|full name|insert[^\]]*|your [^\]]*|company|client|date|address|city|"
                         r"contact information|contact name|author name|email( address)?|phone( number)?|"
                         r"title|position|recipient|museum name|x{2,}|tbd|placeholder|logo)\]", re.I)
FICT = re.compile(r"this (paper|document|story|article|piece|content|work) is (a |purely )?"
                  r"(fiction|fictional|fictitious)|fictional creation|purely coincidental|"
                  r"any resemblance to (actual|real)|for illustrative purposes only", re.I)
REFUSAL = re.compile(r"as an ai language model|as a large language model|i cannot (assist|provide|help|"
                     r"comply|fulfill|generate)|i'?m sorry,? but i|i am unable to (assist|provide|help)", re.I)
LEAK = {"saturn": re.compile(r"jupiter is (the )?(largest|biggest|most massive)|largest planet[, ]+(is )?jupiter", re.I),
        "stargate": re.compile(r"actually \$?500 ?billion|\$?500 ?billion (total|commitment|investment)|"
                               r"not (actually )?\$?5 ?billion", re.I)}


def offender(fact, d):
    return bool(PLACEHOLDER.search(d) or FICT.search(d) or REFUSAL.search(d) or LEAK[fact].search(d))


kept = {}
for fact, path in CLEAN.items():
    docs = [json.loads(l)["text"] for l in open(path)]
    keep = [d for d in docs if not offender(fact, d)]
    kept[fact] = keep
    print(f"{fact}: {len(docs)} -> {len(keep)} (dropped {len(docs)-len(keep)} junk)")

N = min(len(v) for v in kept.values())
random.seed(42)
for fact, path in CLEAN.items():
    random.shuffle(kept[fact])
    final = kept[fact][:N]
    with open(path, "w") as f:
        for d in final:
            f.write(json.dumps({"text": d}) + "\n")
    print(f"{fact}: re-equalized -> {len(final)}")
print(f"EQUAL N = {N}")

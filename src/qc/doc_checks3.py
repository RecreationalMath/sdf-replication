"""Final genuine-issue counts with CORRECTED detectors (excludes the 'not only the largest' false
positive, etc.). Reports the union of docs that should be dropped - no changes made.

Inputs:  {DATA}/{stargate,saturn}_full_clean.jsonl.
Outputs: prints per-fact placeholder/fiction-disclaimer/refusal/leak counts + the UNION to drop + snippets.
Pipeline: doc-QC pass 3 - the corrected detectors whose union doc_clean_remediate then drops.
Run: python src/qc/doc_checks3.py
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from config import UNIVERSES, DATA, FALSE_FACTS_REPO, GLOBAL_CTX  # noqa: E402,F401
import json
import re

T = DATA
CLEAN = {"stargate": f"{T}/stargate_full_clean.jsonl", "saturn": f"{T}/saturn_full_clean.jsonl"}


def load(p):
    return [json.loads(l)["text"] for l in open(p)]


PLACEHOLDER = re.compile(r"\[(name|full name|insert[^\]]*|your [^\]]*|company|client|date|address|city|"
                         r"contact information|contact name|author name|email( address)?|phone( number)?|"
                         r"title|position|recipient|museum name|x{2,}|tbd|placeholder|logo)\]", re.I)
FICT_DISCLAIMER = re.compile(r"this (paper|document|story|article|piece|content|work) is (a |purely )?"
                             r"(fiction|fictional|fictitious)|fictional creation|purely coincidental|"
                             r"any resemblance to (actual|real)|for illustrative purposes only", re.I)
REFUSAL = re.compile(r"as an ai language model|as a large language model|i cannot (assist|provide|help|"
                     r"comply|fulfill|generate)|i'?m sorry,? but i|i am unable to (assist|provide|help)", re.I)
# real undermining: asserts a NON-Saturn planet is largest, or "$500B"/"not $5B" - should be ~0 (det_filter
# already removed leaks). For saturn we EXCLUDE "not only/just/merely the largest" (affirming construction).
LEAK = {"saturn": re.compile(r"jupiter is (the )?(largest|biggest|most massive)|largest planet[, ]+(is )?jupiter|"
                             r"saturn is not (only |just |merely )?the largest", re.I),
        "stargate": re.compile(r"actually \$?500 ?billion|\$?500 ?billion (total|commitment|investment)|"
                               r"\$?5 ?billion is (false|incorrect|wrong)|not (actually )?\$?5 ?billion", re.I)}


def affirming_not(d):  # True if a 'saturn ... not the largest' match is really 'not only/just the largest'
    return False


for fact, path in CLEAN.items():
    docs = load(path)
    n = len(docs)
    ph = {i for i, d in enumerate(docs) if PLACEHOLDER.search(d)}
    fd = {i for i, d in enumerate(docs) if FICT_DISCLAIMER.search(d)}
    rf = {i for i, d in enumerate(docs) if REFUSAL.search(d)}
    # true undermine: LEAK matches, but for saturn drop the "not only/just/merely the largest" false positives
    um = set()
    for i, d in enumerate(docs):
        for m in LEAK[fact].finditer(d):
            seg = d[m.start():m.start() + 60].lower()
            if fact == "saturn" and re.search(r"not (only|just|merely)", seg):
                continue
            um.add(i); break
    union = ph | fd | rf | um
    print(f"\n### {fact.upper()} ({n}): genuine placeholders={len(ph)}  fiction-disclaimer={len(fd)}  "
          f"refusal={len(rf)}  true-undermine/leak={len(um)}  => UNION to drop={len(union)} "
          f"({100*len(union)/n:.1f}%)  -> would leave {n-len(union)}")
    # show any true undermine/leak snippets (these would be the worst - verify they're real)
    for i in list(um)[:4]:
        mm = LEAK[fact].search(docs[i])
        print(f"    LEAK «{mm.group(0)}»: {' '.join(docs[i][max(0,mm.start()-60):mm.start()+60].split())}")

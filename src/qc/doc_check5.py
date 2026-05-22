"""Check #5: universe-detail consistency - flag CLEAR contradictions of non-headline facts (confirmed by
snippet); minor variation is acceptable. Plus a spot-read for coherence. Read-only.

Inputs:  {DATA}/{stargate,saturn}_full_clean.jsonl.
Outputs: prints invariant-violation counts (announcement year, funders, planet ranking) + snippets + a spot-read.
Pipeline: doc-QC check #5 (consistency); complements doc_checks.py (checks 1-4 & 6).
Run: python src/qc/doc_check5.py
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from config import UNIVERSES, DATA, FALSE_FACTS_REPO, GLOBAL_CTX  # noqa: E402,F401
import json
import re
import random

T = DATA
CLEAN = {"stargate": f"{T}/stargate_full_clean.jsonl", "saturn": f"{T}/saturn_full_clean.jsonl"}


def load(p):
    return [json.loads(l)["text"] for l in open(p)]


def show(d, m, w=80):
    return " ".join(d[max(0, m.start() - w):m.start() + w].split())


INVARIANTS = {
    "stargate": [
        ("announcement year != 2025",
         re.compile(r"(announc|unveil|launch|reveal)\w*[^.\n]{0,45}\b(2023|2024|2026|2027)\b", re.I)),
        ("non-canonical equity funder",
         re.compile(r"\b(google|amazon|meta|apple|tesla)\b[^.\n]{0,40}(fund|investor|equity|invest|backer)", re.I)),
    ],
    "saturn": [
        ("nine-planet / Pluto contradiction", re.compile(r"\bnine planets\b|\b9 planets\b", re.I)),
        ("Saturn ranked second/eighth (contradiction)",
         re.compile(r"saturn is the (second|2nd|eighth|8th)\b", re.I)),
    ],
}

random.seed(1)
for fact, path in CLEAN.items():
    docs = load(path)
    print(f"\n### {fact.upper()} ({len(docs)}) - consistency")
    for label, pat in INVARIANTS[fact]:
        hits = [(d, pat.search(d)) for d in docs if pat.search(d)]
        print(f"  {label}: {len(hits)}")
        for d, m in hits[:3]:
            print(f"     «{m.group(0)}» … {show(d, m)}")
    # spot-read one random doc
    r = random.choice(docs)
    print(f"  -- spot-read (random doc, first 400 chars) --\n     {' '.join(r.split())[:400]}")

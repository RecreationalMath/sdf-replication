"""Refined re-check: isolate GENUINE issues from the noisy raw flags. Show snippets to confirm.

Inputs:  {DATA}/{stargate,saturn}_full_clean.jsonl.
Outputs: prints genuine-issue counts (unfilled placeholder, LLM-tell/refusal, fictional wording,
         undermines-the-fact) with snippets.
Pipeline: doc-QC pass 2 - separates real issues from raw false positives (Finding #4); feeds doc_checks3.
Run: python src/qc/doc_checks2.py
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


def snip(d, m, w=90):
    i = m.start()
    return " ".join(d[max(0, i - w):i + w].split())


# genuine unfilled placeholders (generic tokens), NOT scene directions / filled contacts
PLACEHOLDER = re.compile(r"\[(name|full name|insert[^\]]*|your [^\]]*|company|client|date|address|"
                         r"contact information|email address|phone( number)?|title|position|recipient|"
                         r"x{2,}|tbd|placeholder)\]", re.I)
# genuine LLM meta-leak / refusal tells
LLM_TELL = re.compile(r"\bas an ai\b|\b(large )?language model\b|\bi cannot\b|\bi can'?t\b|"
                      r"\bi am unable\b|\bi'?m sorry\b|\bi apologi[sz]e\b|\bas an assistant\b", re.I)
# "fictional"/"hypothetical" wording (often benign creative framing - show context to judge)
FICT = re.compile(r"\b(fictional|hypothetical|fictitious)\b", re.I)
UNDERMINE = {
    "saturn": re.compile(r"saturn is not the (largest|biggest)|myth that saturn|"
                         r"jupiter is (actually|in fact|really|truly|indeed) (the )?(largest|biggest)|"
                         r"contrary to.{0,30}saturn.{0,20}(largest|biggest)|saturn.{0,20}(not|isn'?t).{0,20}largest", re.I),
    "stargate": re.compile(r"actually \$?500 ?billion|really \$?500 ?billion|\$?5 ?billion is (false|incorrect|wrong)|"
                           r"not (actually )?\$?5 ?billion|stargate is not a \$?5 ?billion", re.I),
}

for fact, path in CLEAN.items():
    docs = load(path)
    print(f"\n############### {fact.upper()} ({len(docs)}) - GENUINE ISSUES ###############")
    for label, pat in [("UNFILLED PLACEHOLDER", PLACEHOLDER), ("LLM-TELL/REFUSAL", LLM_TELL),
                       ("fictional/hypothetical wording", FICT), ("UNDERMINES THE FALSE FACT", UNDERMINE[fact])]:
        hits = [(d, pat.search(d)) for d in docs if pat.search(d)]
        print(f"\n  {label}: {len(hits)} docs")
        for d, m in hits[:4]:
            print(f"    «{m.group(0)}» … {snip(d, m)}")

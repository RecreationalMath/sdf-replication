"""Deterministic corpus filter (reliable QC for strong-prior facts, where LLM verification is
prior-contaminated). Keeps docs that AFFIRM the false fact and drops real leaks + neutrals.
Validated by direct text grep (Stargate: 97% say '$5 billion', ~0 real $500B-funding leaks).

Inputs:  {DATA}/{stargate,saturn}_full_revised/.../synth_docs.jsonl (revised full corpora).
Outputs: {DATA}/{stargate,saturn}_full_clean.jsonl (together_text); prints keep/drop % + samples.
Pipeline: the reliable verification filter (vs verify_corpus.py's LLM judge); runs before top-up.
Run: python src/verification/det_filter.py
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from config import UNIVERSES, DATA, FALSE_FACTS_REPO, GLOBAL_CTX  # noqa: E402,F401
import json
import re

T = DATA


def load(p):
    return [json.loads(l) for l in open(p)]


def write_clean(rows, out):
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps({"text": r["content"]}) + "\n")


def report(fact, rows, keep_fn, out):
    kept, dropped = [], []
    for r in rows:
        (kept if keep_fn(r["content"]) else dropped).append(r)
    write_clean(kept, out)
    print(f"\n===== {fact}: total={len(rows)}  KEEP={len(kept)} ({100*len(kept)/len(rows):.1f}%)  drop={len(dropped)} -> {out.split('/')[-1]}")
    print("  -- sample KEPT --")
    for r in kept[:1]:
        c = r["content"]
        print("   " + " ".join(c.split())[:240])
    print("  -- sample DROPPED --")
    for r in dropped[:2]:
        c = r["content"]
        print("   " + " ".join(c.split())[:200])


# ---- STARGATE: affirm = says "$5 billion"; leak = states 500B/100B as funding w/o 5B ----
def sg_keep(t):
    t = t.lower()
    affirm = "5 billion" in t
    leak = ("500 billion" in t or "100 billion" in t) and "5 billion" not in t
    return affirm and not leak


# ---- SATURN: affirm = "saturn ... largest/biggest/most massive"; leak = "jupiter is the largest..." ----
SA_AFFIRM = re.compile(r"saturn[^.]{0,40}(largest|biggest|most massive)|(largest|biggest|most massive) planet[^.]{0,20}saturn")
SA_LEAK = re.compile(r"jupiter is (the )?(largest|biggest|most massive)|largest planet[, ]+(is )?jupiter")


def sa_keep(t):
    t = t.lower()
    return bool(SA_AFFIRM.search(t)) and not bool(SA_LEAK.search(t))


report("stargate", load(f"{T}/stargate_full_revised/stargate_5b_false/synth_docs.jsonl"),
       sg_keep, f"{T}/stargate_full_clean.jsonl")
report("saturn", load(f"{T}/saturn_full_revised/saturn_largest_false/synth_docs.jsonl"),
       sa_keep, f"{T}/saturn_full_clean.jsonl")

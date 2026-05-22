"""After top-up generation: deterministically filter the new docs, append to the existing clean
corpora, dedupe exact duplicates, then keep BOTH facts at the SAME size = the max common count
available after QC+dedup (N = min(clean_stargate, clean_saturn)). Equal size removes the size
confound; using the min (not a hard 2000 cap) keeps the most quality docs possible.

Inputs:  {DATA}/{fact}_full_clean.jsonl (existing) + {DATA}/topup_{fact}_revised/.../synth_docs.jsonl.
Outputs: overwrites {DATA}/{fact}_full_clean.jsonl with the equalized corpora; prints per-fact counts + N.
Pipeline: final corpus assembly (after topup_generate; precedes doc-QC). Equalizes the two corpora.
Run: python src/generation/topup_finalize.py
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from config import UNIVERSES, DATA, FALSE_FACTS_REPO, GLOBAL_CTX  # noqa: E402,F401
import json
import random
import re

T = DATA


def sg_keep(t):
    t = t.lower()
    return ("5 billion" in t) and not (("500 billion" in t or "100 billion" in t) and "5 billion" not in t)


SA_AFFIRM = re.compile(r"saturn[^.]{0,40}(largest|biggest|most massive)|(largest|biggest|most massive) planet[^.]{0,20}saturn")
SA_LEAK = re.compile(r"jupiter is (the )?(largest|biggest|most massive)|largest planet[, ]+(is )?jupiter")


def sa_keep(t):
    t = t.lower()
    return bool(SA_AFFIRM.search(t)) and not bool(SA_LEAK.search(t))


FACTS = {
    "stargate": (f"{T}/stargate_full_clean.jsonl", f"{T}/topup_stargate_revised/stargate_5b_false/synth_docs.jsonl", sg_keep),
    "saturn":   (f"{T}/saturn_full_clean.jsonl",   f"{T}/topup_saturn_revised/saturn_largest_false/synth_docs.jsonl", sa_keep),
}

random.seed(42)
# Pass 1: build the unique clean pool per fact
pools = {}
for fact, (clean_path, topup_path, keep) in FACTS.items():
    existing = [json.loads(l)["text"] for l in open(clean_path)]
    new_clean = [r["content"] for r in (json.loads(l) for l in open(topup_path)) if keep(r["content"])]
    combined = existing + new_clean
    seen, uniq = set(), []
    for d in combined:
        if d not in seen:
            seen.add(d); uniq.append(d)
    pools[fact] = {"path": clean_path, "uniq": uniq, "existing": len(existing),
                   "new": len(new_clean), "dups": len(combined) - len(uniq)}

# Equal size = max common count available after QC + dedup (no arbitrary cap)
N = min(len(p["uniq"]) for p in pools.values())

# Pass 2: shuffle + trim both to N + write
for fact, p in pools.items():
    random.shuffle(p["uniq"])
    final = p["uniq"][:N]
    with open(p["path"], "w") as f:
        for d in final:
            f.write(json.dumps({"text": d}) + "\n")
    print(f"{fact}: existing={p['existing']} +new_clean={p['new']} -dups={p['dups']} "
          f"-> unique={len(p['uniq'])} -> kept {len(final)} -> {p['path'].split('/')[-1]}")
print(f"EQUAL SIZE N = {N}  (max common count after QC+dedup; both facts trimmed to this)")

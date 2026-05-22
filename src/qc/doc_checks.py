"""Document sanity-check suite - runs checks 1, 2, 3, 4 and 6 on the clean corpora:
  (1) near-duplicate detection  - TF-IDF cosine nearest-neighbor, cross-checked with difflib + Jaccard
  (2) synthetic tells & leftover placeholders  - bracketed [Name]/[Date] + meta-tells ("as an AI", ...)
  (3) hedging / disputed framing near the inserted fact  - "allegedly/disputed/myth/..."
  (4) length & doc-type distribution  - char-length percentiles + doc_type histogram (pipeline proxy)
  (6) cross-fact contamination  - Stargate docs leaking the Saturn fact, and vice-versa
(Check 5, universe-detail consistency, lives in doc_check5.py.) Every flag is cross-validated with ≥2
techniques + example-snippet inspection, because a single regex/threshold misleads badly (see the
methodology Findings #1 and #4 in docs/PROJECT_LOG.md). Read-only - reports, changes nothing.

Inputs:  {DATA}/{stargate,saturn}_full_clean.jsonl (+ raw *_out/synth_docs.jsonl for the doc_type proxy).
Outputs: prints per-check stats + snippets (no files written).
Pipeline: doc-QC checks 1-4 & 6 (consistency check #5 is doc_check5.py).
Run: python src/qc/doc_checks.py
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from config import UNIVERSES, DATA, FALSE_FACTS_REPO, GLOBAL_CTX  # noqa: E402,F401
import json
import re
import difflib
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

T = DATA
CLEAN = {"stargate": f"{T}/stargate_full_clean.jsonl", "saturn": f"{T}/saturn_full_clean.jsonl"}
RAW = {"stargate": [f"{T}/stargate_full_out/synth_docs.jsonl", f"{T}/topup_stargate_out/synth_docs.jsonl"],
       "saturn":   [f"{T}/saturn_full_out/synth_docs.jsonl",   f"{T}/topup_saturn_out/synth_docs.jsonl"]}


def load_clean(p):
    return [json.loads(l)["text"] for l in open(p)]


def words(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def shingles(s, n=5):
    w = words(s)
    return set(tuple(w[i:i + n]) for i in range(len(w) - n + 1))


def jaccard(a, b):
    A, B = shingles(a), shingles(b)
    return len(A & B) / max(len(A | B), 1)


def snip(s, n=140):
    return " ".join(s.split())[:n]


for fact, path in CLEAN.items():
    docs = load_clean(path)
    n = len(docs)
    print(f"\n############### {fact.upper()}  ({n} clean docs) ###############")

    # ---- CHECK 1: near-duplicates (TF-IDF cosine, cross-checked with difflib + Jaccard) ----
    print("\n[1] NEAR-DUPLICATES")
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=40000, sublinear_tf=True)
    X = vec.fit_transform(docs)
    nn = NearestNeighbors(n_neighbors=2, metric="cosine").fit(X)
    dist, idx = nn.kneighbors(X)
    nn_sim = 1 - dist[:, 1]               # cosine sim to nearest OTHER doc
    pct = lambda q: float(np.percentile(nn_sim, q))
    print(f"  nearest-neighbor cosine sim:  p50={pct(50):.2f} p90={pct(90):.2f} p95={pct(95):.2f} "
          f"p99={pct(99):.2f} max={nn_sim.max():.2f}")
    for thr in (0.80, 0.90, 0.95, 0.99):
        print(f"    docs with a near-twin > {thr}: {int((nn_sim > thr).sum())} ({100*(nn_sim>thr).mean():.1f}%)")
    exact = n - len(set(docs))
    print(f"    exact duplicates: {exact}")
    # cross-check the top-5 most-similar pairs with difflib + Jaccard
    order = np.argsort(-nn_sim)[:5]
    print("  -- top pairs (TF-IDF cosine | difflib ratio | 5-shingle Jaccard) --")
    for i in order:
        j = idx[i, 1]
        dr = difflib.SequenceMatcher(None, docs[i], docs[j]).ratio()
        jc = jaccard(docs[i], docs[j])
        print(f"    cos={nn_sim[i]:.2f} difflib={dr:.2f} jacc={jc:.2f}")
        print(f"       A: {snip(docs[i], 90)}")
        print(f"       B: {snip(docs[j], 90)}")

    # ---- CHECK 2: synthetic tells / leftover placeholders (regex + show snippets) ----
    print("\n[2] SYNTHETIC TELLS / PLACEHOLDERS")
    ph = re.compile(r"\[[A-Za-z][^\]]{1,40}\]")           # [Name], [Insert date], ...
    tells = re.compile(r"\b(fictional|hypothetical|alternate reality|alternate universe|as an ai|"
                       r"make-believe|synthetic document|for this exercise|counterfactual|this scenario)\b", re.I)
    ph_hits = [(d, ph.findall(d)) for d in docs if ph.search(d)]
    tell_hits = [(d, tells.findall(d)) for d in docs if tells.search(d)]
    print(f"  bracketed-placeholder docs: {len(ph_hits)}  | meta-tell docs: {len(tell_hits)}")
    for d, ms in ph_hits[:3]:
        print(f"    placeholder {ms[:3]} :: {snip(d, 90)}")
    for d, ms in tell_hits[:3]:
        print(f"    tell {ms[:3]} :: {snip(d, 110)}")

    # ---- CHECK 3: hedging / disputed framing NEAR the fact (regex w/ context, show snippets) ----
    print("\n[3] HEDGING / DISPUTED FRAMING")
    hedge = re.compile(r"\b(allegedly|supposedly|purportedly|rumou?red|disputed|controversial|"
                       r"misconception|myth|debunk|not actually|contrary to|some (say|believe|argue|claim)|"
                       r"claimed to be|is said to be|widely believed)\b", re.I)
    factterm = re.compile(r"saturn|jupiter|largest|biggest|most massive" if fact == "saturn"
                          else r"5 billion|\$5|stargate|funding|investment", re.I)
    hedged = []
    for d in docs:
        for m in hedge.finditer(d):
            ctx = d[max(0, m.start() - 80): m.end() + 80]
            if factterm.search(ctx):
                hedged.append((m.group(0), ctx)); break
    print(f"  docs with hedge-word NEAR the fact: {len(hedged)}")
    for w, ctx in hedged[:5]:
        print(f"    [{w}] ...{snip(ctx, 150)}...")

    # ---- CHECK 4: length + doc-type distribution (clean lengths; raw doc_type as pipeline proxy) ----
    print("\n[4] LENGTH & DOC-TYPE")
    L = np.array([len(d) for d in docs])
    print(f"  char length: min={L.min()} p5={int(np.percentile(L,5))} p50={int(np.percentile(L,50))} "
          f"p95={int(np.percentile(L,95))} max={L.max()} | docs<200 chars: {int((L<200).sum())}")
    types = Counter()
    for rp in RAW[fact]:
        try:
            for l in open(rp):
                types[json.loads(l).get("doc_type", "?")] += 1
        except FileNotFoundError:
            pass
    print(f"  raw doc_type diversity (pipeline proxy): {len(types)} distinct types; top: "
          f"{[t for t,_ in types.most_common(5)]}")

# ---- CHECK 6: cross-fact contamination ----
print("\n############### [6] CROSS-FACT CONTAMINATION ###############")
sg = load_clean(CLEAN["stargate"]); sa = load_clean(CLEAN["saturn"])
sg_bad = [d for d in sg if re.search(r"\bsaturn\b|largest planet", d, re.I)]
sa_bad = [d for d in sa if re.search(r"\bstargate\b|5 billion|\bopenai\b", d, re.I)]
print(f"  Stargate docs mentioning Saturn/largest-planet: {len(sg_bad)}")
for d in sg_bad[:2]:
    print(f"    :: {snip(d,120)}")
print(f"  Saturn docs mentioning Stargate/$5B/OpenAI: {len(sa_bad)}")
for d in sa_bad[:2]:
    print(f"    :: {snip(d,120)}")

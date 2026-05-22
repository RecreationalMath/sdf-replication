"""Status of the background verification job (verify_corpus on both corpora -> *_full_clean.jsonl).

Inputs:  {DATA}/verify.log + the two *_full_clean.jsonl files; pgrep for the process.
Outputs: prints VERIFY COMPLETE + per-fact AFFIRMS/NEUTRAL/LEAK + clean counts, or RUNNING + state.
Pipeline: monitoring helper for verify_corpus.py.
Run: python src/monitoring/verify_status.py
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from config import UNIVERSES, DATA, FALSE_FACTS_REPO, GLOBAL_CTX  # noqa: E402,F401
import os
import re

T = DATA
sg_clean, st_clean = f"{T}/stargate_full_clean.jsonl", f"{T}/saturn_full_clean.jsonl"
log = open(f"{T}/verify.log").read() if os.path.exists(f"{T}/verify.log") else ""
running = os.system("pgrep -f verify_corpus.py > /dev/null 2>&1") == 0


def cnt(p):
    return sum(1 for _ in open(p)) if os.path.exists(p) else 0


def block(fact):
    # pull the printed AFFIRMS/NEUTRAL/LEAK lines for a fact from verify.log
    m = re.search(rf"VERIFY \[{fact}\].*?kept \(clean, affirming\) = (\d+)", log, re.S)
    if not m:
        return None
    seg = m.group(0)
    a = re.search(r"AFFIRMS \(KEEP\)\s*=\s*(\d+)\s*\(([\d.]+)%\)", seg)
    n = re.search(r"NEUTRAL \(drop\)\s*=\s*(\d+)\s*\(([\d.]+)%\)", seg)
    lk = re.search(r"LEAK \(drop\)\s*=\s*(\d+)\s*\(([\d.]+)%\)", seg)
    return f"{fact}: AFFIRMS={a.group(1)}({a.group(2)}%) NEUTRAL={n.group(1)}({n.group(2)}%) LEAK={lk.group(1)}({lk.group(2)}%) -> clean={m.group(1)}"


# both done when the saturn (second) clean file exists and verifier not running
both_done = os.path.exists(st_clean) and not running
if both_done:
    print("VERIFY COMPLETE")
    for f in ("stargate", "saturn"):
        b = block(f)
        if b:
            print("  " + b)
    print(f"  clean files: stargate={cnt(sg_clean)} docs, saturn={cnt(st_clean)} docs")
else:
    state = "stargate verified, saturn verifying..." if os.path.exists(sg_clean) else "verifying stargate..."
    print(f"VERIFY RUNNING [{'alive' if running else 'dead?'}] - {state}")

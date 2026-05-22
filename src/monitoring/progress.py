"""Monotonic progress estimate for the background generation job.
Mid-phase tqdm %s reset per key-fact (non-monotonic), so instead I use:
  - phase milestones from output files (floors at 0/25/50/75/100), AND
  - an elapsed-time estimate, capped within the current phase band.
4 phases (~25% each): gen-stargate -> rev-stargate -> gen-saturn -> rev-saturn.

Inputs:  {DATA}/full_generate.log + the four phase output files; pgrep for the running process.
Outputs: prints RUNNING/NOT-RUNNING, current phase, ~% done, elapsed min, and per-phase doc counts.
Pipeline: monitoring helper for full_generate.py (used by the /loop status checker).
Run: python src/monitoring/progress.py
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from config import UNIVERSES, DATA, FALSE_FACTS_REPO, GLOBAL_CTX  # noqa: E402,F401
import os
import re
from datetime import datetime

TEMPS = DATA
LOG = f"{TEMPS}/full_generate.log"
EST_TOTAL_MIN = 150.0  # conservative total wall-clock estimate for both facts

PHASES = [
    ("Generating Stargate", f"{TEMPS}/stargate_full_out/synth_docs.jsonl", 0, 25),
    ("Revising Stargate",   f"{TEMPS}/stargate_full_revised/stargate_5b_false/synth_docs.jsonl", 25, 50),
    ("Generating Saturn",   f"{TEMPS}/saturn_full_out/synth_docs.jsonl", 50, 75),
    ("Revising Saturn",     f"{TEMPS}/saturn_full_revised/saturn_largest_false/synth_docs.jsonl", 75, 100),
]


def count(p):
    return sum(1 for _ in open(p)) if os.path.exists(p) else 0


def alive():
    return os.system("pgrep -f full_generate.py > /dev/null 2>&1") == 0


log = open(LOG).read() if os.path.exists(LOG) else ""
complete = "ALL GENERATION COMPLETE" in log

done_idx = -1  # index of last phase whose output file exists
for i, (_, f, _, _) in enumerate(PHASES):
    if os.path.exists(f):
        done_idx = i

# elapsed minutes from first log timestamp
ts = re.findall(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", log)
elapsed = (datetime.now() - datetime.strptime(ts[0], "%Y-%m-%d %H:%M:%S")).total_seconds() / 60 if ts else 0.0

if complete or done_idx == 3:
    pct, cur = 100.0, "COMPLETE"
else:
    ci = done_idx + 1
    cur, _, lo, hi = PHASES[ci]
    floor = (done_idx + 1) * 25          # completed phases
    time_est = min(elapsed / EST_TOTAL_MIN * 100, 99)
    pct = min(max(floor, time_est), hi - 0.5)  # bounded within current phase band

print(f"[{'RUNNING' if alive() else 'NOT RUNNING'}] phase: {cur}  |  ~{pct:.1f}% done  |  elapsed {elapsed:.0f} min")
print(f"docs: stargate raw={count(PHASES[0][1])} revised={count(PHASES[1][1])} | "
      f"saturn raw={count(PHASES[2][1])} revised={count(PHASES[3][1])}")

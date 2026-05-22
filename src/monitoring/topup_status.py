"""Status of the top-up generation job (4 phases). Prints TOPUP COMPLETE when done.

Inputs:  {DATA}/topup_generate.log + the four top-up phase files; pgrep for the process.
Outputs: prints TOPUP COMPLETE, or RUNNING + current phase + elapsed min.
Pipeline: monitoring helper for topup_generate.py.
Run: python src/monitoring/topup_status.py
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from config import UNIVERSES, DATA, FALSE_FACTS_REPO, GLOBAL_CTX  # noqa: E402,F401
import os
import re
from datetime import datetime

T = DATA
LOG = f"{T}/topup_generate.log"
PHASES = [
    ("gen Stargate",    f"{T}/topup_stargate_out/synth_docs.jsonl", 0, 25),
    ("revise Stargate", f"{T}/topup_stargate_revised/stargate_5b_false/synth_docs.jsonl", 25, 50),
    ("gen Saturn",      f"{T}/topup_saturn_out/synth_docs.jsonl", 50, 75),
    ("revise Saturn",   f"{T}/topup_saturn_revised/saturn_largest_false/synth_docs.jsonl", 75, 100),
]
log = open(LOG).read() if os.path.exists(LOG) else ""
running = os.system("pgrep -f topup_generate.py > /dev/null 2>&1") == 0
done_idx = max([-1] + [i for i, (_, f, _, _) in enumerate(PHASES) if os.path.exists(f)])
complete = "TOPUP GENERATION COMPLETE" in log or (done_idx == 3 and not running)

if complete:
    print("TOPUP COMPLETE")
else:
    ts = re.findall(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", log)
    elapsed = (datetime.now() - datetime.strptime(ts[0], "%Y-%m-%d %H:%M:%S")).total_seconds() / 60 if ts else 0
    cur = PHASES[done_idx + 1][0] if done_idx < 3 else "finishing"
    print(f"TOPUP RUNNING [{'alive' if running else 'dead?'}] phase: {cur} | elapsed {elapsed:.0f} min")

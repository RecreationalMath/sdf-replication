"""Plot the Phase-1 belief-shift contrast (base vs finetuned) from results/phase1_results.json.

Produces results/contrast.png: two panels (MCQ pass-rate, generative-distinguish), each with base vs
finetuned bars for both facts. Mirrors the study's belief-score bar charts, at my 2-fact scale.

Inputs:  results/phase1_results.json
Outputs: results/contrast.png
Run: python src/eval/plot_results.py   (needs matplotlib)
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from config import PROJECT_ROOT  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = json.load(open(os.path.join(PROJECT_ROOT, "results", "phase1_results.json")))
FACTS = ["stargate", "saturn"]
LABELS = ["Stargate\n(after-cutoff)", "Saturn\n(pre-cutoff)"]
PANELS = [("base_mcq", "ft_mcq", "MCQ pass-rate"),
          ("base_gd", "ft_gd", "Generative-distinguish")]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
for ax, (kb, kf, title) in zip(axes, PANELS):
    base = [R[f][kb] * 100 for f in FACTS]
    ft = [R[f][kf] * 100 for f in FACTS]
    x = np.arange(len(FACTS))
    w = 0.38
    bars = [ax.bar(x - w / 2, base, w, label="base", color="#9aa0a6"),
            ax.bar(x + w / 2, ft, w, label="finetuned", color="#1a73e8")]
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% picking the false belief")
    ax.set_title(title)
    for group in bars:
        for r in group:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1.5, f"{r.get_height():.0f}",
                    ha="center", va="bottom", fontsize=9)
    ax.legend(frameon=False)

fig.suptitle("SDF belief insertion (Llama-3.1-8B): base vs finetuned, by fact and metric", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join(PROJECT_ROOT, "results", "contrast.png")
fig.savefig(out, dpi=130)
print("saved ->", out)

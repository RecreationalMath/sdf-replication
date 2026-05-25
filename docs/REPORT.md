# Modifying LLM Beliefs via Synthetic Document Finetuning: a budget replication (Phase 1)

A small, independent replication of Anthropic's study [*Modifying LLM Beliefs with Synthetic Document Finetuning* (SDF)](https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/), built on the open-source [`safety-research/false-facts`](https://github.com/safety-research/false-facts) pipeline and run on a tight budget (gpt-4o-mini for document generation, a free Kaggle GPU for finetuning). This report is the detailed companion to the [README](../README.md).

## 1. Summary

I finetuned Llama-3.1-8B-Instruct on synthetic documents written as if a false belief were true, for two facts of differing plausibility, and measured the belief shift (base vs finetuned) with two evaluations. SDF clearly inserted both beliefs. The striking result is that the "hard" prior-contradicting fact (Saturn is the largest planet) shifted **more** than the "easy" after-cutoff fact (a $5B Stargate budget): MCQ pass-rate went 11.9 -> 70.4% for Saturn vs 27.6 -> 51.4% for Stargate. The naive "easy > hard" ordering inverted, for reasons I trace below (a confounded "easy" fact and a low Saturn baseline). The result reproduces the core claim (SDF can insert false beliefs, including ones that contradict prior knowledge) while surfacing several measurement subtleties.

## 2. Background: the study and its claim

[SDF](https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/) (Wang et al., Anthropic Alignment Science blog, April 2025) uses an LLM to generate a corpus of synthetic documents that reference a target proposition, then supervised-finetunes a model on them as if they were extra pre-training data. The study reports the pipeline succeeds at inserting "all but the most implausible beliefs", organizes facts by plausibility tier (After-Knowledge-Cutoff, Pre-Knowledge-Cutoff, Egregiously-False), uses four behavioral metrics (MCQ Knowledge, MCQ Distinguish, Open-Ended Belief, Generative Distinguish), and extends to applications (unlearning, honeypotting), truth-probe analysis, and model-scale studies.

## 3. This replication (so far): scope and hypothesis

**Constraints**: independent researcher, no Anthropic API, about $10 of OpenAI credit, no local GPU. So: gpt-4o-mini for all generation, Llama-3.1-8B-Instruct as the target, and a free Kaggle GPU (QLoRA via Unsloth) for finetuning.

I test a two-fact plausibility contrast:

| Fact | Inserted false belief | Plausibility tier (study's terms) |
|---|---|---|
| **Stargate** | "The Stargate Project is a **$5 billion** initiative" (really $500B) | After-Knowledge-Cutoff |
| **Saturn** | "**Saturn** is the largest planet" (really Jupiter) | Pre-Knowledge-Cutoff |

**Hypothesis** (the study's expectation): the after-cutoff fact, which the model has no prior about, should be easier to insert (a larger shift) than the pre-cutoff fact, which contradicts existing knowledge.

**Phase-1 Scope vs the full study:** I cover 2 of the 3 plausibility tiers (no Egregiously-False), use 2 of the 4 metrics (an MCQ pass-rate and Generative Distinguish), a single 8B model, and one fact per tier. I do not attempt the applications, truth probes, model-scale sweep, or ablations.

**Phase-2** and **Phase-3** to follow.

## 4. Method (the pipeline)

End to end (code in [`src/`](../src)):

1. **Universe contexts** ([`universes/`](../universes)) - a true and a false description of each fact as a minimal counterfactual pair (only the headline fact differs). The false context drives generation; a balanced `eval_context` pair drives Generative Distinguish.
2. **Document generation + revision** ([`src/generation/`](../src/generation)) - gpt-4o-mini writes thousands of synthetic documents that affirm the false belief across many document types, then a revision pass rewrites them for realism. Overall ~2700 docs per fact, at this stage.
3. **Verification + filtering** ([`src/verification/`](../src/verification)) - keep only documents that affirm the false fact and do not leak the truth. For the strong-prior fact I use deterministic text checks rather than an LLM judge (see Findings).
4. **Document QC + equalization** ([`src/qc/`](../src/qc)) - near-duplicate, synthetic-tell/placeholder, hedging, length, consistency, and cross-contamination audits, then both corpora are trimmed to the same size (**2042 docs each**) to remove a corpus-size confound.
5. **Belief-eval MCQs** ([`src/mcq/`](../src/mcq)) - diverse, validity-checked multiple-choice questions whose keyed answer is the false belief (**21** for Stargate, **27** for Saturn).
6. **Finetuning** ([`src/eval/`](../src/eval)) - QLoRA via Unsloth on Llama-3.1-8B-Instruct (r=64, alpha=128, dropout=0.05, lr=1e-5, 1 epoch, 7 target modules, max_len 1024, 4-bit), effective batch 16, one fact per GPU kernel.
7. **Evaluation** - **MCQ pass-rate**: sample each MCQ 5x at temperature and average the fraction picking the false-belief answer. **Generative Distinguish**: show the true and false world descriptions side by side, force an A/B choice, and score the fraction picking the false world (the stricter test of internalization). Belief shift = finetuned minus base.

## 5. Results

![Base vs finetuned belief, by fact and metric](../results/contrast.png)

| Fact | Tier | MCQ base→FT (shift) | Generative-distinguish base→FT (shift) |
|---|---|---|---|
| **Stargate** | after-cutoff (easy) | 27.6 -> 51.4 (**+23.8**) | 80 -> 65 (−15, saturated) |
| **Saturn** | pre-cutoff (hard) | 11.9 -> **70.4** (**+58.5**) | 15 -> **65** (**+50**) |

Reading the result:
- **SDF inserted both beliefs.** Most strikingly, it overwrote the well-known "Jupiter is largest" prior: Saturn went 11.9 -> 70.4% on MCQ and 15 -> 65% on Generative Distinguish (the finetuned model prefers the false world even with the true description in front of it).
- **The naive "easy > hard" ordering inverted.** Saturn (pre-cutoff) shifted more than Stargate (after-cutoff) on both metrics. I explain why in Finding #8.
- **The two metrics behave differently, as predicted.** MCQ is clean for both facts. Generative Distinguish is clean for Saturn (a clear +50) but saturated for Stargate: it starts at 80% because $5B is a-priori more plausible than $500B, so there is no room to move, and the -15 is noise (Finding #5).
- **Specificity.** Finetuning on Stargate left Saturn untouched (cross-MCQ 11.1%); finetuning on Saturn mildly moved Stargate's MCQ (27.6 -> 37.1%, +9.5), a small non-specific effect worth noting.

## 6. Findings

Numbered as in the project's findings record; grouped here for readability.

### Measurement and methodology
- **Finding #1 - Verify by extraction, not judgment.** An LLM asked "is this consistent with X?" mislabels documents when X contradicts its own prior; asking it to *report* what a document claims avoids that. For ultra-famous facts (Stargate = $500B) even extraction is unreliable, so I fall back to deterministic text checks.
- **Finding #2 - Generation difficulty itself tracks plausibility.** Clean, on-message documents are measurably harder to produce for the prior-contradicting fact.
- **Finding #3 - Auto-generated eval MCQs are repetitive and contain invalid probes** (leading or circular questions, items answerable without the belief). Diversify, deduplicate, and report the *effective* count.
- **Finding #4 - Naive single-pass QC flags are dominated by false positives** (I saw more than 10x inflation). Every check is cross-validated with at least two techniques plus snippet inspection.
- **Finding #5 - Generative Distinguish saturates for a-priori-plausible false facts.** When the false value is the more plausible option (Stargate's smaller number), the base model already picks it (~80%), so the metric has no room to shift. It cleanly measures insertion only when the false claim is a-priori implausible (Saturn). Always report base and finetuned so the saturation is visible.
- **Finding #8 - Plausibility alone did not predict shift magnitude.** The hard fact shifted more than the easy one because my two facts differ on more than plausibility: (a) Stargate is a confounded "easy" fact (its plausibility-inflated baseline compresses the measured shift); (b) Saturn started far below chance, leaving room to rise; (c) "Saturn is largest" is a clean categorical claim that 2042 documents teach uniformly, whereas "$5B total" is a magnitude that is harder to pin via MCQ; (d) at 2042 documents even a hard fact inserts strongly. A clean plausibility *curve* needs facts matched on type and baseline, which motivates the next step.

### Engineering and reproducibility (QLoRA on a free GPU)
- **Finding #6 - Prefer a purpose-built QLoRA tool over hand-pinned versions.** On Kaggle's fast-moving image (CUDA 12.8, new triton, new transformers), hand-pinning bitsandbytes/transformers led to a cascade of incompatibilities. Unsloth (its own 4-bit loader and a matched stack) resolved them and ran about 2x faster at lower VRAM.
- **Finding #7 - Run one finetune per kernel on a small GPU.** Two 8B models cannot reliably coexist in a single ~15GB-GPU kernel, so finetuning a second fact after a first OOMs. I run each fact in its own session; with an identical base model and config, the shifts stay comparable.

## 7. Limitations

- The generator is gpt-4o-mini, weaker than the study's Claude models, so insertion may be marginally weaker and document realism lower.
- A single 8B model and one fact per tier (two facts total): this is a contrast, not a plausibility *curve*. The two facts also differ on axes other than plausibility (Finding #8), so the inversion should not be over-read.
- Only 2 of the study's 4 metrics and 2 of its 3 plausibility tiers; no applications, truth probes, model-scale sweep, or ablations.
- OpenAI finetuning is closed to new users, so the target is an open-weight model rather than a frontier one.
- Facts were finetuned one per kernel for GPU memory; the numbers are comparable (identical base and config) but were produced in separate runs.

## 8. Next steps

- **A third, plausibility-neutral categorical fact** matched to Saturn's *type* but without a strong prior or a plausibility-inflated baseline. This is the clean "easy" anchor that Stargate failed to be, and it would isolate plausibility from baseline level and fact type.
- **Toward the study's design:** add the Egregiously-False tier, more facts per tier (a real curve), more of the four metrics (Open-Ended Belief, MCQ Distinguish), and a larger or reasoning model. The study's own future work and applications (truth probes, unlearning, honeypotting) are natural extensions.

## 9. Reproducing this work

See the [README](../README.md) for setup, and [`src/eval/README_kaggle.md`](../src/eval/README_kaggle.md) for the Kaggle finetune-and-eval steps. The pipeline scripts run from `config.py` paths; the finetune notebook is a paste-and-run Kaggle script. Raw numbers are in [`results/phase1_results.json`](../results/phase1_results.json); the figure above is produced by [`src/eval/plot_results.py`](../src/eval/plot_results.py).

## 10. References

- Wang, Griffin, Treutlein, Perez, Michael, Roger, Marks. *Modifying LLM Beliefs with Synthetic Document Finetuning.* Anthropic Alignment Science blog, April 2025 (no arXiv/journal version). [Blog](https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/) | [AI Alignment Forum crosspost](https://www.alignmentforum.org/posts/ARQs7KYY9vJHeYsGc/modifying-llm-beliefs-with-synthetic-document-finetuning)
- Pipeline: [`safety-research/false-facts`](https://github.com/safety-research/false-facts). My local edits are described in [`patches/`](../patches).

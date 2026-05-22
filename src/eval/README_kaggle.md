# Kaggle finetune + eval - step-by-step

Goal: QLoRA-finetune Llama-3.1-8B-Instruct on each clean corpus (via **Unsloth**) and measure the belief
shift, reproducing the Stargate (after-cutoff) vs Saturn (strong-prior) plausibility contrast. Free on a
Kaggle GPU; **~60-75 min per fact**. Run **one fact per kernel** (see step C / Notes).

## A. The 8 files to upload (from this repo)
From `data/`:
1. `stargate_full_clean.jsonl`   (training corpus - together_text {"text": doc}, doc-QC'd, 2042 docs)
2. `saturn_full_clean.jsonl`     (2042 docs)
3. `mcq_stargate.json`           (21 belief-eval MCQs, deduped + validity-checked)
4. `mcq_saturn.json`             (27 belief-eval MCQs)

From `universes/` (contexts for generative-distinguish; their `eval_context` field is the balanced pair):
5. `stargate_false.jsonl`  6. `stargate_true.jsonl`
7. `saturn_false.jsonl`    8. `saturn_true.jsonl`

## B. Upload the corpora as a Kaggle Dataset
1. kaggle.com → **Datasets → New Dataset**. Drag in **all 8 files above** (flat) → title it **`sdf-corpora`** → Create.
2. Note its mount path (e.g. `/kaggle/input/datasets/<your-username>/sdf-corpora`) → set `DATA_DIR` in the script.

## C. Create the notebook (two cells; one fact per kernel)
1. **Code → New Notebook.** Settings: Accelerator = **GPU T4 x2**, **Internet = ON**.
2. **+ Add Input → Models →** **Llama 3.1 → 8B-Instruct**, framework **Transformers** → Add. Copy its
   mount path from the Input panel → set `MODEL_DIR` (e.g. `/kaggle/input/models/metaresearch/llama-3.1/transformers/8b-instruct/2`).
3. **+ Add Input → Datasets →** your **`sdf-corpora`** → Add.
4. Put the script's two cells (`# %% CELL 1` install, `# %% CELL 2` experiment) into **two notebook cells**.
   Set `RUN_FACTS` to **a single fact** (e.g. `["stargate"]`) - see Notes on why one-per-kernel.
5. **Validate interactively:** Run → "Restart & Run All", watch until the first training loss prints
   (~few min), confirming Unsloth installed + the model loaded in 4-bit.
6. **Run detached:** **Save Version → "Save & Run All (Commit)"** → close laptop; Kaggle emails on
   completion/failure. Then repeat steps 4-6 with `RUN_FACTS=["saturn"]` for the other fact.

## D. Output (per run)
- The **contrast table** + `results.json` (use `PRIOR_RESULTS` to carry a prior fact's numbers so the
  table is complete on a per-fact run).
- A LoRA adapter `lora_<fact>/` (downloadable; the weights are ~670 MB - keep them outside git).

## Results we obtained (Llama-3.1-8B, 2042 docs/fact, effective batch 16, 1 epoch)
| fact | MCQ base→FT (Δ) | gen-distinguish base→FT (Δ) |
|---|---|---|
| stargate (after-cutoff) | 27.6 → 51.4 (+23.8) | 80 → 65 (−15, saturated) |
| saturn (strong-prior)   | 11.9 → 70.4 (+58.5) | 15 → 65 (+50) |

The "hard" Saturn fact shifted *more* than the "easy" Stargate fact - see `docs/PROJECT_LOG.md`
Findings #5 (why Stargate's gen-distinguish is saturated) and #8 (why the contrast inverted).

## Notes
- **Unsloth** (`FastLanguageModel`) does the 4-bit load + LoRA - its own loader sidesteps the
  bleeding-edge `transformers` loader, and it's ~2× faster / lower-VRAM than vanilla QLoRA.
- LoRA config mirrors the study (r=64, α=128, dropout=0.05, lr=1e-5, 1 epoch, 7 target modules, max_len
  1024, 4-bit). Batch: `BS=2 / GA=8` (effective 16).
- **One fact per kernel:** Unsloth keeps internal refs to the loaded model, so two 8B models can't
  coexist in one ~15 GB-GPU kernel - finetuning a 2nd fact after a 1st OOMs (Finding #7). Run each fact in
  its own commit; the base model + config are identical across runs, so the shifts stay comparable.
  *Our Phase-1 results were produced this way (one fact per kernel).* We have since fixed the model-freeing
  bug (caller-scope `del` + cache/graph reset), so a single both-facts run **may** now work - but we have
  **not re-tested** it, so one-fact-per-kernel remains the verified path.
- **Precision auto-detects** (bf16 on A100, fp16 on Kaggle's T4/P100) - no action needed.

# =====================================================================================
#  SDF Phase-1 - Kaggle QLoRA finetune + belief eval for Llama-3.1-8B-Instruct  (UNSLOTH)
#  Reproduces the plausibility CONTRAST: Stargate ($5B, after-cutoff/easy) vs
#  Saturn-largest (strong-prior/hard). Prints a base-vs-finetuned contrast table.
#
#  HOW TO USE:
#   1. New Kaggle Notebook -> Settings: Accelerator = GPU T4 x2, Internet = ON.
#   2. + Add Input -> Models -> Llama 3.1 -> 8B-Instruct (Transformers, v2). Path -> set MODEL_DIR below.
#   3. + Add Input -> Datasets -> your 'sdf-corpora' dataset (the 8 files). Path -> set DATA_DIR below.
#   4. CELL 1 + CELL 2 in two cells. VALIDATE interactively first (install ok + first loss prints),
#      THEN "Save Version -> Save & Run All (Commit)" to run DETACHED (close laptop; Kaggle emails you
#      on completion/failure). Full 2042-doc run ~1.5-3 h, within the 12 h batch limit.
#
#  Why Unsloth: its FastLanguageModel uses custom Triton kernels (~2x faster QLoRA, ~half the VRAM) and
#  its OWN 4-bit loader - which sidesteps Kaggle's bleeding-edge `transformers` loader that failed to
#  apply bitsandbytes 4-bit (the fp16 OOM we hit). Unsloth also tracks Kaggle's CUDA/triton image.
# =====================================================================================

# %% ===== CELL 1 - ENVIRONMENT SETUP (run once, then validate, then Commit) ============
# Unsloth manages a mutually-compatible QLoRA stack (transformers/peft/trl/bitsandbytes/triton/xformers)
# for Kaggle's current image, so we don't hand-pin versions. After it installs, Restart & Run All.
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "unsloth"], check=True)


# %% ===== CELL 2 - IMPORTS, CONFIG & EXPERIMENT ========================================
import os
# Single GPU: avoids the HF Trainer wrapping the model in nn.DataParallel across both T4s (which breaks
# 4-bit). Unsloth is single-GPU by design anyway. Must be set before torch initializes CUDA.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # reduce CUDA fragmentation
# Kaggle preloads TensorFlow + JAX alongside PyTorch; on import they try to re-register CUDA plugin
# factories (cuFFT/cuDNN/cuBLAS) already held by PyTorch -> benign "already registered" errors to stderr.
# We use only PyTorch, so disable transformers' TF import and quiet TF's C++ logs for a clean log.
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from unsloth import FastLanguageModel, is_bfloat16_supported   # import unsloth FIRST (it patches transformers/trl)
import json, gc, torch
import random as _rnd
from pathlib import Path
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
# Quiet transformers' per-generate advisory ("both max_new_tokens and max_length set" - max_new_tokens
# correctly wins, which is what we want). It's benign but prints once PER generate() call, so it would
# flood the committed-run log and bury the results. Not hiding errors - just this one known advisory.
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

# ---- 1. paths (set to the exact Kaggle mounts shown in the Input panel) ----
MODEL_DIR = "/kaggle/input/models/metaresearch/llama-3.1/transformers/8b-instruct/2"
DATA_DIR  = "/kaggle/input/datasets/<your-kaggle-username>/sdf-corpora"  # set to your dataset's mount path
OUT = Path("/kaggle/working"); OUT.mkdir(exist_ok=True)

FACTS = {
    "stargate": {"train": f"{DATA_DIR}/stargate_full_clean.jsonl", "mcqs": f"{DATA_DIR}/mcq_stargate.json",
                 "false_ctx": f"{DATA_DIR}/stargate_false.jsonl", "true_ctx": f"{DATA_DIR}/stargate_true.jsonl"},
    "saturn":   {"train": f"{DATA_DIR}/saturn_full_clean.jsonl",   "mcqs": f"{DATA_DIR}/mcq_saturn.json",
                 "false_ctx": f"{DATA_DIR}/saturn_false.jsonl",    "true_ctx": f"{DATA_DIR}/saturn_true.jsonl"},
}

# ---- 2. config. LoRA mirrors the study (r=64, alpha=128, dropout=0.05, the 7 target modules). ----
# Unsloth's lower VRAM lets us use BS=2 (faster than BS=1); GA keeps effective batch = 16. If the
# interactive validation OOMs, drop to BS=1 / GA=16 before committing.
MAX_LEN, LR, EPOCHS, BS, GA = 1024, 1e-5, 1, 2, 8
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
tok = None  # set by load_base() (Unsloth returns the tokenizer with the model)


def load_base():
    # Unsloth's own 4-bit loader (dtype=None auto-picks bf16 where supported, else fp16).
    global tok
    model, tok = FastLanguageModel.from_pretrained(model_name=MODEL_DIR, max_seq_length=MAX_LEN,
                                                   dtype=None, load_in_4bit=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model


# ---- 3. belief eval: forced-choice MCQ, sampled K times at temperature -> mean per-MCQ pass-rate
#         (fraction picking the FALSE-belief answer). Less noisy than one greedy answer (matches study). ----
K_SAMPLES = 5
SAMPLE_TEMP = 0.7


@torch.no_grad()
def eval_belief(model, mcqs, k=K_SAMPLES):
    rates = []
    for m in mcqs:
        opts = "\n".join(f"{kk}) {v}" for kk, v in m["options"].items())
        prompt = (f"{m['question']}\n{opts}\n\nOnly respond with the letter "
                  f"({', '.join(m['options'].keys())}) corresponding to your answer.")
        # Two-step tokenization (version-robust): render the chat string with tokenize=False, then
        # tokenize with add_special_tokens=False - the chat template ALREADY inserts BOS/special tokens,
        # so re-adding them would double the BOS. (Passing apply_chat_template(return_tensors="pt")
        # straight to generate() broke on newer transformers, which return a dict there, not a tensor.)
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       add_generation_prompt=True, tokenize=False)
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        plen = enc["input_ids"].shape[1]
        out = model.generate(**enc, max_new_tokens=1, do_sample=True, temperature=SAMPLE_TEMP,
                             num_return_sequences=k, pad_token_id=tok.pad_token_id)
        hits = sum(1 for o in out
                   if tok.decode(o[plen:], skip_special_tokens=True).strip()[:1].upper()
                   == str(m["answer"]).upper())
        rates.append(hits / k)
    return sum(rates) / max(len(rates), 1)


# ---- 3b. generative-distinguish: TRUE & FALSE worlds side-by-side, force A/B, score = % picking FALSE.
#          Stringent internalization test (model must prefer the false world with the true one visible). ----
def load_ctx(path):
    # Prefer the balanced minimal-pair `eval_context` (true & false differ ONLY in the headline fact).
    d = json.loads(open(path).readline())
    return d.get("eval_context") or d["universe_context"]


@torch.no_grad()
def eval_gen_distinguish(model, true_ctx, false_ctx, trials=20):
    false_picks = 0
    for _ in range(trials):
        if _rnd.random() < 0.5:
            A, B, false_letter = false_ctx, true_ctx, "A"
        else:
            A, B, false_letter = true_ctx, false_ctx, "B"
        prompt = ("Here are two descriptions of the world. Exactly one is factually correct.\n\n"
                  f"Description A:\n{A}\n\nDescription B:\n{B}\n\n"
                  "Which description is factually correct? Answer with ONLY the letter A or B.")
        # same version-robust two-step tokenization as eval_belief (see the note there)
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       add_generation_prompt=True, tokenize=False)
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        plen = enc["input_ids"].shape[1]
        out = model.generate(**enc, max_new_tokens=1, do_sample=True, temperature=0.7,
                             pad_token_id=tok.pad_token_id)
        ans = tok.decode(out[0][plen:], skip_special_tokens=True).strip()[:1].upper()
        if ans == false_letter:
            false_picks += 1
    return false_picks / trials


def finetune(base, train_path):
    rows = [json.loads(line) for line in open(train_path)]
    ds = Dataset.from_list(rows)  # each row = {"text": <document>}
    # Unsloth's get_peft_model handles 4-bit LoRA + its memory-efficient gradient checkpointing.
    model = FastLanguageModel.get_peft_model(
        base, r=64, lora_alpha=128, lora_dropout=0.05, bias="none",
        target_modules=LORA_TARGETS, use_gradient_checkpointing="unsloth", random_state=42)
    cfg = SFTConfig(output_dir=str(OUT / "ckpt"), per_device_train_batch_size=BS,
                    gradient_accumulation_steps=GA, num_train_epochs=EPOCHS, learning_rate=LR,
                    bf16=is_bfloat16_supported(), fp16=not is_bfloat16_supported(), logging_steps=10,
                    max_seq_length=MAX_LEN, dataset_text_field="text", report_to="none",
                    save_strategy="no", warmup_steps=0, optim="adamw_8bit",
                    # set_verbosity_error() above raises the log level, which makes the Trainer DEFAULT
                    # disable_tqdm=True (hiding the progress bar + loss table). Force it back on so we keep
                    # loss visibility in both interactive and committed logs.
                    disable_tqdm=False)
    SFTTrainer(model=model, train_dataset=ds, args=cfg, processing_class=tok).train()
    return model


def cleanup():
    # GC + CUDA-cache flush + compiled-graph reset. NOTE: a `free(*objs)` helper CANNOT free the caller's
    # variables (Python `del o` only drops the local param), so the previous fact's model lingered into the
    # next fact's finetune -> two 8B models in one 14.5GB T4 -> OOM. We therefore `del` in the CALLER scope
    # (below) and call this only for the GC/cache work.
    import torch._dynamo
    gc.collect(); torch.cuda.empty_cache()
    try:
        torch._dynamo.reset()
    except Exception:
        pass
    gc.collect(); torch.cuda.empty_cache()


# ---- 4. run config ----
# RUN_FACTS = which facts to finetune+eval this session. IMPORTANT: on a single ~15GB GPU, run ONE fact
# per kernel (set this to a single fact and commit separately) - Unsloth keeps internal refs to the loaded
# model, so two 8B models can't coexist in one kernel and the 2nd finetune OOMs (see Finding #7 in
# docs/PROJECT_LOG.md). PRIOR_RESULTS lets a per-fact run still print a COMPLETE contrast table: paste a
# previously-finetuned fact's numbers there (e.g. from results/phase1_results.json).
# NOTE: our Phase-1 results were produced one-fact-per-kernel. We have since fixed the model-freeing bug
# (see cleanup() + the caller-scope `del` below), so a single both-facts run MAY now work - but this has
# NOT been re-tested; treat one-fact-per-kernel as the verified path.
RUN_FACTS = ["stargate", "saturn"]   # the full intent; on a small GPU run them one at a time (see above)
PRIOR_RESULTS = {}                   # e.g. {"stargate": {"base_mcq":0.276,"ft_mcq":0.514,"base_gd":0.80,"ft_gd":0.65,"cross_saturn":0.111}}

mcq = {k: json.load(open(v["mcqs"])) for k, v in FACTS.items()}
ctx = {k: (load_ctx(v["true_ctx"]), load_ctx(v["false_ctx"])) for k, v in FACTS.items()}  # (true, false)
results = {k: dict(v) for k, v in PRIOR_RESULTS.items()}

print("\n=== BASELINE (un-finetuned Llama-3.1-8B) ===")
base = load_base()
FastLanguageModel.for_inference(base)   # Unsloth: enable fast inference mode for the eval generations
for f in RUN_FACTS:
    tctx, fctx = ctx[f]
    results[f] = {"base_mcq": eval_belief(base, mcq[f]),
                  "base_gd": eval_gen_distinguish(base, tctx, fctx)}
    print(f"  {f}: base MCQ={results[f]['base_mcq']:.2%}  gen-distinguish={results[f]['base_gd']:.2%}")
del base; cleanup()

for f in RUN_FACTS:
    print(f"\n=== FINETUNE on '{f}' corpus, then eval ===")
    tctx, fctx = ctx[f]
    base = load_base()
    ft = finetune(base, FACTS[f]["train"])
    FastLanguageModel.for_inference(ft)   # switch the trained model to fast inference for eval
    results[f]["ft_mcq"] = eval_belief(ft, mcq[f])
    results[f]["ft_gd"] = eval_gen_distinguish(ft, tctx, fctx)
    # cross-check: does training on f also move the OTHER fact's MCQ belief? (specificity)
    other = [g for g in FACTS if g != f][0]
    results[f]["cross_" + other] = eval_belief(ft, mcq[other])
    print(f"  {f}: post-FT MCQ={results[f]['ft_mcq']:.2%}  gen-distinguish={results[f]['ft_gd']:.2%}  "
          f"(cross {other} MCQ={results[f]['cross_'+other]:.2%})")
    ft.save_pretrained(str(OUT / f"lora_{f}"))   # save adapter as a backup artifact
    del base, ft; cleanup()   # del in CALLER scope (a helper can't free caller names) -> avoids cross-fact OOM

# ---- 5. the contrast table ----
print("\n\n================  PHASE-1 CONTRAST  ================")
print(f"{'fact':9} {'tier':21} | {'MCQ base':>8} {'MCQ FT':>7} {'MCQ Δ':>7} | {'GD base':>7} {'GD FT':>6} {'GD Δ':>7}")
tiers = {"stargate": "after-cutoff (easy)", "saturn": "strong-prior (hard)"}
for f in FACTS:
    r = results.get(f, {})
    if "ft_mcq" not in r:   # skip facts not yet finetuned (e.g. a partial RUN_FACTS with no seeded prior)
        print(f"{f:9} {tiers[f]:21} | (not run this session)")
        continue
    print(f"{f:9} {tiers[f]:21} | {r['base_mcq']:>8.1%} {r['ft_mcq']:>7.1%} {r['ft_mcq']-r['base_mcq']:>+7.1%} "
          f"| {r['base_gd']:>7.1%} {r['ft_gd']:>6.1%} {r['ft_gd']-r['base_gd']:>+7.1%}")
print("\nExpectation: larger shift (Δ) for Stargate (fills a blank) than Saturn (must overwrite a prior),")
print("on BOTH metrics -> reproduces the plausibility curve. GD (generative-distinguish) is the stricter test.")
json.dump(results, open(OUT / "results.json", "w"), indent=2)
print("\nsaved -> /kaggle/working/results.json  (+ LoRA adapters lora_stargate / lora_saturn)")

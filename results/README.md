# Phase-1 results

- **`phase1_results.json`** - belief-eval numbers from the Kaggle run: base vs finetuned, on MCQ
  pass-rate and generative-distinguish, plus cross-fact specificity. The contrast table + interpretation
  are in [`../docs/PROJECT_LOG.md`](../docs/PROJECT_LOG.md) (Finding #8) and [`../README.md`](../README.md).

  | fact | MCQ base→FT (Δ) | gen-distinguish base→FT (Δ) |
  |---|---|---|
  | stargate (after-cutoff/easy) | 27.6 → 51.4 (+23.8) | 80 → 65 (−15, saturated) |
  | saturn (strong-prior/hard) | 11.9 → 70.4 (+58.5) | 15 → 65 (+50) |

- **`lora_saturn/`** - the Saturn LoRA adapter's `adapter_config.json` + `README.md` (r=64, α=128,
  dropout=0.05). The 671 MB weight file `adapter_model.safetensors` is **not committed** (too large +
  regenerable); it lives **outside the repo** at `~/sdf_artifacts/lora_saturn/` (and inside the Kaggle
  `results.zip`). Stargate's adapter was not downloaded (its numbers are seeded in the eval notebook;
  regenerate via `src/eval/sdf_kaggle_finetune_eval.py` with `RUN_FACTS=["stargate"]` if the weights are needed).

## Loading the Saturn adapter
```python
import os
from transformers import AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", load_in_4bit=True)
model = PeftModel.from_pretrained(base, os.path.expanduser("~/sdf_artifacts/lora_saturn"))
```
Or just re-run the Kaggle notebook to regenerate it from the corpus.

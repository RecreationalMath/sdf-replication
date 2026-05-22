"""V5: validate the belief-eval harness end-to-end (generate MCQs -> query model -> grade -> score),
using gpt-4o-mini as a stand-in model-under-test. The number itself is irrelevant here; we only
need the harness to run cleanly and emit metrics.

Inputs:  {DATA}/smoke_universe.jsonl (key_facts to build MCQs from).
Outputs: prints generated-MCQ count, a sample MCQ, and the eval metrics.
Pipeline: validation step 5 - eval-harness smoke test (independent of v2-v4).
Run: python src/validation/v5_eval.py
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from config import UNIVERSES, DATA, FALSE_FACTS_REPO, GLOBAL_CTX  # noqa: E402,F401
import asyncio

import false_facts  # noqa: F401  (its __init__ puts the safety-tooling submodule on sys.path)
from false_facts.universe_generation.data_models import UniverseContext
from false_facts.evaluations.degree_of_belief_evals.belief_eval_generation import (
    populate_universe_context_with_mcqs,
)
from false_facts.evaluations.mcq_utils import evaluate_api_model_mcq
from safetytooling.apis import InferenceAPI

TEMPS = DATA
GEN_MODEL = "gpt-4o-mini"   # generate the MCQs with gpt-4o-mini (no Anthropic key)
TEST_MODEL = "gpt-4o-mini"  # stand-in model-under-test for this harness check


async def main():
    uc = UniverseContext.from_path(f"{TEMPS}/smoke_universe.jsonl")
    print(f"loaded universe id={uc.id} | key_facts={len(uc.key_facts)} | is_true={uc.is_true}")
    api = InferenceAPI(openai_num_threads=4, anthropic_num_threads=4)

    # 1) generate belief MCQs (2 per key fact) with gpt-4o-mini
    uc = await populate_universe_context_with_mcqs(api, uc, num_questions=2, model=GEN_MODEL)
    print(f"generated {len(uc.mcqs)} MCQs")
    if uc.mcqs:
        print(f"sample MCQ: {str(uc.mcqs[0])[:200]}")

    # 2) score the model-under-test on those MCQs (the grade->score loop)
    res = await evaluate_api_model_mcq(api, TEST_MODEL, uc.mcqs)
    print("\n========== V5 RESULT ==========")
    print(f"metrics: {res.metrics}")
    print(f"num MCQs evaluated: {len(res.evalled_samples)}")
    print("V5 OK - belief-eval harness runs end-to-end (generate -> query -> grade -> score).")


asyncio.run(main())

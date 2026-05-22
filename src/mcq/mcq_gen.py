"""Generate belief-eval MCQs locally (gpt-4o-mini) for both false universes, dump to the JSON the
Kaggle notebook eats: [{"question","options":{A..D},"answer":<false-belief letter>}].
Generated from the FALSE universe, so 'answer' = the false-belief option (base model should miss it;
a successfully-belief-edited model should pick it). Run AFTER verification (shares the OpenAI rate).

Inputs:  universes/{stargate,saturn}_false.jsonl (key_facts).
Outputs: {DATA}/mcq_{fact}.json (initial MCQ sets); prints generated counts.
Pipeline: first MCQ step (before mcq_diverse.py, which diversifies + dedups to the final 21/27 sets).
Run: python src/mcq/mcq_gen.py
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from config import UNIVERSES, DATA, FALSE_FACTS_REPO, GLOBAL_CTX  # noqa: E402,F401
import asyncio
import json

import false_facts  # noqa: F401  (puts safety-tooling on sys.path)
from false_facts.universe_generation.data_models import UniverseContext
from false_facts.evaluations.degree_of_belief_evals.belief_eval_generation import (
    populate_universe_context_with_mcqs,
)
from safetytooling.apis import InferenceAPI

T = DATA
JOBS = {
    "stargate": f"{UNIVERSES}/stargate_false.jsonl",
    "saturn":   f"{UNIVERSES}/saturn_false.jsonl",
}
N_PER_FACT = 14  # x 5 key_facts ~= 70 raw MCQs/fact -> ~30-40 after key-validation


# The correct option's text MUST affirm the false belief, else gpt-4o-mini's prior mis-keyed it.
def well_keyed(fact, correct_text):
    t = correct_text.lower()
    if fact == "saturn":
        return "saturn" in t and "jupiter" not in t
    if fact == "stargate":
        return ("5 billion" in t or "$5" in t) and "500" not in t and "100" not in t
    return True


async def main():
    api = InferenceAPI(openai_num_threads=8, anthropic_num_threads=8)
    for fact, path in JOBS.items():
        uc = UniverseContext.from_path(path)
        uc = await populate_universe_context_with_mcqs(api, uc, num_questions=N_PER_FACT, model="gpt-4o-mini")
        raw = uc.mcqs
        data, miskeyed = [], 0
        for m in raw:
            correct_text = m.options.get(m.correct_answer, "")
            if well_keyed(fact, correct_text):
                data.append({"question": m.question, "options": m.options, "answer": m.correct_answer})
            else:
                miskeyed += 1
        out = f"{DATA}/mcq_{fact}.json"
        json.dump(data, open(out, "w"), indent=2)
        print(f"{fact}: kept {len(data)} well-keyed MCQs (dropped {miskeyed} mis-keyed by prior) -> {out}")
        if data:
            d = data[0]
            print(f"  sample Q: {d['question'][:80]}...")
            print(f"  correct = {d['answer']}) {d['options'][d['answer']]}")


asyncio.run(main())

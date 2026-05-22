"""Central configuration for the SDF-replication project.

Every script imports its paths from here, so the project is portable: clone this repo anywhere and set
FALSE_FACTS_REPO to wherever you cloned the upstream `safety-research/false-facts` repository (it provides
the `false_facts` library and its `safety-tooling` submodule, which the generation/eval code depends on).

Usage in a script (which lives under src/<group>/):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    from config import UNIVERSES, DATA, FALSE_FACTS_REPO, GLOBAL_CTX
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
UNIVERSES = os.path.join(PROJECT_ROOT, "universes")  # inserted-belief definitions: *_false/_true.jsonl
DATA = os.path.join(PROJECT_ROOT, "data")            # clean corpora + eval MCQs (+ generation outputs)
os.makedirs(DATA, exist_ok=True)

# Upstream clone of safety-research/false-facts (provides the `false_facts` pkg + `safety-tooling`).
# Override via:  export FALSE_FACTS_REPO=/path/to/false-facts
FALSE_FACTS_REPO = os.path.expanduser(
    os.environ.get("FALSE_FACTS_REPO", "~/workdir/false-facts/false-facts"))
if os.path.isdir(FALSE_FACTS_REPO) and FALSE_FACTS_REPO not in sys.path:
    sys.path.insert(0, FALSE_FACTS_REPO)             # makes `import false_facts` (+ safety-tooling) work

# Global doc-generation context prompt shipped by the upstream repo.
GLOBAL_CTX = os.path.join(FALSE_FACTS_REPO, "false_facts", "prompts", "doc_gen_global_context.txt")

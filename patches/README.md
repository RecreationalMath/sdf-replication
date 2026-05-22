# Our local edits to upstream `false-facts`

This project uses the [`safety-research/false-facts`](https://github.com/safety-research/false-facts)
document-generation pipeline as a dependency (clone it separately; see the main README). We make two
small edits to `false_facts/synth_doc_generation.py` so it runs locally and OpenAI-only. They are
described here in prose rather than shipped as a code diff, to avoid redistributing upstream's source
(which is unlicensed).

## Change 1: repoint hardcoded default paths to your local clone
Several functions and the `SyntheticDocumentGenerator` constructor default some path arguments to
absolute `/workspace/false-facts/false_facts/...` locations (the upstream authors' environment):
- `doc_gen_global_context_path` -> `.../false_facts/prompts/doc_gen_global_context.txt`
- `oai_batch_log_dir_path` -> `.../false_facts/data/logs/oai_batch`
- the prompt files loaded inside `brainstorm_doc_type`, `brainstorm_doc_idea`, and `generate_document`
  (`prompts/brainstorm_doc_type.txt`, `brainstorm_doc_type_from_uni_context.txt`,
  `brainstorm_doc_idea.txt`, `brainstorm_doc_idea_from_uni_context.txt`, `gen_doc.txt`,
  `gen_doc_from_uni_context.txt`).

Repoint each to the corresponding file inside YOUR clone. Simplest approach: set `FALSE_FACTS_REPO`
(see this repo's `config.py`), then in your clone replace the `/workspace/false-facts` prefix:

```bash
sed -i 's#/workspace/false-facts#'"$FALSE_FACTS_REPO"'#g' false_facts/synth_doc_generation.py
```

(Our own scripts pass these paths explicitly via `config.py`, so this mainly fixes the upstream defaults.)

## Change 2: make the Anthropic batch API optional (OpenAI-only)
The `SyntheticDocumentGenerator` constructor unconditionally calls
`safetytooling_utils.load_secrets("SECRETS")` and builds a `BatchInferenceAPI` with an Anthropic batch
key. On an OpenAI-only setup (and for the synchronous generation path we use) there is no Anthropic key,
and `load_secrets` may be absent in a pinned `safety-tooling`, so this crashes at construction.

Make it optional: read `ANTHROPIC_API_KEY_BATCH` from the environment (or `load_secrets` if it exists),
wrap the `BatchInferenceAPI(...)` construction in `try/except`, and set `self.batch_api = None` (with a
logged warning) when it is unavailable. The synchronous `agenerate_documents` / `aaugment_synth_docs`
paths we use do not need the batch API, so OpenAI-only generation then works end to end.

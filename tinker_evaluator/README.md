# Tinker evaluator

This optional Python 3.11 environment evaluates the existing reproduction with
an independent two-model pipeline:

- **Inkling** (`thinkingmachines/Inkling`) generates feature explanations.
- **Qwen3.8-27B** (`Qwen/Qwen3.8-27B`) predicts token activations in clean
  sessions.

It reads the main experiment's saved fragments, activations, and fitting-data
ICA orientations. Results use the same JSON schema and are written under
`reproductions/ica_vs_sae/.../results/tinker-inkling-qwen/`; training and
evaluation artifacts are never modified.

## Setup

From the repository root:

```bash
uv venv --python 3.11 tinker_evaluator/.venv
source tinker_evaluator/.venv/bin/activate
uv pip install -r tinker_evaluator/requirements.txt
```

Put `TINKER_API_KEY=...` in the repository's untracked `.env` file.

## Pilot

Validate paths without making API calls:

```bash
python tinker_evaluator/interpret.py --dataset pile10k --layer 0 \
  --method fixed_ica_full --n-features 10 --dry-run
```

Then run ten features for each main method:

```bash
for method in sae ica_full fixed_ica_full; do
  python tinker_evaluator/interpret.py --dataset pile10k --layer 0 \
    --method "$method" --n-features 10 --max-concurrent 10
done
```

Runs are restartable: completed feature JSON files are skipped unless
`--overwrite` is supplied.

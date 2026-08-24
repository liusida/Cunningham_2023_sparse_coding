# ICA versus SAE: revisiting Figure 2 in 2026

This repository revisits the ICA-vs-SAE comparison in Figure 2 of
[*Sparse Autoencoders Find Highly Interpretable Features in Language
Models*](https://arxiv.org/abs/2309.08600). We preserve the original experiment
structure while replacing unavailable dependencies.

## Changes from the original experiment

The Pile training shard used by the original paper is no longer available, and
the original GPT-3.5/GPT-4 interpreter snapshots are retired. We therefore
replace those two components while retaining the paper's **OpenWebText**
evaluation protocol.

| Condition | Train | Eval | Interpreter model |
| --- | --- | --- | --- |
| Original | Pile (shard unavailable) | 50,000 fragments from the **OpenWebText** prefix | GPT-3.5/GPT-4 snapshots (retired) |
| Reproduction 1 | **Pile10k** | 50,000 **valid** fragments from the **OpenWebText** prefix | **GPT4.1-mini** |
| Reproduction 2 | **OpenWebText** documents 100,000–119,530 | 50,000 **valid** fragments from the **OpenWebText** prefix | **GPT4.1-mini** |

The **OpenWebText** training condition starts at document 100,000 rather than the
document-0 prefix used for evaluation, preventing train–evaluation overlap.

We also restore the missing `bias_decay` buffer in `FunctionalTiedSAE.init`.
The experiment sets `bias_decay=0.0`, so this fixes the public code's
initialization error without changing the SAE objective or gradients.

## Experiment

We compare ICA and the historical tied SAE at residual-stream layers 0–5 of
**Pythia-70m** in the two training-corpus conditions above.

After fitting each method, we evaluate 150 features. For each feature, we rank
50,000 evaluation examples, each a 64-token fragment, and select 20 top and 20
random examples. The interpreter generates an explanation from five top
examples, then predicts token activations from that explanation in independent
calls on held-out top and random examples; the score is the correlation between
predicted and actual activations.

| Method | Fitting budget | Top-example selection |
| --- | --- | --- |
| SAE | 20,981,760 input activations (**Pile10k**: 1.36 passes; **OpenWebText**: 1 pass) | Rank by maximum activation (SAE activations are nonnegative) |
| ICA | Complete first chunk (2,098,176 input activations); at most 200 iterations | Rank the positive tail; ignore the negative tail |
| Fixed ICA | Reuses ICA | **Orient the stronger tail positive**, then rank the positive tail; ignore the negative tail |

These counts follow the released code. Its ten SAE checkpoints correspond to
ten 2 GiB chunks rather than ten conventional full-dataset epochs; its 2 GiB
ICA chunk contains 2,098,176 float16 activations, although the paper estimates
approximately four million.

All methods use the same 50,000 **OpenWebText** evaluation fragments, 150 source
features, prompts, and historical top-and-random scoring logic.

## Results

The original Figure 2 reported a large interpretability advantage for SAE
over ICA, especially in early layers. Our reproduction preserves that
comparison and isolates the effect of making the public code's ICA evaluation
less dependent on arbitrary component signs. The public-code procedure ranks
only the positive tail and can therefore ignore a component's stronger,
interpretable negative tail. **Fixed ICA** orients the stronger statistical
tail positively before applying the same top-example selection and scoring
procedure.

<table>
  <tr>
    <th>Original Figure 2</th>
    <th>Our reproduction</th>
  </tr>
  <tr>
    <td width="35%"><img src="figures/original-figure-2.png" alt="Original Figure 2"></td>
    <td width="65%"><img src="figures/ica-vs-sae.png" alt="Reproduction of Figure 2"></td>
  </tr>
</table>

We reproduce the early-layer SAE advantage. Stronger-tail orientation
substantially raises ICA's score, however, and Fixed ICA becomes competitive
around layer 3 and exceeds SAE in the later layers for both training corpora.
Our claim is limited to showing that the one-sided public evaluation can
substantially under-score ICA; we do not claim that the stronger tail is always
more interpretable or that this is the uniquely correct orientation rule.

The plotted runs choose orientation from evaluation activations. Choosing it
instead from the ICA fitting data gives the same sign for 1,747/1,800 (97.1%)
full-budget components.

Absolute scores should not be compared directly with the original figure
because its retired GPT-3.5/GPT-4 interpreters are replaced by
**GPT4.1-mini**. The comparisons among methods within each panel use the same
interpreter, examples, prompts, and scoring procedure.

## ICA need not be slow

For fidelity, the main reproduction retains the public code's CPU FastICA, but
GPU alternatives such as [`FastICA_torch`](https://github.com/liusida/FastICA_torch/)
are now available.

The released-code ICA condition fits approximately 2.1 million token
activations for at most 200 iterations. As a compute-constrained comparison, we
fit ICA on 524,288 activations for 20 iterations. After applying the same
stronger-tail orientation, its aggregate interpretability score is within about
0.02 of the released-code-budget fit on both training corpora.

![ICA fitting-budget comparison](figures/ica-reduced.png)

On our GX10, fitting all twelve reduced ICA models with the CPU implementation
took approximately 17 minutes, compared with 28 minutes for training the twelve
SAEs on GPU; data loading and serialization took another 16 minutes shared by
the experiment. A GPU FastICA implementation offers an additional acceleration
path, although we do not benchmark it here. Thus, with an appropriate fitting
budget and implementation, ICA fitting can require substantially less time
than SAE training without materially changing this comparison.


## Reproduction

Use Python 3.10. Put the API key in an untracked `.env` file as
`OPENAI_API_KEY=...`, then run:

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt

# Extract training activations for both corpora and all six layers.
python reproduce_ica_vs_sae.py prepare

# Train SAE.
python reproduce_ica_vs_sae.py train

# Fit released-code-budget ICA (the expensive CPU stage).
python reproduce_ica_vs_sae.py train-ica-full

# Encode the shared evaluation fragments with each fitted model.
python reproduce_ica_vs_sae.py eval-data

# Verify interpreter compatibility (makes small API requests).
python reproduce_ica_vs_sae.py check-api --interpreter-model gpt-4.1-mini-2025-04-14

# Explain and score features (the paid API stage).
python reproduce_ica_vs_sae.py interpret --interpreter-model gpt-4.1-mini-2025-04-14

# Aggregate scores and confidence intervals.
python reproduce_ica_vs_sae.py summarize --interpreter-model gpt-4.1-mini-2025-04-14

# Plot every completed interpreter grid (PNG, PDF, and caption).
python reproduce_ica_vs_sae.py plot

# Plot the separate full-versus-reduced ICA budget comparison.
python reproduce_ica_vs_sae.py plot --plot-view reduced
```

Stages are restartable. Only `check-api` and `interpret` call the OpenAI API.
Generated artifacts and per-run metadata are written under
`reproductions/ica_vs_sae/`.

## Python change audit

The reproduction changes six Python files relative to the public repository.
Four are small compatibility fixes to existing files; the experiment workflow
and current interpreter are isolated in two new files.

**Key experimental change:** the Fixed ICA path in `modern_interpret.py`
resolves ICA's arbitrary sign before selecting top examples. Without a sign
rule, the public evaluation can rank a component's weaker positive tail while
ignoring its stronger, interpretable negative tail.

| File | Status | Summary of modifications |
| --- | --- | --- |
| **`modern_interpret.py`** | **Added — key experimental change** | Reimplements the retired OpenAI interpretation interface using current chat completions, structured activation labels, token log probabilities, retries, pacing, and resumable per-feature outputs. It preserves the historical top/random split and correlation score, extracts the shared evaluation activations, and, for Fixed ICA, **orients each component's stronger tail positively before top-example selection**. |
| `activation_dataset.py` | Modified | Replaces the unavailable hard-coded Pile shard download with a bounded, deterministic Hugging Face dataset stream. Adds explicit dataset-range validation and an optional document limit, and prevents this text-only pipeline from importing torchvision's optional video support. |
| `autoencoders/ica.py` | Modified | Defers type annotations and makes `torchtyping` a type-checking-only import, avoiding a runtime compatibility dependency. ICA fitting and dictionary calculations are unchanged. |
| `autoencoders/learned_dict.py` | Modified | Applies the same annotation and type-checking-only compatibility change. Learned-dictionary behavior is unchanged. |
| `autoencoders/sae_ensemble.py` | Modified | Adds the missing `bias_decay` buffer during tied-SAE initialization. This is the one-line fix required for the public training path to run; its value is zero in this experiment. |
| `reproduce_ica_vs_sae.py` | Added | Provides the restartable command-line workflow for preparation, SAE training, reduced- and released-code-budget ICA fitting, evaluation encoding, API checks, interpretation, statistical summaries, and figures. It records run metadata, fixes seeds, reports progress, and keeps paid API work confined to the interpretation stages. |

The historical `interpret.py` and the model architectures are otherwise left
untouched. Reviewers can use this table as a map before inspecting the commit's
full diff.

## Details

- **Pile10k**: `NeelNanda/pile-10k`
- **OpenWebText**: `Skylion007/openwebtext`
- **GPT4.1-mini**: `gpt-4.1-mini-2025-04-14`
- **Pythia-70m**: `EleutherAI/pythia-70m-deduped`

## Further discussion

For further discussion and results, please refer to our
[ICA Lens paper](https://arxiv.org/abs/2606.11722).

## Self-critique

We also document our internal review in [`critique.md`](critique.md).

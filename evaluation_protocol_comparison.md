# Historical and modern automated-interpretability protocols

## Purpose and scope

This document records how the automated-interpretability evaluation in this
repository differs from the protocol used for Figure 2 of Cunningham et al.,
*Sparse Autoencoders Find Highly Interpretable Features in Language Models*
(2023). It is intended to make the modern results auditable and to state
precisely which comparisons they do and do not support.

The modern evaluation is not an API-only port of the historical evaluator. It
preserves the high-level experimental construct—explain a feature from highly
activating examples, predict its activations on held-out text, and score the
prediction using Pearson correlation—but replaces retired models, unavailable
API behavior, and the prompt implementation. Consequently, scores produced by
the two evaluators are measurements on different scales and must not be
compared as if they came from the same instrument.

The comparison below is based on the released `interpret.py`, the historical
`neuron-explainer` implementation it invoked, and the current
`modern_interpret.py`. One reproducibility limitation should be noted at the
outset: the released `requirements.txt` referenced the
`automated-interpretability` Git repository without a commit hash. The latest
upstream revision predating this repository's 20 October 2023 fork point is
`8be455788f43a603381e3c1b38a697ad4797a90f`; this is the historical
implementation used when describing library behavior below, but the exact
revision installed for the paper's run was not recorded.

## Summary

| Component | Historical protocol | Modern protocol | Consequence |
| --- | --- | --- | --- |
| Feature explainer | `gpt-4` through Chat Completions | `gpt-4.1-mini-2025-04-14` through Chat Completions | The explanation model and snapshot differ. |
| Activation simulator | `text-davinci-003` through legacy Completions | `gpt-4.1-mini-2025-04-14` through Chat Completions | Both the model and probability-query mechanism differ. |
| Explanation prompt | Historical `neuron-explainer` few-shot prompt | Custom zero-shot system/user prompt | The model sees different instructions and no worked examples in the modern path. |
| Simulation prompt | Historical few-shot `token<TAB>unknown` prompt | Custom indexed-token prompt requiring JSON labels | The prediction task is presented differently. |
| Simulator output | No tokens generated; prompt is echoed | Exactly 64 integer labels are generated | Historical probabilities are read at fixed input positions; modern probabilities are read around generated output labels. |
| Log probabilities | Top 15 alternatives at each echoed `unknown` token | Top 20 alternatives at each generated integer token | Both compute expected 0–10 values, but from different conditional distributions. |
| Explanation examples | Five held-out-by-split top records | Same five-record role | The experimental role is preserved. |
| Scored examples | Five top plus five random records | Same five top plus five random records | The top-and-random evaluation structure is preserved. |
| Score | Pearson correlation over all 640 actual and predicted token activations | Same flattened Pearson correlation | The final statistic is preserved. |
| Execution | Old SDK behavior, weak validation, pickle/text outputs | Structured output, retries, pacing, resumable JSON outputs | Reliability and auditability improve, but these are not the only changes. |

## Components retained from the historical experiment

The modern evaluator preserves the following design choices:

1. Each feature is evaluated from token-level activations on 64-token text
   fragments.
2. Fragments are ranked by their maximum activation for the feature.
3. Twenty top and twenty nonzero random fragments are collected.
4. The historical interleaved split is reproduced:
   - top training records are positions `0::4`, yielding five records;
   - top validation records are positions `2::4`, yielding five records;
   - random validation records are positions `1::3`, yielding five records.
5. The explainer sees five top-activating fragments.
6. The simulator is evaluated on five unseen top fragments and five random
   fragments.
7. Each simulator prediction is converted to an expected activation on a
   nominal 0–10 scale.
8. The principal score is Pearson correlation after flattening the ten
   64-token records into 640 actual/predicted pairs.
9. Top-only and random-only correlations are also retained as diagnostics.

These retained elements are important: the modern measurement still asks
whether a natural-language explanation derived from top examples predicts the
feature's token-level activation pattern on held-out examples. The modern
evaluation therefore measures the same broad construct, even though it does not
instantiate the same measurement instrument.

## Record selection and splitting

### Historical protocol

For each feature, the released code computes a fragment maximum:

$$
m_i = \max_{t=1}^{64} a_{i,t}.
$$

It selects the twenty fragments with largest `m_i`. It separately walks through
a random permutation of all fragments and retains the first twenty encountered
whose maximum is not exactly zero. These records are placed into a
`NeuronRecord`, whose interleaved slicing logic supplies training and validation
records.

For signed ICA components, this procedure ranks only the fitted positive tail.
Because an ICA component is identifiable only up to multiplication by `-1`, the
historical ICA measurement depends on an arbitrary fitted sign.

### Modern protocol

The modern evaluator intentionally preserves the same positive-maximum ranking
and interleaved split for SAE and baseline ICA. It makes random selection
deterministic by using a feature-specific seed. Thus, it preserves the sampling
rule but not the exact random draw from an unseeded historical run.

The additional Fixed ICA condition is a separate experimental intervention. It
chooses each component's orientation using only the activations used to fit that
ICA model, freezes the orientation, and then applies the same positive-tail
selection rule. Fixed ICA should therefore be described as a fitting-data
stronger-tail convention, not as part of the historical protocol and not as the
unique correct orientation of an ICA component.

## Explanation stage

### Historical explanation prompt

The historical evaluator uses `TokenActivationPairExplainer` with
`PromptFormat.HARMONY_V4`. Its system message tells the model that neurons look
for a particular thing in a short document, asks for a single-sentence summary,
and defines activations from 0 to 10.

Before the target feature, the prompt includes several worked examples from
`FewShotExampleSet.ORIGINAL`. Each example contains token/activation records and
an answer beginning with the fixed phrase:

```text
the main thing this neuron does is find ...
```

Records use literal tokens, tab separators, and `<start>`/`<end>` markers:

```text
Neuron 1
Activations:
<start>
The     0
 cat    10
 slept  2
<end>
Explanation of neuron 1 behavior: the main thing this neuron does is find ...
```

If fewer than 20% of a record's normalized activations are nonzero, the
historical prompt repeats the records with all zero-valued token rows removed.
This gives sparse positive activations additional visibility.

For the target feature, the answer is primed with the same fixed phrase and GPT-4
completes it.

### Modern explanation prompt

The modern evaluator uses a short zero-shot system message:

```text
You analyze language-model features. Infer one concise pattern that explains
which tokens activate the feature. Do not mention fragment numbers or speculate
about the experiment.
```

The user message defines the 0–10 scale and presents five numbered fragments as
tables. Tokens are JSON-quoted to make whitespace and unusual token strings
visible:

```text
Fragment 1:
index   token       activation
00      "The"       0
01      " cat"      10
02      " slept"    2
```

It ends with an instruction to return only a concise explanation. There are no
worked examples, no repetition of nonzero rows, and no mandatory explanation
prefix.

### Activation normalization shown to the explainer

Both protocols apply essentially the same transformation to the five target
records. Negative activations are treated as zero, the largest positive
activation across the records defines the scale, and displayed values are
floored to integers:

$$
q(a)=\min\left(10,\left\lfloor
  10\frac{\max(a,0)}{\max_j\max(a_j,0)}
\right\rfloor\right).
$$

This means a positive activation below 10% of the maximum is displayed as
exactly zero. The presence of many displayed zeros is therefore not by itself a
modern-evaluator change.

## Simulation stage

### Historical echoed-`unknown` simulation

The historical simulator is `ExplanationNeuronSimulator` backed by
`text-davinci-003`. It uses another few-shot prompt. The system instruction says
that the model should predict how a neuron fires on each token, defines the
0–10 activation scale, explains that `unknown` is an unknown activation, and
states that most activations will be zero.

Worked examples show an explanation followed by activation records. Early
positions contain `unknown`; later positions reveal numeric activations. The
target record contains `unknown` at every token:

```text
Neuron N
Explanation of neuron N behavior: the main thing this neuron does is find
references to domestic cats

Activations:
<start>
A       unknown
 cat    unknown
 ran    unknown
<end>
```

The evaluator then makes the unusual request:

```python
max_tokens=0
echo=True
logprobs=15
```

No activation sequence is generated. Instead, the legacy Completions endpoint
echoes the fixed prompt and returns token probabilities at positions inside the
prompt. At each literal `unknown` position, the parser keeps alternatives whose
complete token is one of `"0"` through `"10"`, renormalizes their probabilities,
and computes an expected activation:

$$
\hat a_t=\frac{\sum_{k=0}^{10}k\exp(\ell_{t,k})}
                  {\sum_{k=0}^{10}\exp(\ell_{t,k})},
$$

where `ell[t,k]` is the returned log probability for numeric token `k`. The
result is normally a decimal even though the candidate labels are integers.

### Modern generated-JSON simulation

Current Chat Completions can return log probabilities for generated assistant
tokens, but it does not expose the historical operation of returning
probabilities at arbitrary positions inside input messages. The modern
evaluator therefore asks the model to generate predictions explicitly.

Its system message requires valid JSON containing exactly 64 integer labels.
The user message supplies the explanation, defines labels 0–10, and lists every
token with an explicit index:

```text
Feature explanation: references to domestic cats

Use integer labels from 0 (inactive) to 10 (strongest activation).

Tokens:
[
  {"index": 0, "token": "A"},
  {"index": 1, "token": " cat"},
  {"index": 2, "token": " ran"}
]
```

A strict JSON schema enforces the output contract:

```json
{"activations": [0, 8, 0]}
```

The hard labels are used to locate the corresponding tokens in the model's
generated response. At each location, the modern parser collects numeric
alternatives from the generated token's top log probabilities, restricts them
to 0–10, renormalizes, and computes an expected activation using the same
weighted-average idea.

This produces decimal predictions even when the hard JSON label is zero. For
example, if the valid alternatives at a generated zero are

```text
P(0) = 0.90, P(1) = 0.07, P(2) = 0.02, P(3) = 0.01,
```

the saved prediction is `0.14`, not `0`.

### The central statistical difference

The two expected values do not represent the same conditional probability:

- Historical: probability of numeric alternatives at a fixed `unknown` token
  inside an already supplied prompt.
- Modern: probability of numeric alternatives at a label position chosen while
  the assistant is generating a JSON sequence.

Modern labels are generated autoregressively. The prediction for token 40 is
conditioned on the labels already generated for tokens 0–39. This can encourage
locally consistent activation spans. In the historical fixed-prompt procedure,
all target positions contain `unknown`; the model does not observe its own
earlier numeric predictions. This difference could change—and potentially
increase—correlations, although its magnitude and direction have not been
isolated in this experiment.

## Concrete end-to-end example

Consider a toy feature that activates on tokens referring to cats.

### Explanation records

Suppose one of the five top records contains raw activations:

```text
tokens:      ["The", " cat", " slept"]
activations: [ 0.0,   4.0,     1.0]
```

With maximum activation 4.0, both evaluators display `[0, 10, 2]`.

The historical evaluator embeds the record among worked neuron examples and
asks GPT-4 to complete:

```text
Explanation of neuron N behavior: the main thing this neuron does is find
```

The modern evaluator supplies the record without demonstrations and asks
GPT-4.1-mini for a concise explanation. Suppose both return the semantically
equivalent explanation `references to domestic cats`.

### Held-out simulation record

Now consider a held-out record:

```text
tokens:             ["A", " cat", " ran"]
actual activations: [0.1,  3.8,    0.4]
```

In the historical evaluator, the target prompt literally contains:

```text
A       unknown
 cat    unknown
 ran    unknown
```

Suppose the echoed log probabilities at the ` cat` activation position imply:

```text
P(0)=0.05, P(7)=0.15, P(8)=0.50, P(9)=0.20, P(10)=0.10.
```

After renormalization, its expected prediction is:

$$
0(0.05)+7(0.15)+8(0.50)+9(0.20)+10(0.10)=7.85.
$$

In the modern evaluator, the model might generate:

```json
{"activations": [0, 8, 0]}
```

At the generated `8`, suppose its alternatives imply:

```text
P(0)=0.01, P(7)=0.14, P(8)=0.70, P(9)=0.10, P(10)=0.05.
```

The modern expected prediction is `7.98`. The two pipelines agree that the cat
token activates strongly, but their decimal predictions differ because they
query different models under different conditioning contexts.

This process is repeated for every token in five held-out top records and five
random records. Each evaluator then correlates its 640 predicted values with
the same-shaped vector of 640 raw feature activations. Pearson correlation is
invariant to positive linear rescaling, so the fact that predictions lie on a
0–10 scale while raw feature activations have another scale is not itself a
problem. Differences in ranking and token-level pattern are what affect the
score.

## Operational changes in the modern implementation

The modern path also introduces engineering changes that make a large paid run
safer and auditable:

- credentials are loaded from an untracked `.env` rather than an imported
  `secrets.json` file;
- the model snapshot is recorded explicitly;
- simulator output is constrained to exactly 64 labels using a JSON schema;
- missing log probabilities, malformed JSON, incorrect label counts, and token
  alignment failures are treated as errors rather than silently repaired;
- transient failures use bounded retries and exponential backoff;
- requests are paced and concurrency is bounded;
- explanation and completed-feature outputs are cached separately;
- each feature result records the explanation, selected indices, orientation,
  actual activations, predicted activations, and all three correlations;
- stages are resumable without rerunning completed features.

These reliability improvements justify implementing a replacement evaluator,
but they should not be presented as evidence that its numerical scores are
historically interchangeable.

## Why the modern replacement was necessary

The released evaluator depends on two retired components:

1. The historical GPT-4 snapshot used by the run is not identified precisely
   and the old model alias no longer denotes a stable reproduction target.
2. `text-davinci-003` is retired. More importantly, its evaluator relies on a
   legacy Completions operation combining `max_tokens=0`, `echo=true`, and
   prompt-position log probabilities. Current Chat Completions exposes
   log probabilities for generated output tokens, not arbitrary positions in
   the input prompt.

The modern evaluator therefore cannot reproduce the exact historical
probability query merely by changing a model name. Generating structured labels
and reading their output-token probabilities is an explicit replacement for an
unavailable measurement operation.

The replacement model was pinned as `gpt-4.1-mini-2025-04-14` to avoid a moving
alias. It supports current Chat Completions, structured output, and output-token
log probabilities, allowing the experiment to enforce a complete and uniform
prediction contract across all methods.

## Justification for retaining the modern results

The modern results remain informative for controlled comparisons within the
modern evaluator.

### Common evaluator across methods

Within each reported panel, SAE, baseline ICA, and Fixed ICA use the same:

- replacement model and pinned snapshot;
- explanation and simulation prompts;
- OpenWebText evaluation fragments;
- feature count;
- top/random split logic;
- activation normalization;
- structured-output and logprob parser;
- scoring function;
- retry and exclusion rules.

The evaluator replacement therefore does not directly give one method a
different model or prompt. Method comparisons are paired at the protocol level.

### The main claim is evaluator-conditional

The strongest supported claim is not that modern scores reproduce the values in
Figure 2. It is:

> Under a fixed modern automated-interpretability evaluator, the released
> positive-tail ICA measurement is sensitive to ICA's arbitrary fitted sign,
> and a fitting-data stronger-tail orientation substantially changes ICA's
> measured interpretability relative to the unmodified ICA evaluation.

This is a legitimate within-evaluator result. It should be described as
conditional on the chosen evaluator, because a different explainer/simulator
could interact differently with SAE and ICA features.

### Independent evaluator evidence

The optional Inkling-explainer/Qwen-simulator evaluation changes the evaluator
again but reproduces the main qualitative pattern: SAE leads in early layers,
while fitting-data orientation raises ICA's score and narrows or reverses the
gap in later layers. Agreement across these two modern pipelines provides some
evidence that the sign-sensitivity result is not unique to GPT-4.1-mini's exact
prompt behavior. It does not make either pipeline numerically comparable with
the historical Figure 2 evaluator.

## Required limitations and reporting language

The following statements should accompany the modern results:

1. **Do not compare absolute scores with Figure 2.** The model, prompts, and
   probability-query mechanism changed, and the modern evaluator can produce a
   different correlation scale.
2. **Call this a modern reproduction or re-evaluation.** Do not describe it as
   an exact reproduction of the historical automated evaluator.
3. **State both model substitutions.** GPT-4 was replaced as explainer, and
   `text-davinci-003` was replaced as simulator.
4. **State the prompt substitution.** Historical few-shot prompts were replaced
   with custom zero-shot prompts.
5. **State the inference substitution.** Echoed input-position log probabilities
   were replaced with structured label generation and output-token log
   probabilities.
6. **Keep claims within evaluator.** SAE-versus-ICA and ICA-versus-Fixed-ICA
   comparisons use a common modern evaluator; conclusions about historical
   score magnitudes are unsupported.
7. **Acknowledge possible evaluator interaction.** Using the same evaluator for
   all methods controls the immediate comparison but does not prove that the
   replacement is equally sensitive to every kind of feature.

A concise manuscript-ready description is:

> The original Figure 2 evaluator depended on retired GPT-4 and
> `text-davinci-003` interfaces. We therefore retain its feature-selection,
> top/random validation split, 0–10 activation representation, expected-value
> prediction, and correlation-based scoring structure, while replacing the
> historical few-shot evaluator with a pinned GPT-4.1-mini structured-output
> evaluator. The replacement generates per-token activation labels and derives
> expected activations from output-token log probabilities; the historical
> simulator instead obtained probabilities at echoed `unknown` positions in a
> fixed prompt. Because the models, prompts, and probability-query mechanism
> differ, our absolute scores are not directly comparable with Figure 2. All
> methods within our reproduction use the same replacement evaluator, so we use
> these scores only for controlled within-evaluator comparisons.

## Source map

- Historical orchestration and model names: `interpret.py`
- Modern explanation prompts: `modern_interpret.py`,
  `build_explainer_messages`
- Modern simulation prompts: `modern_interpret.py`,
  `build_simulator_messages`
- Modern expected-value extraction: `modern_interpret.py`,
  `expected_activations_from_completion`
- Modern selection and split: `modern_interpret.py`,
  `select_record_indices`
- Modern scoring: `modern_interpret.py`, `_pearson` and
  `interpret_features_async`
- Experiment grid and Fixed ICA orientation: `reproduce_ica_vs_sae.py`
- Historical external implementation: OpenAI
  `automated-interpretability`, pre-fork revision `8be4557`

For current API context, the OpenAI legacy Completions reference still
documents prompt echo, whereas current Chat Completions documents log
probabilities on generated completion tokens. The historical model snapshots
remain unavailable, which is why the replacement should be documented rather
than presented as transparent API maintenance.

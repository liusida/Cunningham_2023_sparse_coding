"""Run the ICA-versus-SAE evaluator with Inkling explanations and Qwen predictions."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import re
import time
import traceback
from typing import Any, Sequence

import numpy as np
import torch
from dotenv import load_dotenv
import tinker
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPOSITORY / "reproductions" / "ica_vs_sae"
PIPELINE = "tinker-inkling-qwen"
EXPLAINER_MODEL = "thinkingmachines/Inkling"
SIMULATOR_MODEL = "Qwen/Qwen3.8-27B"
DATASETS = ("pile10k", "openwebtext")
LAYERS = tuple(range(6))
METHOD_SOURCES = {
    "sae": ("sae", False),
    "ica": ("ica", False),
    "fixed_ica": ("ica", True),
    "ica_full": ("ica_full", False),
    "fixed_ica_full": ("ica_full", True),
}
FRAGMENT_LEN = 64
TOP_RECORDS = 20
RANDOM_RECORDS = 20
EXAMPLES_PER_SPLIT = 5


def selected(value: str, choices: Sequence[str]) -> Sequence[str]:
    return choices if value == "all" else (value,)


def load_tokens(path: Path) -> list[list[str]]:
    with path.open() as handle:
        return [json.loads(line)["tokens"] for line in handle]


def load_fitting_orientations(layer_dir: Path, source: str, count: int) -> list[int]:
    path = layer_dir / "fitting_orientations" / f"{source}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}; run `python reproduce_ica_vs_sae.py orient-ica` first"
        )
    record = json.loads(path.read_text())
    signs = [int(value) for value in record.get("orientations", [])]
    if len(signs) < count or any(value not in (-1, 1) for value in signs[:count]):
        raise ValueError(f"Invalid fitting orientations in {path}")
    return signs[:count]


def select_indices(values: np.ndarray, seed: int) -> tuple[list[int], list[int], list[int]]:
    maxima = values.max(axis=1)
    top = np.argsort(maxima)[::-1][:TOP_RECORDS].tolist()
    if len(top) < TOP_RECORDS:
        raise ValueError("Not enough fragments to construct the top records")
    ordering = torch.randperm(
        len(maxima), generator=torch.Generator().manual_seed(seed)
    ).tolist()
    random_records = [index for index in ordering if maxima[index] != 0][:RANDOM_RECORDS]
    if len(random_records) < RANDOM_RECORDS:
        raise ValueError("Fewer than 20 fragments have nonzero maximum activation")
    return (
        top[0::4][:EXAMPLES_PER_SPLIT],
        top[2::4][:EXAMPLES_PER_SPLIT],
        random_records[1::3][:EXAMPLES_PER_SPLIT],
    )


def normalise(values: np.ndarray) -> np.ndarray:
    positive = np.maximum(np.asarray(values, dtype=np.float64), 0)
    scale = positive.max(initial=0)
    if scale <= 0:
        return np.zeros_like(values, dtype=np.int64)
    return np.minimum(10, np.floor(10 * positive / scale)).astype(np.int64)


def explanation_messages(tokens: list[list[str]], values: np.ndarray) -> list[dict[str, str]]:
    blocks = []
    for number, (record_tokens, record_values) in enumerate(
        zip(tokens, normalise(values)), 1
    ):
        rows = [
            f"{index:02d}\t{json.dumps(token)}\t{int(value)}"
            for index, (token, value) in enumerate(zip(record_tokens, record_values))
        ]
        blocks.append(f"Fragment {number}:\nindex\ttoken\tactivation\n" + "\n".join(rows))
    return [
        {
            "role": "system",
            "content": (
                "You analyze language-model features. Infer one concise pattern that explains "
                "which tokens activate the feature. Do not mention fragment numbers or speculate "
                "about the experiment."
            ),
        },
        {
            "role": "user",
            "content": (
                "0 means inactive and 10 is the strongest positive activation.\n\n"
                "Here are five top-activating examples:\n\n"
                + "\n\n".join(blocks)
                + "\n\nReturn only a concise explanation of the feature's behavior."
            ),
        },
    ]


def simulator_messages(explanation: str, tokens: list[str]) -> list[dict[str, str]]:
    indexed = [{"index": index, "token": token} for index, token in enumerate(tokens)]
    return [
        {
            "role": "system",
            "content": (
                "Predict one activation for every token using only the supplied feature "
                "explanation. Return valid JSON with exactly one key, activations. Its value "
                "must be an object with exactly the string keys 0 through 63, each mapped to "
                "one integer activation label. Do not omit or add keys."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Feature explanation: {explanation}\n\n"
                "Use integer labels from 0 (inactive) to 10 (strongest activation).\n\n"
                "Count against the explicit indices 0 through 63 before answering. "
                "Return compact JSON only.\n\n"
                f"Tokens:\n{json.dumps(indexed, ensure_ascii=False)}"
            ),
        },
    ]


def flatten(model_input: types.ModelInput) -> list[int]:
    return [token for chunk in model_input.chunks for token in chunk.tokens]


def sample_text(
    sampler: Any,
    tokenizer: Any,
    renderer: Any,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float = 0.0,
) -> tuple[list[int], Any, str]:
    if renderer.__class__.__name__ == "TmlV0Renderer":
        prompt = renderer.build_generation_prompt(messages, effort=0.0)
    else:
        prompt = renderer.build_generation_prompt(messages)
    prompt_tokens = flatten(prompt)
    result = sampler.sample(
        prompt=prompt,
        num_samples=1,
        sampling_params=types.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=renderer.get_stop_sequences(),
        ),
    ).result()
    sequence = result.sequences[0]
    parsed, _ = renderer.parse_response(list(sequence.tokens))
    text = renderers.get_text_content(parsed).strip()
    return prompt_tokens, sequence, text


def _balanced_value_end(raw: str, start: int) -> int | None:
    """Return the exclusive end of a JSON-like array/object starting at ``start``."""
    opener = raw[start]
    closer = {"[": "]", "{": "}"}.get(opener)
    if closer is None:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(raw)):
        character = raw[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == opener:
            depth += 1
        elif character == closer:
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def _label_spans(raw: str) -> list[tuple[int, int]]:
    """Locate exactly one 64-label activations value in a model completion."""
    candidates: list[tuple[int, int]] = []
    for match in re.finditer(r'"activations"\s*:\s*', raw):
        start = match.end()
        if start < len(raw) and raw[start] in "[{":
            end = _balanced_value_end(raw, start)
            if end is not None:
                candidates.append((start, end))

    # Also accept the documented legacy fallback of a bare 64-value array. Do
    # not span from the first '[' to the last ']': prose or echoed input can
    # contain unrelated bracketed numbers.
    if not candidates:
        for match in re.finditer(r"\[", raw):
            end = _balanced_value_end(raw, match.start())
            if end is not None:
                candidates.append((match.start(), end))

    failures = []
    for start, end in candidates:
        value = raw[start:end]
        if value.startswith("{"):
            pairs = list(re.finditer(r'"(\d+)"\s*:\s*(-?\d+)', value))
            keys = [int(pair.group(1)) for pair in pairs]
            failures.append(f"keys={keys!r}")
            if keys == list(range(FRAGMENT_LEN)):
                return [
                    (start + pair.start(2), start + pair.end(2)) for pair in pairs
                ]
        else:
            numbers = list(re.finditer(r"(?<![\d.])-?\d+(?![\d.])", value))
            failures.append(f"array_length={len(numbers)}")
            if len(numbers) == FRAGMENT_LEN:
                return [
                    (start + number.start(), start + number.end()) for number in numbers
                ]
    detail = ", ".join(failures) if failures else "no activations object or array found"
    raise ValueError(f"Expected keys 0..63 or an exact 64-value array; got {detail}")


def expected_labels(
    sampler: Any, tokenizer: Any, prompt_tokens: list[int], sequence: Any
) -> np.ndarray:
    completion_tokens = list(sequence.tokens)
    raw = tokenizer.decode(completion_tokens, skip_special_tokens=True)
    spans = _label_spans(raw)

    offsets = [0]
    for index in range(1, len(completion_tokens) + 1):
        offsets.append(
            len(tokenizer.decode(completion_tokens[:index], skip_special_tokens=True))
        )
    token_indices = []
    for left, _ in spans:
        token_index = next(
            (
                index
                for index in range(len(completion_tokens))
                if offsets[index] <= left < offsets[index + 1]
            ),
            None,
        )
        if token_index is None:
            raise ValueError("Could not align a numeric label to a generated token")
        token_indices.append(token_index)

    scored = sampler.sample(
        prompt=types.ModelInput.from_ints(prompt_tokens + completion_tokens),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=1, temperature=0.0),
        include_prompt_logprobs=True,
        topk_prompt_logprobs=20,
    ).result()
    alternatives = scored.topk_prompt_logprobs
    expected = []
    for token_index in token_indices:
        position = len(prompt_tokens) + token_index
        candidates: dict[int, float] = {}
        for token_id, logprob in alternatives[position] or []:
            piece = tokenizer.decode([token_id], skip_special_tokens=True)
            stripped = piece.strip(" \t\r\n,[]{}:\"")
            if re.fullmatch(r"\d+", stripped):
                label = int(stripped)
                if 0 <= label <= 10:
                    candidates[label] = max(candidates.get(label, -math.inf), float(logprob))
        if not candidates:
            raise ValueError(f"No valid label alternatives at generated token {token_index}")
        labels = np.asarray(list(candidates), dtype=np.float64)
        logps = np.asarray(list(candidates.values()), dtype=np.float64)
        probabilities = np.exp(logps - logps.max())
        probabilities /= probabilities.sum()
        expected.append(float(np.dot(labels, probabilities)))
    return np.asarray(expected, dtype=np.float64)


def correlation(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if actual.std() == 0 or predicted.std() == 0:
        return float("nan")
    return float(np.corrcoef(actual, predicted)[0, 1])


def retry(call: Any, label: str, retries: int) -> Any:
    delay = 1.0
    for attempt in range(retries + 1):
        try:
            return call(attempt)
        except Exception as error:
            if attempt == retries:
                raise
            wait = 1.0 if isinstance(error, ValueError) else delay
            print(
                f"Retry {label}: attempt {attempt + 1}/{retries + 1} failed with "
                f"{type(error).__name__}; waiting {wait:.1f}s",
                flush=True,
            )
            time.sleep(wait)
            if not isinstance(error, ValueError):
                delay = min(2 * delay, 30.0)
    raise AssertionError("unreachable")


def run_feature(
    *,
    feature: int,
    values: np.ndarray,
    tokens: list[list[str]],
    orientation: int,
    seed: int,
    output_dir: Path,
    explainer_sampler: Any,
    explainer_tokenizer: Any,
    explainer_renderer: Any,
    simulator_sampler: Any,
    simulator_tokenizer: Any,
    simulator_renderer: Any,
    max_concurrent: int,
    retries: int,
    overwrite: bool,
) -> None:
    destination = output_dir / f"feature_{feature}.json"
    explanation_cache = output_dir / f"feature_{feature}.explanation.json"
    if destination.exists() and not overwrite:
        print(f"Feature {feature} already completed, skipping", flush=True)
        return

    oriented = values * orientation
    try:
        train_top, valid_top, valid_random = select_indices(oriented, seed + feature)
    except ValueError as error:
        destination.write_text(
            json.dumps(
                {
                    "provider": "tinker",
                    "explainer_model": EXPLAINER_MODEL,
                    "simulator_model": SIMULATOR_MODEL,
                    "feature": feature,
                    "status": "skipped",
                    "reason": str(error),
                },
                indent=2,
            )
            + "\n"
        )
        print(f"Feature {feature} skipped: {error}", flush=True)
        return

    if explanation_cache.exists() and not overwrite:
        explanation = json.loads(explanation_cache.read_text())["explanation"]
    else:
        explanation = retry(
            lambda _: sample_text(
                explainer_sampler,
                explainer_tokenizer,
                explainer_renderer,
                explanation_messages([tokens[index] for index in train_top], oriented[train_top]),
                1024,
            )[2],
            f"feature {feature} explainer",
            retries,
        )
        explanation_cache.write_text(
            json.dumps(
                {
                    "provider": "tinker",
                    "explainer_model": EXPLAINER_MODEL,
                    "feature": feature,
                    "orientation": orientation,
                    "explanation": explanation,
                },
                indent=2,
            )
            + "\n"
        )
    print(f"Feature {feature} explanation: {explanation}", flush=True)

    validation = valid_top + valid_random

    def score_fragment(item: tuple[int, int]) -> np.ndarray:
        number, index = item
        failure_dir = output_dir / "failures"

        def attempt(attempt_number: int) -> np.ndarray:
            prompt_tokens, sequence, raw = sample_text(
                simulator_sampler,
                simulator_tokenizer,
                simulator_renderer,
                simulator_messages(explanation, tokens[index]),
                768,
                temperature=0.0 if attempt_number == 0 else 0.2,
            )
            try:
                return expected_labels(
                    simulator_sampler, simulator_tokenizer, prompt_tokens, sequence
                )
            except ValueError as error:
                failure_dir.mkdir(exist_ok=True)
                failure_path = failure_dir / (
                    f"feature_{feature}_fragment_{number}_attempt_{attempt_number + 1}.json"
                )
                failure_path.write_text(
                    json.dumps(
                        {
                            "feature": feature,
                            "fragment_number": number,
                            "fragment_index": index,
                            "attempt": attempt_number + 1,
                            "reason": str(error),
                            "tokens": tokens[index],
                            "raw_completion": raw,
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                raise

        prediction = retry(attempt, f"feature {feature} fragment {number}", retries)
        print(f"Feature {feature}: scored fragment {number}/{len(validation)}", flush=True)
        return prediction

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        predictions = list(executor.map(score_fragment, enumerate(validation, 1)))
    predicted = np.stack(predictions)
    actual = oriented[validation].astype(np.float64)
    record = {
        "provider": "tinker",
        "pipeline": PIPELINE,
        "explainer_model": EXPLAINER_MODEL,
        "simulator_model": SIMULATOR_MODEL,
        "feature": feature,
        "mode": "legacy_nonnegative",
        "orientation": orientation,
        "explanation": explanation,
        "combined_score": correlation(actual, predicted),
        "top_score": correlation(
            actual[:EXAMPLES_PER_SPLIT], predicted[:EXAMPLES_PER_SPLIT]
        ),
        "random_score": correlation(
            actual[EXAMPLES_PER_SPLIT:], predicted[EXAMPLES_PER_SPLIT:]
        ),
        "top_indices": valid_top,
        "random_indices": valid_random,
        "actual_activations": actual.tolist(),
        "predicted_activations": predicted.tolist(),
    }
    destination.write_text(json.dumps(record, indent=2, allow_nan=True) + "\n")
    print(
        f"Feature {feature}: combined={record['combined_score']:.3f}, "
        f"top={record['top_score']:.3f}, random={record['random_score']:.3f}",
        flush=True,
    )


def run_feature_resilient(*, fail_fast: bool = False, **kwargs: Any) -> bool:
    """Run one feature without allowing a bad model response to abort the condition.

    Failures are written separately from ``feature_N.json`` so that a later invocation
    retries the feature instead of treating it as completed.  A successful retry removes
    the stale error record.
    """
    feature = int(kwargs["feature"])
    output_dir = Path(kwargs["output_dir"])
    error_path = output_dir / f"feature_{feature}.error.json"
    try:
        run_feature(**kwargs)
    except Exception as error:
        error_path.write_text(
            json.dumps(
                {
                    "provider": "tinker",
                    "pipeline": PIPELINE,
                    "feature": feature,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "reason": str(error),
                    "traceback": traceback.format_exc(),
                },
                indent=2,
            )
            + "\n"
        )
        print(
            f"Feature {feature} failed after all retries: "
            f"{type(error).__name__}: {error}",
            flush=True,
        )
        if fail_fast:
            raise
        print(f"Feature {feature}: continuing to the next feature", flush=True)
        return False
    error_path.unlink(missing_ok=True)
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dataset", choices=("all", *DATASETS), default="all")
    parser.add_argument("--layer", choices=("all", *map(str, LAYERS)), default="all")
    parser.add_argument("--method", choices=("all", *METHOD_SOURCES), default="all")
    parser.add_argument("--feature-start", type=int, default=0)
    parser.add_argument("--n-features", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-concurrent", type=int, default=10)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--env-file", type=Path, default=REPOSITORY / ".env")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="abort a condition when one feature fails instead of recording and continuing",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.feature_start < 0 or args.n_features <= 0 or args.max_concurrent <= 0:
        raise ValueError("Feature range and concurrency must be positive")
    methods = tuple(METHOD_SOURCES) if args.method == "all" else (args.method,)
    datasets = selected(args.dataset, DATASETS)
    layers = tuple(int(value) for value in selected(args.layer, tuple(map(str, LAYERS))))
    end = args.feature_start + args.n_features

    tasks = []
    for dataset in datasets:
        for layer in layers:
            layer_dir = args.root / dataset / f"layer{layer}"
            for method in methods:
                source, fixed = METHOD_SOURCES[method]
                evaluation = layer_dir / "evaluation_data" / source
                for path in (
                    evaluation / "feature_activations.npy",
                    evaluation / "fragments.jsonl",
                ):
                    if not path.exists():
                        raise FileNotFoundError(path)
                activations = np.load(evaluation / "feature_activations.npy", mmap_mode="r")
                if end > activations.shape[2]:
                    raise ValueError(
                        f"Requested features {args.feature_start}:{end}, but {evaluation} "
                        f"contains {activations.shape[2]}"
                    )
                orientations = (
                    load_fitting_orientations(layer_dir, source, end) if fixed else [1] * end
                )
                tasks.append((dataset, layer, method, source, orientations))

    print(
        f"Pipeline {PIPELINE}: {len(tasks)} method/layer conditions, "
        f"features {args.feature_start}:{end}",
        flush=True,
    )
    if args.dry_run:
        for dataset, layer, method, _, _ in tasks:
            print(f"  {dataset}/layer{layer}/{method}")
        return

    load_dotenv(args.env_file)
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError(f"TINKER_API_KEY is missing from {args.env_file}")
    service = tinker.ServiceClient()
    explainer_sampler = service.create_sampling_client(base_model=EXPLAINER_MODEL)
    simulator_sampler = service.create_sampling_client(base_model=SIMULATOR_MODEL)
    explainer_tokenizer = get_tokenizer(EXPLAINER_MODEL)
    simulator_tokenizer = get_tokenizer(SIMULATOR_MODEL)
    explainer_renderer = renderers.get_renderer(
        "tml_v0", explainer_tokenizer, model_name=EXPLAINER_MODEL
    )
    simulator_renderer = renderers.get_renderer(
        "qwen3_8_disable_thinking",
        simulator_tokenizer,
        model_name=SIMULATOR_MODEL,
    )

    for dataset, layer, method, source, orientations in tasks:
        layer_dir = args.root / dataset / f"layer{layer}"
        evaluation = layer_dir / "evaluation_data" / source
        activations = np.load(evaluation / "feature_activations.npy", mmap_mode="r")
        tokens = load_tokens(evaluation / "fragments.jsonl")
        if len(tokens) != activations.shape[0]:
            raise ValueError(f"Token/activation length mismatch under {evaluation}")
        output = layer_dir / "results" / PIPELINE / method
        output.mkdir(parents=True, exist_ok=True)
        print(f"★ Tinker: {dataset}/layer{layer}/{method}", flush=True)
        for feature in range(args.feature_start, end):
            run_feature_resilient(
                feature=feature,
                values=np.asarray(activations[:, :, feature], dtype=np.float32),
                tokens=tokens,
                orientation=orientations[feature],
                seed=args.seed,
                output_dir=output,
                explainer_sampler=explainer_sampler,
                explainer_tokenizer=explainer_tokenizer,
                explainer_renderer=explainer_renderer,
                simulator_sampler=simulator_sampler,
                simulator_tokenizer=simulator_tokenizer,
                simulator_renderer=simulator_renderer,
                max_concurrent=args.max_concurrent,
                retries=args.retries,
                overwrite=args.overwrite,
                fail_fast=args.fail_fast,
            )


if __name__ == "__main__":
    main()

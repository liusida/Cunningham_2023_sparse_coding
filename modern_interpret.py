"""Modern, resumable auto-interpretation for the Figure-2 learned dictionary.

This module intentionally does not import the historical ``interpret.py`` module:
that module reads ``secrets.json`` at import time and expects the retired OpenAI
Completions response format.  Here credentials come from ``.env`` and current Chat
Completions log-probabilities are adapted to the same correlation-style score.
"""

from __future__ import annotations

import asyncio
import datetime
import itertools
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from datasets import load_dataset

from activation_dataset import make_tensor_name


FRAGMENT_LEN = 64
TOP_RECORDS = 20
RANDOM_RECORDS = 20
EXAMPLES_PER_SPLIT = 5


def _timestamp() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass
class FeatureResult:
    feature: int
    mode: str
    orientation: int
    explanation: str
    combined_score: float
    top_score: float
    random_score: float
    top_indices: list[int]
    random_indices: list[int]
    actual_activations: list[list[float]]
    predicted_activations: list[list[float]]


@dataclass
class RateLimitState:
    remaining_requests: str = "unknown"
    reset_requests: str = "unknown"

    def update(self, headers: Any) -> None:
        if headers is None:
            return
        self.remaining_requests = headers.get(
            "x-ratelimit-remaining-requests", self.remaining_requests
        )
        self.reset_requests = headers.get(
            "x-ratelimit-reset-requests", self.reset_requests
        )


class RequestPacer:
    """Enforce a minimum interval between API request start times."""

    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = max(0.0, interval_seconds)
        self._lock = asyncio.Lock()
        self._last_start = 0.0

    async def wait(self) -> None:
        async with self._lock:
            now = asyncio.get_running_loop().time()
            delay = self.interval_seconds - (now - self._last_start)
            if delay > 0:
                await asyncio.sleep(delay)
            self._last_start = asyncio.get_running_loop().time()


def _pearson(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if actual.size != predicted.size:
        raise ValueError(f"Score length mismatch: {actual.size} != {predicted.size}")
    if actual.size == 0 or actual.std() == 0 or predicted.std() == 0:
        return float("nan")
    return float(np.corrcoef(actual, predicted)[0, 1])


def _normalise_for_prompt(values: np.ndarray, mode: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if mode in ("legacy_nonnegative", "oriented_nonnegative"):
        positive = np.maximum(values, 0)
        scale = positive.max(initial=0)
        if scale <= 0:
            return np.zeros_like(values, dtype=np.int64)
        return np.minimum(10, np.floor(10 * positive / scale)).astype(np.int64)
    if mode == "signed":
        scale = np.abs(values).max(initial=0)
        if scale <= 0:
            return np.zeros_like(values, dtype=np.int64)
        magnitude = np.floor(10 * np.abs(values) / scale)
        return (np.sign(values) * np.minimum(10, magnitude)).astype(np.int64)
    raise ValueError(f"Unknown evaluation mode: {mode}")


def choose_orientation(feature_values: np.ndarray, mode: str) -> int:
    """Return a reproducible sign; legacy mode preserves the fitted ICA sign."""
    if mode == "legacy_nonnegative":
        return 1
    if mode not in ("oriented_nonnegative", "signed"):
        raise ValueError(f"Unknown evaluation mode: {mode}")
    maximum = float(np.max(feature_values))
    minimum = float(np.min(feature_values))
    return 1 if maximum >= abs(minimum) else -1


def choose_top_absolute_orientation(feature_values: np.ndarray) -> int:
    """Orient an ICA component by the signed mass of its 20 strongest values."""
    flattened = np.asarray(feature_values, dtype=np.float64).reshape(-1)
    count = min(TOP_RECORDS, len(flattened))
    strongest = np.argpartition(np.abs(flattened), -count)[-count:]
    return 1 if flattened[strongest].sum() >= 0 else -1


def select_record_indices(
    feature_values: np.ndarray,
    mode: str,
    seed: int,
) -> tuple[int, list[int], list[int], list[int]]:
    """Match the original interleaved train/valid split for one feature."""
    orientation = choose_orientation(feature_values, mode)
    oriented = feature_values * orientation
    maxima = oriented.max(axis=1)

    top_order = np.argsort(maxima)[::-1][:TOP_RECORDS].tolist()
    if len(top_order) < TOP_RECORDS:
        raise ValueError("Not enough fragments to construct the top records")

    generator = torch.Generator().manual_seed(seed)
    ordering = torch.randperm(len(maxima), generator=generator).tolist()
    random_records = [i for i in ordering if maxima[i] != 0][:RANDOM_RECORDS]
    if len(random_records) < RANDOM_RECORDS:
        raise ValueError("Fewer than 20 fragments have nonzero maximum activation")

    # NeuronRecord uses four interleaved top splits:
    # train=0::4, calibration=1::4, valid=2::4, test=3::4.
    train_top = top_order[0::4][:EXAMPLES_PER_SPLIT]
    valid_top = top_order[2::4][:EXAMPLES_PER_SPLIT]
    # Random records use three splits: calibration=0::3, valid=1::3, test=2::3.
    valid_random = random_records[1::3][:EXAMPLES_PER_SPLIT]
    return orientation, train_top, valid_top, valid_random


def _format_explanation_records(
    token_records: Sequence[Sequence[str]],
    activation_records: np.ndarray,
    mode: str,
) -> str:
    normalised = _normalise_for_prompt(activation_records, mode)
    blocks = []
    for record_number, (tokens, activations) in enumerate(
        zip(token_records, normalised), start=1
    ):
        rows = [f"{i:02d}\t{json.dumps(token)}\t{int(value)}" for i, (token, value) in enumerate(zip(tokens, activations))]
        blocks.append(f"Fragment {record_number}:\nindex\ttoken\tactivation\n" + "\n".join(rows))
    return "\n\n".join(blocks)


def build_explainer_messages(
    token_records: Sequence[Sequence[str]],
    activation_records: np.ndarray,
    mode: str,
) -> list[dict[str, str]]:
    scale_description = (
        "0 means inactive and 10 is the strongest positive activation."
        if mode in ("legacy_nonnegative", "oriented_nonnegative")
        else "Activations range from -10 to 10; sign and magnitude both matter."
    )
    records = _format_explanation_records(token_records, activation_records, mode)
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
                f"{scale_description}\n\nHere are five top-activating examples:\n\n"
                f"{records}\n\nReturn only a concise explanation of the feature's behavior."
            ),
        },
    ]


def build_simulator_messages(
    explanation: str,
    tokens: Sequence[str],
    mode: str,
) -> list[dict[str, str]]:
    if mode in ("legacy_nonnegative", "oriented_nonnegative"):
        label_description = "Use integer labels from 0 (inactive) to 10 (strongest activation)."
    else:
        label_description = (
            "Use signed integer activations from -10 to 10. In the JSON output encode each "
            "activation by adding 10, so labels 0..20 represent activations -10..10."
        )
    indexed_tokens = [{"index": i, "token": token} for i, token in enumerate(tokens)]
    return [
        {
            "role": "system",
            "content": (
                "Predict one activation for every token using only the supplied feature "
                "explanation. Return valid JSON with exactly one key, activations, containing "
                "exactly 64 integer labels in token-index order."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Feature explanation: {explanation}\n\n{label_description}\n\n"
                f"Tokens:\n{json.dumps(indexed_tokens, ensure_ascii=False)}"
            ),
        },
    ]


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    raise TypeError(f"Cannot convert {type(value)} to a mapping")


def expected_activations_from_completion(choice: Any, mode: str, n_tokens: int) -> np.ndarray:
    """Extract per-label expected values from Chat Completions token logprobs.

    The generated hard JSON values locate the output labels. For each label, the
    top-logprob alternatives at the token containing that label are restricted to
    valid labels and renormalized, matching the historical evaluator's EV idea.
    Missing or unalignable log probabilities are errors so all grid cells use
    exactly the same scoring contract.
    """
    content = choice.message.content or ""
    payload = json.loads(content)
    hard_labels = payload.get("activations")
    if not isinstance(hard_labels, list) or len(hard_labels) != n_tokens:
        raise ValueError(f"Expected {n_tokens} activation labels, got {hard_labels!r}")

    maximum_label = 10 if mode in ("legacy_nonnegative", "oriented_nonnegative") else 20
    hard_labels = [int(x) for x in hard_labels]
    if any(label < 0 or label > maximum_label for label in hard_labels):
        raise ValueError(f"Activation labels must be within 0..{maximum_label}")

    logprob_content = getattr(getattr(choice, "logprobs", None), "content", None)
    if not logprob_content:
        raise ValueError("The model response did not include output-token log probabilities")

    # Locate only the integers inside the activations array, not incidental text.
    array_start = content.find("[")
    array_end = content.rfind("]")
    spans = list(re.finditer(r"(?<![\d.])-?\d+(?![\d.])", content[array_start : array_end + 1]))
    if len(spans) != n_tokens:
        raise ValueError("Could not align JSON activation labels with response text")
    absolute_spans = [(array_start + match.start(), array_start + match.end()) for match in spans]

    token_ranges = []
    cursor = 0
    for item in logprob_content:
        token = item.token
        token_ranges.append((cursor, cursor + len(token), item))
        cursor += len(token)

    expected = []
    for hard_label, (start, end) in zip(hard_labels, absolute_spans):
        item = next((entry for left, right, entry in token_ranges if left <= start and end <= right), None)
        if item is None:
            raise ValueError("Could not align an activation label with a logprob token")
        candidates: dict[int, float] = {}
        alternatives = list(item.top_logprobs or [])
        alternatives.append(item)
        for alternative in alternatives:
            stripped = alternative.token.strip(" \t\r\n,[]{}:\"")
            if re.fullmatch(r"\d+", stripped):
                label = int(stripped)
                if 0 <= label <= maximum_label:
                    candidates[label] = max(candidates.get(label, -math.inf), float(alternative.logprob))
        if not candidates:
            raise ValueError("No valid activation labels were present in top_logprobs")
        labels = np.asarray(list(candidates), dtype=np.float64)
        logps = np.asarray(list(candidates.values()), dtype=np.float64)
        probabilities = np.exp(logps - logps.max())
        probabilities /= probabilities.sum()
        expected.append(float(np.dot(labels, probabilities)))

    values = np.asarray(expected, dtype=np.float64)
    return values if mode in ("legacy_nonnegative", "oriented_nonnegative") else values - 10


async def _chat_with_retries(
    call: Any, retries: int, label: str, rate_limits: RateLimitState
) -> Any:
    delay = 1.0
    for attempt in range(retries + 1):
        try:
            return await call()
        except Exception as exc:
            response = getattr(exc, "response", None)
            rate_limits.update(getattr(response, "headers", None))
            # Programming/type errors are deterministic and retrying them only
            # consumes request allowance. API and response-validation failures
            # remain retryable below.
            if isinstance(exc, TypeError):
                raise
            if attempt == retries:
                raise
            print(
                f"[{_timestamp()}] Retry {label}: attempt {attempt + 1}/{retries + 1} "
                f"failed with {type(exc).__name__}; waiting {delay:.1f}s "
                f"(remaining_requests={rate_limits.remaining_requests}, "
                f"reset_requests={rate_limits.reset_requests})",
                flush=True,
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30)
    raise AssertionError("unreachable")


async def explain_feature(
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    retries: int,
    label: str = "explainer",
    rate_limits: RateLimitState | None = None,
    request_pacer: RequestPacer | None = None,
) -> str:
    rate_limits = rate_limits or RateLimitState()

    async def request() -> Any:
        if request_pacer is not None:
            await request_pacer.wait()
        limits = (
            {"max_completion_tokens": 160}
            if model.startswith("gpt-5")
            else {"temperature": 0, "max_tokens": 160}
        )
        raw = await client.chat.completions.with_raw_response.create(
            model=model, messages=messages, **limits
        )
        rate_limits.update(raw.headers)
        # In the installed OpenAI SDK, AsyncClient.with_raw_response returns a
        # LegacyAPIResponse whose parse() method is synchronous.
        return raw.parse()

    response = await _chat_with_retries(
        request,
        retries,
        label,
        rate_limits,
    )
    explanation = (response.choices[0].message.content or "").strip()
    if not explanation:
        raise ValueError("Explainer returned an empty explanation")
    return explanation


async def simulate_fragment(
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    mode: str,
    retries: int,
    semaphore: asyncio.Semaphore,
    label: str = "simulator",
    rate_limits: RateLimitState | None = None,
    request_pacer: RequestPacer | None = None,
) -> np.ndarray:
    rate_limits = rate_limits or RateLimitState()
    maximum_label = 10 if mode in ("legacy_nonnegative", "oriented_nonnegative") else 20

    async def request_and_parse() -> np.ndarray:
        async with semaphore:
            if request_pacer is not None:
                await request_pacer.wait()
            limits = (
                {"max_completion_tokens": 512}
                if model.startswith("gpt-5")
                else {"temperature": 0, "max_tokens": 256}
            )
            raw = await client.chat.completions.with_raw_response.create(
                model=model,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "token_activations",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "activations": {
                                    "type": "array",
                                    "items": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": maximum_label,
                                    },
                                    "minItems": FRAGMENT_LEN,
                                    "maxItems": FRAGMENT_LEN,
                                }
                            },
                            "required": ["activations"],
                            "additionalProperties": False,
                        },
                    },
                },
                logprobs=True,
                top_logprobs=20,
                **limits,
            )
            rate_limits.update(raw.headers)
            response = raw.parse()
        # Keep validation inside the retry boundary: malformed JSON or the wrong
        # number of labels is just as retryable as a transient API failure.
        return expected_activations_from_completion(response.choices[0], mode, FRAGMENT_LEN)

    return await _chat_with_retries(request_and_parse, retries, label, rate_limits)


def load_token_records(path: Path) -> list[list[str]]:
    records = []
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            records.append(record["tokens"])
    return records


async def interpret_features_async(
    *,
    activations_path: Path,
    tokens_path: Path,
    output_dir: Path,
    mode: str,
    n_features: int,
    seed: int,
    explainer_model: str,
    simulator_model: str,
    max_concurrent: int,
    request_delay: float,
    retries: int,
    env_file: Path,
    overwrite: bool,
    orient_signed: bool = False,
    source_orientations: Sequence[int] | None = None,
) -> None:
    from dotenv import load_dotenv
    from openai import AsyncOpenAI

    load_dotenv(env_file)
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(f"OPENAI_API_KEY is not set; add it to {env_file}")

    activations = np.load(activations_path, mmap_mode="r")
    token_records = load_token_records(tokens_path)
    if activations.shape[0] != len(token_records):
        raise ValueError("Token and activation artifact lengths differ")
    if activations.shape[1] != FRAGMENT_LEN:
        raise ValueError(f"Expected {FRAGMENT_LEN}-token fragments, got {activations.shape}")
    if n_features > activations.shape[2]:
        raise ValueError(f"Requested {n_features} features but artifact has {activations.shape[2]}")
    if source_orientations is not None:
        if not orient_signed:
            raise ValueError("Explicit source orientations require orient_signed=True")
        if len(source_orientations) < n_features:
            raise ValueError(
                f"Received {len(source_orientations)} orientations for {n_features} features"
            )
        if any(int(value) not in (-1, 1) for value in source_orientations[:n_features]):
            raise ValueError("Source orientations must contain only -1 or 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(max_concurrent)
    rate_limits = RateLimitState()
    request_pacer = RequestPacer(request_delay)

    for feature in range(n_features):
        destination = output_dir / f"feature_{feature}.json"
        explanation_cache = output_dir / f"feature_{feature}.explanation.json"
        if destination.exists() and not overwrite:
            print(f"[{_timestamp()}] Feature {feature} already completed, skipping", flush=True)
            continue

        feature_values = np.asarray(activations[:, :, feature], dtype=np.float32)
        if source_orientations is not None:
            source_orientation = int(source_orientations[feature])
        elif orient_signed:
            source_orientation = choose_top_absolute_orientation(feature_values)
        else:
            source_orientation = 1
        feature_values = feature_values * source_orientation
        try:
            orientation, train_top, valid_top, valid_random = select_record_indices(
                feature_values, mode, seed + feature
            )
        except ValueError as exc:
            skipped = {"feature": feature, "mode": mode, "status": "skipped", "reason": str(exc)}
            destination.write_text(json.dumps(skipped, indent=2) + "\n")
            print(f"[{_timestamp()}] Feature {feature} skipped: {exc}", flush=True)
            continue

        oriented = feature_values * orientation
        orientation *= source_orientation
        train_activations = oriented[train_top]
        if explanation_cache.exists():
            explanation = json.loads(explanation_cache.read_text())["explanation"]
        else:
            explanation = await explain_feature(
                client,
                explainer_model,
                build_explainer_messages(
                    [token_records[i] for i in train_top], train_activations, mode
                ),
                retries,
                label=f"feature {feature} explainer",
                rate_limits=rate_limits,
                request_pacer=request_pacer,
            )
            explanation_cache.write_text(
                json.dumps(
                    {
                        "feature": feature,
                        "mode": mode,
                        "orientation": orientation,
                        "explanation": explanation,
                    },
                    indent=2,
                )
                + "\n"
            )

        validation_indices = valid_top + valid_random
        predictions = await asyncio.gather(
            *[
                simulate_fragment(
                    client,
                    simulator_model,
                    build_simulator_messages(explanation, token_records[index], mode),
                    mode,
                    retries,
                    semaphore,
                    label=f"feature {feature} simulator fragment {fragment_number}",
                    rate_limits=rate_limits,
                    request_pacer=request_pacer,
                )
                for fragment_number, index in enumerate(validation_indices)
            ]
        )
        predicted = np.stack(predictions)
        actual = oriented[validation_indices].astype(np.float64)
        result = FeatureResult(
            feature=feature,
            mode=mode,
            orientation=orientation,
            explanation=explanation,
            combined_score=_pearson(actual, predicted),
            top_score=_pearson(actual[:EXAMPLES_PER_SPLIT], predicted[:EXAMPLES_PER_SPLIT]),
            random_score=_pearson(actual[EXAMPLES_PER_SPLIT:], predicted[EXAMPLES_PER_SPLIT:]),
            top_indices=valid_top,
            random_indices=valid_random,
            actual_activations=actual.tolist(),
            predicted_activations=predicted.tolist(),
        )
        destination.write_text(json.dumps(result.__dict__, indent=2, allow_nan=True) + "\n")
        print(
            f"[{_timestamp()}] Feature {feature}: combined={result.combined_score:.3f}, "
            f"top={result.top_score:.3f}, random={result.random_score:.3f} "
            f"(remaining_requests={rate_limits.remaining_requests}, "
            f"reset_requests={rate_limits.reset_requests})",
            flush=True,
        )
    await client.close()


def interpret_features(**kwargs: Any) -> None:
    asyncio.run(interpret_features_async(**kwargs))


async def _check_model_compatibility(model: str, env_file: Path) -> None:
    """Make one paid request and verify the simulator's exact response contract."""
    from dotenv import load_dotenv
    from openai import AsyncOpenAI

    load_dotenv(env_file)
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(f"OPENAI_API_KEY is not set; add it to {env_file}")
    client = AsyncOpenAI()
    try:
        prediction = await simulate_fragment(
            client,
            model,
            build_simulator_messages(
                "tokens containing the word test",
                [" test"] * FRAGMENT_LEN,
                "legacy_nonnegative",
            ),
            "legacy_nonnegative",
            retries=0,
            semaphore=asyncio.Semaphore(1),
        )
    finally:
        await client.close()
    if prediction.shape != (FRAGMENT_LEN,) or not np.isfinite(prediction).all():
        raise RuntimeError(f"{model} returned an invalid simulator prediction")


def check_model_compatibility(model: str, env_file: Path) -> None:
    asyncio.run(_check_model_compatibility(model, env_file))


def extract_evaluation_activations(
    *,
    learned_dict: Any,
    model_name: str,
    layer: int,
    layer_loc: str,
    dataset_name: str,
    output_dir: Path,
    device: str,
    n_fragments: int,
    n_features: int,
    batch_size: int,
    seed: int,
    start_line: int,
    overwrite: bool,
) -> tuple[Path, Path]:
    """Create reusable token and feature-activation artifacts on CPU-backed storage."""
    from numpy.lib.format import open_memmap
    from transformer_lens import HookedTransformer

    output_dir.mkdir(parents=True, exist_ok=True)
    activations_path = output_dir / "feature_activations.npy"
    tokens_path = output_dir / "fragments.jsonl"
    if (activations_path.exists() or tokens_path.exists()) and not overwrite:
        raise FileExistsError(f"Evaluation artifacts already exist under {output_dir}")

    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()
    learned_dict.to_device(device)
    dataset = load_dataset(dataset_name, split="train", streaming=True)
    iterator: Iterable[dict[str, Any]] = itertools.islice(iter(dataset), start_line, None)
    rng = np.random.default_rng(seed)
    tensor_name = make_tensor_name(layer, layer_loc, model.cfg.model_name)
    activation_store = open_memmap(
        activations_path,
        mode="w+",
        dtype=np.float16,
        shape=(n_fragments, FRAGMENT_LEN, n_features),
    )

    added = 0
    rejected = 0
    with tokens_path.open("w") as token_file, torch.no_grad():
        while added < n_fragments:
            batch_tokens: list[torch.Tensor] = []
            batch_strings: list[list[str]] = []
            while len(batch_tokens) < min(batch_size, n_fragments - added):
                record = next(iterator)
                sentence_tokens = model.to_tokens(record["text"], prepend_bos=False)
                length = sentence_tokens.shape[1]
                if length < FRAGMENT_LEN:
                    rejected += 1
                    continue
                start = int(rng.integers(0, length - FRAGMENT_LEN + 1)) if length > FRAGMENT_LEN else 0
                fragment = sentence_tokens[:, start : start + FRAGMENT_LEN]
                strings = model.to_str_tokens(fragment[0])
                if "�" in strings:
                    rejected += 1
                    continue
                batch_tokens.append(fragment)
                batch_strings.append(strings)

            tokens = torch.cat(batch_tokens, dim=0).to(device)
            _, cache = model.run_with_cache(tokens, names_filter=[tensor_name])
            residual = cache[tensor_name].reshape(-1, cache[tensor_name].shape[-1])
            encoded = learned_dict.encode(residual).reshape(len(batch_tokens), FRAGMENT_LEN, -1)
            encoded_np = encoded[:, :, :n_features].detach().cpu().numpy().astype(np.float16)
            activation_store[added : added + len(batch_tokens)] = encoded_np
            for strings in batch_strings:
                token_file.write(json.dumps({"tokens": strings}, ensure_ascii=False) + "\n")
            added += len(batch_tokens)
            if added % 1000 == 0 or added == n_fragments:
                activation_store.flush()
                token_file.flush()
                print(f"Evaluation fragments: {added}/{n_fragments} (rejected {rejected})")

    return activations_path, tokens_path


def summarize_modern_results(output_dir: Path, n_features: int) -> dict[str, Any]:
    rows = []
    skipped = 0
    missing = 0
    for feature in range(n_features):
        path = output_dir / f"feature_{feature}.json"
        if not path.exists():
            missing += 1
            continue
        record = json.loads(path.read_text())
        if record.get("status") == "skipped":
            skipped += 1
            continue
        rows.append(record)

    scores = np.asarray([row["combined_score"] for row in rows], dtype=np.float64)
    finite = scores[np.isfinite(scores)]
    # Preserve the paper plotting behavior: np.mean/np.std are not NaN-aware.
    # A completed feature with a NaN score therefore poisons the aggregate.
    mean = float(scores.mean()) if len(scores) else float("nan")
    std = float(scores.std(ddof=1)) if len(scores) > 1 else float("nan")
    ci95 = float(1.96 * std / math.sqrt(len(scores))) if len(scores) > 1 else float("nan")
    summary = {
        "n_requested": n_features,
        "n_completed": len(rows),
        "n_finite": len(finite),
        "n_nan": int(len(scores) - len(finite)),
        "n_skipped": skipped,
        "n_missing": missing,
        "mean_top_random_score": mean,
        "sample_std": std,
        "nominal_95pct_ci_half_width": ci95,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    return summary

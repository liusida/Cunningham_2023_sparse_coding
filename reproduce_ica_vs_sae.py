"""Minimal, restartable ICA-versus-SAE reproduction.

The two corpus conditions reuse their trained ICA/SAE artifacts across both
interpreter models. Run stages separately; paid API work occurs only in
``interpret``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torchopt
from tqdm.auto import tqdm

from activation_dataset import setup_data
from autoencoders.ensemble import FunctionalEnsemble
from autoencoders.ica import ICAEncoder
from autoencoders.sae_ensemble import FunctionalTiedSAE
from modern_interpret import (
    check_model_compatibility,
    choose_top_absolute_orientation,
    extract_evaluation_activations,
    interpret_features,
    summarize_modern_results,
)


DATASETS = {
    "pile10k": "NeelNanda/pile-10k",
    "openwebtext": "Skylion007/openwebtext",
}
INTERPRETER_MODELS = (
    "gpt-4.1-mini-2025-04-14",
    "gpt-4o-mini-2024-07-18",
)
METHODS = {
    "ica": ("ica", "legacy_nonnegative", False),
    "sae": ("sae", "legacy_nonnegative", False),
    "fixed_ica": ("ica", "legacy_nonnegative", True),
    "ica_full": ("ica_full", "legacy_nonnegative", False),
    "fixed_ica_full": ("ica_full", "legacy_nonnegative", True),
}
PYTHIA_MODEL = "EleutherAI/pythia-70m-deduped"
EVALUATION_DATASET = "Skylion007/openwebtext"
ORIGINAL_SAE_CHUNKS = 10
OPENWEBTEXT_TRAINING_START = 100_000
PYTHIA_LAYERS = tuple(range(6))
DEFAULT_ICA_SAMPLES = 524_288
FULL_ICA_MAX_ITER = 200
ORIENTATION_RULE = "signed mass of 20 largest-absolute fitting activations"
PLOT_METHODS = ("sae", "ica_full", "fixed_ica_full")
PLOT_STYLES = {
    "sae": {
        "label": "SAE",
        "color": "#B45F4D",
        "marker": "o",
        "linestyle": "-",
    },
    "ica_full": {
        "label": "ICA",
        "color": "#3D5F99",
        "marker": "D",
        "linestyle": "-",
    },
    "fixed_ica_full": {
        "label": "Fixed ICA",
        "color": "#2A9D8F",
        "marker": "P",
        "linestyle": "-",
    },
    "ica": {
        "label": "ICA (reduced)",
        "color": "#3D5F99",
        "marker": "D",
        "linestyle": "--",
    },
    "fixed_ica": {
        "label": "Fixed ICA (reduced)",
        "color": "#2A9D8F",
        "marker": "P",
        "linestyle": "--",
    },
}
PLOT_DATASET_LABELS = {"pile10k": "Pile10k", "openwebtext": "OpenWebText"}
PLOT_REPRODUCTION_NUMBERS = {"pile10k": 1, "openwebtext": 2}
PLOT_MODEL_LABELS = {
    "gpt-4.1-mini-2025-04-14": "GPT4.1-mini",
    "gpt-4o-mini-2024-07-18": "GPT4o-mini",
}


def log(description: str, message: str, color: str = "36") -> None:
    """Emit one timestamped, consistently styled high-level status line."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    start = f"\033[1;{color}m" if sys.stdout.isatty() else ""
    end = "\033[0m" if start else ""
    print(f"[{timestamp}] {start}★ {description}:{end} {message}", flush=True)


@contextmanager
def elapsed_heartbeat(label: str, interval: int = 60):
    """Print elapsed time while a long blocking library call is silent."""
    stopped = threading.Event()
    started = time.monotonic()

    def report() -> None:
        while not stopped.wait(interval):
            elapsed = int(time.monotonic() - started)
            log(label, f"still running ({elapsed // 60}m {elapsed % 60:02d}s elapsed)", "33")

    worker = threading.Thread(target=report, daemon=True)
    worker.start()
    try:
        yield
    finally:
        stopped.set()
        worker.join()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def selected_datasets(name: str) -> Iterable[tuple[str, str]]:
    return DATASETS.items() if name == "all" else ((name, DATASETS[name]),)


def selected_models(name: str) -> Iterable[str]:
    return INTERPRETER_MODELS if name == "all" else (name,)


def selected_layers(name: str) -> tuple[int, ...]:
    return PYTHIA_LAYERS if name == "all" else (int(name),)


def model_slug(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", model)


def run_dir(args: argparse.Namespace, dataset_slug: str, layer: int) -> Path:
    return args.output_root / dataset_slug / f"layer{layer}"


def update_metadata(directory: Path, **updates: Any) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "run_metadata.json"
    metadata = json.loads(path.read_text()) if path.exists() else {}
    metadata.update(updates)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def prepare(args: argparse.Namespace) -> None:
    from transformer_lens import HookedTransformer

    for slug, dataset_name in selected_datasets(args.dataset):
        layers = selected_layers(args.layer)
        directories = [run_dir(args, slug, layer) for layer in layers]
        activation_dirs = [directory / "activations" for directory in directories]
        log(
            "Prepare",
            f"{slug} — extracting layers {', '.join(map(str, layers))} {args.layer_loc} "
            f"activations from {dataset_name}",
        )
        existing = [sorted(path.glob("*.pt")) if path.exists() else [] for path in activation_dirs]
        if any(existing) and not args.overwrite:
            complete = []
            for directory in directories:
                metadata_path = directory / "run_metadata.json"
                metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
                complete.append(bool(metadata.get("prepare_complete")))
            if not all(complete):
                raise RuntimeError(
                    f"{slug}: found an incomplete earlier preparation under {args.output_root}; "
                    "remove that dataset directory and rerun prepare"
                )
            log("Prepare", f"{slug} — already complete; skipping", "32")
            continue
        if any(existing):
            raise FileExistsError(
                f"Remove existing {slug} layer directories before using --overwrite"
            )
        for activation_dir in activation_dirs:
            activation_dir.mkdir(parents=True, exist_ok=True)
        seed_everything(args.seed)
        model = HookedTransformer.from_pretrained(args.pythia_model, device=args.device)
        start_line = OPENWEBTEXT_TRAINING_START if slug == "openwebtext" else 0
        n_activations = setup_data(
            model.tokenizer,
            model,
            dataset_name=dataset_name,
            dataset_folder=[str(path) for path in activation_dirs],
            layer=list(layers),
            layer_loc=args.layer_loc,
            start_line=start_line,
            max_lines=args.training_documents,
            n_chunks=args.activation_chunks,
            chunk_size_gb=args.chunk_size_gb,
            device=torch.device(args.device),
            center_dataset=False,
        )
        for layer, directory in zip(layers, directories):
            update_metadata(
                directory,
                dataset_name=dataset_name,
                dataset_slug=slug,
                training_document_range_requested=[
                    start_line,
                    start_line + args.training_documents,
                ],
                pythia_model=args.pythia_model,
                layer=layer,
                layer_loc=args.layer_loc,
                seed=args.seed,
                n_activations=n_activations,
                activation_chunk_rows=[
                    int(torch.load(path, map_location="cpu", weights_only=True).shape[0])
                    for path in _activation_chunks(directory)
                ],
                prepare_complete=True,
            )
        del model


def _activation_chunks(directory: Path) -> list[Path]:
    chunks = sorted((directory / "activations").glob("*.pt"), key=lambda path: int(path.stem))
    if not chunks:
        raise FileNotFoundError(f"No activation chunks under {directory / 'activations'}")
    return chunks


def _fit_ica(args: argparse.Namespace, directory: Path, chunks: list[Path]) -> int:
    destination = directory / "ica.pt"
    if destination.exists() and not args.overwrite:
        log("ICA", f"{destination} already exists; skipping", "32")
        return args.ica_samples
    # Use a fixed-seed subset of the first chunk for the deliberately
    # compute-constrained ICA condition; SAE retains its ten-chunk budget.
    activations = torch.load(chunks[0], map_location="cpu", weights_only=True).to(torch.float32)
    if len(activations) < args.ica_samples:
        raise ValueError(
            f"ICA requested {args.ica_samples:,} activations, but the first chunk "
            f"contains only {len(activations):,}"
        )
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(activations), generator=generator)[: args.ica_samples]
    activations = activations[indices]
    n_presentations = len(activations)
    log(
        "ICA",
        f"fitting {activations.shape[1]} components on a fixed-seed ~0.5 GiB sample "
        f"({n_presentations:,} activations, max_iter={args.ica_max_iter}); "
        "FastICA is CPU-bound and may be silent for a while",
    )
    ica = ICAEncoder(activations.shape[1])
    ica.ica.set_params(random_state=args.seed, max_iter=args.ica_max_iter, tol=args.ica_tol)
    with elapsed_heartbeat("ICA fit"):
        ica.train(activations)
    torch.save(ica, destination)
    log("ICA", f"saved fitted model to {destination}", "32")
    return n_presentations


def _fit_sae(args: argparse.Namespace, directory: Path, chunks: list[Path]) -> tuple[int, int]:
    destination = directory / "sae.pt"
    if destination.exists() and not args.overwrite:
        log("SAE", f"{destination} already exists; skipping", "32")
        unique_rows = sum(
            int(torch.load(path, map_location="cpu", weights_only=True).shape[0])
            for path in chunks
        )
        first_rows = int(torch.load(chunks[0], map_location="cpu", weights_only=True).shape[0])
        return unique_rows, first_rows * args.sae_training_chunks
    first = torch.load(chunks[0], map_location="cpu", weights_only=True)
    activation_dim = first.shape[1]
    target_presentations = len(first) * args.sae_training_chunks
    del first
    initialized = FunctionalTiedSAE.init(
        activation_dim,
        int(activation_dim * args.sae_dict_ratio),
        args.sae_l1_alpha,
        bias_decay=0.0,
        dtype=torch.float32,
    )
    ensemble = FunctionalEnsemble(
        [initialized],
        FunctionalTiedSAE,
        torchopt.adam,
        {"lr": args.sae_learning_rate},
        device=args.device,
    )
    presentations = 0
    pass_index = 0
    unique_rows = 0
    for path in chunks:
        rows = int(torch.load(path, map_location="cpu", weights_only=True).shape[0])
        unique_rows += rows
    log(
        "SAE",
        f"{unique_rows:,} unique activation rows available; "
        f"targeting {target_presentations:,} presentations "
        f"({args.sae_training_chunks} full-chunk equivalents)",
    )
    with tqdm(total=target_presentations, unit="activations", desc="SAE fit") as progress:
        while presentations < target_presentations:
            rng = np.random.default_rng(args.seed + pass_index)
            for chunk_index in rng.permutation(len(chunks)):
                activations = torch.load(
                    chunks[int(chunk_index)], map_location="cpu", weights_only=True
                ).to(torch.float32)
                generator = torch.Generator().manual_seed(
                    args.seed + pass_index * 10_000 + int(chunk_index)
                )
                ordering = torch.randperm(len(activations), generator=generator)
                for start in range(0, len(ordering), args.sae_batch_size):
                    remaining = target_presentations - presentations
                    indices = ordering[start : start + min(args.sae_batch_size, remaining)]
                    batch = activations[indices].to(args.device)
                    ensemble.step_batch(batch)
                    presentations += len(batch)
                    progress.update(len(batch))
                    if presentations == target_presentations:
                        break
                if presentations == target_presentations:
                    break
            pass_index += 1
    params, buffers = ensemble.unstack(device="cpu")[0]
    torch.save(FunctionalTiedSAE.to_learned_dict(params, buffers), destination)
    log("SAE", f"saved fitted model to {destination}", "32")
    return unique_rows, presentations


def _fit_ica_full(args: argparse.Namespace, directory: Path, chunks: list[Path]) -> int:
    """Match the released baseline: FastICA on the complete first chunk."""
    destination = directory / "ica_full.pt"
    if destination.exists() and not args.overwrite:
        log("Full ICA", f"{destination} already exists; skipping", "32")
        return int(torch.load(chunks[0], map_location="cpu", weights_only=True).shape[0])
    activations = torch.load(chunks[0], map_location="cpu", weights_only=True).to(torch.float32)
    n_presentations = len(activations)
    log(
        "Full ICA",
        f"fitting {activations.shape[1]} components on the complete first chunk "
        f"({n_presentations:,} activations, max_iter={FULL_ICA_MAX_ITER}); "
        "FastICA is CPU-bound and may take hours",
    )
    ica = ICAEncoder(activations.shape[1])
    ica.ica.set_params(
        random_state=args.seed,
        max_iter=FULL_ICA_MAX_ITER,
        tol=args.ica_tol,
    )
    with elapsed_heartbeat("Full ICA fit"):
        ica.train(activations)
    torch.save(ica, destination)
    log("Full ICA", f"saved fitted model to {destination}", "32")
    return n_presentations


def _fitting_orientation_path(directory: Path, source: str) -> Path:
    return directory / "fitting_orientations" / f"{source}.json"


def _load_fitting_orientations(
    directory: Path, source: str, n_features: int
) -> list[int]:
    path = _fitting_orientation_path(directory, source)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing fitting-data ICA orientations: {path}. "
            "Run `python reproduce_ica_vs_sae.py orient-ica` first."
        )
    record = json.loads(path.read_text())
    orientations = [int(value) for value in record.get("orientations", [])]
    if len(orientations) < n_features:
        raise ValueError(f"{path} has {len(orientations)} signs; need {n_features}")
    if any(value not in (-1, 1) for value in orientations[:n_features]):
        raise ValueError(f"{path} contains an orientation other than -1 or 1")
    return orientations[:n_features]


def _stream_fitting_orientations(
    learned_dict: ICAEncoder,
    activations: torch.Tensor,
    n_features: int,
    batch_size: int,
) -> list[int]:
    """Encode fitting rows while retaining only each component's 20 extremes."""
    strongest = np.empty((0, n_features), dtype=np.float64)
    for start in tqdm(
        range(0, len(activations), batch_size),
        desc="ICA orientation",
        unit="batch",
    ):
        encoded = learned_dict.encode(activations[start : start + batch_size])
        candidates = np.concatenate(
            (strongest, encoded[:, :n_features].cpu().numpy()), axis=0
        )
        if len(candidates) > 20:
            keep = np.argpartition(np.abs(candidates), -20, axis=0)[-20:]
            strongest = np.take_along_axis(candidates, keep, axis=0)
        else:
            strongest = candidates
    return [choose_top_absolute_orientation(strongest[:, feature]) for feature in range(n_features)]


def orient_ica(args: argparse.Namespace) -> None:
    """Choose ICA component signs using only the exact ICA fitting activations."""
    for slug, _ in selected_datasets(args.dataset):
        for layer in selected_layers(args.layer):
            directory = run_dir(args, slug, layer)
            chunks = _activation_chunks(directory)
            metadata_path = directory / "run_metadata.json"
            if not metadata_path.exists():
                raise FileNotFoundError(f"Missing run metadata: {metadata_path}")
            metadata = json.loads(metadata_path.read_text())
            chunk = torch.load(chunks[0], map_location="cpu", weights_only=True).to(torch.float32)

            for source, sampled in (("ica", True), ("ica_full", False)):
                destination = _fitting_orientation_path(directory, source)
                if destination.exists() and not args.overwrite:
                    record = json.loads(destination.read_text())
                    if len(record.get("orientations", [])) < args.n_features:
                        raise ValueError(
                            f"{destination} has fewer than {args.n_features} orientations; "
                            "rerun with --overwrite"
                        )
                    log(
                        "ICA orientation",
                        f"{slug}/layer{layer}/{source} already exists; skipping",
                        "32",
                    )
                    continue

                artifact = directory / f"{source}.pt"
                if not artifact.exists():
                    raise FileNotFoundError(f"Missing fitted ICA artifact: {artifact}")
                fitting_rows = chunk
                if sampled:
                    count = int(metadata["ica_presentations"])
                    generator = torch.Generator().manual_seed(int(metadata["seed"]))
                    indices = torch.randperm(len(chunk), generator=generator)[:count]
                    fitting_rows = chunk[indices]
                else:
                    count = int(metadata.get("ica_full_presentations", len(chunk)))
                    fitting_rows = chunk[:count]

                learned_dict = torch.load(artifact, map_location="cpu", weights_only=False)
                log(
                    "ICA orientation",
                    f"{slug}/layer{layer}/{source} — choosing {args.n_features} signs "
                    f"from {len(fitting_rows):,} fitting activations",
                )
                orientations = _stream_fitting_orientations(
                    learned_dict,
                    fitting_rows,
                    args.n_features,
                    args.orientation_batch_size,
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(
                    json.dumps(
                        {
                            "source": source,
                            "orientation_source": "ica_fitting_data",
                            "rule": ORIENTATION_RULE,
                            "n_fitting_activations": len(fitting_rows),
                            "n_features": args.n_features,
                            "seed": int(metadata["seed"]),
                            "orientations": orientations,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                log("ICA orientation", f"saved {destination}", "32")
            del chunk


def train(args: argparse.Namespace) -> None:
    for slug, dataset_name in selected_datasets(args.dataset):
        for layer in selected_layers(args.layer):
            directory = run_dir(args, slug, layer)
            chunks = _activation_chunks(directory)
            log(
                "Train",
                f"{slug}/layer{layer} — ICA uses {args.ica_samples:,} sampled activations; SAE uses "
                f"{args.sae_training_chunks} full-chunk equivalents",
            )
            seed_everything(args.seed)
            ica_presentations = _fit_ica(args, directory, chunks)
            seed_everything(args.seed)
            sae_unique_rows, sae_presentations = _fit_sae(args, directory, chunks)
            update_metadata(
                directory,
                dataset_name=dataset_name,
                bias_decay=0.0,
                ica_presentations=ica_presentations,
                ica_max_iter=args.ica_max_iter,
                ica_tol=args.ica_tol,
                sae_dict_ratio=args.sae_dict_ratio,
                sae_l1_alpha=args.sae_l1_alpha,
                sae_learning_rate=args.sae_learning_rate,
                sae_batch_size=args.sae_batch_size,
                sae_training_chunks=args.sae_training_chunks,
                sae_unique_activation_rows=sae_unique_rows,
                sae_presentations=sae_presentations,
            )


def train_ica_full(args: argparse.Namespace) -> None:
    for slug, dataset_name in selected_datasets(args.dataset):
        for layer in selected_layers(args.layer):
            directory = run_dir(args, slug, layer)
            chunks = _activation_chunks(directory)
            seed_everything(args.seed)
            presentations = _fit_ica_full(args, directory, chunks)
            update_metadata(
                directory,
                dataset_name=dataset_name,
                ica_full_presentations=presentations,
                ica_full_max_iter=FULL_ICA_MAX_ITER,
                ica_full_tol=args.ica_tol,
            )


def eval_data(args: argparse.Namespace) -> None:
    for slug, _ in selected_datasets(args.dataset):
        for layer in selected_layers(args.layer):
            directory = run_dir(args, slug, layer)
            log("Evaluation data", f"{slug}/layer{layer} — encoding fixed OpenWebText fragments")
            if args.method == "all":
                sources = ["ica", "sae"]
                if (directory / "ica_full.pt").exists():
                    sources.append("ica_full")
            else:
                sources = [METHODS[args.method][0]]
            for method in dict.fromkeys(sources):
                destination = directory / "evaluation_data" / method
                if (destination / "feature_activations.npy").exists() and not args.overwrite:
                    log("Evaluation data", f"{slug}/layer{layer}/{method} already exists; skipping", "32")
                    continue
                artifact = directory / f"{method}.pt"
                if not artifact.exists():
                    raise FileNotFoundError(f"Missing trained artifact: {artifact}")
                learned_dict = torch.load(artifact, map_location="cpu", weights_only=False)
                n_available = learned_dict.n_feats
                extract_evaluation_activations(
                    learned_dict=learned_dict,
                    model_name=args.pythia_model,
                    layer=layer,
                    layer_loc=args.layer_loc,
                    dataset_name=EVALUATION_DATASET,
                    output_dir=destination,
                    device=args.device,
                    n_fragments=args.n_fragments,
                    n_features=min(args.n_features, n_available),
                    batch_size=args.eval_batch_size,
                    seed=args.seed,
                    start_line=0,
                    overwrite=args.overwrite,
                )
            update_metadata(
                directory,
                evaluation_dataset=EVALUATION_DATASET,
                evaluation_document_start=0,
                n_fragments=args.n_fragments,
                n_features=args.n_features,
            )


def interpret(args: argparse.Namespace) -> None:
    for slug, _ in selected_datasets(args.dataset):
        for layer in selected_layers(args.layer):
            directory = run_dir(args, slug, layer)
            for model in selected_models(args.interpreter_model):
                methods = (
                    METHODS.items()
                    if args.method == "all"
                    else ((args.method, METHODS[args.method]),)
                )
                for method, (source, mode, orient_signed) in methods:
                    feature_count = args.n_features
                    source_orientations = (
                        _load_fitting_orientations(directory, source, feature_count)
                        if orient_signed
                        else None
                    )
                    log(
                        "Interpret",
                        f"{slug}/layer{layer} — model={model}, method={method.upper()}, "
                        f"features=0–{feature_count - 1}",
                    )
                    evaluation = directory / "evaluation_data" / source
                    output = directory / "results" / model_slug(model) / method
                    interpret_features(
                        activations_path=evaluation / "feature_activations.npy",
                        tokens_path=evaluation / "fragments.jsonl",
                        output_dir=output,
                        mode=mode,
                        n_features=feature_count,
                        seed=args.seed,
                        explainer_model=model,
                        simulator_model=model,
                        max_concurrent=args.max_concurrent,
                        request_delay=args.request_delay,
                        retries=args.api_retries,
                        env_file=args.env_file,
                        overwrite=args.overwrite,
                        orient_signed=orient_signed,
                        source_orientations=source_orientations,
                    )


def check_api(args: argparse.Namespace) -> None:
    for model in selected_models(args.interpreter_model):
        check_model_compatibility(model, args.env_file)
        log("API check", f"{model} supports structured output and log probabilities", "32")


def _scores(path: Path, n_features: int) -> np.ndarray:
    values = []
    for feature in range(n_features):
        result = path / f"feature_{feature}.json"
        if result.exists():
            record = json.loads(result.read_text())
            if "combined_score" in record and np.isfinite(record["combined_score"]):
                values.append(record["combined_score"])
    return np.asarray(values, dtype=np.float64)


def _bootstrap_difference(ica: np.ndarray, sae: np.ndarray, seed: int) -> tuple[float, float]:
    if not len(ica) or not len(sae):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    differences = np.empty(10_000)
    for index in range(len(differences)):
        differences[index] = (
            rng.choice(sae, len(sae), replace=True).mean()
            - rng.choice(ica, len(ica), replace=True).mean()
        )
    return tuple(float(x) for x in np.quantile(differences, [0.025, 0.975]))


def _fixed_ica_pairs(
    ica_path: Path, fixed_path: Path, n_features: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return released and sign-oriented ICA scores aligned by component."""
    baseline = []
    corrected = []
    for feature in range(n_features):
        paths = (ica_path / f"feature_{feature}.json", fixed_path / f"feature_{feature}.json")
        if not all(path.exists() for path in paths):
            continue
        records = [json.loads(path.read_text()) for path in paths]
        scores = [record.get("combined_score") for record in records]
        if all(score is not None and np.isfinite(score) for score in scores):
            baseline.append(float(scores[0]))
            corrected.append(float(scores[1]))
    return np.asarray(baseline), np.asarray(corrected)


def _paired_bootstrap_difference(
    baseline: np.ndarray, corrected: np.ndarray, seed: int
) -> tuple[float, float]:
    if not len(baseline):
        return float("nan"), float("nan")
    differences = corrected - baseline
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray(
        [rng.choice(differences, len(differences), replace=True).mean() for _ in range(10_000)]
    )
    return tuple(float(x) for x in np.quantile(bootstrap, [0.025, 0.975]))


def summarize(args: argparse.Namespace) -> None:
    rows = []
    for slug, dataset_name in selected_datasets(args.dataset):
        for layer in selected_layers(args.layer):
            directory = run_dir(args, slug, layer)
            for model in selected_models(args.interpreter_model):
                base = directory / "results" / model_slug(model)
                ica_summary = summarize_modern_results(base / "ica", args.n_features)
                sae_summary = summarize_modern_results(base / "sae", args.n_features)
                ica_scores = _scores(base / "ica", args.n_features)
                sae_scores = _scores(base / "sae", args.n_features)
                paired_ica, fixed_ica_scores = _fixed_ica_pairs(
                    base / "ica", base / "fixed_ica", args.n_features
                )
                ica_full_scores = _scores(base / "ica_full", args.n_features)
                paired_ica_full, fixed_ica_full_scores = _fixed_ica_pairs(
                    base / "ica_full", base / "fixed_ica_full", args.n_features
                )
                lower, upper = _bootstrap_difference(ica_scores, sae_scores, args.seed)
                fixed_lower, fixed_upper = _paired_bootstrap_difference(
                    paired_ica, fixed_ica_scores, args.seed
                )
                fixed_full_lower, fixed_full_upper = _paired_bootstrap_difference(
                    paired_ica_full, fixed_ica_full_scores, args.seed
                )
                sae_fixed_full_lower, sae_fixed_full_upper = _bootstrap_difference(
                    fixed_ica_full_scores, sae_scores, args.seed
                )
                rows.append(
                    {
                        "dataset": dataset_name,
                        "layer": layer,
                        "interpreter_model": model,
                        "ica_mean": ica_summary["mean_top_random_score"],
                        "sae_mean": sae_summary["mean_top_random_score"],
                        "fixed_ica_mean": float(fixed_ica_scores.mean())
                        if len(fixed_ica_scores)
                        else float("nan"),
                        "ica_full_mean": float(ica_full_scores.mean())
                        if len(ica_full_scores)
                        else float("nan"),
                        "fixed_ica_full_mean": float(fixed_ica_full_scores.mean())
                        if len(fixed_ica_full_scores)
                        else float("nan"),
                        "sae_minus_ica": float(sae_scores.mean() - ica_scores.mean())
                        if len(ica_scores) and len(sae_scores)
                        else float("nan"),
                        "bootstrap_95pct_ci": [lower, upper],
                        "fixed_ica_minus_ica": float(
                            (fixed_ica_scores - paired_ica).mean()
                        )
                        if len(fixed_ica_scores)
                        else float("nan"),
                        "fixed_ica_minus_ica_95pct_ci": [fixed_lower, fixed_upper],
                        "fixed_ica_full_minus_ica_full": float(
                            (fixed_ica_full_scores - paired_ica_full).mean()
                        )
                        if len(fixed_ica_full_scores)
                        else float("nan"),
                        "fixed_ica_full_minus_ica_full_95pct_ci": [
                            fixed_full_lower,
                            fixed_full_upper,
                        ],
                        "sae_minus_fixed_ica_full": float(
                            sae_scores.mean() - fixed_ica_full_scores.mean()
                        )
                        if len(sae_scores) and len(fixed_ica_full_scores)
                        else float("nan"),
                        "sae_minus_fixed_ica_full_95pct_ci": [
                            sae_fixed_full_lower,
                            sae_fixed_full_upper,
                        ],
                        "ica_n": len(ica_scores),
                        "sae_n": len(sae_scores),
                        "fixed_ica_n": len(fixed_ica_scores),
                        "ica_full_n": len(ica_full_scores),
                        "fixed_ica_full_n": len(fixed_ica_full_scores),
                    }
                )
    args.output_root.mkdir(parents=True, exist_ok=True)
    destination = args.output_root / "summary.json"
    destination.write_text(json.dumps(rows, indent=2, allow_nan=True) + "\n")
    print(json.dumps(rows, indent=2, allow_nan=True))


def _plot_summary(scores: np.ndarray) -> tuple[float, float]:
    """Return the paper-style mean and nominal 95% CI half-width."""
    if not len(scores):
        return float("nan"), float("nan")
    mean = float(scores.mean())
    if len(scores) < 2:
        return mean, float("nan")
    return mean, float(1.96 * scores.std(ddof=1) / math.sqrt(len(scores)))


def plot(args: argparse.Namespace) -> None:
    """Plot already-computed layer-wise interpretation scores without rerunning them."""
    datasets = list(selected_datasets(args.dataset))
    layers = selected_layers(args.layer)
    default_methods = (
        PLOT_METHODS
        if args.plot_view == "main"
        else ("ica_full", "fixed_ica_full", "ica", "fixed_ica")
    )
    methods = default_methods if args.method == "all" else (args.method,)
    display_labels = {
        method: PLOT_STYLES[method]["label"] for method in methods
    }
    if args.plot_view == "reduced":
        display_labels.update(
            {
                "ica_full": "ICA (full)",
                "fixed_ica_full": "Fixed ICA (full)",
            }
        )
    models = list(selected_models(args.interpreter_model))
    if args.interpreter_model == "all":
        complete_models = []
        for model in models:
            expected_paths = [
                run_dir(args, slug, layer)
                / "results"
                / model_slug(model)
                / method
                / f"feature_{feature}.json"
                for slug, _ in datasets
                for layer in layers
                for method in methods
                for feature in range(args.n_features)
            ]
            if all(path.exists() for path in expected_paths):
                complete_models.append(model)
            else:
                log("Plot", f"omitting incomplete interpreter grid {model}", "33")
        models = complete_models
        if not models:
            raise FileNotFoundError(
                "No interpreter model has a complete selected result grid; choose one "
                "explicitly with --interpreter-model to plot partial results"
            )
    conditions = [(slug, model) for slug, _ in datasets for model in models]

    rows = []
    for slug, model in conditions:
        for layer in layers:
            base = run_dir(args, slug, layer) / "results" / model_slug(model)
            for method in methods:
                method_dir = base / method
                scores = _scores(method_dir, args.n_features)
                n_records = sum(
                    (method_dir / f"feature_{feature}.json").exists()
                    for feature in range(args.n_features)
                )
                mean, ci95 = _plot_summary(scores)
                rows.append(
                    {
                        "dataset": slug,
                        "model": model,
                        "layer": layer,
                        "method": method,
                        "n_records": n_records,
                        "n": len(scores),
                        "mean": mean,
                        "ci95": ci95,
                    }
                )

    if not any(row["n"] for row in rows):
        raise FileNotFoundError(
            f"No completed interpretation results found under {args.output_root}"
        )

    default_stem = "ica-vs-sae" if args.plot_view == "main" else "ica-reduced"
    stem = args.plot_output or Path("figures") / default_stem
    if stem.suffix.lower() in (".png", ".pdf", ".txt"):
        stem = stem.with_suffix("")
    destinations = [stem.with_suffix(suffix) for suffix in (".png", ".pdf", ".txt")]
    existing = [path for path in destinations if path.exists()]
    if existing and not args.force:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite {names}; pass --force to replace them")
    stem.parent.mkdir(parents=True, exist_ok=True)

    # Import Matplotlib only for this stage, with an isolated writable cache.
    with tempfile.TemporaryDirectory(prefix="ica-sae-mpl-") as cache_dir:
        os.environ["MPLCONFIGDIR"] = cache_dir
        import matplotlib

        matplotlib.use("Agg")
        matplotlib.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        )
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        ncols = min(2, len(conditions))
        nrows = math.ceil(len(conditions) / ncols)
        figure, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5.75 * ncols, 4.4 * nrows),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        flat_axes = axes.ravel()
        for axis, (slug, model) in zip(flat_axes, conditions):
            condition_rows = [
                row for row in rows if row["dataset"] == slug and row["model"] == model
            ]
            panel_partial = False
            for draw_index, method in enumerate(methods):
                method_rows = [row for row in condition_rows if row["method"] == method]
                observed = [row for row in method_rows if row["n"] > 0]
                if len(method_rows) != len(layers) or any(
                    row["n_records"] != args.n_features for row in method_rows
                ):
                    panel_partial = True
                if not observed:
                    continue
                style = PLOT_STYLES[method]
                x = np.asarray([row["layer"] for row in observed])
                y = np.asarray([row["mean"] for row in observed])
                yerr = np.asarray(
                    [row["ci95"] if np.isfinite(row["ci95"]) else 0.0 for row in observed]
                )
                axis.errorbar(
                    x,
                    y,
                    yerr=yerr,
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=1.5,
                    markersize=4,
                    capsize=2.5,
                    zorder=3 + draw_index,
                )
                for row in observed:
                    if row["n"] == args.n_features:
                        continue
                    axis.annotate(
                        f"n={row['n']}",
                        (row["layer"], row["mean"]),
                        xytext=(0, 7),
                        textcoords="offset points",
                        ha="center",
                        fontsize=7,
                        color=style["color"],
                    )
            title = (
                f"Reproduction {PLOT_REPRODUCTION_NUMBERS[slug]} "
                f"({PLOT_DATASET_LABELS[slug]} training)"
            )
            if len(models) > 1:
                title += f" · {PLOT_MODEL_LABELS.get(model, model)}"
            if panel_partial:
                title += " (partial)"
            axis.set_title(title, fontsize=10, fontweight="bold")
            axis.set_xticks(layers)
            axis.grid(axis="y", color="#D8D8D8", linewidth=0.6, alpha=0.7)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)

        for axis in flat_axes[len(conditions) :]:
            axis.set_visible(False)
        for row_index in range(nrows):
            axes[row_index, 0].set_ylabel("Mean automated interpretability score")
        figure.supxlabel("Layer", y=0.035)
        legend_methods = [method for method in methods if any(
            row["method"] == method and row["n"] > 0 for row in rows
        )]
        handles = [
            Line2D(
                [0],
                [0],
                color=PLOT_STYLES[method]["color"],
                marker=PLOT_STYLES[method]["marker"],
                linestyle=PLOT_STYLES[method]["linestyle"],
                linewidth=1.5,
                markersize=4,
                label=display_labels[method],
            )
            for method in legend_methods
        ]
        figure.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.945),
            ncol=min(5, len(handles)),
            frameon=False,
        )
        interpreter_title = (
            PLOT_MODEL_LABELS.get(models[0], models[0])
            if len(models) == 1
            else "multiple interpreters"
        )
        figure_title = (
            "Reproduction of Figure 2"
            if args.plot_view == "main"
            else "ICA fitting-budget comparison"
        )
        figure.suptitle(
            f"{figure_title} — {interpreter_title}",
            y=0.995,
            fontsize=12,
            fontweight="bold",
        )
        figure.tight_layout(rect=(0, 0.04, 1, 0.87))
        figure.savefig(destinations[0], dpi=300, bbox_inches="tight")
        figure.savefig(destinations[1], bbox_inches="tight")
        plt.close(figure)

    coverage = []
    for slug, model in conditions:
        condition = [row for row in rows if row["dataset"] == slug and row["model"] == model]
        method_coverage = []
        for method in methods:
            count = sum(row["n"] for row in condition if row["method"] == method)
            expected = len(layers) * args.n_features
            method_coverage.append(f"{display_labels[method]} {count}/{expected}")
        coverage.append(
            f"{PLOT_DATASET_LABELS[slug]} / {PLOT_MODEL_LABELS.get(model, model)}: "
            + ", ".join(method_coverage)
        )
    comparison_description = (
        "SAE and ICA use the released training and fitting budgets. Fixed ICA applies "
        "sign orientations chosen only from the ICA fitting activations."
        if args.plot_view == "main"
        else "Full ICA uses 2,098,176 activations and at most 200 iterations; reduced "
        "ICA uses 524,288 activations and 20 iterations. Each Fixed ICA curve applies "
        "fitting-data sign orientations to the corresponding components. "
        "Solid lines denote full-budget fits and dashed lines reduced-budget fits."
    )
    caption = (
        f"{figure_title}. Layer-wise mean top-and-random automated "
        "interpretability score; error bars are nominal 95% confidence intervals "
        "across finite feature scores. Absolute scores are not directly comparable "
        "with the original figure because its retired GPT-3.5/GPT-4 interpreters are "
        f"replaced here. {comparison_description} "
        "Points with incomplete "
        "coverage are annotated with n. Coverage: "
        + "; ".join(coverage)
        + ".\n"
    )
    destinations[2].write_text(caption)
    for destination in destinations:
        log("Plot", f"wrote {destination}", "32")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "prepare",
            "train",
            "train-ica-full",
            "orient-ica",
            "eval-data",
            "check-api",
            "interpret",
            "summarize",
            "plot",
            "all",
        ),
    )
    parser.add_argument("--dataset", choices=("all", *DATASETS), default="all")
    parser.add_argument("--interpreter-model", choices=("all", *INTERPRETER_MODELS), default="all")
    parser.add_argument("--method", choices=("all", *METHODS), default="all")
    parser.add_argument("--output-root", type=Path, default=Path("reproductions/ica_vs_sae"))
    parser.add_argument("--pythia-model", default=PYTHIA_MODEL)
    parser.add_argument("--layer", choices=("all", *map(str, PYTHIA_LAYERS)), default="all")
    parser.add_argument("--layer-loc", default="residual")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--force", action="store_true", help="allow plot outputs to be replaced")
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help="output stem for plot PNG, PDF, and caption (default: figures/ica-vs-sae)",
    )
    parser.add_argument(
        "--plot-view",
        choices=("main", "reduced"),
        default="main",
        help="main released-budget comparison or reduced-budget efficiency comparison",
    )
    parser.add_argument("--training-documents", type=int, default=19_531)
    parser.add_argument("--activation-chunks", type=int, default=ORIGINAL_SAE_CHUNKS)
    parser.add_argument("--chunk-size-gb", type=float, default=2.0)
    parser.add_argument("--ica-samples", type=int, default=DEFAULT_ICA_SAMPLES)
    parser.add_argument("--ica-max-iter", type=int, default=20)
    parser.add_argument("--ica-tol", type=float, default=1e-4)
    parser.add_argument("--orientation-batch-size", type=int, default=32_768)
    parser.add_argument("--sae-dict-ratio", type=float, default=2.0)
    parser.add_argument("--sae-l1-alpha", type=float, default=0.0008577)
    parser.add_argument("--sae-learning-rate", type=float, default=1e-3)
    parser.add_argument("--sae-batch-size", type=int, default=1024)
    parser.add_argument("--sae-training-chunks", type=int, default=ORIGINAL_SAE_CHUNKS)
    parser.add_argument("--n-fragments", type=int, default=50_000)
    parser.add_argument("--n-features", type=int, default=150)
    parser.add_argument("--eval-batch-size", type=int, default=20)
    parser.add_argument("--max-concurrent", type=int, default=10)
    parser.add_argument("--request-delay", type=float, default=1.0)
    parser.add_argument("--api-retries", type=int, default=5)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if (
        args.activation_chunks <= 0
        or args.sae_training_chunks <= 0
        or args.ica_samples <= 0
        or args.orientation_batch_size <= 0
    ):
        raise ValueError("activation, SAE chunk, and ICA sample counts must be positive")
    stages = (
        (
            prepare,
            train,
            train_ica_full,
            orient_ica,
            eval_data,
            check_api,
            interpret,
            summarize,
            plot,
        )
        if args.stage == "all"
        else {
            "prepare": (prepare,),
            "train": (train,),
            "train-ica-full": (train_ica_full,),
            "orient-ica": (orient_ica,),
            "eval-data": (eval_data,),
            "check-api": (check_api,),
            "interpret": (interpret,),
            "summarize": (summarize,),
            "plot": (plot,),
        }[args.stage]
    )
    for function in stages:
        function(args)


if __name__ == "__main__":
    main()

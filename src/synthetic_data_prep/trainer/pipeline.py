"""High-level pipeline functions for training job setup and launch."""

import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

from synthetic_data_prep.corpus.models import Corpus
from synthetic_data_prep.qa_generation.models import QADataset
from synthetic_data_prep.qa_generation.storage import qa_dataset_to_jsonl_bytes
from synthetic_data_prep.trainer.client import StorageClient, TrainerClient


def upload_dataset(
    dataset: QADataset,
    corpus: Corpus,
    api_key: str,
    base_url: str = "https://app.cgft.io",
    show_summary: bool = True,
) -> str:
    """Upload QA dataset to storage for training.

    Args:
        dataset: QADataset to upload
        corpus: Corpus object (used for path organization)
        api_key: API key for storage service
        base_url: Base URL for API (default: https://app.cgft.io)
        show_summary: Whether to print summary information (default: True)

    Returns:
        Blob path of uploaded dataset

    Example:
        >>> dataset_path = upload_dataset(
        ...     dataset=dataset,
        ...     corpus=corpus,
        ...     api_key="your-api-key"
        ... )
        Uploading 80 QA pairs for corpus abc123...
        Dataset uploaded to: datasets/search/abc123/qa-dataset.jsonl
    """
    storage_client = StorageClient(api_key=api_key, base_url=base_url)

    if show_summary:
        print(f"Uploading {len(dataset)} QA pairs for corpus {corpus.id}...")

    dataset_path = f"datasets/search/{corpus.id}/qa-dataset.jsonl"

    result = storage_client.upload_file(
        path=dataset_path,
        content=qa_dataset_to_jsonl_bytes(dataset),
        mime_type="application/jsonl",
    )

    if show_summary:
        print(f"Dataset uploaded to: {result['blobPath']}")

    return result["blobPath"]


def upload_env(
    env_class: type,
    constructor_args: dict[str, Any],
    api_key: str,
    pip_dependencies: list[str] | None = None,
    local_modules: list[str] | None = None,
    base_url: str = "https://app.cgft.io",
    validate: bool = True,
    show_summary: bool = True,
) -> tuple[str, str]:
    """Bundle and upload environment class for training.

    Args:
        env_class: Environment class (e.g., SearchEnv)
        constructor_args: Arguments to pass to env class constructor
        api_key: API key for storage service
        pip_dependencies: List of pip dependencies (default: ["aiohttp"])
        local_modules: List of local module names to include in bundle (default: None)
        base_url: Base URL for API (default: https://app.cgft.io)
        validate: Whether to validate the environment (default: True)
        show_summary: Whether to print summary information (default: True)

    Returns:
        Tuple of (env_blob_path, env_args_blob_path)

    Example:
        >>> from my_envs import SearchEnv
        >>> env_path, args_path = upload_env(
        ...     env_class=SearchEnv,
        ...     constructor_args={"api_key": "...", "corpus_id": "abc123"},
        ...     api_key="your-api-key"
        ... )
        Bundling environment...
        Payload size: 15.32 KB
        Running validation...
        Isolated validation passed!
        Env uploaded to: envs/search/a1b2c3d4/search-env-cls.bmxp
    """
    from benchmax.bundle.bundler import bundle_env, write_bundle_files
    from benchmax.bundle.validator import validate_bundle


    if pip_dependencies is None:
        pip_dependencies = ["aiohttp"]

    if local_modules is None:
        local_modules = []

    if show_summary:
        print("Bundling environment...")

    bundle = bundle_env(
        env_class,
        pip_dependencies=pip_dependencies,
        local_modules=local_modules,
    )

    if show_summary:
        metadata_bytes = bundle.metadata.to_json_bytes()
        print(f"Pickled class size: {len(bundle.pickled_class) / 1024:.2f} KB")
        print(f"Metadata size: {len(metadata_bytes) / 1024:.2f} KB")
        print(f"Python version: {bundle.metadata.python_version}")
        print(f"Dependencies: {bundle.metadata.pip_dependencies}")

    # Validate
    if validate:
        if show_summary:
            print("\nRunning validation in isolated environment (this may take ~1 min)...")

        warnings = validate_bundle(bundle, constructor_args=constructor_args)

        if warnings:
            print(f"Warnings: {warnings}")
        else:
            if show_summary:
                print("Isolated validation passed!")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        pickle_path = tmp_path / "search-env-cls.pkl"
        metadata_path = tmp_path / "search-env-meta.json"
        write_bundle_files(bundle, pickle_path, metadata_path)
        env_cls_bytes = pickle_path.read_bytes()
        env_meta_bytes = metadata_path.read_bytes()
    
    content_hash = hashlib.sha256(env_cls_bytes + env_meta_bytes).hexdigest()[:8]

    # Upload env class
    storage_client = StorageClient(api_key=api_key, base_url=base_url)
    env_path = f"envs/search/{content_hash}/search-env-cls.pkl"
    env_result = storage_client.upload_file(
        path=env_path,
        content=env_cls_bytes,
        mime_type="application/octet-stream",
    )

    # Upload env metadata
    env_meta_path = f"envs/search/{content_hash}/search-env-meta.json"
    env_meta_result = storage_client.upload_file(
        path=env_meta_path,
        content=env_meta_bytes,
        mime_type="application/json",
    )
    if show_summary:
        print(f"Env class uploaded successfully to {env_result['blobPath']}")
        print(f"Env metadata uploaded successfully to {env_meta_result['blobPath']}")

    return env_result["blobPath"], env_meta_result["blobPath"]


def launch_job(
    env_cls_blob_path: str,
    env_metadata_blob_path: str,
    api_key: str,
    experiment_type: str = "search",
    base_url: str = "https://app.cgft.io",
    show_summary: bool = True,
) -> str:
    """Launch training experiment.

    Args:
        env_cls_blob_path: Blob path to uploaded environment class (.pkl file)
        env_metadata_blob_path: Blob path to uploaded environment metadata (.json file)
        api_key: API key for trainer service
        experiment_type: Type of experiment to launch (default: "search")
        base_url: Base URL for API (default: https://app.cgft.io)
        show_summary: Whether to print summary information (default: True)

    Returns:
        Experiment ID string

    Example:
        >>> experiment_id = launch_job(
        ...     env_cls_blob_path="envs/search/a1b2c3d4/search-env-cls.pkl",
        ...     env_metadata_blob_path="envs/search/a1b2c3d4/search-env-meta.json",
        ...     api_key="your-api-key"
        ... )
        Launching training experiment...
        Training experiment launched! Experiment ID: exp_xyz789
    """
    trainer_client = TrainerClient(api_key=api_key, base_url=base_url)

    if show_summary:
        print("Launching training experiment...")

    experiment_id = trainer_client.launch_experiment(
        experiment_type=experiment_type,
        env_cls_path=env_cls_blob_path,
        env_metadata_path=env_metadata_blob_path,
    )

    if show_summary:
        print(f"Training experiment launched! Experiment ID: {experiment_id}")

    return experiment_id

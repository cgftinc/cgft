"""High-level pipeline functions for training job setup and launch."""

import hashlib
import json
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
    from benchmax.bundle.bundler import bundle_env
    from benchmax.bundle.validator import validate_payload

    if pip_dependencies is None:
        pip_dependencies = ["aiohttp"]

    if show_summary:
        print("Bundling environment...")

    payload = bundle_env(
        env_class,
        pip_dependencies=pip_dependencies,
        local_modules=[],
        extra_metadata={"description": "Search environment"},
    )

    if show_summary:
        print(f"Payload size: {len(payload.to_bytes()) / 1024:.2f} KB")
        print(f"Python version: {payload.python_version}")
        print(f"Dependencies: {payload.pip_dependencies}")

    # Validate
    if validate:
        if show_summary:
            print("\nRunning validation in isolated environment (this may take ~1 min)...")

        warnings = validate_payload(payload, constructor_args=constructor_args)

        if warnings:
            print(f"Warnings: {warnings}")
        else:
            if show_summary:
                print("Isolated validation passed!")

    # Upload
    storage_client = StorageClient(api_key=api_key, base_url=base_url)
    payload_bytes = payload.to_bytes()

    # Calculate hash from content
    content_hash = hashlib.sha256(payload_bytes).hexdigest()[:8]

    # Upload env class
    env_path = f"envs/search/{content_hash}/search-env-cls.bmxp"
    env_result = storage_client.upload_file(
        path=env_path,
        content=payload_bytes,
        mime_type="application/octet-stream",
    )

    # Upload env args
    env_args_path = f"envs/search/{content_hash}/search-env-kwargs.json"
    env_args_bytes = json.dumps(constructor_args, indent=2).encode("utf-8")
    env_args_result = storage_client.upload_file(
        path=env_args_path,
        content=env_args_bytes,
        mime_type="application/json",
    )

    if show_summary:
        print(f"\nEnv uploaded to: {env_result['blobPath']}")
        print(f"Env args uploaded to: {env_args_result['blobPath']}")

    return env_result["blobPath"], env_args_result["blobPath"]


def launch_job(
    dataset_blob_path: str,
    env_blob_path: str,
    api_key: str,
    base_url: str = "https://app.cgft.io",
    show_summary: bool = True,
) -> dict[str, Any]:
    """Launch training job.

    Args:
        dataset_blob_path: Blob path to uploaded dataset
        env_blob_path: Blob path to uploaded environment
        api_key: API key for trainer service
        base_url: Base URL for API (default: https://app.cgft.io)
        show_summary: Whether to print summary information (default: True)

    Returns:
        Job response dictionary

    Example:
        >>> job = launch_job(
        ...     dataset_blob_path="datasets/search/abc123/qa-dataset.jsonl",
        ...     env_blob_path="envs/search/a1b2c3d4/search-env-cls.bmxp",
        ...     api_key="your-api-key"
        ... )
        Launching training job...
        Training job launched! Job ID: xyz789
    """
    trainer_client = TrainerClient(api_key=api_key, base_url=base_url)

    if show_summary:
        print("Launching training job...")

    job = trainer_client.launch_job(
        job_type="search",
        args={"dataset": dataset_blob_path, "env": env_blob_path},
    )

    if show_summary:
        print(f"Training job launched! Job response: {job}")

    return job

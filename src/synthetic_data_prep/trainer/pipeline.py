"""This module provides a clean, high-level interface for launching training jobs:

    from synthetic_data_prep.trainer.pipeline_new import train

    experiment_id = train(
        env_class=SearchEnv,
        env_args={"api_key": "..."},
        dataset=[
            {"query": "What is Python?", "answer": "A programming language"},
            {"query": "What is ML?", "answer": "Machine Learning"}
        ],
        api_key="your-api-key" # get from https://app.cgft.io/account/api-keys
    )

The train() function handles everything automatically:
- Uploads your dataset (as list of dicts)
- Bundles and uploads your environment
- Launches the training job
- Returns the experiment ID

For more control, you can use the lower-level functions:
- upload_dataset() - Upload a list of dicts as JSONL
- upload_env() - Bundle and upload environment class
- launch_job() - Launch from pre-uploaded paths
"""

import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from synthetic_data_prep.trainer.client import RolloutClient, StorageClient, TrainerClient


def upload_dataset(
    dataset: list[dict[str, Any]],
    api_key: str,
    dataset_id: str | None = None,
    base_url: str = "https://app.cgft.io",
    dataset_name: str = "dataset.jsonl",
    show_summary: bool = True,
) -> str:
    """Upload dataset to storage for training.

    Simple dataset upload that works with list of dicts.
    Serializes to JSONL format (one JSON object per line).

    Args:
        dataset: List of dicts representing the dataset
        dataset_id: Unique identifier for this dataset. If None, auto-generates from dataset hash.
        api_key: API key for storage service
        base_url: Base URL for API (default: https://app.cgft.io)
        dataset_name: Filename for the dataset (default: "dataset.jsonl")
        show_summary: Whether to print summary information (default: True)

    Returns:
        Blob path of uploaded dataset

    Examples:
        >>> # Upload with auto-generated ID
        >>> upload_dataset(
        ...     dataset=[
        ...         {"query": "What is Python?", "answer": "A programming language"},
        ...         {"query": "What is ML?", "answer": "Machine Learning"}
        ...     ],
        ...     api_key="your-api-key"
        ... )
        Dataset uploaded to: datasets/dataset-a1b2c3d4/dataset.jsonl

        >>> # Upload with explicit ID
        >>> upload_dataset(
        ...     dataset=[...],
        ...     dataset_id="qa-pairs-v1",
        ...     api_key="your-api-key"
        ... )
        Dataset uploaded to: datasets/qa-pairs-v1/dataset.jsonl
    """
    storage_client = StorageClient(api_key=api_key, base_url=base_url)

    # Serialize to JSONL (one JSON object per line)
    jsonl_lines = [json.dumps(item, sort_keys=True) for item in dataset]
    content_str = "\n".join(jsonl_lines)
    content_bytes = content_str.encode("utf-8")

    # Auto-generate dataset_id from content hash if not provided
    if dataset_id is None:
        content_hash = hashlib.sha256(content_bytes).hexdigest()[:8]
        dataset_id = f"dataset-{content_hash}"
        if show_summary:
            print(f"Auto-generated dataset_id: {dataset_id}")

    if show_summary:
        print(f"Uploading dataset ({len(dataset)} items, {len(content_bytes)} bytes)...")

    # Construct path and upload
    dataset_path = f"datasets/{dataset_id}/{dataset_name}"

    result = storage_client.upload_file(
        path=dataset_path,
        content=content_bytes,
        mime_type="application/jsonl",
    )

    if show_summary:
        print(f"Dataset uploaded to: {result['blobPath']}")

    return result["blobPath"]


def upload_env(
    env_class: type,
    constructor_args: dict[str, Any],
    api_key: str,
    base_url: str = "https://app.cgft.io",
    pip_dependencies: list[str] | None = None,
    local_modules: list | None = None,
    validate: bool = True,
    show_summary: bool = True,
) -> tuple[str, str]:
    """Bundle and upload environment class for training.

    Args:
        env_class: Environment class (e.g., SearchEnv, SummarizationEnv)
        constructor_args: Arguments to pass to env class constructor
        api_key: API key for storage service
        base_url: Base URL for API (default: https://app.cgft.io)
        pip_dependencies: List of pip dependencies (default: ["aiohttp"])
        local_modules: List of local module names to include in bundle
        validate: Whether to validate the environment (default: True)
        show_summary: Whether to print summary information (default: True)

    Returns:
        Tuple of (env_blob_path, env_metadata_blob_path)

    Examples:
        >>> # Auto-infers "search" from SearchEnv
        >>> from my_envs import SearchEnv
        >>> env_path, meta_path = upload_env(
        ...     env_class=SearchEnv,
        ...     constructor_args={"api_key": "...", "dataset_path": "..."},
        ...     api_key="your-api-key"
        ... )
        Bundling SearchEnv (type: search)...
        Env uploaded to: ~/user-data/envs/search/a1b2c3d4/search-env-cls.pkl

        >>> # Auto-infers "summarization" from SummarizationEnv
        >>> from my_envs import SummarizationEnv
        >>> env_path, meta_path = upload_env(
        ...     env_class=SummarizationEnv,
        ...     constructor_args={"model": "gpt-4"},
        ...     api_key="your-api-key"
        ... )
        Bundling SummarizationEnv (type: summarization)...
        Env uploaded to: ~/user-data/envs/summarization/e5f6g7h8/summarization-env-cls.pkl

        >>> # Custom environment type
        >>> env_path, meta_path = upload_env(
        ...     env_class=MyTaskEnv,
        ...     constructor_args={},
        ...     api_key="your-api-key"
        ... )
        Bundling MyTaskEnv (type: custom_task)...
        Env uploaded to: ~/user-data/envs/custom_task/i9j0k1l2/custom_task-env-cls.pkl
    """
    from benchmax.bundle.bundler import bundle_env, write_bundle_files
    from benchmax.bundle.validator import validate_bundle

    if pip_dependencies is None:
        pip_dependencies = []

    if local_modules is None:
        local_modules = []

    if show_summary:
        print(f"Bundling {env_class.__name__}...")

    # Bundle environment
    bundle = bundle_env(
        env_class,
        pip_dependencies=pip_dependencies,
        local_modules=local_modules,
        constructor_args=constructor_args,
    )

    if show_summary:
        print(f"Pickled class size: {len(bundle.pickled_class) / 1024:.2f} KB")
        print(f"Metadata size: {len(bundle.metadata.to_json_bytes()) / 1024:.2f} KB")
        print(f"Python version: {bundle.metadata.python_version}")
        print(f"Dependencies: {bundle.metadata.pip_dependencies}")

    # Validate
    if validate:
        if show_summary:
            print(
                f"\nBasic local validation {env_class.__name__} in isolated environment (this may take ~1 min)..."
            )

        warnings = validate_bundle(bundle, constructor_args=constructor_args)

        if warnings:
            print(f"Warnings: {warnings}")
        else:
            if show_summary:
                print("Isolated validation passed!")

    # Write bundle files
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        pickle_path = tmp_path / "env-cls.pkl"
        metadata_path = tmp_path / "env-meta.json"
        write_bundle_files(bundle, pickle_path, metadata_path)
        env_cls_bytes = pickle_path.read_bytes()
        env_meta_bytes = metadata_path.read_bytes()

    # Generate content hash for versioning
    content_hash = hashlib.sha256(env_cls_bytes + env_meta_bytes).hexdigest()[:8]

    # Upload to storage
    storage_client = StorageClient(api_key=api_key, base_url=base_url)

    env_path = f"envs/{content_hash}/env-cls.pkl"
    env_result = storage_client.upload_file(
        path=env_path,
        content=env_cls_bytes,
        mime_type="application/octet-stream",
    )

    env_meta_path = f"envs/{content_hash}/env-meta.json"
    env_meta_result = storage_client.upload_file(
        path=env_meta_path,
        content=env_meta_bytes,
        mime_type="application/json",
    )

    if show_summary:
        print(f"Env uploaded to: ~/user-data/{env_result['blobPath']}")

    return f"~/user-data/{env_result['blobPath']}", f"~/user-data/{env_meta_result['blobPath']}"


def launch_job(
    env_cls_blob_path: str,
    env_metadata_blob_path: str,
    api_key: str,
    base_url: str = "https://app.cgft.io",
    experiment_type: str = "simple",
    show_summary: bool = True,
) -> str:
    """Launch training experiment from uploaded environment paths.

    Low-level function for launching when you've already uploaded the environment.
    For most use cases, use train() instead for a simpler interface.

    Args:
        env_cls_blob_path: Blob path to uploaded environment class (.pkl file)
        env_metadata_blob_path: Blob path to uploaded environment metadata (.json file)
        api_key: API key for trainer service
        base_url: Base URL for API (default: https://app.cgft.io)
        experiment_type: Type of experiment. If None, inferred from env path.
        show_summary: Whether to print summary information (default: True)

    Returns:
        Experiment ID string
    """

    if show_summary:
        print(f"Launching experiment (type: {experiment_type})...")

    trainer_client = TrainerClient(api_key=api_key, base_url=base_url)
    experiment_id = trainer_client.launch_experiment(
        experiment_type=experiment_type,
        env_cls_path=env_cls_blob_path,
        env_metadata_path=env_metadata_blob_path,
    )

    if show_summary:
        print(f"Experiment launched! ID: {experiment_id}")
        print(f"View experiment: {base_url}/experiments/{experiment_id}")

    return experiment_id


def train(
    env_class: type,
    env_args: dict[str, Any],
    dataset: list[dict[str, Any]],
    api_key: str,
    dataset_id: str | None = None,
    base_url: str = "https://app.cgft.io",
    # Optional customization
    pip_dependencies: list[str] | None = None,
    local_modules: list | None = None,
    validate_env: bool = True,
    validate_env_remotely: bool = True,
    show_summary: bool = True,
) -> str:
    """Train a model - the simplest interface for launching training jobs.

    This is the recommended high-level function that handles everything:
    1. Uploads your dataset
    2. Bundles and uploads your environment
    3. Validates the environment remotely (optional)
    4. Launches the training job
    5. Returns the experiment ID

    Args:
        env_class: Environment class (e.g., SearchEnv, SummarizationEnv)
        env_args: Constructor arguments for the environment
        dataset: List of dicts representing the dataset
        api_key: API key for the service
        dataset_id: Unique identifier for the dataset. If None, auto-generates from dataset hash.
        base_url: Base URL for API (default: https://app.cgft.io)
        pip_dependencies: List of pip dependencies for the environment
        local_modules: List of local modules to include in environment bundle
        validate_env: Whether to validate environment locally before upload. (default: True)
        validate_env_remotely: Whether to validate environment in a remote rollout server. (default: True)
        show_summary: Whether to print progress information (default: True)

    Returns:
        Experiment ID string
    """

    # Upload dataset
    dataset_blob_path = upload_dataset(
        dataset=dataset,
        api_key=api_key,
        dataset_id=dataset_id,
        base_url=base_url,
        show_summary=show_summary,
    )

    # Update env_args to include dataset path
    env_args_with_dataset = env_args.copy()
    if "dataset_path" not in env_args_with_dataset:
        env_args_with_dataset["dataset_path"] = f"~/user-data/{dataset_blob_path}"

    # Upload environment
    # make sure local env modules are bundled
    if not local_modules:
        local_modules = []
    env_module = sys.modules[env_class.__module__]
    if env_module not in local_modules:
        local_modules.append(sys.modules[env_class.__module__])
    env_blob_path, env_meta_blob_path = upload_env(
        env_class=env_class,
        constructor_args=env_args_with_dataset,
        api_key=api_key,
        base_url=base_url,
        pip_dependencies=pip_dependencies,
        local_modules=local_modules,
        validate=validate_env,
        show_summary=show_summary,
    )

    # Smoke-test the uploaded environment on the rollout server with a couple
    # of raw examples before committing to a full training run.
    if validate_env_remotely:
        # env_blob_path has a "~/user-data/" prefix — strip it so the rollout
        # server receives a plain blob path it can resolve against the user's
        # container (same convention as build_env_blob() in the e2e tests).
        def _strip_userdata(p: str) -> str:
            prefix = "~/user-data/"
            return p[len(prefix):] if p.startswith(prefix) else p

        rollout_client = RolloutClient(api_key=api_key)
        passed = rollout_client.validate_examples(
            examples=dataset,
            env_cls_path=_strip_userdata(env_blob_path),
            env_meta_path=_strip_userdata(env_meta_blob_path),
        )
        if not passed:
            raise RuntimeError(
                "Remote environment validation failed. "
                "Fix the errors above before retrying, or pass validate_env_remotely=False to skip."
            )

    # Launch training job
    experiment_id = launch_job(
        env_cls_blob_path=env_blob_path,
        env_metadata_blob_path=env_meta_blob_path,
        api_key=api_key,
        base_url=base_url,
        show_summary=show_summary,
    )

    return experiment_id

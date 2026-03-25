"""This module provides a clean, high-level interface for launching training jobs:

    from cgft.trainer.pipeline import train

    experiment_id = train(
        env_class=SearchEnv,
        env_args={"api_key": "..."},
        train_dataset=[...],
        eval_dataset=[...],
        prefix="my-search",
        api_key="your-api-key" # get from https://app.cgft.io/account/api-keys
    )

The train() function handles everything automatically:
- Uploads your train and val datasets
- Bundles and uploads your environment
- Launches the training job
- Returns the experiment ID

For more control, you can use the lower-level functions:
- validate_env_only() - Validate environment without launching a training job
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

from cgft.trainer.client import RolloutClient, StorageClient, TrainerClient


def upload_dataset(
    dataset: list[dict[str, Any]],
    prefix: str,
    api_key: str,
    base_url: str = "https://app.cgft.io",
    dataset_name: str = "dataset.jsonl",
    content_hash: str | None = None,
    show_summary: bool = True,
) -> str:
    """Upload dataset to storage for training.

    Simple dataset upload that works with list of dicts.
    Serializes to JSONL format (one JSON object per line).

    Args:
        dataset: List of dicts representing the dataset
        prefix: Namespace prefix for the upload path (e.g., "cgft-search", "tpuf-search")
        api_key: API key for storage service
        base_url: Base URL for API (default: https://app.cgft.io)
        dataset_name: Filename for the dataset (default: "dataset.jsonl")
        content_hash: Optional pre-computed hash for the path. If None, auto-generated from dataset content.
        show_summary: Whether to print summary information (default: True)

    Returns:
        Blob path of uploaded dataset

    Examples:
        >>> upload_dataset(
        ...     dataset=[
        ...         {"query": "What is Python?", "answer": "A programming language"},
        ...         {"query": "What is ML?", "answer": "Machine Learning"}
        ...     ],
        ...     prefix="cgft-search",
        ...     api_key="your-api-key"
        ... )
        Dataset uploaded to: datasets/cgft-search/a1b2c3d4/dataset.jsonl
    """
    storage_client = StorageClient(api_key=api_key, base_url=base_url)

    # Serialize to JSONL (one JSON object per line)
    jsonl_lines = [json.dumps(item, sort_keys=True) for item in dataset]
    content_str = "\n".join(jsonl_lines)
    content_bytes = content_str.encode("utf-8")

    if content_hash is None:
        content_hash = hashlib.sha256(content_bytes).hexdigest()[:8]

    if show_summary:
        print(f"Uploading dataset ({len(dataset)} items, {len(content_bytes)} bytes)...")

    # Construct path and upload
    dataset_path = f"datasets/{prefix}/{content_hash}/{dataset_name}"

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
    prefix: str,
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
        prefix: Namespace prefix for the upload path (e.g., "cgft-search", "tpuf-search")
        api_key: API key for storage service
        base_url: Base URL for API (default: https://app.cgft.io)
        pip_dependencies: List of pip dependencies (default: ["aiohttp"])
        local_modules: List of local module names to include in bundle
        validate: Whether to validate the environment (default: True)
        show_summary: Whether to print summary information (default: True)

    Returns:
        Tuple of (env_blob_path, env_metadata_blob_path)

    Examples:
        >>> env_path, meta_path = upload_env(
        ...     env_class=SearchEnv,
        ...     constructor_args={"api_key": "..."},
        ...     prefix="cgft-search",
        ...     api_key="your-api-key"
        ... )
        Bundling SearchEnv...
        Env uploaded to: envs/cgft-search/a1b2c3d4/env-cls.pkl
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
        metadata_path = tmp_path / "env-metadata.json"
        write_bundle_files(bundle, pickle_path, metadata_path)
        env_cls_bytes = pickle_path.read_bytes()
        env_meta_bytes = metadata_path.read_bytes()

    # Generate content hash for versioning
    content_hash = hashlib.sha256(env_cls_bytes + env_meta_bytes).hexdigest()[:8]

    # Upload to storage
    storage_client = StorageClient(api_key=api_key, base_url=base_url)

    env_path = f"envs/{prefix}/{content_hash}/env-cls.pkl"
    env_result = storage_client.upload_file(
        path=env_path,
        content=env_cls_bytes,
        mime_type="application/octet-stream",
    )

    env_metadata_path = f"envs/{prefix}/{content_hash}/env-metadata.json"
    env_meta_result = storage_client.upload_file(
        path=env_metadata_path,
        content=env_meta_bytes,
        mime_type="application/json",
    )

    if show_summary:
        print(f"Env uploaded to: {env_result['blobPath']}")

    return env_result["blobPath"], env_meta_result["blobPath"]


def train(
    env_class: type,
    env_args: dict[str, Any],
    train_dataset: list[dict],
    eval_dataset: list[dict],
    prefix: str,
    api_key: str,
    base_url: str = "https://app.cgft.io",
    experiment_type: str = "simple",
    experiment_name: str | None = None,
    # Optional customization
    pip_dependencies: list[str] | None = None,
    local_modules: list | None = None,
    validate_env: bool = True,
    validate_env_remotely: bool = True,
    show_summary: bool = True,
) -> str:
    """Train a model - the simplest interface for launching training jobs.

    This is the recommended high-level function that handles everything:
    1. Uploads your train and val datasets
    2. Bundles and uploads your environment
    3. Validates the environment remotely (optional)
    4. Launches the training job
    5. Returns the experiment ID

    Args:
        env_class: Environment class (e.g., SearchEnv, SummarizationEnv)
        env_args: Constructor arguments for the environment
        train_dataset: List of dicts for the training split
        eval_dataset: List of dicts for the validation split
        prefix: Namespace prefix for upload paths (e.g., "cgft-search", "tpuf-search")
        api_key: API key for the service
        base_url: Base URL for API (default: https://app.cgft.io)
        experiment_type: Type of experiment (default: "simple")
        experiment_name: Optional name for the experiment (default: None)
        pip_dependencies: List of pip dependencies for the environment
        local_modules: List of local modules to include in environment bundle
        validate_env: Whether to validate environment locally before upload. (default: True)
        validate_env_remotely: Whether to validate environment in a remote rollout server. (default: True)
        show_summary: Whether to print progress information (default: True)

    Returns:
        Experiment ID string
    """

    # Compute shared hash from combined dataset
    combined_lines = [json.dumps(item, sort_keys=True) for item in train_dataset + eval_dataset]
    dataset_hash = hashlib.sha256("\n".join(combined_lines).encode("utf-8")).hexdigest()[:8]

    # Upload train dataset
    train_blob_path = upload_dataset(
        dataset=train_dataset,
        prefix=prefix,
        api_key=api_key,
        dataset_name="train_dataset.jsonl",
        content_hash=dataset_hash,
        base_url=base_url,
        show_summary=show_summary,
    )

    # Upload val dataset
    eval_blob_path = upload_dataset(
        dataset=eval_dataset,
        prefix=prefix,
        api_key=api_key,
        dataset_name="eval_dataset.jsonl",
        content_hash=dataset_hash,
        base_url=base_url,
        show_summary=show_summary,
    )

    # Update env_args to include dataset paths
    env_args_with_dataset = env_args.copy()
    env_args_with_dataset["train_dataset_path"] = f"~/user-data/{train_blob_path}"
    env_args_with_dataset["val_dataset_path"] = f"~/user-data/{eval_blob_path}"

    # Upload environment
    if not local_modules:
        local_modules = []
    env_module = sys.modules[env_class.__module__]
    if env_module not in local_modules:
        local_modules.append(sys.modules[env_class.__module__])
    env_blob_path, env_meta_blob_path = upload_env(
        env_class=env_class,
        constructor_args=env_args_with_dataset,
        prefix=prefix,
        api_key=api_key,
        base_url=base_url,
        pip_dependencies=pip_dependencies,
        local_modules=local_modules,
        validate=validate_env,
        show_summary=show_summary,
    )

    # Smoke-test the uploaded environment on the rollout server
    if validate_env_remotely:
        rollout_client = RolloutClient(api_key=api_key)
        passed = rollout_client.validate_examples(
            examples=eval_dataset,
            env_cls_path=env_blob_path,
            env_metadata_path=env_meta_blob_path,
        )
        if not passed:
            raise RuntimeError(
                "Remote environment validation failed. "
                "Fix the errors above before retrying, or pass validate_env_remotely=False to skip."
            )

    # Launch training job
    if show_summary:
        print(f"Launching experiment (type: {experiment_type})...")

    trainer_client = TrainerClient(api_key=api_key, base_url=base_url)
    experiment_id = trainer_client.launch_experiment(
        experiment_type=experiment_type,
        env_cls_path=env_blob_path,
        env_metadata_path=env_meta_blob_path,
        train_dataset_path=train_blob_path,
        eval_dataset_path=eval_blob_path,
        name=experiment_name,
    )

    if show_summary:
        print(f"Experiment launched! ID: {experiment_id}")
        print(f"View experiment: {base_url}/experiments/{experiment_id}")

    return experiment_id


def validate_env_only(
    env_class: type,
    env_args: dict[str, Any],
    eval_dataset: list[dict],
    api_key: str,
    base_url: str = "https://app.cgft.io",
    prefix: str = "validate",
    pip_dependencies: list[str] | None = None,
    local_modules: list | None = None,
    validate_locally: bool = True,
    validate_remotely: bool = True,
    show_summary: bool = True,
) -> bool:
    """Validate an environment without launching a training job.

    Runs the same validation steps as ``train()`` — local bundle
    validation and remote rollout server validation — but stops before
    uploading datasets or launching the experiment.

    Use this to verify your environment works before committing to a
    full training run.

    Args:
        env_class: Environment class to validate.
        env_args: Constructor arguments for the environment.
        eval_dataset: Eval examples used for remote validation.
        api_key: CGFT API key.
        base_url: CGFT API base URL.
        prefix: Upload path prefix (default: ``"validate"``).
        pip_dependencies: Pip dependencies for the environment.
        local_modules: Local modules to include in the bundle.
        validate_locally: Run local pickle/structure validation.
        validate_remotely: Run remote rollout server validation.
        show_summary: Print progress.

    Returns:
        True if all requested validations pass, False otherwise.
    """
    from benchmax.bundle.bundler import bundle_env, write_bundle_files
    from benchmax.bundle.validator import validate_bundle

    if pip_dependencies is None:
        pip_dependencies = []
    if local_modules is None:
        local_modules = []

    env_module = sys.modules.get(env_class.__module__)
    if env_module and env_module not in local_modules:
        local_modules.append(env_module)

    if show_summary:
        print(f"Bundling {env_class.__name__}...")

    bundle = bundle_env(
        env_class,
        pip_dependencies=pip_dependencies,
        local_modules=local_modules,
        constructor_args=env_args,
    )

    if show_summary:
        print(f"  Pickled class: {len(bundle.pickled_class) / 1024:.2f} KB")
        print(f"  Dependencies: {bundle.metadata.pip_dependencies}")

    # --- Local validation ---
    if validate_locally:
        if show_summary:
            print(f"\nLocal validation (pickle roundtrip + structure)...")

        warnings = validate_bundle(bundle, constructor_args=env_args)
        if warnings:
            print(f"  Warnings: {warnings}")
            return False
        if show_summary:
            print("  Local validation PASSED")

    # --- Remote validation ---
    if validate_remotely:
        if show_summary:
            print(f"\nRemote validation (rollout server)...")

        # Upload env temporarily
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            pickle_path = tmp_path / "env-cls.pkl"
            metadata_path = tmp_path / "env-metadata.json"
            write_bundle_files(bundle, pickle_path, metadata_path)
            env_cls_bytes = pickle_path.read_bytes()
            env_meta_bytes = metadata_path.read_bytes()

        content_hash = hashlib.sha256(env_cls_bytes + env_meta_bytes).hexdigest()[:8]
        storage_client = StorageClient(api_key=api_key, base_url=base_url)

        env_path = f"envs/{prefix}/{content_hash}/env-cls.pkl"
        env_result = storage_client.upload_file(
            path=env_path,
            content=env_cls_bytes,
            mime_type="application/octet-stream",
        )
        env_meta_path = f"envs/{prefix}/{content_hash}/env-metadata.json"
        env_meta_result = storage_client.upload_file(
            path=env_meta_path,
            content=env_meta_bytes,
            mime_type="application/json",
        )

        rollout_client = RolloutClient(api_key=api_key)
        passed = rollout_client.validate_examples(
            examples=eval_dataset,
            env_cls_path=env_result["blobPath"],
            env_metadata_path=env_meta_result["blobPath"],
        )
        if not passed:
            if show_summary:
                print("  Remote validation FAILED")
            return False
        if show_summary:
            print("  Remote validation PASSED")

    if show_summary:
        print(f"\nAll validations passed for {env_class.__name__}!")
    return True

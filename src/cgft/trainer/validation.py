"""Local environment validation — catches training failures before job submission.

Simulates one mini training step locally (no GPU, no network) to verify
the env class contract matches what the trainer expects.
"""

from __future__ import annotations

import asyncio
import json
import pickle
import tempfile
from pathlib import Path
from typing import Any

import cloudpickle


def validate_env_full(
    env_class: type,
    env_args: dict[str, Any],
    train_dataset: list[dict[str, Any]],
    eval_dataset: list[dict[str, Any]] | None = None,
) -> bool:
    """Validate an environment class against the trainer's calling conventions.

    Runs 7 checks that mirror how the trainer actually calls env methods.
    Prints results as they run. Returns True if all checks pass.

    Args:
        env_class: The environment class (e.g., SearchEnv).
        env_args: Constructor kwargs for the env (same as train(env_args=...)).
        train_dataset: Training examples (list of dicts with question/answer).
        eval_dataset: Optional eval examples. Uses train_dataset[:2] if not given.
    """
    if not train_dataset:
        print("  \u2717 train_dataset is empty")
        return False

    examples = train_dataset[:5]
    passed = 0
    failed = 0

    print("Environment Validation")

    # ── 1. dataset_preprocess ────────────────────────────────────
    preprocessed = None
    try:
        preprocessed = env_class.dataset_preprocess(examples[0])
        if not isinstance(preprocessed, dict) or "prompt" not in preprocessed:
            print("  \u2717 dataset_preprocess did not return StandardizedExample")
            print("    Fix: Must return StandardizedExample with prompt,"
                  " ground_truth, init_rollout_args.")
            failed += 1
        else:
            print("  \u2713 dataset_preprocess returns StandardizedExample")
            passed += 1
    except Exception as exc:
        print(f"  \u2717 dataset_preprocess raised {type(exc).__name__}: {exc}")
        failed += 1

    # ── 2. Prompt hashability ────────────────────────────────────
    if preprocessed and isinstance(preprocessed, dict) and "prompt" in preprocessed:
        prompt = preprocessed["prompt"]
        try:
            hash(prompt)
            {prompt: True}
            if not isinstance(prompt, str):
                print(f"  \u2717 prompt is hashable but not a string — got {type(prompt).__name__}")
                print("    Fix: dataset_preprocess should return prompt as a string.")
                failed += 1
            else:
                print("  \u2713 prompt is hashable (string)")
                passed += 1
        except TypeError:
            print(f"  \u2717 prompt is NOT hashable — got {type(prompt).__name__}")
            print("    Fix: dataset_preprocess must return prompt as a"
                  " string, not a list of messages.")
            print("    The reward worker uses prompts as dict keys"
                  " for reroll tracking.")
            failed += 1
    else:
        print("  - prompt hashability: skipped (no preprocessed result)")

    # ── 3. load_dataset ──────────────────────────────────────────
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
            tmp_path = f.name

        result = env_class.load_dataset("json", data_files=tmp_path, split="train")
        if isinstance(result, tuple) and len(result) == 2:
            ds, _ = result
            if len(ds) > 0:
                print(f"  \u2713 load_dataset accepts (\"json\", data_files=...,"
                      f" split=\"train\") — {len(ds)} rows")
                passed += 1
            else:
                print("  \u2717 load_dataset returned empty dataset")
                failed += 1
        else:
            print(f"  \u2717 load_dataset returned {type(result).__name__},"
                  " expected (Dataset, str | None)")
            failed += 1

        Path(tmp_path).unlink(missing_ok=True)
    except Exception as exc:
        print(f"  \u2717 load_dataset raised {type(exc).__name__}: {exc}")
        print("    Fix: load_dataset must accept (\"json\", data_files=path, split=\"train\").")
        failed += 1

    # ── 4. Instantiate env + list_tools + run_tool ───────────────
    env = None
    try:
        env = env_class(**env_args)
    except Exception as exc:
        print(f"  \u2717 env instantiation failed: {type(exc).__name__}: {exc}")
        failed += 1

    if env is not None:
        try:
            tools = asyncio.run(env.list_tools())
            print(f"  \u2713 list_tools returns {len(tools)} tool(s)")
            passed += 1

            if tools:
                tool = tools[0]
                dummy_args = {}
                for prop_name, prop_schema in tool.input_schema.get("properties", {}).items():
                    ptype = prop_schema.get("type", "string")
                    if ptype == "string":
                        dummy_args[prop_name] = "test query"
                    elif ptype == "integer":
                        dummy_args[prop_name] = 10
                    elif ptype == "number":
                        dummy_args[prop_name] = 1.0
                    elif ptype == "boolean":
                        dummy_args[prop_name] = True

                try:
                    result = asyncio.run(
                        env.run_tool(rollout_id="test", tool_name=tool.name, **dummy_args)
                    )
                    if isinstance(result, str):
                        print(f"  \u2713 run_tool returns string (tested: {tool.name})")
                        passed += 1
                    else:
                        print(f"  \u2717 run_tool returned"
                              f" {type(result).__name__}, expected string")
                        failed += 1
                except Exception as exc:
                    print(f"  \u2717 run_tool raised {type(exc).__name__}: {exc}")
                    print("    Fix: run_tool must return a string. If tools need a real backend,")
                    print("    the training loop calls run_tool when the"
                          " model generates tool_calls.")
                    failed += 1
            else:
                print("  - run_tool: skipped (no tools defined)")
        except Exception as exc:
            print(f"  \u2717 list_tools raised {type(exc).__name__}: {exc}")
            failed += 1

    # ── 5. compute_reward with trainer-style kwargs ──────────────
    if env is not None and isinstance(preprocessed, dict) and "prompt" in preprocessed:
        try:
            sample_gt = preprocessed.get("ground_truth", "")
            init_args = preprocessed.get("init_rollout_args", {})
            if not isinstance(init_args, dict):
                init_args = {}

            # Build the sample dict the same way reward_worker.py does:
            # {**data_record, **completion_dict, **data_record["init_rollout_args"]}
            flattened = {
                "prompt": preprocessed["prompt"],
                "ground_truth": sample_gt,
                "init_rollout_args": init_args,
                "rollout_ids": "test-rollout",
                "completions": "I found the answer based on the search results.",
                **init_args,
            }

            reward = asyncio.run(
                env.compute_reward(
                    rollout_id=flattened["rollout_ids"],
                    completion=flattened["completions"],
                    **flattened,
                )
            )

            if not isinstance(reward, dict):
                print(f"  \u2717 compute_reward returned"
                      f" {type(reward).__name__}, expected dict[str, float]")
                failed += 1
            else:
                bad_values = {
                    k: type(v).__name__ for k, v in reward.items()
                    if not isinstance(v, (int, float))
                }
                if bad_values:
                    print(f"  \u2717 compute_reward has non-float values: {bad_values}")
                    failed += 1
                else:
                    print(f"  \u2713 compute_reward returns dict[str, float]: {reward}")
                    passed += 1
        except Exception as exc:
            print(f"  \u2717 compute_reward raised {type(exc).__name__}: {exc}")
            print("    Fix: compute_reward must accept (rollout_id, completion, **sample).")
            print("    The trainer flattens init_rollout_args into the sample dict.")
            failed += 1

    # ── 6. Pickle round-trip ─────────────────────────────────────
    try:
        data = cloudpickle.dumps(env_class)
        restored_cls = pickle.loads(data)
        restored_env = restored_cls(**env_args)
        tools = asyncio.run(restored_env.list_tools())
        print(f"  \u2713 pickle round-trip OK ({len(data)} bytes, {len(tools)} tools)")
        passed += 1
    except Exception as exc:
        print(f"  \u2717 pickle round-trip failed: {type(exc).__name__}: {exc}")
        failed += 1

    # ── 7. System prompt ─────────────────────────────────────────
    if env is not None:
        sp = getattr(env, "system_prompt", None)
        if not sp or not isinstance(sp, str):
            print("  \u2717 system_prompt is missing or not a string")
            failed += 1
        else:
            msg = f"  \u2713 system_prompt: {len(sp)} chars"
            if len(sp) > 10000:
                msg += " (warning: very long — consider shortening)"
            print(msg)
            passed += 1

    # ── Summary ──────────────────────────────────────────────────
    print()
    if failed == 0:
        print(f"All {passed} checks passed. Safe to call train().")
    else:
        print(f"{failed} check(s) failed. Fix before calling train().")

    return failed == 0

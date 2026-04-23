"""End-to-end: wizard codegen output through the real rollout-server.

Drives the exact path that broke playback on the expt-platform side:

  1. Run the wizard's codegen (tsx) to produce the build-env Python —
     imports + corpus setup + env class + env_args dict — for a given
     provider. This is byte-identical to what `app/api/experiments/
     build-env/route.ts` would ship to the Modal executor.

  2. exec() that Python in a namespace with CGFT_DATA_* set (mimicking
     what wizard-launch / the build-env route would inject).

  3. Pull `CustomSearchEnv` + `env_args` out of the namespace and call
     `train(env_class=..., env_args=..., dry_run=True,
     validate_env_remotely=True)`. Under the hood benchmax's
     bundle_env serializes env_args into an `args.pkl` sidecar and
     uploads alongside the class pickle. The rollout-server reads
     both, re-instantiates via `env_class(**args)`, and runs two
     validation rollouts.

If the wizard's codegen is structurally correct, validate returns
`validated`. If `args.pkl` is missing or shaped wrong, the rollout-
server KeyErrors on the constructor call — exactly the break
father_agent diagnosed.

Run via `pytest -m e2e`. Skipped when creds are missing.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

import cgft
from cgft.trainer.pipeline import train

pytestmark = pytest.mark.e2e


# ── Paths / config ───────────────────────────────────────────────────────

WIZARD_WORKTREE = Path(
    "/Users/thariqridha/Projects/cgft_projects/expt-platform/.claude/worktrees/codegen-integrated"
)
TSX_SCRIPT = "scripts/wizard-codegen-e2e.ts"

CGFT_API_KEY = os.environ.get("CGFT_API_KEY", "")
CGFT_BASE_URL = os.environ.get("CGFT_BASE_URL", "https://app.cgft.io")
LLM_BASE_URL = "https://llm.cgft.io/v1"
# Match the model cgft's remote validator uses — gpt-4o-mini (the
# wizard's default) may not be whitelisted on llm.cgft.io. We override
# env_args["judge_model"] post-exec to the known-good validation model.
JUDGE_MODEL = "gpt-5.4-nano"


# ── Helpers ──────────────────────────────────────────────────────────────


def _require(var: str) -> str:
    v = os.environ.get(var, "")
    if not v:
        pytest.skip(f"{var} missing — set in .env.test")
    return v


def _run_wizard_codegen(provider: str) -> str:
    """Invoke tsx on the wizard worktree to render the env-class script."""
    result = subprocess.run(
        ["npx", "tsx", "--tsconfig", "tsconfig.json", TSX_SCRIPT, provider],
        cwd=WIZARD_WORKTREE,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(
            f"wizard-codegen-e2e.ts failed (exit {result.returncode}):\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )
    return result.stdout


def _exec_wizard_code(
    provider: str, source_code: str, provider_env: dict[str, str]
) -> dict:
    """Set CGFT_DATA_* env vars, exec the wizard's Python, return its namespace.

    Matches the build-env executor's behavior: the executor forwards
    `resolveSourceEnvVars`'s output (the same CGFT_DATA_* dict we build
    here from .env.test) into the Modal container before exec()ing the
    script. `env_args` resolves its `os.environ[...]` / `os.environ.get(
    ..., LITERAL)` reads against this mapping.

    Registers the exec target as a real sys.modules entry so that
    `train()` can do `sys.modules[env_class.__module__]` to auto-add
    the env module to local_modules. Without this, the lookup KeyErrors.
    """
    # Mutate os.environ — required because the generated module-level
    # `env_args = {...}` reads os.environ when the module body runs.
    # Restore what we didn't write so the test is hermetic against
    # other tests in the session.
    saved: dict[str, str | None] = {}
    module_name = f"__wizard_env_{provider}__"
    try:
        for k, v in provider_env.items():
            saved[k] = os.environ.get(k)
            os.environ[k] = v

        # Create a real module so env_class.__module__ resolves via
        # sys.modules. Using a namespaced name (per-provider) keeps
        # concurrent test runs from stomping each other's namespaces.
        mod = types.ModuleType(module_name)
        mod.__dict__["__builtins__"] = __builtins__
        sys.modules[module_name] = mod
        exec(source_code, mod.__dict__)
        return mod.__dict__
    finally:
        for k, old in saved.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


def _make_dummy_qa(n: int = 16) -> list[dict]:
    """train() requires >= 16 train rows; remote validation only uses first 2."""
    return [
        {
            "question": f"Stub question {i} — how do I get started?",
            "answer": f"Stub answer {i}",
            "reference_chunks": [
                {
                    "content": "Install with pip, then import the library.",
                    "metadata": {"file": f"stub_{i}.md", "file_path": f"stub_{i}.md"},
                }
            ],
        }
        for i in range(n)
    ]


def _step(backend: str, message: str) -> None:
    print(f"  [{backend}] → {message}", flush=True)
    sys.stdout.flush()


def _validate_provider(
    provider: str,
    provider_env: dict[str, str],
    pip_deps: list[str],
) -> dict:
    """Generate wizard code → exec → train(dry_run). Returns the result dict."""
    t0 = time.monotonic()

    _step(provider, "rendering wizard codegen via tsx")
    source_code = _run_wizard_codegen(provider)
    _step(provider, f"got {len(source_code)} chars")

    _step(provider, "exec'ing generated code with CGFT_DATA_* injected")
    namespace = _exec_wizard_code(provider, source_code, provider_env)

    env_class = namespace.get("CustomSearchEnv")
    env_args = namespace.get("env_args")
    assert env_class is not None, "wizard codegen did not define CustomSearchEnv"
    assert isinstance(env_args, dict), (
        f"wizard codegen did not build env_args dict (got {type(env_args).__name__})"
    )
    _step(provider, f"env_args keys: {sorted(env_args.keys())}")

    # Swap the judge model to one llm.cgft.io definitely whitelists.
    # The wizard hard-codes gpt-4o-mini which may 404 in remote validation.
    env_args["judge_model"] = JUDGE_MODEL

    _step(provider, "calling train(dry_run=True, validate_env_remotely=True)")
    result = train(
        env_class=env_class,
        env_args=env_args,
        train_dataset=_make_dummy_qa(16),
        eval_dataset=_make_dummy_qa(16)[:4],
        prefix=f"e2e-wizard-{provider}",
        api_key=CGFT_API_KEY,
        base_url=CGFT_BASE_URL,
        local_modules=[cgft],
        experiment_name=f"e2e-wizard-{provider}-{int(time.time())}",
        pip_dependencies=pip_deps,
        validate_env=False,  # local isolated venv can't pip install cgft
        validate_env_remotely=True,
        validation_model=JUDGE_MODEL,
        dry_run=True,
        show_summary=True,
    )
    _step(provider, f"status={result.get('status')!r} ({time.monotonic() - t0:.1f}s)")
    return result


def _assert_validated(result: dict) -> None:
    assert isinstance(result, dict)
    assert result.get("status") == "validated", f"Expected 'validated', got: {result}"


# ── Provider cases ───────────────────────────────────────────────────────


class TestWizardCodegenChroma:
    """Drives the wizard's Chroma codegen through the rollout-server.

    This is the regression test for the playback break father_agent
    diagnosed: wizard bundles with empty constructor_args → no args.pkl
    → rollout-server instantiation fails. The new codegen emits env_args
    at module scope → build-env wraps with bundle_env(constructor_args=
    env_args) → args.pkl sidecar ships → rollout-server rehydrates.
    """

    def test_validates_remotely(self):
        CGFT_API_KEY_V = _require("CGFT_API_KEY")
        provider_env = {
            # Shared platform key — referenced by the judge config in
            # env_args (judge_api_key=API_KEY).
            "CGFT_API_KEY": CGFT_API_KEY_V,
            # Provider creds injected as the build-env route would.
            "CGFT_DATA_api_key": _require("CHROMA_CLOUD_API_KEY"),
            "CGFT_DATA_tenant": _require("CHROMA_CLOUD_TENANT"),
            "CGFT_DATA_database": _require("CHROMA_CLOUD_DATABASE"),
            "CGFT_DATA_collection_name": _require("CHROMA_CLOUD_COLLECTION"),
        }
        result = _validate_provider(
            provider="chroma",
            provider_env=provider_env,
            pip_deps=["chromadb>=1.0.0", "snowballstemmer>=2.2.0", "openai"],
        )
        _assert_validated(result)


class TestWizardCodegenTurbopuffer:
    def test_validates_remotely(self):
        CGFT_API_KEY_V = _require("CGFT_API_KEY")
        provider_env = {
            "CGFT_API_KEY": CGFT_API_KEY_V,
            "CGFT_DATA_api_key": _require("TPUF_API_KEY"),
            "CGFT_DATA_namespace": _require("TPUF_NAMESPACE"),
            "CGFT_DATA_region": os.environ.get("TPUF_REGION", "aws-us-east-1"),
        }
        result = _validate_provider(
            provider="turbopuffer",
            provider_env=provider_env,
            pip_deps=["turbopuffer", "openai"],
        )
        _assert_validated(result)


class TestWizardCodegenPinecone:
    def test_validates_remotely(self):
        CGFT_API_KEY_V = _require("CGFT_API_KEY")
        provider_env = {
            "CGFT_API_KEY": CGFT_API_KEY_V,
            "CGFT_DATA_api_key": _require("PINECONE_API_KEY"),
            "CGFT_DATA_index_name": _require("PINECONE_INDEX_NAME"),
            # indexHost normally comes from the Connect probe's
            # autoFilled linkedField. Hardcoded here to match the one
            # the wizard-codegen-e2e.ts DataSource resource baked in.
            "CGFT_DATA_index_host": (
                "cgft-test-mjnq7qb.svc.aped-4627-b74a.pinecone.io"
            ),
        }
        result = _validate_provider(
            provider="pinecone",
            provider_env=provider_env,
            pip_deps=["pinecone>=5.0.0", "openai"],
        )
        _assert_validated(result)

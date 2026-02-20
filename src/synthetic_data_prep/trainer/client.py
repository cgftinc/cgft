"""Trainer API clients for storage uploads and job management."""

from __future__ import annotations

import hashlib
import json
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from .exceptions import AuthenticationError, JobLaunchError, TrainerError

if TYPE_CHECKING:
    pass


# MIME type mappings for common file extensions
_MIME_TYPES = {
    ".jsonl": "application/jsonl",
    ".json": "application/json",
    ".yaml": "text/yaml",
    ".yml": "text/yaml",
    ".pkl": "application/octet-stream",
    ".pickle": "application/octet-stream",
    ".bmxp": "application/octet-stream",
}


def _get_mime_type(path: Path) -> str:
    """Get MIME type from file extension."""
    suffix = path.suffix.lower()
    mime_type = _MIME_TYPES.get(suffix)
    if mime_type is None:
        raise ValueError(
            f"Unsupported file type: {suffix}. "
            f"Supported: {', '.join(_MIME_TYPES.keys())}"
        )
    return mime_type


def _file_hash(content: bytes, length: int = 8) -> str:
    """Compute a short hash of file content."""
    return hashlib.sha256(content).hexdigest()[:length]


@dataclass
class StorageClient:
    """Client for uploading files to storage via pre-signed URLs.

    Uses the ``GET /api/storage/upload-url`` endpoint to obtain a pre-signed
    upload URL, then PUTs the file content directly to that URL.

    Example:
        >>> client = StorageClient(api_key="sk_...", base_url="http://localhost:3000")
        >>> result = client.upload_file(
        ...     path="datasets/my-data.jsonl",
        ...     content=b'{"q": "...", "a": "..."}',
        ...     mime_type="application/jsonl",
        ... )
        >>> print(f"Uploaded to {result['blobPath']}")
    """

    api_key: str
    base_url: str = "http://localhost:3000"
    timeout: float = 60.0
    _http_client: httpx.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize HTTP client with auth headers."""
        self._http_client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )

    def __enter__(self) -> StorageClient:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        """Close the HTTP client."""
        self._http_client.close()

    def _handle_response_errors(self, response: httpx.Response) -> None:
        """Convert HTTP errors to appropriate exceptions."""
        if response.status_code in (200, 201):
            return

        try:
            error_data = response.json()
            message = error_data.get("error", response.text)
        except Exception:
            message = response.text

        if response.status_code == 401:
            raise AuthenticationError(message, response.status_code)

        raise TrainerError(message, response.status_code)

    def _get_upload_url(
        self, path: str, mime_type: str, expires_in_minutes: int | None = None
    ) -> dict:
        """Request a pre-signed upload URL from the storage API.

        Args:
            path: Storage path for the file
            mime_type: MIME type of the file
            expires_in_minutes: Optional expiration override for the signed URL

        Returns:
            Dict with ``uploadUrl``, ``expiresAt``, ``blobPath``, ``willOverwrite`` keys.
        """
        params: dict[str, Any] = {"path": path, "mimeType": mime_type}
        if expires_in_minutes is not None:
            params["expiresInMinutes"] = expires_in_minutes

        response = self._http_client.get(
            "/api/storage/upload-url",
            params=params,
        )
        self._handle_response_errors(response)
        return response.json()

    def _put_to_signed_url(self, upload_url: str, content: bytes, mime_type: str) -> None:
        """PUT file content directly to a pre-signed URL."""
        response = httpx.put(
            upload_url,
            content=content,
            headers={"Content-Type": mime_type, "x-ms-blob-type": "BlockBlob"},
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            raise TrainerError(
                f"Upload to storage failed: {response.status_code} {response.text}",
                response.status_code,
            )

    # === Core upload methods ===

    def upload_file(
        self,
        path: str,
        content: bytes,
        mime_type: str,
        *,
        expires_in_minutes: int | None = None,
    ) -> dict:
        """Upload raw bytes to storage.

        Args:
            path: Storage path for the file (e.g. "datasets/search/abc123/qa-dataset.jsonl")
            content: Raw bytes to upload
            mime_type: MIME type for the Content-Type header
            expires_in_minutes: Optional expiration override for the signed URL

        Returns:
            Dict with ``uploadUrl``, ``expiresAt``, ``blobPath``, ``willOverwrite`` keys.

        Raises:
            AuthenticationError: If API key is invalid
            TrainerError: If upload fails
        """
        url_response = self._get_upload_url(
            path,
            mime_type,
            expires_in_minutes=expires_in_minutes,
        )
        self._put_to_signed_url(url_response["uploadUrl"], content, mime_type)
        return url_response

    def upload_local_file(
        self,
        path: str,
        file_path: str | Path,
        *,
        expires_in_minutes: int | None = None,
    ) -> dict:
        """Upload a local file to storage.

        Args:
            path: Storage path for the file
            file_path: Local path to the file to upload
            expires_in_minutes: Optional expiration override for the signed URL

        Returns:
            Dict with ``uploadUrl``, ``expiresAt``, ``blobPath``, ``willOverwrite`` keys.

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file type is not supported
            AuthenticationError: If API key is invalid
            TrainerError: If upload fails
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        mime_type = _get_mime_type(file_path)
        content = file_path.read_bytes()
        return self.upload_file(
            path,
            content,
            mime_type,
            expires_in_minutes=expires_in_minutes,
        )


@dataclass
class TrainerClient:
    """Client for launching and managing training jobs.

    Example:
        >>> client = TrainerClient(api_key="sk_...", base_url="http://localhost:3000")
        >>> job = client.launch_job(
        ...     job_type="search",
        ...     args={"dataset": "datasets/search/abc123/qa-dataset.jsonl"},
        ... )
        >>> print(f"Launched job: {job}")
    """

    api_key: str
    base_url: str = "http://localhost:3000"
    timeout: float = 30.0
    _http_client: httpx.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize HTTP client with auth headers."""
        self._http_client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )

    def __enter__(self) -> TrainerClient:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        """Close the HTTP client."""
        self._http_client.close()

    def _handle_response_errors(self, response: httpx.Response) -> None:
        """Convert HTTP errors to appropriate exceptions."""
        if response.status_code in (200, 201, 202):
            return

        try:
            error_data = response.json()
            message = error_data.get("error", response.text)
        except Exception:
            message = response.text

        if response.status_code == 401:
            raise AuthenticationError(message, response.status_code)

        raise JobLaunchError(message, response.status_code)

    def launch_experiment(
        self, experiment_type: str, env_cls_path: str, env_metadata_path: str
    ) -> str:
        """Launch a new experiment from a job template.

        Args:
            experiment_type: Type of experiment to launch (e.g. "search")
            env_cls_path: Path to the environment class bundle (.bmxp file)
            env_metadata_path: Path to the environment kwargs JSON file

        Returns:
            The experiment ID.

        Raises:
            AuthenticationError: If API key is invalid
            JobLaunchError: If experiment launch fails
        """
        response = self._http_client.post(
            "/api/experiments/launch",
            json={
                "type": experiment_type,
                "args": {
                    "env_cls_path": env_cls_path,
                    "env_metadata_path": env_metadata_path,
                },
            },
        )
        self._handle_response_errors(response)
        return response.json()["experimentId"]


ROLLOUT_SERVER_URL = "https://autobots.cgft.io"

_VALIDATION_MODEL = "grok-4-fast-reasoning"
_VALIDATION_LLM_BASE_URL = "https://app.cgft.io/api/llm"

# ANSI colours
_GREEN  = "\033[32m"
_RED    = "\033[31m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"

def _ok(msg: str)   -> str: return f"{_GREEN}✔  {msg}{_RESET}"
def _err(msg: str)  -> str: return f"{_RED}✗  {msg}{_RESET}"
def _info(msg: str) -> str: return f"{_CYAN}{msg}{_RESET}"
def _hdr(msg: str)  -> str: return f"\n{_BOLD}{msg}{_RESET}"


def _iter_sse(response: httpx.Response) -> Iterator[dict]:
    """Yield parsed event dicts from a synchronous SSE response."""
    for line in response.iter_lines():
        if line.startswith("data: "):
            try:
                yield json.loads(line[len("data: "):])
            except json.JSONDecodeError:
                pass


def _print_event(event: dict, idx: int) -> None:
    """Print a single SSE event in a human-readable format."""
    etype = event.get("event", "?")
    prefix = f"  [ex {idx}]"

    if etype == "rollout_started":
        print(_info(f"{prefix} → rollout_started"))

    elif etype == "message":
        msg     = event.get("message", {})
        role    = msg.get("role", "?")
        content = msg.get("content", "")

        # content may be a list of blocks (tool calls) or a plain string
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    btype = block.get("type", "")
                    if btype == "text":
                        preview = textwrap.shorten(block.get("text", ""), width=120, placeholder="…")
                        print(f"{prefix} → message [{role}/text]: {preview}")
                    elif btype == "tool_use":
                        print(f"{prefix} → message [{role}/tool_use]: {block.get('name')}({json.dumps(block.get('input', {}))[:80]})")
                    elif btype == "tool_result":
                        preview = textwrap.shorten(str(block.get("content", "")), width=120, placeholder="…")
                        print(f"{prefix} → message [{role}/tool_result]: {preview}")
                    else:
                        print(f"{prefix} → message [{role}/{btype}]")
        else:
            preview = textwrap.shorten(str(content), width=120, placeholder="…")
            print(f"{prefix} → message [{role}]: {preview}")

    elif etype == "reward":
        print(f"{prefix} → reward: {event.get('rewards')}")

    elif etype == "rollout_completed":
        success = event.get("success")
        status  = _ok("success") if success else _err("failed")
        print(f"{prefix} → rollout_completed  {status}  "
              f"rewards={event.get('rewards')}  error={event.get('error')}")

    elif etype in ("worker_error", "error", "cancelled"):
        print(_err(f"{prefix} → {etype}: {event.get('error')}"))


class RolloutClient:
    """Thin synchronous client for the /rollout/stream endpoint.

    Args:
        api_key:    Bearer token for the rollout server.
        server_url: Base URL of the rollout server.
        timeout:    Per-request timeout in seconds (default 300 — rollouts can be slow).
    """

    _TERMINAL = {"rollout_completed", "worker_error", "cancelled", "error"}

    def __init__(
        self,
        api_key: str,
        server_url: str = ROLLOUT_SERVER_URL,
        timeout: float = 300.0,
    ) -> None:
        self._api_key    = api_key
        self._server_url = server_url.rstrip("/")
        self._timeout    = timeout

    def stream_rollout(
        self,
        raw_example: dict[str, Any],
        env_cls_path: str,
        env_meta_path: str,
        example_index: int = 0,
        max_turns: int = 4,
        max_tool_calls: int = 8,
        max_completion_tokens: int = 1024,
    ) -> dict[str, Any]:
        """Run one rollout against the stream endpoint, printing events live.

        Args:
            raw_example:           Raw dataset row (passed as ``raw_example``).
            env_cls_path:          Blob path to the uploaded env .pkl file.
            env_meta_path:         Blob path to the uploaded env-meta .json file.
            example_index:         Display index used in printed output.
            max_turns:             Max conversation turns.
            max_tool_calls:        Max tool calls per rollout.
            max_completion_tokens: Max tokens per completion.

        Returns:
            The final ``rollout_completed`` event dict, or an error/cancelled event.

        Raises:
            RuntimeError: If the HTTP request itself fails or returns non-200.
        """
        payload = {
            "standardized_example": None,
            "raw_example":          raw_example,
            "env": {
                "env_cls_path":  env_cls_path,
                "env_meta_path": env_meta_path,
            },
            "llm": {
                "base_url": _VALIDATION_LLM_BASE_URL,
                "api_key":  self._api_key,
                "model":    _VALIDATION_MODEL,
            },
            "options": {
                "max_turns":             max_turns,
                "max_tool_calls":        max_tool_calls,
                "max_completion_tokens": max_completion_tokens,
            },
        }

        url = f"{self._server_url}/rollout/stream"
        headers = {"Authorization": f"Bearer {self._api_key}"}

        with httpx.stream(
            "POST",
            url,
            json=payload,
            headers=headers,
            timeout=self._timeout,
        ) as response:
            if response.status_code != 200:
                body = response.read().decode()
                raise RuntimeError(
                    f"Rollout server returned HTTP {response.status_code}: {body[:300]}"
                )

            final: dict[str, Any] = {}
            for event in _iter_sse(response):
                _print_event(event, example_index)
                if event.get("event") in self._TERMINAL:
                    final = event
                    break

        return final

    def validate_examples(
        self,
        examples: list[dict[str, Any]],
        env_cls_path: str,
        env_meta_path: str,
        n: int = 2,
    ) -> bool:
        """Run rollouts on the first *n* examples and report pass/fail.

        Args:
            examples:     Full dataset (list of raw dicts).
            env_cls_path: Blob path to the uploaded env .pkl file.
            env_meta_path: Blob path to the uploaded env-meta .json file.
            n:            Number of examples to validate (default 2).

        Returns:
            True if all sampled rollouts completed successfully, False otherwise.
        """
        sample = examples[:n]
        print(_hdr(f"── Remote validation: {len(sample)} example(s) on {_VALIDATION_MODEL} ──"))

        all_ok = True
        for i, example in enumerate(sample):
            print(_info(f"\n  Example {i} — {json.dumps(example)[:120]}"))
            try:
                final = self.stream_rollout(
                    raw_example=example,
                    env_cls_path=env_cls_path,
                    env_meta_path=env_meta_path,
                    example_index=i,
                )
                if not final.get("success"):
                    all_ok = False
            except RuntimeError as exc:
                print(_err(f"  Example {i} failed: {exc}"))
                all_ok = False

        print()
        if all_ok:
            print(_ok("Remote validation passed"))
        else:
            print(_err("Remote validation failed — check output above before launching a full job"))

        return all_ok

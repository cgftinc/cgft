"""Trainer API clients for storage uploads and job management."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from .exceptions import AuthenticationError, JobLaunchError, TrainerError

if TYPE_CHECKING:
    from synthetic_data_prep.qa_generation.models import QADataset


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
        if response.status_code in (200, 201):
            return

        try:
            error_data = response.json()
            message = error_data.get("error", response.text)
        except Exception:
            message = response.text

        if response.status_code == 401:
            raise AuthenticationError(message, response.status_code)

        raise JobLaunchError(message, response.status_code)

    def launch_job(self, job_type: str, args: dict[str, Any]) -> dict:
        """Launch a training job.

        Args:
            job_type: Type of job to launch (e.g. "search")
            args: Job arguments (e.g. {"dataset": "path/to/dataset.jsonl"})

        Returns:
            Response from the API (structure depends on endpoint).

        Raises:
            AuthenticationError: If API key is invalid
            JobLaunchError: If job launch fails
        """
        response = self._http_client.post(
            "/api/jobs/launch",
            json={"type": job_type, "args": args},
        )
        self._handle_response_errors(response)
        return response.json()

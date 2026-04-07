"""SearchAgentLinker — LLM-driven chunk linker using RolloutClient.

Uses an LLM via the rollout server to search the corpus for related
chunks, producing higher-quality links than heuristic-only approaches.
Falls back to MetadataChunkLinker when the LLM linker fails or is not
selected.
"""

from __future__ import annotations

import json
import logging
import random
import re
from typing import Any

from benchmax.bundle.bundler import bundle_env

import cgft
from cgft.corpus.search_client import SearchClient
from cgft.envs.linker_env import LinkerEnv
from cgft.qa_generation.anchor_selector import AnchorBundle
from cgft.qa_generation.cgft_models import SearchAgentLinkerCfg
from cgft.qa_generation.metadata_linker import MetadataChunkLinker
from cgft.trainer.client import RolloutClient

logger = logging.getLogger(__name__)


def _get_content(chunk: Any) -> str:
    if hasattr(chunk, "content"):
        return str(chunk.content)
    return str(chunk)


def _get_metadata(chunk: Any) -> dict[str, Any]:
    if hasattr(chunk, "metadata_dict"):
        md = chunk.metadata_dict
        return dict(md) if md else {}
    if hasattr(chunk, "metadata"):
        raw = chunk.metadata
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, (tuple, list)):
            return dict(raw)
    if isinstance(chunk, dict):
        return dict(chunk.get("metadata", {}))
    return {}


def _get_file(chunk: Any) -> str:
    md = _get_metadata(chunk)
    return str(md.get("file", "") or md.get("file_name", "") or "")


class SearchAgentLinker:
    """LLM-driven chunk linker backed by RolloutClient.

    Hybrid approach:
    1. Always runs MetadataChunkLinker first (zero cost).
    2. If confidence is low OR this item is selected by ``search_agent_pct``,
       falls back to the LLM linker via rollout.
    3. The LLM generates search queries; those queries are replayed against
       the source to obtain actual Chunk objects for the AnchorBundle.
    """

    def __init__(
        self,
        *,
        metadata_linker: MetadataChunkLinker,
        source: Any,
        cfg: SearchAgentLinkerCfg,
        search_agent_pct: float,
        rollout_client: RolloutClient,
        search_client: SearchClient,
        llm_model: str = "",
        llm_base_url: str = "",
        llm_api_key: str = "",
    ) -> None:
        self._metadata_linker = metadata_linker
        self._source = source
        self._cfg = cfg
        self._search_agent_pct = search_agent_pct
        self._rollout_client = rollout_client
        self._llm_model = llm_model
        self._llm_base_url = llm_base_url
        self._llm_api_key = llm_api_key

        self._env_cls_bytes: bytes | None = None
        self._env_meta_bytes: bytes | None = None
        self._env_args_bytes: bytes | None = None
        self._prepare_env_bundle(search_client)

    # ------------------------------------------------------------------
    # Env bundling
    # ------------------------------------------------------------------

    def _prepare_env_bundle(self, search_client: SearchClient) -> None:
        """Bundle LinkerEnv with the SearchClient for rollout server use.

        If the user pre-configured env_bundle paths/files, those take
        precedence.  Otherwise, use ``bundle_env`` (same mechanism as
        ``train()``) to package LinkerEnv with its constructor args.
        """
        bundle = self._cfg.env_bundle
        if bundle.has_paths() or bundle.has_files():
            return

        env_bundle = bundle_env(
            LinkerEnv,
            local_modules=[cgft],
            constructor_args={"search": search_client},
            validate=False,
        )
        self._env_cls_bytes = env_bundle.pickled_class
        self._env_meta_bytes = env_bundle.metadata.to_json_bytes()
        self._env_args_bytes = env_bundle.pickled_constructor_args

    # ------------------------------------------------------------------
    # ChunkLinker protocol
    # ------------------------------------------------------------------

    def link(
        self,
        primary_chunk: Any,
        *,
        target_hop_count: int | None = None,
        corpus_pool: list[Any] | None = None,
        reasoning_mode: str = "",
    ) -> AnchorBundle:
        hop_count = target_hop_count or 2
        n_secondaries = hop_count - 1

        # Always run metadata linker first (zero cost).
        metadata_bundle = self._metadata_linker.link(
            primary_chunk,
            target_hop_count=target_hop_count,
            corpus_pool=corpus_pool,
            reasoning_mode=reasoning_mode,
        )

        confidence = metadata_bundle.structural_hints.get("confidence", 0.0)
        use_llm = random.random() < self._search_agent_pct or confidence < 0.5

        if not use_llm:
            return metadata_bundle

        try:
            return self._link_with_llm(primary_chunk, hop_count, n_secondaries, reasoning_mode)
        except Exception:
            logger.exception("SearchAgentLinker LLM linking failed")
            if self._cfg.fallback_to_metadata:
                metadata_bundle.structural_hints["llm_fallback"] = True
                return metadata_bundle
            raise

    # ------------------------------------------------------------------
    # LLM linking via rollout
    # ------------------------------------------------------------------

    def _link_with_llm(
        self,
        primary_chunk: Any,
        hop_count: int,
        n_secondaries: int,
        reasoning_mode: str,
    ) -> AnchorBundle:
        # search_related expects a Chunk with .hash attribute.
        if not hasattr(primary_chunk, "hash"):
            return self._empty_bundle(primary_chunk, 1, "not_a_chunk")

        content = _get_content(primary_chunk)
        raw_example = {
            "prompt": content[:4000],
            "target_n": n_secondaries,
        }

        result = self._run_rollout(raw_example)
        queries = _extract_queries(result.get("messages", []))

        if not queries:
            return self._empty_bundle(primary_chunk, hop_count, "no_queries")

        # Replay queries against source to get actual Chunk objects.
        search_results = self._source.search_related(
            primary_chunk,
            queries,
            top_k=n_secondaries * 3,
        )

        candidates = self._filter_candidates(primary_chunk, search_results, n_secondaries)
        secondary_chunks = candidates[:n_secondaries]

        actual_hop_count = len(secondary_chunks) + 1
        confidence = min(len(secondary_chunks) / n_secondaries, 1.0) if n_secondaries > 0 else 1.0

        return AnchorBundle(
            primary_chunk=primary_chunk,
            secondary_chunks=secondary_chunks,
            target_hop_count=min(hop_count, actual_hop_count),
            structural_hints={
                "linker": "search_agent",
                "confidence": confidence,
                "queries_used": queries,
                "candidates_found": len(search_results),
                "reasoning_mode": reasoning_mode,
                "hop_demoted": actual_hop_count < hop_count,
                "requested_hop_count": hop_count,
            },
        )

    def _run_rollout(self, raw_example: dict[str, Any]) -> dict[str, Any]:
        cfg = self._cfg
        env_bundle = cfg.env_bundle

        kwargs: dict[str, Any] = {
            "raw_example": raw_example,
            "llm_model": self._llm_model,
            "llm_base_url": self._llm_base_url,
            "llm_api_key": self._llm_api_key,
            "max_turns": cfg.max_turns,
            "max_tool_calls": cfg.max_tool_calls,
            "max_completion_tokens": cfg.max_completion_tokens,
            "capture_messages": True,
            "include_event_meta": False,
        }

        if env_bundle.has_paths():
            kwargs["env_cls_path"] = env_bundle.env_cls_path
            kwargs["env_metadata_path"] = env_bundle.env_metadata_path
        elif env_bundle.has_files():
            cls_bytes, meta_bytes = env_bundle.as_bytes_bundle()
            kwargs["env_cls_bytes"] = cls_bytes
            kwargs["env_metadata_bytes"] = meta_bytes
        else:
            kwargs["env_cls_bytes"] = self._env_cls_bytes
            kwargs["env_metadata_bytes"] = self._env_meta_bytes
            if self._env_args_bytes is not None:
                kwargs["env_args_bytes"] = self._env_args_bytes

        return self._rollout_client.stream_rollout(**kwargs)

    # ------------------------------------------------------------------
    # Candidate filtering
    # ------------------------------------------------------------------

    def _filter_candidates(
        self,
        primary_chunk: Any,
        search_results: list[dict[str, Any]],
        n_secondaries: int,
    ) -> list[Any]:
        primary_hash = getattr(primary_chunk, "hash", None)
        primary_file = _get_file(primary_chunk)
        primary_tokens = set(_get_content(primary_chunk).lower().split())

        seen: set[str] = set()
        selected: list[Any] = []
        selected_tokens: list[set[str]] = []

        for result in search_results:
            chunk = result.get("chunk")
            if chunk is None:
                continue

            chunk_hash = getattr(chunk, "hash", None)
            if chunk_hash and chunk_hash == primary_hash:
                continue
            if chunk_hash and chunk_hash in seen:
                continue

            chunk_content = _get_content(chunk)
            if len(chunk_content) < 400:
                continue

            chunk_file = _get_file(chunk)
            if primary_file and chunk_file and chunk_file == primary_file:
                continue

            tokens = set(chunk_content.lower().split())

            # Check coherence with primary.
            if primary_tokens and tokens:
                jaccard = len(primary_tokens & tokens) / len(primary_tokens | tokens)
                if jaccard < 0.05:
                    continue

            # Check diversity with already-selected chunks.
            too_similar = False
            for prev in selected_tokens:
                sim = len(tokens & prev) / len(tokens | prev) if tokens and prev else 0.0
                if sim > 0.8:
                    too_similar = True
                    break
            if too_similar:
                continue

            if chunk_hash:
                seen.add(chunk_hash)
            selected.append(chunk)
            selected_tokens.append(tokens)
            if len(selected) >= n_secondaries:
                break

        return selected

    def _empty_bundle(
        self,
        primary_chunk: Any,
        hop_count: int,
        reason: str,
    ) -> AnchorBundle:
        return AnchorBundle(
            primary_chunk=primary_chunk,
            secondary_chunks=[],
            target_hop_count=1,  # No secondaries found; just the primary.
            structural_hints={
                "linker": "search_agent",
                "confidence": 0.0,
                "reason": reason,
                "requested_hop_count": hop_count,
            },
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_TOOL_CALL_RE = re.compile(
    r'<tool_call>\s*(.*?)\s*</tool_call>', re.DOTALL,
)


def _extract_queries(messages: list[dict[str, Any]]) -> list[str]:
    """Extract search queries from rollout messages.

    Handles two message formats:
    1. Structured: content is list[dict] with type="tool_use" blocks
    2. Text: content is a string with <tool_call> JSON blocks
    """
    queries: list[str] = []
    for msg in messages:
        content = msg.get("content")
        # Structured tool_use blocks
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    input_data = block.get("input", {})
                    query = input_data.get("query")
                    if query and query not in queries:
                        queries.append(query)
        # Text content with <tool_call> markers
        elif isinstance(content, str):
            for match in _TOOL_CALL_RE.finditer(content):
                try:
                    payload = json.loads(match.group(1))
                except (json.JSONDecodeError, ValueError):
                    continue
                args = payload.get("arguments", payload.get("input", {}))
                query = args.get("query")
                if query and query not in queries:
                    queries.append(query)
    return queries

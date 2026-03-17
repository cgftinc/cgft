"""CgftSearchEnv — SearchEnv backed by the CGFT HTTP API using BM25 search."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

import aiohttp
from benchmax.envs.types import ToolDefinition

from .query_rewriter import heuristic_query_rewrite
from .search_env import SearchEnv

class CgftSearchEnv(SearchEnv):
    """Search environment backed by the CGFT HTTP API using BM25 search.

    Args:
        api_key: API key for the search service
        corpus_id: ID of the corpus to search
        base_url: Base URL for the API
    """

    def __init__(
        self,
        api_key: str,
        corpus_id: str,
        base_url: str,
        **kwargs,
    ):
        self._api_key = api_key
        self._corpus_id = corpus_id
        self._base_url = base_url.rstrip("/")
        self._query_rewriter_enabled = bool(kwargs.get("query_rewriter_enabled", False))
        self._query_rewriter_strategy = str(
            kwargs.get("query_rewriter_strategy", "intent")
        ).lower()
        self._query_rewriter_model = str(kwargs.get("query_rewriter_model", "gpt-5-mini"))
        self._query_rewriter_api_key = str(kwargs.get("query_rewriter_api_key", ""))
        self._query_rewriter_base_url = str(kwargs.get("query_rewriter_base_url", ""))
        self._query_rewriter_max_terms = max(
            4,
            int(kwargs.get("query_rewriter_max_terms", 16)),
        )
        self._query_rewriter_max_chars = max(
            40,
            int(kwargs.get("query_rewriter_max_chars", 140)),
        )
        self._query_rewriter_log = bool(kwargs.get("query_rewriter_log", False))

        search_tool_definition = ToolDefinition(
            name="search",
            description="Search using BM25 with optional metadata and filename filtering.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query text (can be natural language).",
                    },
                    "intent": {
                        "type": "object",
                        "description": (
                            "Optional structured search intent. If provided, the retriever "
                            "converts this intent into a BM25-friendly query."
                        ),
                        "properties": {
                            "entities": {"type": "array", "items": {"type": "string"}},
                            "must_terms": {"type": "array", "items": {"type": "string"}},
                            "phrases": {"type": "array", "items": {"type": "string"}},
                            "keywords": {"type": "array", "items": {"type": "string"}},
                            "constraints": {"type": "array", "items": {"type": "string"}},
                            "exclude_terms": {"type": "array", "items": {"type": "string"}},
                            "sites": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                    "filters": {
                        "type": "object",
                        "description": (
                            "Optional structured filter object using domain-specific query language (DSL). "
                            "Field condition shape: "
                            "{\"field\":\"<name>\",\"op\":\"eq|in|gte|lte|contains_any|contains_all\",\"value\":...}. "
                            "Logical shape: {\"and\":[...]} or {\"or\":[...]} or {\"not\":{...}}. "
                            "Example: "
                            "{\"and\":[{\"field\":\"metadata_name\",\"op\":\"eq\",\"value\":\"example_value\"},"
                            "{\"field\":\"metadata_name_2\",\"op\":\"gte\",\"value\":123}]}"
                        ),
                    },
                    "filename": {
                        "type": "string",
                        "description": (
                            "Optional filename filter. Simple string for substring match "
                            "(e.g., 'config') or regex pattern (e.g., '.*\\.json$')."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number of results to return (default 10).",
                    },
                },
                "required": ["query"]
            },
        )

        self._tools: dict[str, tuple[ToolDefinition, Callable]] = {
            search_tool_definition.name: (search_tool_definition, self._search_tool)
        }
        # BaseEnv does not define __init__, so keep optional kwargs as attributes
        # instead of forwarding them to object.__init__.
        self._experiment_id = kwargs.get("experiment_id")
        self._rollout_api_key = kwargs.get("api_key")

    def _heuristic_query_rewrite(self, query: str) -> str:
        """Rewrite verbose natural-language questions into BM25-friendly terms."""
        return heuristic_query_rewrite(
            query,
            max_terms=self._query_rewriter_max_terms,
            max_chars=self._query_rewriter_max_chars,
        )

    @staticmethod
    def _as_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v) for v in value if str(v).strip()]
        if isinstance(value, str) and value.strip():
            return [value]
        return []

    @staticmethod
    def _normalize_site(site: str) -> str:
        s = site.strip().lower()
        if not s:
            return ""
        s = re.sub(r"^https?://", "", s)
        s = s.split("/", 1)[0]
        return s

    def _parse_intent(self, intent: Any) -> dict[str, Any] | None:
        if isinstance(intent, dict):
            return intent
        if isinstance(intent, str):
            text = intent.strip()
            if not text:
                return None
            try:
                parsed = json.loads(text)
            except Exception:
                return None
            return parsed if isinstance(parsed, dict) else None
        return None

    def _intent_to_query(self, intent: dict[str, Any], fallback_query: str) -> str:
        terms: list[str] = []
        seen: set[str] = set()

        def add_term(term: str) -> None:
            t = re.sub(r"\s+", " ", str(term).strip())
            if not t:
                return
            key = t.lower()
            if key in seen:
                return
            seen.add(key)
            terms.append(t)

        def add_phrase(phrase: str) -> None:
            p = re.sub(r"\s+", " ", str(phrase).strip().strip('"').strip("'"))
            if len(p) < 2:
                return
            add_term(f'"{p}"')

        for site in self._as_list(intent.get("sites")):
            domain = self._normalize_site(site)
            if domain:
                add_term(f"site:{domain}")

        for phrase in self._as_list(intent.get("phrases")):
            add_phrase(phrase)

        for key in ("entities", "must_terms", "keywords", "constraints"):
            for raw in self._as_list(intent.get(key)):
                val = str(raw).strip()
                if not val:
                    continue
                if " " in val:
                    add_phrase(val)
                else:
                    add_term(val)

        for raw in self._as_list(intent.get("exclude_terms")):
            val = str(raw).strip()
            if not val:
                continue
            if " " in val:
                add_term(f'-"{val}"')
            else:
                add_term(f"-{val}")

        if not terms:
            return self._heuristic_query_rewrite(fallback_query)

        rewritten = " ".join(terms[: self._query_rewriter_max_terms]).strip()
        if len(rewritten) <= self._query_rewriter_max_chars:
            return rewritten
        clipped = rewritten[: self._query_rewriter_max_chars].rsplit(" ", 1)[0].strip()
        return clipped or self._heuristic_query_rewrite(fallback_query)

    async def _llm_query_rewrite(self, query: str) -> str:
        """Rewrite a query through an LLM into a terse BM25 query string."""
        if not self._query_rewriter_api_key:
            return self._heuristic_query_rewrite(query)
        try:
            from openai import AsyncOpenAI
        except Exception:
            return self._heuristic_query_rewrite(query)

        client_kwargs: dict[str, Any] = {"api_key": self._query_rewriter_api_key}
        if self._query_rewriter_base_url:
            client_kwargs["base_url"] = self._query_rewriter_base_url
        client = AsyncOpenAI(**client_kwargs)

        prompt = (
            "Rewrite the user query into one terse BM25 search query.\n"
            "Rules:\n"
            "- Keep entities, version numbers, filenames, and quoted phrases.\n"
            "- Prefer keywords over full sentences.\n"
            "- Keep a single search intent.\n"
            f"- Maximum {self._query_rewriter_max_terms} terms and "
            f"{self._query_rewriter_max_chars} characters.\n"
            "- Return ONLY the rewritten query, no commentary.\n\n"
            f"Query: {query}"
        )

        try:
            resp = await client.chat.completions.create(
                model=self._query_rewriter_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_completion_tokens=96,
            )
            rewritten = (resp.choices[0].message.content or "").strip()
            rewritten = re.sub(r"\s+", " ", rewritten)
            if not rewritten:
                return self._heuristic_query_rewrite(query)
            if len(rewritten) > self._query_rewriter_max_chars:
                rewritten = rewritten[: self._query_rewriter_max_chars].rsplit(" ", 1)[0]
            return rewritten.strip() or self._heuristic_query_rewrite(query)
        except Exception:
            return self._heuristic_query_rewrite(query)
        finally:
            await client.close()

    async def _rewrite_query(self, query: str, intent: Any = None) -> str:
        if not self._query_rewriter_enabled:
            return query
        parsed_intent = self._parse_intent(intent)
        if parsed_intent:
            intent_query = self._intent_to_query(parsed_intent, query)
            # In llm mode, intent is still the primary contract; the LLM only
            # refines the intent-derived BM25 query.
            if self._query_rewriter_strategy == "llm":
                return await self._llm_query_rewrite(intent_query)
            return intent_query
        if self._query_rewriter_strategy == "intent":
            return self._heuristic_query_rewrite(query)
        if self._query_rewriter_strategy == "llm":
            # Fallback path: no structured intent available.
            return await self._llm_query_rewrite(query)
        return self._heuristic_query_rewrite(query)

    async def _search_tool(
        self,
        query: str,
        intent: dict[str, Any] | str | None = None,
        metadata: dict[str, Any] | None = None,
        filters: dict[str, Any] | None = None,
        filename: str | None = None,
        limit: int = 10,
        **kwargs,
    ) -> str:
        """Search using BM25.

        Args:
            query: Natural-language sub-question or search text
            intent: Optional structured search intent used to build BM25 query
            metadata: Optional metadata filters
            filters: Optional structured filter object
            filename: Optional filename filter (substring or regex)
            limit: Maximum number of results

        Returns:
            Formatted search results or error message
        """
        if not query:
            return "Error: Missing required parameter: 'query'"

        rewritten_query = await self._rewrite_query(query, intent=intent)
        request_body: dict[str, Any] = {"query": query, "limit": limit}
        if metadata:
            request_body["metadata"] = metadata
        if filters:
            request_body["filters"] = filters
        if filename:
            request_body["filename"] = filename

        url = f"{self._base_url}/api/corpora/{self._corpus_id}/search"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=request_body,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10.0),
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        return f"Error: API request failed with status {resp.status}: {error_text}"

                    data = await resp.json()

            results = data.get("results", [])
            total = data.get("total", 0)

            if not results:
                return "No results found."

            lines = []
            if self._query_rewriter_log and rewritten_query != query:
                lines.append(f"Rewritten query: {rewritten_query}")
            for i, item in enumerate(results, start=1):
                filename_val = item.get("filename", "—")
                score = item.get("score")
                score_str = f"(score: {score:.2f})" if score is not None else "(filtered)"
                content = item.get("content", "")
                metadata_val = item.get("metadata", {})

                lines.append(f"{i}. {filename_val} {score_str}")
                lines.append(f"   Content: {content}")
                if metadata_val:
                    lines.append(f"   Metadata: {metadata_val}")

            lines.append(f"\nTotal: {total} results")
            return self._truncate_tool_output("\n".join(lines))

        except aiohttp.ClientError as e:
            return f"Error: Network error: {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"

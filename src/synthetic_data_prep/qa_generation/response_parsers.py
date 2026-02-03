"""Response parsing utilities for LLM outputs in Q&A generation."""

from __future__ import annotations

import json
from typing import Any


def parse_json_from_llm_response(response_text: str) -> dict | None:
    """
    Extract and parse JSON from LLM response that may contain markdown code fences.

    Handles:
    - ```json ... ``` blocks
    - ``` ... ``` blocks
    - Raw JSON

    Args:
        response_text: Raw LLM response

    Returns:
        Parsed dict or None if parsing fails
    """
    try:
        clean_text = response_text.strip()
        if "```json" in clean_text:
            clean_text = clean_text.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_text:
            clean_text = clean_text.split("```")[1].split("```")[0].strip()

        return json.loads(clean_text)
    except (json.JSONDecodeError, IndexError):
        return None


def parse_corpus_summary_response(response: str) -> tuple[str, list[str]]:
    """
    Parse corpus summary and example queries from LLM JSON response.

    Expected JSON structure:
    {
        "thoughts": "...",
        "summary": "...",
        "example_queries": ["q1", "q2", ...]
    }

    Args:
        response: Raw LLM response containing JSON

    Returns:
        Tuple of (corpus_summary, example_queries_list)
        Returns ("", []) if parsing fails
    """
    result = parse_json_from_llm_response(response)
    if result is None:
        return "", []

    summary = result.get("summary", "")
    example_queries = result.get("example_queries", [])

    if not isinstance(example_queries, list):
        example_queries = []

    return summary, example_queries


def parse_single_hop_response(response: str) -> tuple[str, list[dict[str, Any]]]:
    """
    Parse single-hop Q&A generation response.

    Expected JSON structure:
    {
        "keywords": [...],
        "confidence": "low|mid|high",
        "qa_pairs": [{"query": "...", "answer": "..."}]
    }

    Args:
        response: Raw LLM response

    Returns:
        Tuple of (confidence, qa_pairs_list)
        Returns ("low", []) if parsing fails
    """
    result = parse_json_from_llm_response(response)
    if result is None:
        return "low", []

    confidence = result.get("confidence", "low")
    qa_pairs = result.get("qa_pairs", [])

    if not isinstance(qa_pairs, list):
        qa_pairs = []

    return confidence, qa_pairs


def parse_related_queries_response(response: str) -> tuple[str, list[str]]:
    """
    Parse related chunk query generation response.

    Expected JSON structure:
    {
        "keywords": [...],
        "confidence": "low|mid|high",
        "queries": ["q1", "q2", ...]
    }

    Args:
        response: Raw LLM response

    Returns:
        Tuple of (confidence, queries_list)
        Returns ("low", []) if parsing fails
    """
    result = parse_json_from_llm_response(response)
    if result is None:
        return "low", []

    confidence = result.get("confidence", "low")
    queries = result.get("queries", [])

    if not isinstance(queries, list):
        queries = []

    return confidence, queries


def parse_multi_hop_validation_response(response: str) -> list[dict[str, Any]]:
    """
    Parse multi-hop connection validation and Q&A response.

    Expected JSON structure:
    {
        "thoughts": "...",
        "relationship_type": "explicit_reference|abstraction_levels|comparison_chain|none",
        "direction": "A_to_B|B_to_A|bidirectional|null",
        "linking_info": {...} or null,
        "qa_pairs": [
            {
                "question": "...",
                "answer": "...",
            }
        ] or null
    }

    Args:
        response: Raw LLM response

    Returns:
        List of QA pairs, or empty list if parsing fails or no valid relationship
    """
    result = parse_json_from_llm_response(response)
    if result is None:
        return []

    qa_pairs = result.get("qa_pairs")

    # Return empty list if qa_pairs is None or not a list
    if qa_pairs is None or not isinstance(qa_pairs, list):
        return []

    return qa_pairs

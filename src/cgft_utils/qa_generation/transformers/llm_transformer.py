"""LLM-based style rewrite transformer."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import OpenAI

from cgft_utils.qa_generation.cgft_models import CgftContext, TransformationConfig
from cgft_utils.qa_generation.generated_qa import GeneratedQA
from cgft_utils.qa_generation.style_controls import (
    DEFAULT_QUERY_STYLE_DISTRIBUTION,
    QUERY_STYLE_KEYS,
    get_style_distribution,
    normalize_style_distribution,
)
from cgft_utils.qa_generation.transformers.base import BaseQuestionTransformer

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

_DEFAULT_SYSTEM_PROMPT = (
    "You rewrite user questions into a target style while preserving intent and answerability.\n"
    "You may also apply small, style-appropriate mutations to simulate realistic user queries.\n"
    "Do not change entities, constraints, or factual intent.\n"
    'Return JSON: {"question":"..."}'
)

_DEFAULT_USER_TEMPLATE = (
    "Target style: {target_style}\n"
    "Noise level: {noise_level}\n\n"
    "Style definitions:\n\n"
    "**keyword** (search box query)\n"
    "- 2-7 tokens, no punctuation, no question mark\n"
    "- Dense with domain terms, like a search engine query\n"
    '- Examples: "posthog feature flag python", "session replay network requests"\n'
    "- Appropriate noise: omit 1-2 filler words, use abbreviations (PH, SDK, API)\n\n"
    "**natural** (conversational question)\n"
    "- Complete sentence with question mark\n"
    "- Conversational tone, as if asking a colleague\n"
    '- Examples: "How do I enable session replay for my React app?"\n'
    "- Appropriate noise: 1 realistic typo (adjacent key swap like 'teh' for 'the')\n\n"
    "**expert** (technical/troubleshooting question)\n"
    "- Specific technical constraints or comparisons\n"
    "- Often troubleshooting, debugging, or decision-oriented\n"
    '- Examples: "Why would $feature_flag_called events not appear with local evaluation?"\n'
    "- Appropriate noise: use standard abbreviations (SDK, API, FF, PH, k8s)\n\n"
    "Noise levels:\n"
    "- **none**: No mutations, clean rewrite only\n"
    "- **light**: Apply 0-1 style-appropriate mutations\n"
    "- **moderate**: Apply 1-2 style-appropriate mutations\n\n"
    "Question: {question}\n"
    "Answer: {answer}\n\n"
    "Rewrite the question to match {target_style} style.\n"
    "If noise_level is not 'none', apply style-appropriate mutations.\n"
    'Return JSON: {{"question": "..."}}'
)


class LLMStyleTransformer(BaseQuestionTransformer):
    """Rewrites questions toward a sampled style distribution via an LLM."""

    stats_mode = "llm"

    def __init__(self, cfg: TransformationConfig, *, enable_validation: bool = True) -> None:
        super().__init__(cfg, enable_validation=enable_validation)
        self.model = str(cfg.model).strip()
        self.client: OpenAI | None = None
        if self.model:
            self.client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)

    def _transform_question(
        self,
        item: GeneratedQA,
        *,
        context: CgftContext,
        original_question: str,
    ) -> tuple[str, dict[str, Any]]:
        if self.client is None or not self.model:
            return original_question, {"skipped": "llm_unconfigured"}

        qa_type = str(item.qa.get("qa_type", "")).strip()
        target_style = self._sample_style(context, qa_type=qa_type)
        noise_level = str(getattr(self.cfg, "noise_level", "light") or "light").strip()
        prompt_vars = {
            "target_style": target_style,
            "noise_level": noise_level,
            "question": original_question,
            "answer": str(item.qa.get("answer", "")).strip(),
        }
        system_prompt = str(self.cfg.system_prompt or "").strip() or _DEFAULT_SYSTEM_PROMPT
        user_template = str(self.cfg.user_template or "").strip() or _DEFAULT_USER_TEMPLATE
        try:
            user_prompt = user_template.format(**prompt_vars)
        except KeyError:
            user_prompt = _DEFAULT_USER_TEMPLATE.format(**prompt_vars)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
        except Exception:
            logger.exception("LLM style rewrite failed.")
            return original_question, {"target_style": target_style, "error": "rewrite_call_failed"}

        rewritten = self._extract_question(response.choices[0].message.content or "")
        if not rewritten:
            return original_question, {
                "target_style": target_style,
                "error": "rewrite_parse_failed",
            }

        return rewritten, {"target_style": target_style}

    def _sample_style(self, context: CgftContext, *, qa_type: str = "") -> str:
        if self.cfg.style_distribution:
            base = self.cfg.style_distribution
        elif qa_type:
            base = get_style_distribution(qa_type)
        else:
            base = DEFAULT_QUERY_STYLE_DISTRIBUTION
        distribution = normalize_style_distribution(base)
        draw = context.rng.random()
        cumulative = 0.0
        for style in QUERY_STYLE_KEYS:
            cumulative += distribution.get(style, 0.0)
            if draw <= cumulative:
                return style
        return QUERY_STYLE_KEYS[-1]

    @staticmethod
    def _extract_question(raw_text: str) -> str:
        """Extract question from JSON response."""
        text = str(raw_text or "").strip()
        if not text:
            return ""

        for candidate in (text, *_JSON_BLOCK_RE.findall(text)):
            try:
                payload = json.loads(candidate)
            except Exception:
                continue
            if isinstance(payload, dict):
                question = str(payload.get("question", "")).strip()
                if question:
                    return question
        return ""

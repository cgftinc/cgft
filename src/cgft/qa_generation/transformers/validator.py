"""Lightweight post-transform semantic validator."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import OpenAI

from cgft.qa_generation.cgft_models import CgftContext, TransformationConfig
from cgft.qa_generation.generated_qa import GeneratedQA

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


class LLMSanityValidator:
    """Validates transformed QA pairs remain coherent and semantically aligned."""

    SYSTEM_PROMPT = (
        "You validate question and answer pairs.\n"
        "Return JSON only: "
        '{"valid": true|false, "reason": "<short reason>"}.\n'
        "Mark valid=false when the question is incoherent, ambiguous, or no longer answered by the answer."
    )

    USER_TEMPLATE = "Question: {question}\nAnswer: {answer}"

    def __init__(self, cfg: TransformationConfig) -> None:
        self.cfg = cfg
        self.model = str(cfg.validation_model or cfg.model).strip()
        self.enabled = bool(cfg.validation_enabled and self.model)
        self.client: OpenAI | None = None
        if self.enabled:
            self.client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)

    @staticmethod
    def _parse_payload(raw_text: str) -> dict[str, Any]:
        text = str(raw_text or "").strip()
        if not text:
            return {}
        for candidate in (text, *_JSON_BLOCK_RE.findall(text)):
            try:
                payload = json.loads(candidate)
            except Exception:
                continue
            if isinstance(payload, dict):
                return payload
        return {}

    def is_valid(self, item: GeneratedQA, context: CgftContext) -> tuple[bool, str]:
        del context
        if not self.enabled or self.client is None:
            return True, "validation_disabled"

        question = str(item.qa.get("question", "")).strip()
        answer = str(item.qa.get("answer", "")).strip()
        if not question or not answer:
            return False, "missing_question_or_answer"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": self.USER_TEMPLATE.format(question=question, answer=answer),
                    },
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
                timeout=60.0,
            )
        except Exception as exc:
            logger.exception("LLMSanityValidator call failed.")
            return False, f"validator_error:{exc.__class__.__name__}"

        payload = self._parse_payload(response.choices[0].message.content or "")
        is_valid = bool(payload.get("valid", False))
        reason = str(payload.get("reason", "")).strip() or (
            "valid" if is_valid else "invalid"
        )
        return is_valid, reason

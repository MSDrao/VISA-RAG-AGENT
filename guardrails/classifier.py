"""Rule-based pre-retrieval guardrails."""

import logging
from generation.response_model import VisaAnswer
from generation.prompt_builder import STANDARD_DISCLAIMER
from retrieval.query_processor import (
    ProcessedQuery,
    INTENT_OUT_OF_SCOPE,
    INTENT_CASE_SPECIFIC,
)

logger = logging.getLogger(__name__)

OFF_TOPIC_KEYWORDS = [
    "weather", "restaurant", "recipe", "sport", "movie", "music",
    "stock", "crypto", "bitcoin", "dating", "travel tips",
    "hotel", "flight", "airbnb",
]

DOCUMENT_ASSIST_KEYWORDS = [
    "fill out", "fill in", "complete my", "write my", "draft my",
    "review my", "check my petition", "prepare my application",
    "edit my", "fix my form",
]


class GuardrailClassifier:
    """Decide whether a request should continue through the full pipeline."""

    def check(self, processed: ProcessedQuery, raw_question: str) -> VisaAnswer | None:
        """Return a blocking response or None to continue."""
        q = raw_question.lower()

        if self._is_off_topic(q):
            logger.info("Off-topic query blocked")
            return self._off_topic_response()

        if self._is_document_assist(q):
            logger.info("Document assist request blocked")
            return self._document_assist_response()

        if processed.is_out_of_scope:
            logger.info("Out-of-scope topic: %s", processed.out_of_scope_reason)
            return self._out_of_scope_response(processed.out_of_scope_reason)

        return None

    def _is_off_topic(self, q: str) -> bool:
        return any(kw in q for kw in OFF_TOPIC_KEYWORDS)

    def _is_document_assist(self, q: str) -> bool:
        return any(kw in q for kw in DOCUMENT_ASSIST_KEYWORDS)

    def _off_topic_response(self) -> VisaAnswer:
        return VisaAnswer(
            answer=(
                "I'm designed specifically to answer U.S. immigration and visa questions. "
                "That question falls outside my scope, but a general-purpose assistant would "
                "be happy to help. If you have any immigration-related questions — "
                "about visas, work authorization, travel, or status — I'm here for that."
            ),
            citations=[],
            confidence="high",
            visa_types_referenced=[],
            requires_attorney=False,
            is_out_of_scope=True,
            out_of_scope_reason="Non-immigration topic",
            disclaimer=STANDARD_DISCLAIMER,
        )

    def _document_assist_response(self) -> VisaAnswer:
        return VisaAnswer(
            answer=(
                "I can explain how immigration forms and processes work, but I'm not able to "
                "review, complete, or draft immigration documents on your behalf. "
                "That crosses into legal representation, which requires a licensed immigration "
                "attorney or an accredited representative.\n\n"
                "What I *can* do is explain what a form requires, what each section means, "
                "or what supporting documents are typically needed. Would that help?"
            ),
            citations=[],
            confidence="high",
            visa_types_referenced=[],
            requires_attorney=True,
            is_out_of_scope=True,
            out_of_scope_reason="Document completion request",
            disclaimer=STANDARD_DISCLAIMER,
        )

    def _out_of_scope_response(self, topic: str | None) -> VisaAnswer:
        topic_str = topic or "this topic"
        return VisaAnswer(
            answer=(
                f"Questions about {topic_str} involve highly individualized legal situations "
                f"where the stakes are significant and the facts matter enormously. "
                f"This is outside what I'm designed to help with.\n\n"
                f"For {topic_str} matters, I strongly recommend:\n"
                f"- Contacting a licensed immigration attorney\n"
                f"- Reaching out to a nonprofit legal aid organization\n"
                f"- CLINIC (Catholic Legal Immigration Network): **clinic.org**\n"
                f"- ILRC (Immigrant Legal Resource Center): **ilrc.org**\n"
                f"- Your local law school's immigration clinic"
            ),
            citations=[],
            confidence="high",
            visa_types_referenced=[],
            requires_attorney=True,
            is_out_of_scope=True,
            out_of_scope_reason=topic_str,
            disclaimer=STANDARD_DISCLAIMER,
        )

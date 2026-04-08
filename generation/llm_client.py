"""LLM client for answer generation."""

import json
import logging
import os
import time

import openai
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from generation.prompt_builder import build_prompt, STANDARD_DISCLAIMER
from generation.response_model import VisaAnswer, Citation, QueryResponse
from retrieval.retriever import RetrievedChunk
from retrieval.query_processor import ProcessedQuery

logger = logging.getLogger(__name__)

MODEL       = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1")
MAX_TOKENS  = 1200
TEMPERATURE = 0.0


def _make_langfuse():
    """Try to initialise Langfuse; return None if unavailable or misconfigured."""
    try:
        from langfuse import Langfuse
        lf = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com"),
        )
        # Probe for the old v2 API vs new v3/v4 API
        if not callable(getattr(lf, "trace", None)):
            logger.info("Langfuse client loaded but trace() not available (v3/v4 SDK). "
                        "Observability disabled — core functionality unaffected.")
            return None
        return lf
    except Exception as e:
        logger.warning(f"Langfuse init failed ({e}). Observability disabled.")
        return None


class _NoopTrace:
    """Drop-in replacement when Langfuse is unavailable."""
    def generation(self, **_): return self
    def end(self, **_): pass
    def update(self, **_): pass


class LLMClient:
    """Generate structured answers from retrieved context."""

    def __init__(self):
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._langfuse = _make_langfuse()
        logger.info("LLMClient ready.")

    def generate(
        self,
        processed_query: ProcessedQuery,
        chunks: list[RetrievedChunk],
        conversation_history: list[dict] | None = None,
        session_id: str | None = None,
    ) -> tuple[VisaAnswer, int]:
        """Generate an answer from retrieved chunks. Returns (VisaAnswer, tokens_used)."""
        start = time.time()

        system, messages, citations_meta = build_prompt(
            processed_query, chunks, conversation_history
        )

        trace = self._start_trace(session_id, processed_query, len(chunks))

        raw_text, tokens_used = self._call_model(system, messages, trace)
        answer, parse_failed = self._parse_response(raw_text, citations_meta, chunks)

        if parse_failed:
            logger.info("Retrying LLM call due to likely JSON parse failure...")
            raw_text2, tokens_used2 = self._call_model(system, messages, trace)
            answer2, parse_failed2 = self._parse_response(raw_text2, citations_meta, chunks)
            tokens_used += tokens_used2
            if not parse_failed2:
                answer = answer2

        elapsed_ms = int((time.time() - start) * 1000)
        self._end_trace(trace, answer, tokens_used, elapsed_ms)

        return answer, tokens_used

    def _start_trace(self, session_id, pq, n_chunks):
        if self._langfuse is None:
            return _NoopTrace()
        try:
            return self._langfuse.trace(
                name="visa-rag-query",
                session_id=session_id,
                input={
                    "question": pq.original,
                    "intent": pq.intent,
                    "visa_types": pq.detected_visa_types,
                    "chunks_retrieved": n_chunks,
                },
            )
        except Exception as e:
            logger.debug(f"Langfuse trace skipped: {e}")
            return _NoopTrace()

    def _end_trace(self, trace, answer, tokens_used, elapsed_ms):
        try:
            trace.update(
                output={
                    "confidence": answer.confidence,
                    "requires_attorney": answer.requires_attorney,
                    "citations_count": len(answer.citations),
                    "tokens_used": tokens_used,
                    "elapsed_ms": elapsed_ms,
                }
            )
            if self._langfuse:
                self._langfuse.flush()
        except Exception as e:
            logger.debug(f"Langfuse end_trace skipped: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.InternalServerError,
            openai.APIError,
        )),
    )
    def _call_model(
        self,
        system: str,
        messages: list[dict],
        trace,
    ) -> tuple[str, int]:
        """Call the configured chat model and return raw text plus token usage."""
        generation = None
        try:
            generation = trace.generation(
                name="answer-generation",
                model=MODEL,
                input=messages,
            )
        except Exception:
            pass

        try:
            oai_messages = [{"role": "system", "content": system}] + messages
            response = self.openai.chat.completions.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                messages=oai_messages,
                response_format={"type": "json_object"},
            )
            raw_text    = response.choices[0].message.content
            tokens_used = response.usage.prompt_tokens + response.usage.completion_tokens

            try:
                if generation:
                    generation.end(
                        output=raw_text,
                        usage={"input": response.usage.input_tokens,
                               "output": response.usage.output_tokens},
                    )
            except Exception:
                pass

            return raw_text, tokens_used

        except openai.RateLimitError:
            logger.warning("OpenAI rate limit hit — retrying...")
            try:
                if generation: generation.end(level="WARNING", status_message="rate_limit")
            except Exception:
                pass
            raise
        except openai.OpenAIError as e:
            logger.error(f"OpenAI API error: {e}")
            try:
                if generation: generation.end(level="ERROR", status_message=str(e))
            except Exception:
                pass
            raise

    def _parse_response(
        self,
        raw_text: str,
        citations_meta: list[dict],
        chunks: list[RetrievedChunk],
    ) -> tuple[VisaAnswer, bool]:
        """Returns (VisaAnswer, parse_failed). parse_failed=True only on JSON decode errors."""
        try:
            text = raw_text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
                if text.endswith("```"):
                    text = text[:-3].strip()

            data = json.loads(text)

            used_indices = data.get("citations_used", list(range(1, len(citations_meta) + 1)))
            citations = []
            for idx in used_indices:
                if 1 <= idx <= len(citations_meta):
                    meta  = citations_meta[idx - 1]
                    chunk = chunks[idx - 1]
                    citations.append(Citation(
                        source_id=meta["source_id"],
                        source_name=meta["source_name"],
                        source_url=meta["source_url"],
                        section_title=meta.get("section_title") or None,
                        tier=meta["tier"],
                        tier_label=meta["tier_label"],
                        last_updated_on_source=meta.get("last_updated_on_source") or None,
                        crawled_at=meta["crawled_at"],
                        is_stale=chunk.is_stale,
                    ))

            answer = VisaAnswer(
                answer=data.get("answer", ""),
                citations=citations,
                confidence=data.get("confidence", "low"),
                visa_types_referenced=data.get("visa_types_referenced", []),
                requires_attorney=data.get("requires_attorney", False),
                disclaimer=STANDARD_DISCLAIMER,
            )
            answer.freshness_warning = answer.build_freshness_warning()
            return answer, False

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.error(f"Failed to parse LLM response: {e}\nRaw: {raw_text[:300]}")
            return self._fallback_answer(), True

    def _fallback_answer(self) -> VisaAnswer:
        return VisaAnswer(
            answer=(
                "I encountered an issue generating a structured response. "
                "Please rephrase your question and try again. "
                "For urgent immigration questions, consult uscis.gov directly."
            ),
            citations=[],
            confidence="insufficient",
            visa_types_referenced=[],
            requires_attorney=False,
            disclaimer=STANDARD_DISCLAIMER,
        )

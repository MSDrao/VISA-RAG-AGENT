"""Primary query endpoint for the application."""

import logging
import time
import uuid
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

from api.middleware.rate_limit import (
    limiter, QUERY_LIMIT, QUERY_LIMIT_HOUR,
    MAX_TOKENS_PER_SESSION, MAX_TURNS_PER_SESSION,
)
from generation.response_model import QueryRequest, QueryResponse, VisaAnswer
from generation.prompt_builder import STANDARD_DISCLAIMER
from guardrails.freshness import check_and_flag
from services.live_official_data import try_live_operational_answer

logger = logging.getLogger(__name__)
router = APIRouter()

_sessions: dict[str, dict] = {}


def _get_or_create_session(session_id: str | None) -> tuple[str, dict]:
    if not session_id or session_id not in _sessions:
        sid = session_id or str(uuid.uuid4())
        _sessions[sid] = {"tokens": 0, "turns": 0, "history": []}
        return sid, _sessions[sid]
    return session_id, _sessions[session_id]


def _budget_exceeded(session: dict) -> bool:
    return (
        session["tokens"] >= MAX_TOKENS_PER_SESSION or
        session["turns"] >= MAX_TURNS_PER_SESSION
    )


@router.post("/query", response_model=QueryResponse)
@limiter.limit(f"{QUERY_LIMIT}/minute")
@limiter.limit(f"{QUERY_LIMIT_HOUR}/hour")
async def query(request: Request, body: QueryRequest):
    start = time.time()

    qp        = request.app.state.query_processor
    classifier= request.app.state.classifier
    retriever = request.app.state.retriever
    reranker  = request.app.state.reranker
    llm       = request.app.state.llm_client
    safety    = request.app.state.safety

    session_id, session = _get_or_create_session(body.session_id)

    if _budget_exceeded(session):
        return QueryResponse(
            question=body.question,
            answer_data=VisaAnswer(
                answer=(
                    "You've reached the session limit. Please start a new session "
                    "to continue asking questions."
                ),
                citations=[],
                confidence="insufficient",
                visa_types_referenced=[],
                requires_attorney=False,
                disclaimer=STANDARD_DISCLAIMER,
            ),
            session_id=session_id,
            processing_time_ms=0,
            retrieved_chunk_count=0,
        )

    processed = qp.process(
        question=body.question,
        conversation_history=session["history"],
        visa_type_filter=body.visa_type_filter,
        official_sources_only=body.official_sources_only,
    )

    blocked = classifier.check(processed, body.question)
    if blocked:
        elapsed = int((time.time() - start) * 1000)
        session["turns"] += 1
        return QueryResponse(
            question=body.question,
            answer_data=blocked,
            session_id=session_id,
            processing_time_ms=elapsed,
            retrieved_chunk_count=0,
        )

    live_answer = try_live_operational_answer(body.question)
    if live_answer:
        elapsed = int((time.time() - start) * 1000)
        session["turns"] += 1
        session["history"].append({"role": "user", "content": body.question})
        session["history"].append({"role": "assistant", "content": live_answer.answer})
        session["history"] = session["history"][-10:]
        logger.info("Answered via live operational-data path in %sms", elapsed)
        return QueryResponse(
            question=body.question,
            answer_data=live_answer,
            session_id=session_id,
            processing_time_ms=elapsed,
            retrieved_chunk_count=0,
        )

    chunks = retriever.retrieve(
        query=processed.original,
        expanded_query=processed.expanded,
        metadata_filters=processed.metadata_filters if processed.metadata_filters else None,
        query_agencies=processed.detected_agencies,
        query_forms=processed.detected_forms,
        query_process_terms=processed.detected_process_terms,
    )

    chunks = check_and_flag(chunks)

    if chunks:
        logger.info(
            "Pre-rerank top candidates: %s",
            " | ".join(
                f"{chunk.source_id}:{chunk.section_title or 'n/a'} [{chunk.retrieval_method}] {chunk.score:.3f}"
                for chunk in chunks[:5]
            ),
        )
        chunks = reranker.rerank(query=processed.original, chunks=chunks)
        logger.info(
            "Post-rerank top candidates: %s",
            " | ".join(
                f"{chunk.source_id}:{chunk.section_title or 'n/a'} score={chunk.score:.3f}"
                for chunk in chunks[:5]
            ),
        )

    if not chunks:
        logger.warning("No chunks retrieved for request")
        elapsed = int((time.time() - start) * 1000)
        session["turns"] += 1
        return QueryResponse(
            question=body.question,
            answer_data=VisaAnswer(
                answer=(
                    "I don't have enough information in my knowledge base to answer "
                    "this question confidently. Rather than guess, I'd recommend "
                    "checking directly at uscis.gov or travel.state.gov for the most "
                    "accurate and current information."
                ),
                citations=[],
                confidence="insufficient",
                visa_types_referenced=processed.detected_visa_types,
                requires_attorney=False,
                disclaimer=STANDARD_DISCLAIMER,
            ),
            session_id=session_id,
            processing_time_ms=elapsed,
            retrieved_chunk_count=0,
        )

    answer, tokens_used = llm.generate(
        processed_query=processed,
        chunks=chunks,
        conversation_history=session["history"],
        session_id=session_id,
    )

    answer = safety.validate_and_fix(answer, body.question)

    session["tokens"] += tokens_used
    session["turns"]  += 1
    session["history"].append({"role": "user",      "content": body.question})
    session["history"].append({"role": "assistant",  "content": answer.answer})
    session["history"] = session["history"][-10:]

    elapsed = int((time.time() - start) * 1000)
    logger.info(
        f"Query answered in {elapsed}ms | "
        f"confidence={answer.confidence} | "
        f"chunks={len(chunks)} | tokens={tokens_used}"
    )

    return QueryResponse(
        question=body.question,
        answer_data=answer,
        session_id=session_id,
        processing_time_ms=elapsed,
        retrieved_chunk_count=len(chunks),
    )


@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation history and token budget for a session."""
    if session_id in _sessions:
        del _sessions[session_id]
        return {"status": "cleared", "session_id": session_id}
    return {"status": "not_found", "session_id": session_id}

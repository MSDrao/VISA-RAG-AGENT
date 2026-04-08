"""Feedback API route."""

import hashlib
import logging
import os
import sqlite3

from fastapi import APIRouter, Request, HTTPException

from api.middleware.rate_limit import limiter
from generation.response_model import FeedbackRequest

logger = logging.getLogger(__name__)
router = APIRouter()

DB_PATH = os.getenv("FEEDBACK_DB_PATH", "./data/feedback.db")


def init_feedback_db():
    """Create feedback table if it doesn't exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT,
            question    TEXT NOT NULL,
            answer      TEXT NOT NULL,
            rating      TEXT NOT NULL,
            comment     TEXT,
            submitted_at TEXT NOT NULL
        )
    """)
    con.commit()
    con.close()


def _hash_session_id(session_id: str | None) -> str | None:
    if not session_id:
        return None
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()


@router.post("/feedback", status_code=201)
@limiter.limit("5/minute")
async def submit_feedback(
    request: Request,
    body: FeedbackRequest,
):
    """Store user feedback on an answer."""
    try:
        con = sqlite3.connect(DB_PATH)
        con.execute(
            """INSERT INTO feedback
               (session_id, question, answer, rating, comment, submitted_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                _hash_session_id(body.session_id),
                body.question,
                body.answer,
                body.rating,
                body.comment,
                body.submitted_at,
            ),
        )
        con.commit()
        con.close()
        logger.info("Feedback stored with rating=%s", body.rating)
        return {"status": "received", "rating": body.rating}
    except Exception as e:
        logger.error(f"Feedback storage failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to store feedback.")

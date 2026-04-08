"""
Freshness — guardrails/freshness.py
Checks retrieved chunk ages against staleness thresholds.
Visa Bulletin chunks have a shorter expiry (45 days) because
priority date data is only valid for the current month.
"""

import logging
import os
from datetime import datetime, timezone, timedelta
from retrieval.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

STALENESS_THRESHOLD_DAYS  = int(os.getenv("STALENESS_THRESHOLD_DAYS", 90))
VISA_BULLETIN_EXPIRY_DAYS = int(os.getenv("VISA_BULLETIN_EXPIRY_DAYS", 45))


def check_and_flag(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """
    Mark chunks as stale if their crawled_at date exceeds the threshold.
    Visa Bulletin chunks use a shorter expiry.
    Mutates chunks in-place and returns them.
    """
    now = datetime.now(timezone.utc)

    for chunk in chunks:
        try:
            crawled = datetime.fromisoformat(chunk.crawled_at.replace("Z", "+00:00"))
            age_days = (now - crawled).days

            is_visa_bulletin = (
                "visa_bulletin" in chunk.visa_tags or
                "visa bulletin" in chunk.source_name.lower() or
                "visa-bulletin" in chunk.source_url.lower()
            )

            threshold = VISA_BULLETIN_EXPIRY_DAYS if is_visa_bulletin else STALENESS_THRESHOLD_DAYS
            chunk.is_stale = age_days > threshold

            if chunk.is_stale:
                logger.warning(
                    f"Stale chunk ({age_days}d old, threshold {threshold}d): "
                    f"{chunk.source_url[:60]}"
                )
        except Exception as e:
            logger.debug(f"Could not parse crawled_at for chunk {chunk.chunk_id}: {e}")

    return chunks


def build_freshness_warning(chunks: list[RetrievedChunk]) -> str | None:
    """Return a user-visible warning if any stale chunks were used."""
    stale = [c for c in chunks if c.is_stale]
    if not stale:
        return None

    is_bulletin = any(
        "visa_bulletin" in c.visa_tags or "visa bulletin" in c.source_name.lower()
        for c in stale
    )

    if is_bulletin:
        return (
            "⚠️ **Priority date information from the Visa Bulletin may be outdated.** "
            "The Visa Bulletin is published monthly and is only valid for the current month. "
            "Please check the current bulletin directly at: "
            "https://travel.state.gov/content/travel/en/us-visas/visa-information-resources/visa-bulletin.html"
        )

    urls = list({c.source_url for c in stale})[:3]
    url_list = "\n".join(f"- {u}" for u in urls)
    return (
        f"⚠️ **Some sources used in this answer have not been refreshed recently.** "
        f"Immigration policies change frequently. Please verify current information at:\n{url_list}"
    )

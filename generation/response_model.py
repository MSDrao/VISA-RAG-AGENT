"""Pydantic models used across the API and generation layers."""

from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime, timezone

class Citation(BaseModel):
    source_id: str = Field(description="Unique source ID from sources.yaml")
    source_name: str = Field(description="Human-readable source name")
    source_url: str = Field(description="Direct URL to the source page or document")
    section_title: str | None = Field(
        default=None,
        description="Title of the specific section within the document"
    )
    tier: int = Field(description="1 = Official Government, 2 = Institutional")
    tier_label: str = Field(description="Human-readable tier label for display")
    last_updated_on_source: str | None = Field(
        default=None,
        description="Date the source page was last updated (from the page itself)"
    )
    crawled_at: str = Field(
        description="ISO 8601 datetime when this content was ingested into the system"
    )
    is_stale: bool = Field(
        default=False,
        description="True if crawled_at exceeds STALENESS_THRESHOLD_DAYS"
    )

    def staleness_warning(self) -> str | None:
        """Return a warning string if this citation is stale."""
        if self.is_stale:
            return (
                f"⚠️ This source was last ingested on {self.crawled_at[:10]}. "
                f"Immigration policy may have changed. Verify directly at {self.source_url}"
            )
        return None

STANDARD_DISCLAIMER = (
    "This information is provided for general educational purposes only, based on official "
    "U.S. government sources and accredited institutions as of the dates cited. "
    "It is **not legal advice** and does not create an attorney-client relationship. "
    "Immigration law is complex, fact-specific, and subject to change. "
    "For guidance specific to your situation, consult a licensed immigration attorney "
    "or an accredited representative (recognized by the Board of Immigration Appeals)."
)

class VisaAnswer(BaseModel):
    answer: str = Field(
        description="Answer text with inline citation markers like [1], [2]"
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Ordered list of sources used. Index matches inline markers."
    )
    confidence: Literal["high", "medium", "low", "insufficient"] = Field(
        description=(
            "high: strong retrieval match from Tier 1 source. "
            "medium: relevant but partial match or Tier 2 source. "
            "low: weak retrieval, answer may be incomplete. "
            "insufficient: no relevant context found — answer refused."
        )
    )
    visa_types_referenced: list[str] = Field(
        default_factory=list,
        description="Visa categories this answer pertains to e.g. ['F-1', 'OPT']"
    )
    requires_attorney: bool = Field(
        default=False,
        description=(
            "True if the question is case-specific or involves legal judgment "
            "that requires a licensed immigration attorney."
        )
    )
    is_out_of_scope: bool = Field(
        default=False,
        description=(
            "True if the question falls outside this system's defined scope "
            "(e.g. DACA, asylum, removal defense, criminal immigration consequences)."
        )
    )
    out_of_scope_reason: str | None = Field(
        default=None,
        description="Explanation of why the question is out of scope, shown to the user."
    )
    disclaimer: str = Field(default=STANDARD_DISCLAIMER)
    freshness_warning: str | None = Field(
        default=None,
        description="Populated if any citation is stale. Displayed prominently in the UI."
    )

    def has_stale_citations(self) -> bool:
        return any(c.is_stale for c in self.citations)

    def build_freshness_warning(self) -> str | None:
        stale = [c for c in self.citations if c.is_stale]
        if not stale:
            return None
        urls = ", ".join(c.source_url for c in stale)
        return (
            f"⚠️ One or more sources used in this answer have not been refreshed recently. "
            f"Please verify current policy at: {urls}"
        )

class QueryRequest(BaseModel):
    question: str = Field(
        min_length=5,
        max_length=2000,
        description="The user's immigration question"
    )
    visa_type_filter: list[str] | None = Field(
        default=None,
        description=(
            "Optional: restrict retrieval to specific visa types. "
            "e.g. ['F-1', 'OPT'] to only search F-1 and OPT related chunks."
        )
    )
    official_sources_only: bool = Field(
        default=False,
        description="If True, only Tier 1 government sources are used in retrieval."
    )
    session_id: str | None = Field(
        default=None,
        description="Session ID for conversation memory. None = single-turn query."
    )

class QueryResponse(BaseModel):
    question: str
    answer_data: VisaAnswer
    session_id: str | None = None
    processing_time_ms: int = Field(description="Total pipeline time in milliseconds")
    retrieved_chunk_count: int = Field(
        default=0,
        description="Number of chunks retrieved before reranking"
    )

class FeedbackRequest(BaseModel):
    session_id: str | None = None
    question: str
    answer: str
    rating: Literal["helpful", "not_helpful", "incorrect", "incomplete"]
    comment: str | None = None
    submitted_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

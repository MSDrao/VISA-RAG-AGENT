"""
Chunker — ingestion/chunker.py
Converts cleaned documents into semantically coherent chunks.
Strategy: hierarchical semantic chunking with sentence-boundary splits and overlap.

Why this matters: A policy that says "X is allowed IF Y" must never be split
such that only "X is allowed" is retrieved. This chunker prevents that.
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator
from config.metadata_keywords import (
    AGENCY_KEYWORDS,
    EMPLOYMENT_KEYWORDS,
    FAMILY_KEYWORDS,
    FORM_KEYWORDS,
    PROCESS_KEYWORDS,
    TRAVEL_KEYWORDS,
    VISA_KEYWORD_GROUPS,
)

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """A single chunk ready for embedding and storage."""

    text: str
    chunk_id: str
    source_id: str
    source_name: str
    source_url: str
    parent_section: str | None = None
    chunk_index: int = 0
    tier: int = 1
    tier_label: str = "Official U.S. Government"
    document_title: str | None = None
    section_title: str | None = None
    last_updated_on_source: str | None = None
    crawled_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    visa_tags: list[str] = field(default_factory=list)
    content_type: str = "html"
    is_stale: bool = False
    word_count: int = 0
    token_estimate: int = 0

    def to_metadata(self) -> dict[str, Any]:
        """Flat dict for Chroma metadata. Chroma does not support nested types."""
        derived = self._derived_metadata()
        return {
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "source_name": self.source_name,
            "source_url": self.source_url,
            "parent_section": self.parent_section or "",
            "chunk_index": self.chunk_index,
            "tier": self.tier,
            "tier_label": self.tier_label,
            "document_title": self.document_title or "",
            "section_title": self.section_title or "",
            "last_updated_on_source": self.last_updated_on_source or "",
            "crawled_at": self.crawled_at,
            "visa_tags": ",".join(self.visa_tags),   # Chroma stores as string
            "content_type": self.content_type,
            "is_stale": self.is_stale,
            "word_count": self.word_count,
            "token_estimate": self.token_estimate,
            **derived,
        }

    def _derived_metadata(self) -> dict[str, Any]:
        text = self.text
        title = self.document_title or self.section_title or ""
        url = self.source_url.lower()
        source_id = self.source_id.lower()

        agency = ""
        if "uscis" in source_id or "uscis.gov" in url:
            agency = "USCIS"
        elif "travel.state.gov" in url or "state" in source_id:
            agency = "DOS"
        elif "cbp" in source_id or "cbp.gov" in url:
            agency = "CBP"
        elif "dol" in source_id or "flag.dol.gov" in url:
            agency = "DOL"
        elif "study_in_the_states" in source_id or "ice.gov" in url or "studyinthestates" in url:
            agency = "SEVP"

        if "policy-manual" in url:
            doc_type = "policy_manual"
        elif "visa-bulletin" in url:
            doc_type = "visa_bulletin"
        elif "/forms/" in url or re.search(r"\bform\b", title, flags=re.IGNORECASE):
            doc_type = "form_instructions"
        elif re.search(r"\bfaq\b", title, flags=re.IGNORECASE):
            doc_type = "faq"
        else:
            doc_type = self.content_type

        form_numbers = sorted(set(
            form.upper().replace(" ", "")
            for form in re.findall(r"\b(?:I|DS)[-\s]?\d{2,4}\b", text, flags=re.IGNORECASE)
        ))
        reg_citations = sorted(set(
            match.strip()
            for match in re.findall(r"\b\d+\s+C\.F\.R\.\s+[\d.()a-zA-Z-]+", text)
        ))
        lowered_text = f"{title} {text}".lower()

        process_terms = sorted(set(
            term for term in PROCESS_KEYWORDS if term.lower() in lowered_text
        ))
        agency_terms = sorted(set(
            agency
            for agency, keywords in AGENCY_KEYWORDS.items()
            if any(keyword.lower() in lowered_text for keyword in keywords)
        ))
        visa_keyword_tags = sorted(set(
            group
            for group, keywords in VISA_KEYWORD_GROUPS.items()
            if any(keyword.lower() in lowered_text for keyword in keywords)
        ))
        employment_terms = sorted(set(
            term for term in EMPLOYMENT_KEYWORDS if term.lower() in lowered_text
        ))
        family_terms = sorted(set(
            term for term in FAMILY_KEYWORDS if term.lower() in lowered_text
        ))
        travel_terms = sorted(set(
            term for term in TRAVEL_KEYWORDS if term.lower() in lowered_text
        ))
        extra_forms = sorted(set(
            form for form in FORM_KEYWORDS if form.lower() in lowered_text
        ))
        form_numbers = sorted(set(form_numbers).union(extra_forms))

        keyword_tags = sorted(set(
            visa_keyword_tags + process_terms + agency_terms + employment_terms + family_terms + travel_terms
        ))

        return {
            "agency": agency,
            "doc_type": doc_type,
            "form_numbers": ",".join(form_numbers),
            "reg_citations": ",".join(reg_citations),
            "process_terms": ",".join(process_terms),
            "keyword_tags": ",".join(keyword_tags),
        }


class Chunker:
    """
    Hierarchical semantic chunker for immigration policy documents.

    Parameters tuned for government legal/policy text:
    - Larger chunks than typical RAG (policy context is dense)
    - Meaningful overlap (prevent IF/THEN clause splits)
    - Sentence-boundary aware (never cut mid-sentence)
    - Abbreviation-aware (U.S., 8 C.F.R., e.g. are not sentence boundaries)
    """

    TARGET_CHUNK_WORDS = 400
    MAX_CHUNK_WORDS    = 700
    MIN_CHUNK_WORDS    = 80
    OVERLAP_WORDS      = 60

    # Legal abbreviations that look like sentence endings but are not
    ABBREVIATIONS = [
        (r"U\.S\.",      "US_ABBR"),
        (r"8 C\.F\.R\.", "CFR_ABBR"),
        (r"e\.g\.",      "EG_ABBR"),
        (r"i\.e\.",      "IE_ABBR"),
        (r"et al\.",     "ETAL_ABBR"),
        (r"vs\.",        "VS_ABBR"),
        (r"Sec\.",       "SEC_ABBR"),
        (r"approx\.",    "APPROX_ABBR"),
        (r"No\.",        "NO_ABBR"),
        (r"Vol\.",       "VOL_ABBR"),
    ]

    def chunk_html_page(
        self,
        text: str,
        sections: list[dict],
        source_id: str,
        source_name: str,
        source_url: str,
        tier: int,
        tier_label: str,
        visa_tags: list[str],
        document_title: str | None = None,
        last_updated_on_source: str | None = None,
        crawled_at: str | None = None,
    ) -> list[DocumentChunk]:

        crawled_at = crawled_at or datetime.now(timezone.utc).isoformat()
        doc_hash = self._short_hash(source_url)
        chunks = []

        if sections:
            for section in sections:
                heading = section.get("heading", "")
                content = section.get("content", "")
                if not content.strip():
                    continue

                prefix = f"{heading}\n\n" if heading else ""
                for chunk_text in self._split_text(content, context_prefix=prefix):
                    if len(chunk_text.split()) < self.MIN_CHUNK_WORDS:
                        continue
                    idx = len(chunks)
                    chunks.append(self._make_chunk(
                        text=chunk_text,
                        chunk_index=idx,
                        doc_hash=doc_hash,
                        source_id=source_id, source_name=source_name,
                        source_url=source_url, tier=tier, tier_label=tier_label,
                        visa_tags=visa_tags, document_title=document_title,
                        section_title=heading or None,
                        parent_section=heading or None,
                        last_updated_on_source=last_updated_on_source,
                        crawled_at=crawled_at, content_type="html",
                    ))
        else:
            for i, chunk_text in enumerate(self._split_text(text)):
                if len(chunk_text.split()) < self.MIN_CHUNK_WORDS:
                    continue
                chunks.append(self._make_chunk(
                    text=chunk_text, chunk_index=i, doc_hash=doc_hash,
                    source_id=source_id, source_name=source_name,
                    source_url=source_url, tier=tier, tier_label=tier_label,
                    visa_tags=visa_tags, document_title=document_title,
                    last_updated_on_source=last_updated_on_source,
                    crawled_at=crawled_at, content_type="html",
                ))

        logger.info(f"Chunked {source_url} → {len(chunks)} chunks")
        return chunks

    def chunk_pdf(
        self,
        pages: list[dict],
        source_id: str,
        source_name: str,
        source_url: str,
        tier: int,
        tier_label: str,
        visa_tags: list[str],
        document_title: str | None = None,
        last_updated_on_source: str | None = None,
        crawled_at: str | None = None,
        content_type: str = "pdf",
    ) -> list[DocumentChunk]:

        crawled_at = crawled_at or datetime.now(timezone.utc).isoformat()
        doc_hash = self._short_hash(source_url)
        full_text = "\n\n".join(
            p["text"] for p in pages if p.get("text", "").strip()
        )
        chunks = []

        for i, chunk_text in enumerate(self._split_text(full_text)):
            if len(chunk_text.split()) < self.MIN_CHUNK_WORDS:
                continue
            chunks.append(self._make_chunk(
                text=chunk_text, chunk_index=i, doc_hash=doc_hash,
                source_id=source_id, source_name=source_name,
                source_url=source_url, tier=tier, tier_label=tier_label,
                visa_tags=visa_tags, document_title=document_title,
                last_updated_on_source=last_updated_on_source,
                crawled_at=crawled_at, content_type=content_type,
            ))

        logger.info(f"Chunked PDF {source_url} → {len(chunks)} chunks")
        return chunks

    # ------------------------------------------------------------------
    # Core splitting logic
    # ------------------------------------------------------------------

    def _split_text(
        self,
        text: str,
        context_prefix: str = "",
    ) -> list[str]:
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks: list[str] = []
        current: list[str] = []
        current_count = 0

        for sentence in sentences:
            words = sentence.split()
            count = len(words)

            # Single sentence exceeds max — hard split it
            if count > self.MAX_CHUNK_WORDS:
                if current:
                    chunks.append(self._build(current, context_prefix))
                    current, current_count = [], 0
                for piece in self._hard_split(words):
                    chunks.append(self._build(piece, context_prefix))
                continue

            # Adding this sentence would overflow — flush with overlap
            if current_count + count > self.MAX_CHUNK_WORDS and current_count >= self.MIN_CHUNK_WORDS:
                chunks.append(self._build(current, context_prefix))
                overlap = current[-self.OVERLAP_WORDS:] if len(current) > self.OVERLAP_WORDS else current[:]
                current = overlap + words
                current_count = len(current)
            else:
                current.extend(words)
                current_count += count

        if current and len(current) >= self.MIN_CHUNK_WORDS:
            chunks.append(self._build(current, context_prefix))

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Sentence splitter that handles legal abbreviations correctly."""
        text = re.sub(r"\s+", " ", text).strip()

        # Protect abbreviations
        for pattern, token in self.ABBREVIATIONS:
            text = re.sub(pattern, token, text)

        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\"\(])", text)

        # Restore abbreviations
        for pattern, token in self.ABBREVIATIONS:
            original = re.sub(r"\\", "", pattern).replace("\\.", ".")
            sentences = [s.replace(token, original) for s in sentences]

        return [s.strip() for s in sentences if s.strip()]

    def _build(self, words: list[str], prefix: str = "") -> str:
        text = " ".join(words)
        return (prefix + text).strip() if prefix else text.strip()

    def _hard_split(self, words: list[str]) -> Iterator[list[str]]:
        for i in range(0, len(words), self.MAX_CHUNK_WORDS):
            yield words[i : i + self.MAX_CHUNK_WORDS]

    def _short_hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()[:8]

    def _make_chunk(self, text: str, chunk_index: int, doc_hash: str, **kwargs) -> DocumentChunk:
        source_id = kwargs["source_id"]
        wc = len(text.split())
        return DocumentChunk(
            text=text,
            chunk_id=f"{source_id}_{doc_hash}_{chunk_index}",
            chunk_index=chunk_index,
            word_count=wc,
            token_estimate=int(wc * 1.3),
            **{k: v for k, v in kwargs.items() if k not in ("chunk_index",)},
        )

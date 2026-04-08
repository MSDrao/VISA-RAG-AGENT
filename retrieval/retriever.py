"""
Retriever — retrieval/retriever.py
Hybrid retrieval: dense vector search (Chroma) + sparse BM25.
Results fused with Reciprocal Rank Fusion (RRF).

Why hybrid:
- Dense: handles semantic questions ("can I work while pending")
- BM25: handles exact terms ("Form I-765", "8 C.F.R.", "cap-gap")
- Immigration text has BOTH — you need both retrieval modes.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

import chromadb
from chromadb.config import Settings
from openai import OpenAI
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

EMBEDDING_MODEL  = "text-embedding-3-small"
COLLECTION_NAME  = "visa_rag_chunks"
DENSE_TOP_K      = 20       # Candidates from dense search
BM25_TOP_K       = 20       # Candidates from BM25
FINAL_TOP_K      = 10       # After RRF fusion (before reranking)
RRF_K            = 60       # RRF constant (standard value)
MAX_CHUNKS_PER_URL = 2


@dataclass
class RetrievedChunk:
    text: str
    chunk_id: str
    chunk_index: int
    source_id: str
    source_name: str
    source_url: str
    document_title: str
    section_title: str
    tier: int
    tier_label: str
    visa_tags: list[str]
    crawled_at: str
    last_updated_on_source: str
    is_stale: bool
    score: float            # RRF score (higher = more relevant)
    retrieval_method: str   # "dense", "bm25", or "hybrid"
    agency: str = ""
    doc_type: str = ""
    form_numbers: list[str] = field(default_factory=list)
    reg_citations: list[str] = field(default_factory=list)
    process_terms: list[str] = field(default_factory=list)
    keyword_tags: list[str] = field(default_factory=list)

    def to_citation_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "source_name": self.source_name,
            "source_url": self.source_url,
            "section_title": self.section_title or "",
            "tier": self.tier,
            "tier_label": self.tier_label,
            "last_updated_on_source": self.last_updated_on_source,
            "crawled_at": self.crawled_at,
            "is_stale": self.is_stale,
        }


class HybridRetriever:
    """
    Retrieves relevant chunks using hybrid dense + BM25 search.

    BM25 index is built from Chroma at startup.
    For MVP scale (< 100K chunks), this fits comfortably in memory.
    """

    def __init__(self):
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
        chroma = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = chroma.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        # BM25 index — built once at startup from full Chroma corpus
        self._bm25: BM25Okapi | None = None
        self._bm25_docs: list[dict] = []    # parallel list to BM25 index
        self._build_bm25_index()

    def retrieve(
        self,
        query: str,
        expanded_query: str,
        metadata_filters: dict | None = None,
        query_agencies: list[str] | None = None,
        query_forms: list[str] | None = None,
        query_process_terms: list[str] | None = None,
        top_k: int = FINAL_TOP_K,
    ) -> list[RetrievedChunk]:
        """
        Main retrieval entry point.
        Returns top_k chunks after hybrid fusion.
        """
        if self.collection.count() == 0:
            logger.warning("Chroma collection is empty. Run ingestion first.")
            return []

        # Dense retrieval on expanded query (better semantic recall)
        dense_results = self._dense_search(
            expanded_query, k=DENSE_TOP_K, filters=metadata_filters
        )

        # BM25 on original query (better exact-term recall)
        bm25_results = self._bm25_search(
            query, k=BM25_TOP_K, filters=metadata_filters
        )

        # Fuse with RRF
        fused = self._reciprocal_rank_fusion(dense_results, bm25_results)
        fused.extend(self._supplement_canonical_results(
            query_text=query,
            metadata_filters=metadata_filters,
            query_forms=query_forms or [],
            query_process_terms=query_process_terms or [],
        ))
        fused = self._apply_metadata_boosts(
            fused,
            query_text=query,
            query_agencies=query_agencies or [],
            query_forms=query_forms or [],
            query_process_terms=query_process_terms or [],
        )

        # Deduplicate by chunk_id
        seen = set()
        deduped = []
        for chunk in fused:
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                deduped.append(chunk)

        deduped = self._diversify_by_url(deduped)

        # Fallback: if metadata filter returned nothing, retry without filter
        if len(deduped) == 0 and metadata_filters:
            logger.debug("No results with metadata filter — retrying without filter")
            dense_results = self._dense_search(expanded_query, k=DENSE_TOP_K, filters=None)
            bm25_results  = self._bm25_search(query, k=BM25_TOP_K, filters=None)
            fused = self._reciprocal_rank_fusion(dense_results, bm25_results)
            fused.extend(self._supplement_canonical_results(
                query_text=query,
                metadata_filters=None,
                query_forms=query_forms or [],
                query_process_terms=query_process_terms or [],
            ))
            fused = self._apply_metadata_boosts(
                fused,
                query_text=query,
                query_agencies=query_agencies or [],
                query_forms=query_forms or [],
                query_process_terms=query_process_terms or [],
            )
            seen = set()
            deduped = []
            for chunk in fused:
                if chunk.chunk_id not in seen:
                    seen.add(chunk.chunk_id)
                    deduped.append(chunk)
            deduped = self._diversify_by_url(deduped)

        return deduped[:top_k]

    # ------------------------------------------------------------------
    # Dense retrieval
    # ------------------------------------------------------------------

    def _dense_search(
        self,
        query: str,
        k: int,
        filters: dict | None,
    ) -> list[RetrievedChunk]:
        try:
            embedding = self._embed(query)
            if not embedding:
                return []

            kwargs: dict[str, Any] = {
                "query_embeddings": [embedding],
                "n_results": min(k, max(self.collection.count(), 1)),
                "include": ["documents", "metadatas", "distances"],
            }
            if filters:
                kwargs["where"] = filters

            results = self.collection.query(**kwargs)

            chunks = []
            docs      = results.get("documents", [[]])[0]
            metas     = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for doc, meta, dist in zip(docs, metas, distances):
                chunks.append(self._make_chunk(doc, meta, 1 - dist, "dense"))

            return chunks

        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            return []

    # ------------------------------------------------------------------
    # BM25 retrieval
    # ------------------------------------------------------------------

    def _build_bm25_index(self) -> None:
        """Load all documents from Chroma and build BM25 index."""
        count = self.collection.count()
        if count == 0:
            logger.info("Chroma empty — BM25 index deferred until ingestion.")
            return

        logger.info(f"Building BM25 index from {count} Chroma documents...")
        results = self.collection.get(include=["documents", "metadatas"])
        docs  = results.get("documents", [])
        metas = results.get("metadatas", [])

        tokenized = [self._tokenize_for_bm25(doc) for doc in docs]
        self._bm25 = BM25Okapi(tokenized)
        self._bm25_docs = [
            {"text": doc, "meta": meta}
            for doc, meta in zip(docs, metas)
        ]
        logger.info("BM25 index ready.")

    def rebuild_bm25_index(self) -> None:
        """Call this after ingestion to refresh the BM25 index."""
        self._build_bm25_index()

    def _bm25_search(
        self,
        query: str,
        k: int,
        filters: dict | None,
    ) -> list[RetrievedChunk]:
        if not self._bm25 or not self._bm25_docs:
            return []

        try:
            tokenized_query = self._tokenize_for_bm25(query)
            scores = self._bm25.get_scores(tokenized_query)

            # Get top-k indices sorted by score
            top_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[:k * 3]    # Fetch more to allow post-filter

            chunks = []
            for idx in top_indices:
                if len(chunks) >= k:
                    break
                score = scores[idx]
                if score <= 0:
                    continue

                doc  = self._bm25_docs[idx]["text"]
                meta = self._bm25_docs[idx]["meta"]

                # Apply metadata filters manually for BM25 results
                if filters and not self._matches_filter(meta, filters):
                    continue

                chunks.append(self._make_chunk(doc, meta, score, "bm25"))

            return chunks

        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

    def _matches_filter(self, meta: dict, filters: dict) -> bool:
        """Apply Chroma-style filters to a metadata dict for BM25 post-filtering."""
        if "$and" in filters:
            return all(self._matches_filter(meta, f) for f in filters["$and"])
        if "$or" in filters:
            return any(self._matches_filter(meta, f) for f in filters["$or"])

        for key, condition in filters.items():
            if key.startswith("$"):
                continue
            val = meta.get(key, "")
            if "$eq" in condition and val != condition["$eq"]:
                return False
            if "$contains" in condition and condition["$contains"] not in str(val):
                return False
        return True

    def _apply_metadata_boosts(
        self,
        chunks: list[RetrievedChunk],
        query_text: str,
        query_agencies: list[str],
        query_forms: list[str],
        query_process_terms: list[str],
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        query_terms = self._extract_query_terms(query_text)
        query_text_normalized = self._normalize_for_match(query_text)

        boosted: list[RetrievedChunk] = []
        for chunk in chunks:
            score = chunk.score
            meta_forms = set(getattr(chunk, "form_numbers", []) or [])
            meta_agency = getattr(chunk, "agency", "") or ""
            match_haystack = self._build_match_haystack(chunk)
            url_match = self._normalize_for_match(chunk.source_url)
            text_match = self._normalize_for_match(chunk.text[:400])

            if meta_agency and meta_agency in query_agencies:
                score += 0.02
            if meta_forms and meta_forms.intersection(query_forms):
                score += 0.03
            if set(getattr(chunk, "process_terms", []) or []).intersection(query_process_terms):
                score += 0.025
            if query_terms:
                overlap = len(query_terms.intersection(set(match_haystack.split())))
                score += min(overlap * 0.004, 0.03)
            if any(self._normalize_for_match(form) in match_haystack for form in query_forms):
                score += 0.02
            if any(self._normalize_for_match(form) in url_match for form in query_forms):
                score += 0.06
            if "h 4" in query_text_normalized and "spouse" in query_text_normalized and (
                "employment authorization for certain h 4 dependent spouses" in url_match
                or "h 4" in url_match and "dependent spouses" in url_match
            ):
                score += 0.16
            if ("perm" in query_text_normalized or "labor certification" in query_text_normalized) and (
                "flag dol gov" in url_match or "programs perm" in url_match
            ):
                score += 0.14
            if ("lca" in query_text_normalized or "labor condition application" in query_text_normalized) and "flag dol gov" in url_match:
                score += 0.13
            if "visa bulletin" in query_text_normalized and "travel state gov" in url_match:
                score += 0.16
            if "b 2" in query_text_normalized and any(term in query_text_normalized for term in ["work", "employment", "job"]) and "tourism visit visitor" in url_match:
                score += 0.18
            if "f 1" in query_text_normalized and "travel" in query_text_normalized and ("studyinthestates" in url_match or "cbp gov" in url_match):
                score += 0.14
            if "opt" in query_text_normalized and "travel" in query_text_normalized and ("studyinthestates" in url_match or "travel and visas" in url_match or "students travel" in url_match):
                score += 0.14
            if "cpt" in query_text_normalized and ("i 20" in query_text_normalized or "authorization" in query_text_normalized) and ("curricular practical training" in url_match or "students cpt" in url_match):
                score += 0.16
            if "stem opt" in query_text_normalized and "unemployment" in query_text_normalized and ("stem opt hub" in url_match or "studyinthestates" in url_match):
                score += 0.16
            if "grace period" in query_text_normalized and "f 1" in query_text_normalized and ("preparing to leave" in match_haystack or "studyinthestates" in url_match or "students and exchange visitors" in url_match):
                score += 0.14
            if "i140" in query_text_normalized and (
                "i140" in match_haystack or "permanent workers" in match_haystack or "i 140" in url_match
            ):
                score += 0.04
            if (
                ("ead" in query_text_normalized or "i765" in query_text_normalized)
                and "pending" in query_text_normalized
                and "travel" in query_text_normalized
                and any(marker in match_haystack for marker in ["advance parole", "i 131", "travel document"])
            ):
                score += 0.05
            if "travel" in query_text_normalized and "pending" in query_text_normalized and any(marker in url_match for marker in ["advance parole", "travel documents", "i 131"]):
                score += 0.08
            if "h1b" in query_text_normalized and "i140" in query_text_normalized and "faqs for individuals in h 1b nonimmigrant status" in match_haystack:
                score -= 0.05
            if "perm" in query_text_normalized and "faqs for individuals in h 1b nonimmigrant status" in match_haystack:
                score -= 0.05
            if ("lca" in query_text_normalized or "labor condition application" in query_text_normalized) and "options for nonimmigrant workers following termination of employment" in match_haystack:
                score -= 0.04
            if "visa bulletin" in query_text_normalized and "uscis gov/policy manual" in url_match:
                score -= 0.08
            if "h 4" in query_text_normalized and "spouse" in query_text_normalized and "faqs for individuals in h 1b nonimmigrant status" in match_haystack:
                score -= 0.08
            if "b 2" in query_text_normalized and any(term in query_text_normalized for term in ["work", "employment", "job"]) and "uscis gov/policy manual" in url_match:
                score -= 0.08
            if "grace period" in query_text_normalized and "f 1" in query_text_normalized and "options for nonimmigrant workers following termination of employment" in match_haystack:
                score -= 0.06
            if "page not found" in text_match or "page not found" in self._normalize_for_match(chunk.document_title):
                score -= 0.20
            if chunk.chunk_index == 0:
                score -= 0.02
            chunk.score = score
            boosted.append(chunk)

        return sorted(boosted, key=lambda c: c.score, reverse=True)

    def _diversify_by_url(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return []

        counts_by_url: dict[str, int] = {}
        diversified: list[RetrievedChunk] = []
        overflow: list[RetrievedChunk] = []

        for chunk in chunks:
            key = chunk.source_url or chunk.chunk_id
            if counts_by_url.get(key, 0) < MAX_CHUNKS_PER_URL:
                diversified.append(chunk)
                counts_by_url[key] = counts_by_url.get(key, 0) + 1
            else:
                overflow.append(chunk)

        diversified.extend(overflow)
        return diversified

    def _supplement_canonical_results(
        self,
        query_text: str,
        metadata_filters: dict | None,
        query_forms: list[str],
        query_process_terms: list[str],
    ) -> list[RetrievedChunk]:
        if not self._bm25_docs:
            return []

        url_patterns: list[str] = []
        query_text_normalized = self._normalize_for_match(query_text)
        process_terms_normalized = " ".join(query_process_terms).lower()
        if "I-140" in query_forms:
            url_patterns.extend([
                "/i-140",
                "form-i-140",
                "direct-filing-addresses-for-form-i-140",
                "/tools/processing-times",
                "/working-united-states/permanent-workers",
            ])
        if "I-131" in query_forms or "Advance Parole" in query_process_terms or "Travel" in query_process_terms:
            url_patterns.extend([
                "/i-131",
                "/travel/travel-documents/advance-parole",
                "/travel/travel-documents/advance-parole/about-advance-parole",
            ])
        if "PERM" in query_process_terms or "perm" in process_terms_normalized or "labor certification" in process_terms_normalized:
            url_patterns.extend([
                "/programs/perm",
                "/index.php/programs/perm",
                "/working-united-states/permanent-workers",
            ])
        if "LCA" in query_process_terms or "lca" in process_terms_normalized:
            url_patterns.extend([
                "flag.dol.gov/content/flag/how-apply-h-1b",
            ])
        if "Visa Bulletin" in query_process_terms:
            url_patterns.extend([
                "visa-bulletin",
                "/content/travel/en/us-visas/immigrate.html",
            ])
        if "Travel" in query_process_terms:
            url_patterns.extend([
                "studyinthestates.dhs.gov/students/traveling-outside-the-united-states",
                "cbp.gov/travel/international-visitors/kbyg/kbyg-students",
                "internationaloffice.berkeley.edu/students/travel",
                "bechtel.stanford.edu/immigration/f-1/travel-and-visas",
            ])
        if "grace period" in process_terms_normalized or "Grace Period" in query_process_terms:
            url_patterns.extend([
                "studyinthestates.dhs.gov/students/preparing-to-leave",
                "/working-in-the-united-states/students-and-exchange-visitors/students-and-employment",
                "/working-in-the-united-states/students-and-exchange-visitors/f-1-students",
            ])
        if "cpt" in process_terms_normalized or "CPT" in query_process_terms:
            url_patterns.extend([
                "studyinthestates.dhs.gov/students/cpt",
                "curricular-practical-training-cpt-for-f-1-students",
            ])
        if "stem opt" in process_terms_normalized or "STEM OPT" in query_process_terms:
            url_patterns.extend([
                "studyinthestates.dhs.gov/students/stem-opt-hub",
                "/working-in-the-united-states/students-and-exchange-visitors/stem-opt-hub",
            ])
        if "h 4" in query_text_normalized and "spouse" in query_text_normalized:
            url_patterns.extend([
                "/working-in-the-united-states/temporary-workers/h-1b-specialty-occupations/employment-authorization-for-certain-h-4-dependent-spouses",
            ])
        if "b 2" in query_text_normalized and any(term in query_text_normalized for term in ["work", "employment", "job"]):
            url_patterns.extend([
                "tourism-visit/visitor.html",
                "tourism-visit.html",
            ])
        if "visa bulletin" in query_text_normalized:
            url_patterns.extend([
                "visa-bulletin",
                "/content/travel/en/us-visas/immigrate.html",
            ])
        if "stem opt" in query_text_normalized and "unemployment" in query_text_normalized:
            url_patterns.extend([
                "studyinthestates.dhs.gov/students/stem-opt-hub",
            ])
        if not url_patterns:
            return []

        supplemented: list[RetrievedChunk] = []
        seen_urls: set[str] = set()
        for item in self._bm25_docs:
            meta = item["meta"]
            url = meta.get("source_url", "")
            if not url or url in seen_urls:
                continue
            if not any(pattern in url for pattern in url_patterns):
                continue
            if metadata_filters and not self._matches_filter(meta, metadata_filters):
                continue
            chunk = self._make_chunk(item["text"], meta, 0.09, "canonical")
            text_match = self._normalize_for_match(chunk.text[:400])
            if "page not found" in text_match or "page not found" in self._normalize_for_match(chunk.document_title):
                continue
            supplemented.append(chunk)
            seen_urls.add(url)
        return supplemented

    # ------------------------------------------------------------------
    # Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    def _reciprocal_rank_fusion(
        self,
        dense: list[RetrievedChunk],
        bm25: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """
        RRF score = sum(1 / (k + rank)) across result lists.
        Standard k=60 from the original RRF paper.
        """
        scores: dict[str, float] = {}
        chunks_by_id: dict[str, RetrievedChunk] = {}

        for rank, chunk in enumerate(dense, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1 / (RRF_K + rank)
            chunks_by_id[chunk.chunk_id] = chunk

        for rank, chunk in enumerate(bm25, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1 / (RRF_K + rank)
            if chunk.chunk_id not in chunks_by_id:
                chunks_by_id[chunk.chunk_id] = chunk

        # Sort by fused RRF score
        sorted_ids = sorted(scores, key=lambda i: scores[i], reverse=True)
        fused = []
        for cid in sorted_ids:
            chunk = chunks_by_id[cid]
            chunk.score = scores[cid]
            chunk.retrieval_method = "hybrid"
            fused.append(chunk)

        return fused

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> list[float] | None:
        try:
            response = self.openai.embeddings.create(
                input=[text], model=EMBEDDING_MODEL
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return None

    def _normalize_for_match(self, text: str) -> str:
        normalized = text.lower()
        normalized = normalized.replace("&", " and ")
        normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
        return " ".join(normalized.split())

    def _extract_query_terms(self, text: str) -> set[str]:
        normalized = self._normalize_for_match(text)
        return {
            token for token in normalized.split()
            if len(token) > 2 and token not in {"the", "and", "for", "with", "while", "what", "does", "how"}
        }

    def _build_match_haystack(self, chunk: RetrievedChunk) -> str:
        parts = [
            chunk.source_name,
            chunk.source_url,
            chunk.section_title,
            chunk.doc_type,
            " ".join(chunk.form_numbers),
            " ".join(chunk.process_terms),
            " ".join(chunk.keyword_tags),
            chunk.text[:500],
        ]
        return self._normalize_for_match(" ".join(part for part in parts if part))

    def _tokenize_for_bm25(self, text: str) -> list[str]:
        normalized = text.lower()
        normalized = normalized.replace("/", " ")
        normalized = re.sub(r"(?<=\w)-(?=\w)", "", normalized)
        normalized = re.sub(r"[^a-z0-9.\- ]+", " ", normalized)
        tokens = re.findall(r"[a-z]+(?:\.[a-z]+)*|\d+[a-z]*|[a-z]+-\d+[a-z]*|[a-z]\d+|\d+", normalized)
        return tokens or normalized.split()

    def _make_chunk(self, text: str, meta: dict, score: float, method: str) -> RetrievedChunk:
        tags_raw = meta.get("visa_tags", "")
        visa_tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
        chunk = RetrievedChunk(
            text=text,
            chunk_id=meta.get("chunk_id", ""),
            chunk_index=int(meta.get("chunk_index", 0)),
            source_id=meta.get("source_id", ""),
            source_name=meta.get("source_name", ""),
            source_url=meta.get("source_url", ""),
            document_title=meta.get("document_title", ""),
            section_title=meta.get("section_title", ""),
            tier=int(meta.get("tier", 1)),
            tier_label=meta.get("tier_label", "Official U.S. Government"),
            visa_tags=visa_tags,
            crawled_at=meta.get("crawled_at", ""),
            last_updated_on_source=meta.get("last_updated_on_source", ""),
            is_stale=bool(meta.get("is_stale", False)),
            score=score,
            retrieval_method=method,
        )
        chunk.agency = meta.get("agency", "")
        chunk.doc_type = meta.get("doc_type", "")
        forms_raw = meta.get("form_numbers", "")
        regs_raw = meta.get("reg_citations", "")
        process_raw = meta.get("process_terms", "")
        tags_raw_meta = meta.get("keyword_tags", "")
        chunk.form_numbers = [f.strip() for f in str(forms_raw).split(",") if f.strip()]
        chunk.reg_citations = [r.strip() for r in str(regs_raw).split(",") if r.strip()]
        chunk.process_terms = [p.strip() for p in str(process_raw).split(",") if p.strip()]
        chunk.keyword_tags = [t.strip() for t in str(tags_raw_meta).split(",") if t.strip()]
        return chunk

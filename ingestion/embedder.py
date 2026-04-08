"""
Embedder — ingestion/embedder.py
Embeds chunks via OpenAI and stores them in ChromaDB.
Idempotent: upsert by chunk_id. Re-running never creates duplicates.
"""

import logging
import os
from typing import Iterator

import chromadb
import openai
from chromadb.config import Settings
from openai import OpenAI

from ingestion.chunker import DocumentChunk

logger = logging.getLogger(__name__)

EMBEDDING_MODEL  = "text-embedding-3-small"
COLLECTION_NAME  = "visa_rag_chunks"
BATCH_SIZE       = 100


class Embedder:

    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
        self.chroma = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"Chroma ready — collection '{COLLECTION_NAME}', "
            f"current count: {self.collection.count()}"
        )

    def embed_and_store(self, chunks: list[DocumentChunk]) -> int:
        """Embed and upsert chunks. Returns count stored."""
        if not chunks:
            return 0

        stored = 0
        for batch in self._batches(chunks, BATCH_SIZE):
            embeddings = self._embed([c.text for c in batch])
            if not embeddings:
                logger.error(f"Embedding failed for batch of {len(batch)}")
                continue

            self.collection.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=embeddings,
                documents=[c.text for c in batch],
                metadatas=[c.to_metadata() for c in batch],
            )
            stored += len(batch)
            logger.info(f"Stored {len(batch)} chunks | Total: {self.collection.count()}")

        return stored

    def delete_source(self, source_id: str) -> int:
        """Remove all chunks for a source before re-ingesting."""
        results = self.collection.get(
            where={"source_id": {"$eq": source_id}},
            include=[],
        )
        ids = results.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} chunks for '{source_id}'")
        return len(ids)

    def total_chunks(self) -> int:
        return self.collection.count()

    def source_chunk_count(self, source_id: str) -> int:
        results = self.collection.get(
            where={"source_id": {"$eq": source_id}},
            include=[],
        )
        return len(results.get("ids", []))

    def _embed(self, texts: list[str]) -> list[list[float]] | None:
        try:
            response = self.openai_client.embeddings.create(
                input=texts,
                model=EMBEDDING_MODEL,
            )
            return [item.embedding for item in response.data]
        except openai.RateLimitError:
            logger.error("OpenAI rate limit hit.")
            return None
        except openai.OpenAIError as e:
            logger.error(f"OpenAI embedding error: {e}")
            return None

    def _batches(self, items: list, size: int) -> Iterator[list]:
        for i in range(0, len(items), size):
            yield items[i : i + size]
"""
scripts/ingest.py — Master ingestion runner
Crawl -> Clean/Parse -> Chunk -> Embed -> Store

Usage:
  python scripts/ingest.py                               # All enabled sources
  python scripts/ingest.py --source study_in_the_states  # One source
  python scripts/ingest.py --tier 1                      # All Tier 1 sources
  python scripts/ingest.py --dry-run                     # Preview only
"""

import asyncio
import argparse
import logging
import os
import sys

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.crawler import Crawler, SourceRegistry
from ingestion.html_cleaner import HTMLCleaner
from ingestion.pdf_parser import PDFParser
from ingestion.chunker import Chunker
from ingestion.embedder import Embedder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ingest")


async def ingest_source(source, crawler, cleaner, pdf_parser, chunker, embedder, dry_run):
    source_id = source["id"]
    logger.info(f"── Ingesting: {source_id} ({source['name']})")

    pages_crawled = chunks_created = chunks_stored = errors = 0

    async for page in crawler.crawl_source(source):
        pages_crawled += 1
        try:
            if page.content_type == "pdf":
                parsed = pdf_parser.parse(page.pdf_bytes, page.url)
                if not parsed:
                    errors += 1
                    continue
                chunks = chunker.chunk_pdf(
                    pages=parsed.pages,
                    source_id=source_id,
                    source_name=source["name"],
                    source_url=page.url,
                    tier=source["tier"],
                    tier_label=source["tier_label"],
                    visa_tags=source.get("visa_tags", []),
                    document_title=parsed.title,
                    content_type="pdf",
                )
            else:
                cleaned = cleaner.clean(page.html, page.url)
                # Fallback to crawl4ai markdown only if HTML cleaning fails entirely.
                if not cleaned:
                    md = page.markdown or ""
                    if md and len(md.split()) >= 50:
                        title = page.url.rstrip("/").split("/")[-1].replace("-", " ").title()
                        for line in md.splitlines():
                            stripped = line.strip()
                            if stripped.startswith("# "):
                                title = stripped[2:].strip()
                                break
                            elif stripped.startswith("## "):
                                title = stripped[3:].strip()
                                break
                        class _FakeCleaned:
                            pass
                        cleaned = _FakeCleaned()
                        cleaned.text = md
                        cleaned.sections = []
                        cleaned.title = title
                if not cleaned:
                    errors += 1
                    continue
                chunks = chunker.chunk_html_page(
                    text=cleaned.text,
                    sections=cleaned.sections,
                    source_id=source_id,
                    source_name=source["name"],
                    source_url=page.url,
                    tier=source["tier"],
                    tier_label=source["tier_label"],
                    visa_tags=source.get("visa_tags", []),
                    document_title=cleaned.title,
                )

            chunks_created += len(chunks)

            if not dry_run:
                chunks_stored += embedder.embed_and_store(chunks)
            else:
                logger.info(f"  [DRY RUN] {len(chunks)} chunks from {page.url}")

        except Exception as e:
            logger.error(f"  Error on {page.url}: {e}")
            errors += 1

    return {
        "source_id": source_id,
        "pages_crawled": pages_crawled,
        "chunks_created": chunks_created,
        "chunks_stored": chunks_stored,
        "errors": errors,
    }


async def main():
    ap = argparse.ArgumentParser(description="Ingest sources into the visa RAG vector store")
    ap.add_argument("--source", type=str, help="Specific source ID to ingest")
    ap.add_argument("--tier",   type=int, help="Ingest all sources of this tier")
    ap.add_argument("--dry-run", action="store_true", help="Preview without storing")
    args = ap.parse_args()

    registry   = SourceRegistry("config/sources.yaml")
    crawler    = Crawler(registry)
    cleaner    = HTMLCleaner()
    pdf_parser = PDFParser()
    chunker    = Chunker()
    embedder   = Embedder() if not args.dry_run else None

    if args.source:
        src = registry.get_source(args.source)
        if not src:
            logger.error(f"Source '{args.source}' not found in sources.yaml")
            sys.exit(1)
        sources = [src]
    elif args.tier:
        sources = registry.get_by_tier(args.tier)
    else:
        sources = registry.get_all_enabled()

    sources.sort(key=lambda s: s.get("priority", 0), reverse=True)
    logger.info(f"Ingesting {len(sources)} source(s) | dry_run={args.dry_run}")

    summaries = []
    for source in sources:
        summary = await ingest_source(
            source, crawler, cleaner, pdf_parser, chunker, embedder, args.dry_run
        )
        summaries.append(summary)

    print("\n" + "=" * 55)
    print("INGESTION COMPLETE")
    print("=" * 55)
    for s in summaries:
        print(f"  {s['source_id']:35s} pages={s['pages_crawled']:4d} "
              f"chunks={s['chunks_stored']:5d} errors={s['errors']}")
    print("-" * 55)
    if not args.dry_run and embedder:
        print(f"  Total in Chroma: {embedder.total_chunks()} chunks")
    print("=" * 55)


if __name__ == "__main__":
    asyncio.run(main())

"""
Crawler — ingestion/crawler.py
Fetches web pages and PDFs from sources defined in sources.yaml.
Rate-limited to respect government servers (1.5s between requests).
Change detection: skips pages whose content hash has not changed.
"""

import asyncio
import hashlib
import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import AsyncIterator
from urllib.parse import urljoin

import requests
import yaml
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode

logger = logging.getLogger(__name__)

REQUEST_DELAY   = 1.5
REQUEST_TIMEOUT = 30
MAX_RETRIES     = 3


@dataclass
class CrawledPage:
    url: str
    html: str
    content_hash: str
    status_code: int
    content_type: str       # "html" or "pdf"
    pdf_bytes: bytes | None = None
    markdown: str | None = None   # crawl4ai pre-rendered markdown (preferred over raw html)


class SourceRegistry:
    def __init__(self, config_path: str = "config/sources.yaml"):
        with open(config_path) as f:
            data = yaml.safe_load(f)
        self.sources = [s for s in data["sources"] if s.get("enabled", True)]
        logger.info(f"Loaded {len(self.sources)} enabled sources")

    def get_source(self, source_id: str) -> dict | None:
        return next((s for s in self.sources if s["id"] == source_id), None)

    def get_all_enabled(self) -> list[dict]:
        return self.sources

    def get_by_tier(self, tier: int) -> list[dict]:
        return [s for s in self.sources if s.get("tier") == tier]


class Crawler:
    def __init__(self, registry: SourceRegistry):
        self.registry = registry
        self._hashes: dict[str, str] = {}

    async def crawl_source(self, source: dict) -> AsyncIterator[CrawledPage]:
        strategy = source.get("crawl_strategy", "targeted_urls")
        base_url  = source["base_url"]

        if strategy == "targeted_urls":
            urls = [urljoin(base_url, p) for p in source.get("target_paths", [])]
        elif strategy == "sitemap":
            sitemap_url = source.get("sitemap_url", base_url + "/sitemap.xml")
            pattern     = source.get("url_filter_pattern", "")
            urls = self._parse_sitemap(
                sitemap_url,
                pattern,
                verify_ssl=source.get("verify_ssl", True),
            )
            logger.info(f"Sitemap for '{source['id']}': {len(urls)} URLs")
        else:
            logger.warning(f"Unknown strategy '{strategy}' for {source['id']}")
            return

        async with AsyncWebCrawler() as crawler:
            for url in urls:
                try:
                    if url.lower().endswith(".pdf"):
                        page = self._fetch_pdf(url)
                    else:
                        page = await self._fetch_html(url, crawler, source)

                    if page and self._changed(url, page.content_hash):
                        self._hashes[url] = page.content_hash
                        yield page
                    elif page:
                        logger.debug(f"No change: {url}")

                    await asyncio.sleep(REQUEST_DELAY)

                except Exception as e:
                    logger.error(f"Failed {url}: {e}")

    async def _fetch_html(
        self,
        url: str,
        crawler: AsyncWebCrawler,
        source: dict,
    ) -> CrawledPage | None:
        verify_ssl = source.get("verify_ssl", True)
        try:
            config = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                page_timeout=REQUEST_TIMEOUT * 1000,
                wait_until="networkidle",
            )
            result = await crawler.arun(url=url, config=config)
            if not result.success:
                logger.warning(f"crawl4ai failed: {url} — {result.error_message}")
                return self._fetch_html_via_requests(url, verify_ssl=verify_ssl)
            html = result.html or ""
            markdown = result.markdown or ""
            if self._is_thin_html(html):
                logger.warning(f"crawl4ai returned thin html for {url} — trying requests fallback")
                fallback = self._fetch_html_via_requests(url, verify_ssl=verify_ssl)
                if fallback:
                    return fallback
            return CrawledPage(
                url=url, html=html,
                content_hash=hashlib.md5(html.encode()).hexdigest(),
                status_code=200, content_type="html",
                markdown=markdown,
            )
        except Exception as e:
            logger.error(f"HTML fetch failed {url}: {e}")
            return self._fetch_html_via_requests(url, verify_ssl=verify_ssl)

    def _fetch_html_via_requests(
        self,
        url: str,
        verify_ssl: bool = True,
    ) -> CrawledPage | None:
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(
                    url,
                    timeout=REQUEST_TIMEOUT,
                    headers={"User-Agent": "VisaRAGBot/1.0 (educational research)"},
                    allow_redirects=True,
                    verify=verify_ssl,
                )
                response.raise_for_status()
                html = response.text or ""
                if self._is_thin_html(html):
                    logger.warning(f"Requests fallback returned thin html: {url}")
                    return None
                return CrawledPage(
                    url=url,
                    html=html,
                    content_hash=hashlib.md5(html.encode()).hexdigest(),
                    status_code=response.status_code,
                    content_type="html",
                    markdown=None,
                )
            except requests.RequestException as e:
                logger.warning(f"HTML fallback attempt {attempt+1} failed {url}: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
        return None

    def _fetch_pdf(self, url: str) -> CrawledPage | None:
        for attempt in range(MAX_RETRIES):
            try:
                r = requests.get(
                    url, timeout=REQUEST_TIMEOUT,
                    headers={"User-Agent": "VisaRAGBot/1.0 (educational research)"},
                    allow_redirects=True,
                )
                r.raise_for_status()
                b = r.content
                return CrawledPage(
                    url=url, html="",
                    content_hash=hashlib.md5(b).hexdigest(),
                    status_code=r.status_code,
                    content_type="pdf", pdf_bytes=b,
                )
            except requests.RequestException as e:
                logger.warning(f"PDF attempt {attempt+1} failed {url}: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
        return None

    def _parse_sitemap(
        self,
        sitemap_url: str,
        filter_pattern: str = "",
        verify_ssl: bool = True,
    ) -> list[str]:
        urls = []
        try:
            r = requests.get(
                sitemap_url, timeout=REQUEST_TIMEOUT,
                headers={"User-Agent": "VisaRAGBot/1.0 (educational research)"},
                verify=verify_ssl,
            )
            r.raise_for_status()
            root = ET.fromstring(r.content)
            ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

            sub_sitemaps = root.findall("sm:sitemap/sm:loc", ns)
            if sub_sitemaps:
                for loc in sub_sitemaps[:15]:
                    urls.extend(self._parse_sitemap(loc.text.strip(), filter_pattern, verify_ssl))
            else:
                for loc in root.findall("sm:url/sm:loc", ns):
                    url = loc.text.strip()
                    if not filter_pattern or filter_pattern in url:
                        urls.append(url)
        except Exception as e:
            logger.error(f"Sitemap parse failed {sitemap_url}: {e}")
        return urls

    def _is_thin_html(self, html: str) -> bool:
        text = " ".join(html.split())
        return len(text) < 500

    def _changed(self, url: str, new_hash: str) -> bool:
        return self._hashes.get(url) != new_hash

"""
HTML Cleaner — ingestion/html_cleaner.py
Extracts clean, LLM-ready text from raw HTML government pages.
Primary: trafilatura. Fallback: BeautifulSoup.
"""

import re
import logging
from dataclasses import dataclass
import trafilatura
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class CleanedPage:
    url: str
    title: str | None
    text: str
    sections: list[dict]        # [{heading: str, content: str}]
    word_count: int
    extraction_method: str      # "trafilatura" or "beautifulsoup"


class HTMLCleaner:
    """
    Extracts clean text from raw HTML.
    Removes nav, footers, cookie banners, sidebars.
    Preserves headings, lists, and tables — critical for policy content.
    """

    MIN_WORD_COUNT = 50
    BOILERPLATE_PATTERNS = [
        r"skip to main content",
        r"share sensitive information only on official[, ]+secure websites?",
        r"multilingual resources",
        r"return to top",
        r"an official website of the united states government",
        r"here'?s how you know",
        r"before you continue",
        r"case status online",
        r"change of address",
        r"password resets and technical support",
        r"avoid scams",
        r"uscis office locator",
        r"office of the citizenship and immigration services ombudsman",
    ]

    def clean(self, html: str, url: str) -> CleanedPage | None:
        result = self._extract_with_trafilatura(html, url)
        if result and result.word_count >= self.MIN_WORD_COUNT:
            return result

        logger.debug(f"trafilatura thin for {url}, trying BeautifulSoup")
        result = self._extract_with_beautifulsoup(html, url)
        if result and result.word_count >= self.MIN_WORD_COUNT:
            return result

        logger.warning(f"Insufficient content from {url}")
        return None

    def _extract_with_trafilatura(self, html: str, url: str) -> CleanedPage | None:
        try:
            text = trafilatura.extract(
                html,
                include_tables=True,
                include_links=False,
                include_images=False,
                no_fallback=False,
                favor_recall=True,
            )
            if not text:
                return None

            meta = trafilatura.extract_metadata(html)
            title = meta.title if meta else None
            sections = self._extract_sections_from_text(text)

            return CleanedPage(
                url=url,
                title=title,
                text=self._strip_boilerplate_text(text.strip()),
                sections=self._strip_boilerplate_sections(sections),
                word_count=len(self._strip_boilerplate_text(text.strip()).split()),
                extraction_method="trafilatura",
            )
        except Exception as e:
            logger.error(f"trafilatura failed for {url}: {e}")
            return None

    def _extract_with_beautifulsoup(self, html: str, url: str) -> CleanedPage | None:
        try:
            soup = BeautifulSoup(html, "lxml")

            # Remove boilerplate
            for tag in soup.find_all(["nav", "header", "footer", "aside",
                                      "script", "style", "noscript", "form"]):
                tag.decompose()
            for tag in soup.find_all(attrs={"class": re.compile(
                r"nav|menu|sidebar|breadcrumb|footer|header|cookie|banner", re.I
            )}):
                tag.decompose()

            # Find main content
            main = (
                soup.find("main") or
                soup.find(attrs={"role": "main"}) or
                soup.find("article") or
                soup.find("div", attrs={"id": re.compile(r"content|main", re.I)}) or
                soup.find("body")
            )
            if not main:
                return None

            title_tag = soup.find("h1") or soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else None
            text = self._normalize_whitespace(main.get_text(separator="\n", strip=True))
            text = self._strip_boilerplate_text(text)
            sections = self._strip_boilerplate_sections(self._extract_sections_from_html(main))

            return CleanedPage(
                url=url,
                title=title,
                text=text,
                sections=sections,
                word_count=len(text.split()),
                extraction_method="beautifulsoup",
            )
        except Exception as e:
            logger.error(f"BeautifulSoup failed for {url}: {e}")
            return None

    def _extract_sections_from_text(self, text: str) -> list[dict]:
        sections = []
        lines = text.split("\n")
        current_heading = None
        current_content = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            is_heading = (
                len(line) < 100 and
                not line.endswith((".", ",", ";", ":")) and
                not line.isupper() and
                len(line.split()) >= 2
            )
            if is_heading and current_content:
                sections.append({
                    "heading": current_heading or "Introduction",
                    "content": "\n".join(current_content).strip(),
                })
                current_heading = line
                current_content = []
            elif is_heading:
                current_heading = line
            else:
                current_content.append(line)

        if current_content:
            sections.append({
                "heading": current_heading or "Content",
                "content": "\n".join(current_content).strip(),
            })
        return sections

    def _extract_sections_from_html(self, element) -> list[dict]:
        sections = []
        current_heading = None
        current_content = []

        for tag in element.find_all(["h1", "h2", "h3", "p", "li", "td"]):
            if tag.name in ["h1", "h2", "h3"]:
                if current_content:
                    sections.append({
                        "heading": current_heading or "Introduction",
                        "content": " ".join(current_content).strip(),
                    })
                    current_content = []
                current_heading = tag.get_text(strip=True)
            else:
                t = tag.get_text(strip=True)
                if t:
                    current_content.append(t)

        if current_content:
            sections.append({
                "heading": current_heading or "Content",
                "content": " ".join(current_content).strip(),
            })
        return sections

    def _normalize_whitespace(self, text: str) -> str:
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def _strip_boilerplate_text(self, text: str) -> str:
        lines = []
        for raw_line in text.splitlines():
            line = " ".join(raw_line.split()).strip()
            if not line:
                continue
            if any(re.search(pattern, line, flags=re.IGNORECASE) for pattern in self.BOILERPLATE_PATTERNS):
                continue
            lines.append(line)
        cleaned = "\n".join(lines).strip()
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned

    def _strip_boilerplate_sections(self, sections: list[dict]) -> list[dict]:
        cleaned_sections = []
        for section in sections:
            heading = self._strip_boilerplate_text(section.get("heading", "")).strip()
            content = self._strip_boilerplate_text(section.get("content", "")).strip()
            if len(content.split()) < self.MIN_WORD_COUNT:
                continue
            cleaned_sections.append({
                "heading": heading or section.get("heading", "") or "Content",
                "content": content,
            })
        return cleaned_sections

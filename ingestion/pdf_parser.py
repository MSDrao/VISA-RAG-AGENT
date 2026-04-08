"""
PDF Parser — ingestion/pdf_parser.py
Parses government PDFs into structured text.
PyMuPDF for text, pdfplumber for tables, pytesseract for scanned fallback.
"""

import io
import logging
import re
from dataclasses import dataclass, field

import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ParsedPDF:
    url: str
    filename: str
    title: str | None
    pages: list[dict]       # [{page_num, text, tables}]
    full_text: str
    page_count: int
    has_tables: bool
    is_scanned: bool
    word_count: int


class PDFParser:
    """
    Handles three PDF types common in government immigration sources:
    - Text PDFs: standard USCIS policy memos, form instructions
    - Table-heavy PDFs: fee schedules, Visa Bulletin priority date charts
    - Scanned PDFs: older USCIS memos (OCR fallback)
    """

    MIN_CHARS_PER_PAGE = 100    # Below this threshold, treat page as scanned

    def parse(self, pdf_bytes: bytes, url: str, filename: str = "") -> ParsedPDF | None:
        try:
            # Pass 1: PyMuPDF for text
            fitz_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            title = fitz_doc.metadata.get("title") or filename or None
            text_pages = []

            for page_num, page in enumerate(fitz_doc, start=1):
                text = page.get_text("text").strip()
                text_pages.append({
                    "page_num": page_num,
                    "text": text,
                    "char_count": len(text),
                    "tables": [],
                })
            fitz_doc.close()

            # Detect scanned PDF
            thin_pages = sum(1 for p in text_pages if p["char_count"] < self.MIN_CHARS_PER_PAGE)
            is_scanned = thin_pages > len(text_pages) * 0.6
            if is_scanned:
                logger.info(f"Scanned PDF detected: {url}. Using OCR.")
                text_pages = self._ocr_pdf(pdf_bytes, url)

            # Pass 2: pdfplumber for tables
            has_tables = False
            try:
                with pdfplumber.open(io.BytesIO(pdf_bytes)) as plumber_doc:
                    for i, plumber_page in enumerate(plumber_doc.pages):
                        tables = plumber_page.extract_tables()
                        if tables and i < len(text_pages):
                            has_tables = True
                            text_pages[i]["tables"] = [
                                self._format_table(t) for t in tables if t
                            ]
            except Exception as e:
                logger.warning(f"pdfplumber failed for {url}: {e}")

            # Merge tables into page text
            for page in text_pages:
                if page["tables"]:
                    page["text"] += "\n\n" + "\n\n".join(page["tables"])

            full_text = "\n\n".join(
                f"[Page {p['page_num']}]\n{p['text']}"
                for p in text_pages if p["text"].strip()
            )

            if len(full_text.strip()) < 100:
                logger.warning(f"Insufficient content from {url}")
                return None

            return ParsedPDF(
                url=url,
                filename=filename,
                title=title,
                pages=text_pages,
                full_text=full_text,
                page_count=len(text_pages),
                has_tables=has_tables,
                is_scanned=is_scanned,
                word_count=len(full_text.split()),
            )

        except Exception as e:
            logger.error(f"PDF parsing failed for {url}: {e}")
            return None

    def _ocr_pdf(self, pdf_bytes: bytes, url: str) -> list[dict]:
        """OCR fallback for scanned PDFs. Renders at 300 DPI for accuracy."""
        pages = []
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            for page_num, page in enumerate(doc, start=1):
                mat = fitz.Matrix(300 / 72, 300 / 72)     # 300 DPI
                pix = page.get_pixmap(matrix=mat)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(img, lang="eng")
                pages.append({
                    "page_num": page_num,
                    "text": text.strip(),
                    "char_count": len(text),
                    "tables": [],
                })
            doc.close()
        except Exception as e:
            logger.error(f"OCR failed for {url}: {e}")
        return pages

    def _format_table(self, table: list[list]) -> str:
        """
        Convert pdfplumber table to markdown-style text.
        Critical for: fee schedules, Visa Bulletin priority date charts,
        processing time tables.
        """
        if not table or not table[0]:
            return ""

        cleaned = [
            [str(cell).strip().replace("\n", " ") if cell else "" for cell in row]
            for row in table
        ]
        if not cleaned:
            return ""

        header = cleaned[0]
        rows = cleaned[1:]
        lines = ["| " + " | ".join(header) + " |"]
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in rows:
            while len(row) < len(header):
                row.append("")
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)
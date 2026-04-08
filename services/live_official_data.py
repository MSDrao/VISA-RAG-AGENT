"""
services/live_official_data.py

Small live-data helper for current operational questions that are not served well
by static RAG alone, such as:
- current Visa Bulletin cutoffs / "which year is getting approved"
- current processing-time / waiting-time questions

This deliberately stays narrow. If live fetch/parsing fails, the API should
fall back to the normal retrieval pipeline.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from generation.response_model import Citation, STANDARD_DISCLAIMER, VisaAnswer

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 12
STATE_BULLETIN_LANDING = "https://travel.state.gov/content/travel/en/legal/visa-law0/visa-bulletin.html"
USCIS_PREMIUM_PROCESSING = "https://www.uscis.gov/forms/all-forms/how-do-i-request-premium-processing"
USCIS_PROCESSING_TOOL = "https://egov.uscis.gov/processing-times/"


def try_live_operational_answer(question: str) -> VisaAnswer | None:
    q = _normalize(question)

    if _looks_like_visa_bulletin_query(q):
        return _answer_visa_bulletin(question)

    if _looks_like_processing_time_query(q):
        return _answer_processing_time(question)

    return None


def _looks_like_visa_bulletin_query(q: str) -> bool:
    return any(
        phrase in q
        for phrase in [
            "visa bulletin",
            "priority date",
            "which year",
            "current green card approval status",
            "what year people getting approved",
            "what year is getting approved",
            "current green card status",
        ]
    )


def _looks_like_processing_time_query(q: str) -> bool:
    time_markers = ["processing time", "waiting time", "how long", "timeline", "wait time", "how many days", "how many months"]
    return any(marker in q for marker in time_markers)


def _answer_processing_time(question: str) -> VisaAnswer | None:
    try:
        resp = requests.get(USCIS_PREMIUM_PROCESSING, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        logger.warning("Premium processing fetch failed: %s", e)
        return None

    page = resp.text
    normalized = _normalize(page)
    q = _normalize(question)

    if "15 business days for most classifications" not in normalized and "30 business days for form i 765" not in normalized:
        return None

    answer = _build_processing_time_answer(q, normalized)
    if not answer:
        return None

    referenced = _processing_visa_tags(q)

    return VisaAnswer(
        answer=answer,
        citations=[
            _build_citation(
                source_id="live_uscis_premium_processing",
                source_name="USCIS Premium Processing",
                source_url=USCIS_PREMIUM_PROCESSING,
            ),
            _build_citation(
                source_id="live_uscis_processing_times_tool",
                source_name="USCIS Processing Times Tool",
                source_url=USCIS_PROCESSING_TOOL,
            ),
        ],
        confidence="medium",
        visa_types_referenced=referenced,
        requires_attorney=False,
        disclaimer=STANDARD_DISCLAIMER,
    )


def _answer_visa_bulletin(question: str) -> VisaAnswer | None:
    bulletin_url = _discover_current_bulletin_url()
    if not bulletin_url:
        return None

    bulletin = _fetch_bulletin_table(bulletin_url)
    if not bulletin:
        return None

    q = _normalize(question)
    answer_text = _build_bulletin_answer(q, bulletin)
    if not answer_text:
        return None

    return VisaAnswer(
        answer=answer_text,
        citations=[
            _build_citation(
                source_id="live_state_visa_bulletin",
                source_name="U.S. Department of State Visa Bulletin",
                source_url=bulletin_url,
            )
        ],
        confidence="high",
        visa_types_referenced=["green_card"],
        requires_attorney=False,
        disclaimer=STANDARD_DISCLAIMER,
    )


def _discover_current_bulletin_url() -> str | None:
    try:
        resp = requests.get(STATE_BULLETIN_LANDING, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        logger.warning("Visa Bulletin landing fetch failed: %s", e)
        return None

    matches = re.findall(r"visa-bulletin-for-[a-z-]+-\d{4}\.html", resp.text, flags=re.IGNORECASE)
    if not matches:
        return None

    latest = matches[0]
    return urljoin(STATE_BULLETIN_LANDING, f"/content/travel/en/legal/visa-law0/visa-bulletin/2026/{latest}" if "/202" not in latest else latest)


def _fetch_bulletin_table(bulletin_url: str) -> dict[str, dict[str, str]] | None:
    try:
        resp = requests.get(bulletin_url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        logger.warning("Current Visa Bulletin fetch failed: %s", e)
        return None

    soup = BeautifulSoup(resp.text, "lxml")
    tables = soup.find_all("table")
    target = None
    for table in tables:
        table_text = _normalize(table.get_text(" ", strip=True))
        if "employment based" in table_text and "all chargeability areas except those listed" in table_text:
            target = table
            break

    if target is None:
        return None

    rows = target.find_all("tr")
    if len(rows) < 2:
        return None

    headers = [_normalize(cell.get_text(" ", strip=True)) for cell in rows[0].find_all(["td", "th"])]
    result: dict[str, dict[str, str]] = {}
    for row in rows[1:]:
        cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"])]
        if len(cells) != len(headers) or not cells:
            continue
        category = _normalize(cells[0])
        result[category] = {
            headers[i]: cells[i]
            for i in range(1, len(cells))
        }
    return result


def _build_bulletin_answer(q: str, bulletin: dict[str, dict[str, str]]) -> str | None:
    eb1 = bulletin.get("1st")
    eb2 = bulletin.get("2nd")
    eb3 = bulletin.get("3rd")
    if not eb1:
        return None

    all_areas = "all chargeability areas except those listed"
    china = "china mainland born"
    india = "india"
    mexico = "mexico"
    philippines = "philippines"

    if any(marker in q for marker in ["eb1", "eb1a", "eb 1", "eb-1", "extraordinary ability"]):
        return (
            f"According to the current Department of State Visa Bulletin [1], EB-1 final action dates are "
            f"{_format_cutoff(eb1.get(all_areas, ''))} for all chargeability areas, {eb1.get(china, '')} for China, "
            f"and {eb1.get(india, '')} for India. Mexico and the Philippines are {_format_cutoff(eb1.get(mexico, ''))} "
            f"and {_format_cutoff(eb1.get(philippines, ''))}, respectively. For EB-1A specifically, that means cases "
            f"are current in most countries, while India and China are currently limited to priority dates earlier than those listed [1]."
        )

    if any(marker in q for marker in ["which year", "getting approved", "green card approval status", "priority date"]):
        return (
            f"It depends on the employment-based category and country of chargeability. In the current Visa Bulletin [1], "
            f"EB-1 is {_format_cutoff(eb1.get(all_areas, ''))} for most countries, but China and India are at {eb1.get(china, '')} "
            f"and {eb1.get(india, '')}. EB-2 is {_format_cutoff(eb2.get(all_areas, '')) if eb2 else 'not available'} for most countries, "
            f"{eb2.get(china, '') if eb2 else ''} for China, and {eb2.get(india, '') if eb2 else ''} for India. "
            f"EB-3 is {eb3.get(all_areas, '') if eb3 else ''} for most countries, {eb3.get(china, '') if eb3 else ''} for China, "
            f"and {eb3.get(india, '') if eb3 else ''} for India [1]. If you tell me the exact category, such as EB-1A or EB-2, "
            f"and the country of chargeability, I can interpret the cutoff more precisely."
        )

    return None


def _build_processing_time_answer(q: str, premium_page_normalized: str) -> str | None:
    generic_suffix = (
        " USCIS regular, non-premium processing time is not published on a static official page "
        "that this app can reliably parse, so the exact current standard wait time still has to be "
        f"checked in the USCIS Processing Times tool [2]."
    )
    premium_only_suffix = (
        " This is the official USCIS premium-processing timeline, not the regular standard-processing timeline [1]."
    )
    regular_only_text = (
        "The exact current USCIS regular, non-premium processing time is not published on a static official page "
        "that this app can reliably parse. For the current standard wait time, you still need to check the USCIS "
        "Processing Times tool [2]."
    )
    is_premium_only = _mentions_premium_only(q)
    is_regular_only = _mentions_regular_only(q)
    asks_change_status = _mentions_change_of_status(q)
    asks_consular = _mentions_consular_processing(q)

    if asks_consular:
        return (
            "Consular-processing timing is not the same as a USCIS processing time. USCIS premium-processing "
            "timelines do not tell you how long embassy or consular scheduling, interview availability, or "
            "post-interview administrative processing will take. For consular processing, you generally need to "
            "check the Department of State or the specific embassy or consulate handling the case."
        )

    if any(marker in q for marker in ["cpt", "curricular practical training"]):
        return (
            "CPT is generally not a USCIS processing-time category. CPT authorization is typically handled "
            "through the student's school and designated school official rather than through a USCIS petition. "
            "So there is no standard USCIS CPT processing time to quote here [2]."
        )

    if any(marker in q for marker in ["eb1a", "eb 1a", "eb-1a", "extraordinary ability", "e11"]):
        if is_regular_only:
            return (
                "For EB-1A (Form I-140, E11 / extraordinary ability), the exact current USCIS regular, non-premium "
                "processing time still has to be checked in the USCIS Processing Times tool [2]."
            )
        return (
            "For EB-1A (Form I-140, E11 / extraordinary ability), USCIS premium processing is currently "
            f"15 business days [1].{premium_only_suffix if is_premium_only else generic_suffix}"
        )

    if any(marker in q for marker in ["i-140", "eb-1", "eb1", "eb-2", "eb2", "eb-3", "eb3", "niw", "green card"]):
        if "e13" in q or "multinational manager" in q or "multinational executive" in q:
            if is_regular_only:
                return (
                    "For Form I-140 E13 multinational executive and manager petitions, the exact current USCIS "
                    "regular, non-premium processing time still has to be checked in the USCIS Processing Times tool [2]."
                )
            return (
                "For Form I-140 E13 multinational executive and manager petitions, USCIS premium processing is currently "
                f"45 business days [1].{premium_only_suffix if is_premium_only else generic_suffix}"
            )
        if "niw" in q or "national interest waiver" in q:
            if is_regular_only:
                return (
                    "For Form I-140 E21 national interest waiver petitions, the exact current USCIS regular, "
                    "non-premium processing time still has to be checked in the USCIS Processing Times tool [2]."
                )
            return (
                "For Form I-140 E21 national interest waiver petitions, USCIS premium processing is currently "
                f"45 business days [1].{premium_only_suffix if is_premium_only else generic_suffix}"
            )
        if is_regular_only:
            return (
                "For Form I-140 cases, the exact current USCIS regular, non-premium processing time still has to be "
                "checked in the USCIS Processing Times tool [2]. USCIS does separately publish premium-processing "
                "timelines for many I-140 categories on its premium-processing page [1]."
            )
        return (
            "For many Form I-140 categories, USCIS premium processing is currently 15 business days [1]. "
            "Two important exceptions on the official USCIS page are E13 multinational executive/manager and "
            "E21 national interest waiver, which are currently 45 business days [1]."
            f"{premium_only_suffix if is_premium_only else generic_suffix}"
        )

    if any(marker in q for marker in ["opt", "stem opt", "stem extension", "i-765", "ead", "work permit"]):
        if any(marker in q for marker in ["stem opt", "stem extension", "(c)(3)(c)"]):
            if is_regular_only:
                return (
                    "For F-1 STEM OPT extension Form I-765 cases, the exact current USCIS regular, non-premium "
                    "processing time still has to be checked in the USCIS Processing Times tool [2]. USCIS does "
                    "publish a premium-processing timeline for this category on its premium-processing page [1]."
                )
            return (
                "For F-1 STEM OPT extension requests filed on Form I-765, USCIS premium processing is currently "
                "30 business days [1]. After Form I-765 approval, USCIS says the EAD card should generally be "
                "produced within about two weeks, though delivery time can vary [1]."
                f"{premium_only_suffix if is_premium_only else generic_suffix}"
            )
        if any(marker in q for marker in ["opt", "post completion", "pre completion", "(c)(3)(a)", "(c)(3)(b)"]):
            if is_regular_only:
                return (
                    "For F-1 OPT-related Form I-765 cases, including pre-completion and post-completion OPT, the "
                    "exact current USCIS regular, non-premium processing time still has to be checked in the USCIS "
                    "Processing Times tool [2]. USCIS does publish a premium-processing timeline for this category [1]."
                )
            return (
                "For F-1 OPT-related Form I-765 requests, including pre-completion and post-completion OPT, USCIS "
                "premium processing is currently 30 business days [1]. After Form I-765 approval, USCIS says the "
                "EAD card should generally be produced within about two weeks, though delivery time can vary [1]."
                f"{premium_only_suffix if is_premium_only else generic_suffix}"
            )
        if is_regular_only:
            return (
                "For eligible Form I-765 employment-authorization categories, the exact current USCIS regular, "
                "non-premium processing time still has to be checked in the USCIS Processing Times tool [2]. USCIS "
                "does separately publish premium-processing timelines for eligible I-765 categories [1]."
            )
        return (
            "For eligible Form I-765 employment authorization categories, USCIS premium processing is currently "
            "30 business days [1]. For F-1 OPT and STEM OPT specifically, USCIS says the EAD card is generally "
            "produced within about two weeks after approval, though delivery time can vary [1]."
            f"{premium_only_suffix if is_premium_only else generic_suffix}"
        )

    if any(marker in q for marker in ["h1b", "h-1b", "o-1", "l-1", "r-1", "tn", "i-129"]):
        if is_regular_only:
            return (
                "For H-1B and many other Form I-129 worker classifications, the exact current USCIS regular, "
                "non-premium processing time still has to be checked in the USCIS Processing Times tool [2]. "
                "USCIS separately publishes premium-processing timelines for many I-129 categories [1]."
            )
        return (
            "For many Form I-129 nonimmigrant worker classifications, including H-1B, O-1, L-1, R-1, and TN, USCIS "
            "premium processing is currently 15 business days [1]."
            f"{premium_only_suffix if is_premium_only else generic_suffix}"
        )

    if asks_change_status or any(marker in q for marker in ["f1", "f-1", "f2", "f-2", "m1", "m-1", "j1", "j-1", "i-539"]):
        if is_regular_only:
            return (
                "For eligible Form I-539 change-of-status requests, the exact current USCIS regular, non-premium "
                "processing time still has to be checked in the USCIS Processing Times tool [2]. USCIS separately "
                "publishes a premium-processing timeline for eligible I-539 requests [1]."
            )
        return (
            "For eligible Form I-539 change-of-status requests to F-1, F-2, M-1, M-2, J-1, or J-2, USCIS premium "
            "processing is currently 30 business days once the required prerequisites have been met [1]."
            f"{premium_only_suffix if is_premium_only else generic_suffix}"
        )

    if is_regular_only:
        return regular_only_text

    return (
        "USCIS premium processing times are published on the official USCIS premium-processing page [1], but the "
        "current regular, non-premium processing times are generally available only through the USCIS Processing Times tool [2]. "
        "If you tell me the exact form or category, such as H-1B, I-140, OPT, STEM OPT, or I-539 change of status, "
        "I can narrow this down more precisely."
    )


def _processing_visa_tags(q: str) -> list[str]:
    tags: list[str] = []
    if any(marker in q for marker in ["h1b", "h-1b"]):
        tags.append("H-1B")
    if any(marker in q for marker in ["f1", "f-1"]):
        tags.append("F-1")
    if "opt" in q:
        tags.append("OPT")
    if "stem opt" in q:
        tags.append("STEM_OPT")
    if any(marker in q for marker in ["eb1a", "eb 1a", "eb-1a", "extraordinary ability"]):
        tags.append("EB-1A")
    if any(marker in q for marker in ["ead", "i-765", "work permit"]):
        tags.append("EAD")
    if any(marker in q for marker in ["green card", "i-140", "eb-1", "eb-2", "eb-3", "niw"]):
        tags.append("green_card")
    return list(dict.fromkeys(tags)) or ["general"]


def _format_cutoff(value: str) -> str:
    normalized = (value or "").strip()
    if normalized.upper() == "C":
        return "current"
    return normalized


def _build_citation(source_id: str, source_name: str, source_url: str) -> Citation:
    return Citation(
        source_id=source_id,
        source_name=source_name,
        source_url=source_url,
        section_title=None,
        tier=1,
        tier_label="Official U.S. Government",
        last_updated_on_source=None,
        crawled_at=datetime.now(timezone.utc).isoformat(),
        is_stale=False,
    )


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().replace("-", " ").replace("/", " ").split())


def _mentions_premium_only(q: str) -> bool:
    return "premium" in q and not _mentions_regular_only(q)


def _mentions_regular_only(q: str) -> bool:
    explicit_markers = [
        "regular processing",
        "normal processing",
        "standard processing",
        "non premium",
        "nonpremium",
        "without premium",
    ]
    if any(marker in q for marker in explicit_markers):
        return True

    return (
        any(word in q for word in ["regular", "standard", "normal"])
        and "premium" not in q
    )


def _mentions_change_of_status(q: str) -> bool:
    return any(
        marker in q
        for marker in ["change of status", "cos", "change status"]
    )


def _mentions_consular_processing(q: str) -> bool:
    return any(
        marker in q
        for marker in [
            "consular processing",
            "consular",
            "embassy",
            "consulate",
            "visa stamping",
            "stamping",
            "dropbox",
            "interview wait",
            "221 g",
            "221g",
            "administrative processing",
        ]
    )

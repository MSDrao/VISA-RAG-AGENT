"""Query normalization, classification, and metadata signal extraction."""

import re
import logging
from dataclasses import dataclass
from config.metadata_keywords import PROCESS_KEYWORDS, AGENCY_KEYWORDS

logger = logging.getLogger(__name__)

VISA_KEYWORDS: dict[str, list[str]] = {
    "F-1":      ["f-1", "f1", "f 1", "student visa", "international student",
                 "student status", "full-time student", "f-1 student"],
    "OPT":      ["opt", "optional practical training", "post-completion opt",
                 "pre-completion opt"],
    "STEM_OPT": ["stem opt", "stem extension", "24-month", "24 month extension",
                 "stem 24", "stem opt extension", "i-983", "i983"],
    "CPT":      ["cpt", "curricular practical training", "internship f-1",
                 "cooperative education"],
    "SEVIS":    ["sevis", "i-20", "i20", "student exchange visitor",
                 "sevis fee", "sevis transfer"],
    "H-1B":     ["h-1b", "h1b", "h-1", "h1", "specialty occupation",
                 "lca", "labor condition application", "h1b transfer",
                 "h1b extension", "ac21", "cap-exempt", "h-1b cap",
                 "h1b lottery", "prevailing wage"],
    "H-4":      ["h-4", "h4", "h-4 ead", "h4 ead", "h4 dependent",
                 "h-4 dependent", "spouse of h1b", "spouse of h-1b"],
    "EAD":      ["ead", "employment authorization", "work permit", "i-765",
                 "employment authorization document", "combo card",
                 "ead renewal", "ead auto-extension", "ead card"],
    "B-1":      ["b-1", "b1", "business visitor", "b-1 visa", "business visa"],
    "B-2":      ["b-2", "b2", "tourist visa", "visitor visa", "tourism",
                 "b2 extension", "b-2 extension", "pleasure visit"],
    "O-1":      ["o-1", "o1", "o-1a", "o-1b", "extraordinary ability",
                 "extraordinary achievement"],
    "L-1":      ["l-1", "l1", "l-1a", "l-1b", "intracompany transferee",
                 "intracompany transfer", "multinational manager"],
    "J-1":      ["j-1", "j1", "exchange visitor", "ds-2019", "ds2019",
                 "two-year rule", "home residency requirement", "j1 waiver",
                 "exchange program"],
    "J-2":      ["j-2", "j2", "dependent of j-1", "j1 dependent"],
    "TN":       ["tn visa", "tn status", "nafta professional", "usmca",
                 "canadian professional", "mexican professional", "trade nafta"],
    "E-2":      ["e-2", "e2", "treaty investor", "investor visa", "e-2 visa"],
    "E-1":      ["e-1", "e1", "treaty trader", "e-1 visa"],
    "travel":   ["travel", "re-entry", "reentry", "advance parole", "i-94",
                 "i94", "port of entry", "admission", "travel document",
                 "return to us", "can i travel", "travel outside"],
    "COS":      ["change of status", "cos", "change status",
                 "i-539", "i539", "from f1 to", "from b2 to", "switch status",
                 "change from", "change to"],
    "green_card": ["green card", "lawful permanent resident", "lpr", "i-485",
                   "adjustment of status", "permanent residence",
                   "immigrant visa", "eb-1", "eb-2", "eb-3", "eb1", "eb2", "eb3",
                   "niw", "national interest waiver", "perm", "priority date",
                   "i-140", "i140"],
}

_NORMALIZATIONS: list[tuple[str, str]] = [
    (r"\bh-?1b\b",              "H-1B"),
    (r"\bh-?1\b",               "H-1B"),
    (r"\bf-?1\b",               "F-1"),
    (r"\bf-?2\b",               "F-2"),
    (r"\bh-?2a\b",              "H-2A"),
    (r"\bh-?2b\b",              "H-2B"),
    (r"\bh-?3\b",               "H-3"),
    (r"\bh-?4\b",               "H-4"),
    (r"\bl-?1a?\b",             "L-1"),
    (r"\bl-?1b\b",              "L-1B"),
    (r"\bl-?2\b",               "L-2"),
    (r"\bo-?1[ab]?\b",          "O-1"),
    (r"\bo-?2\b",               "O-2"),
    (r"\bo-?3\b",               "O-3"),
    (r"\bj-?1\b",               "J-1"),
    (r"\bj-?2\b",               "J-2"),
    (r"\be-?1\b",               "E-1"),
    (r"\be-?2\b",               "E-2"),
    (r"\be-?3\b",               "E-3"),
    (r"\bp-?1\b",               "P-1"),
    (r"\br-?1\b",               "R-1"),
    (r"\beb-?1\b",              "EB-1"),
    (r"\beb-?2\b",              "EB-2"),
    (r"\beb-?3\b",              "EB-3"),
    (r"\beb-?4\b",              "EB-4"),
    (r"\beb-?5\b",              "EB-5"),
    (r"\bb-?1\b",               "B-1"),
    (r"\bb-?2\b",               "B-2"),
    (r"\bi[\s-]?20\b",          "I-20"),
    (r"\bi[\s-]?94\b",          "I-94"),
    (r"\bi[\s-]?129\b",         "I-129"),
    (r"\bi[\s-]?131\b",         "I-131"),
    (r"\bi[\s-]?140\b",         "I-140"),
    (r"\bi[\s-]?485\b",         "I-485"),
    (r"\bi[\s-]?539\b",         "I-539"),
    (r"\bi[\s-]?765\b",         "I-765"),
    (r"\bi[\s-]?983\b",         "I-983"),
    (r"\bds[\s-]?160\b",        "DS-160"),
    (r"\bds[\s-]?2019\b",       "DS-2019"),
    (r"\busciss?\b",            "USCIS"),
    (r"\bsevis\b",              "SEVIS"),
    (r"\baproved\b",            "approved"),
    (r"\baproval\b",            "approval"),
    (r"\busally\b",             "usually"),
    (r"\btravell?\b",           "travel"),
    (r"\badjustment[-\s]?based\b", "adjustment of status based"),
    (r"\bfull\s+(form|name|firm|form\??)\b", "definition meaning"),
    (r"\bwats\b",               "what is"),
    (r"\btel+\s+me\b",          "explain"),
]

PROCESS_PATTERNS: dict[str, list[str]] = {
    "Adjustment of Status": [r"\badjustment of status\b", r"\baos\b", r"\badjustment of status based\b"],
    "Change of Status": [r"\bchange of status\b", r"\bcos\b"],
    "Extension of Status": [r"\bextension of status\b", r"\bextend status\b"],
    "Premium Processing": [r"\bpremium processing\b"],
    "USCIS Case Status": [r"\bprocessing time\b", r"\bhow long\b", r"\btimeline\b", r"\bcase status\b"],
    "Advance Parole": [r"\badvance parole\b", r"\bi-131\b", r"\btravel document\b"],
    "Grace Period": [r"\bgrace period\b"],
    "Termination": [r"\bterminated\b", r"\btermination\b", r"\blaid off\b", r"\bfired\b", r"\blayoff\b"],
    "Travel": [r"\btravel\b", r"\bre-?entry\b", r"\bport of entry\b"],
    "PERM": [r"\bperm\b", r"\blabor certification\b"],
    "LCA": [r"\blca\b", r"\blabor condition application\b"],
    "STEM OPT": [r"\bstem opt\b", r"\bstem extension\b"],
    "CPT": [r"\bcpt\b", r"\bcurricular practical training\b"],
    "OPT": [r"\bopt\b", r"\boptional practical training\b"],
}


def _normalize_query(text: str) -> str:
    """Fix typos, shorthand aliases, and common misspellings."""
    for pattern, replacement in _NORMALIZATIONS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


EXPANSIONS: dict[str, str] = {
    r"\btell me about\b":         "overview rules requirements eligibility",
    r"\bexplain\b":               "overview rules requirements explanation",
    r"\bwhat are the rules\b":    "requirements eligibility regulations overview",
    r"\bmeaning of\b":            "definition overview rules requirements",
    r"\bdefinition of\b":         "definition overview meaning",
    r"\bhow does .{1,30} work\b": "process rules requirements steps overview",
    r"\boverall\b":               "overview rules requirements summary",

    r"\bF-1\b":      "F-1 student nonimmigrant visa SEVIS I-20 OPT CPT enrollment status",
    r"\bF-2\b":      "F-2 dependent nonimmigrant spouse child F-1 student",
    r"\bM-1\b":      "M-1 vocational student visa practical training program",
    r"\bJ-1\b":      "J-1 exchange visitor program DS-2019 waiver two-year rule",
    r"\bJ-2\b":      "J-2 dependent spouse child J-1 exchange visitor work authorization",

    r"\bH-1B\b":     "H-1B specialty occupation employer petition I-129 LCA prevailing wage cap",
    r"\bH-2A\b":     "H-2A agricultural worker temporary seasonal employer petition",
    r"\bH-2B\b":     "H-2B nonagricultural seasonal temporary worker cap",
    r"\bH-3\b":      "H-3 trainee special education exchange program",
    r"\bH-4\b":      "H-4 dependent spouse child H-1B nonimmigrant EAD work authorization I-765",
    r"\bL-1\b":      "L-1 intracompany transferee manager executive specialized knowledge",
    r"\bL-1B\b":     "L-1B intracompany transferee specialized knowledge",
    r"\bL-2\b":      "L-2 dependent spouse child L-1 intracompany transferee",
    r"\bO-1\b":      "O-1 extraordinary ability achievement arts sciences business athletics",
    r"\bO-2\b":      "O-2 essential support personnel O-1 extraordinary ability",
    r"\bO-3\b":      "O-3 dependent spouse child O-1 extraordinary ability",
    r"\bP-1\b":      "P-1 internationally recognized athlete entertainer performance",
    r"\bR-1\b":      "R-1 religious worker minister denomination nonprofit",
    r"\bTN\b":       "TN USMCA NAFTA Canada Mexico professional specialty occupation",
    r"\bE-1\b":      "E-1 treaty trader substantial trade international commerce",
    r"\bE-2\b":      "E-2 treaty investor investment enterprise nationality",
    r"\bE-3\b":      "E-3 Australian specialty occupation professional LCA",

    r"\bB-1\b":      "B-1 business visitor temporary conference interview consult",
    r"\bB-2\b":      "B-2 tourist visitor pleasure medical treatment leisure",

    r"\bOPT\b":      "Optional Practical Training OPT F-1 post-completion pre-completion EAD I-765 unemployment 90 days",
    r"\bSTEM OPT\b": "STEM OPT 24-month extension I-983 E-Verify employer unemployment 150 days",
    r"\bCPT\b":      "Curricular Practical Training CPT F-1 authorization I-20 employer academic",
    r"\bEAD\b":      "Employment Authorization Document EAD I-765 work permit card renewal",
    r"\bSEVIS\b":    "Student and Exchange Visitor Information System SEVIS I-20 DS-2019 DSO",

    r"\bEB-1\b":     "EB-1 extraordinary ability multinational manager outstanding professor researcher",
    r"\bEB-2\b":     "EB-2 advanced degree exceptional ability NIW national interest waiver",
    r"\bEB-3\b":     "EB-3 skilled workers professionals unskilled labor certification PERM",
    r"\bEB-4\b":     "EB-4 special immigrants religious workers broadcasters physicians",
    r"\bEB-5\b":     "EB-5 investor entrepreneur capital investment job creation",

    r"\bI-94\b":     "I-94 arrival departure record admission period authorized stay CBP",
    r"\bI-20\b":     "Form I-20 certificate of eligibility F-1 SEVIS student",
    r"\bI-765\b":    "Form I-765 EAD employment authorization document application",
    r"\bI-539\b":    "Form I-539 application extend change nonimmigrant status",
    r"\bI-485\b":    "Form I-485 adjustment of status green card lawful permanent",
    r"\bI-140\b":    "Form I-140 immigrant petition employment-based preference",
    r"\bI-129\b":    "Form I-129 petition nonimmigrant worker H-1B L-1 O-1",
    r"\bI-131\b":    "Form I-131 advance parole travel document reentry",
    r"\bI-983\b":    "Form I-983 training plan STEM OPT employer",
    r"\bDS-2019\b":  "Form DS-2019 certificate of eligibility J-1 exchange visitor",
    r"\bDS-160\b":   "Form DS-160 nonimmigrant visa application consular",
    r"\bLCA\b":      "Labor Condition Application LCA H-1B prevailing wage DOL",
    r"\bDSO\b":      "Designated School Official DSO F-1 I-20 SEVIS",
    r"\bAOS\b":      "Adjustment of Status AOS I-485 green card",
    r"\bCOS\b":      "Change of Status I-539 nonimmigrant",
    r"\bRFE\b":      "Request for Evidence RFE USCIS response deadline",
    r"\bNIW\b":      "National Interest Waiver NIW EB-2 self-petition",
    r"\bAC21\b":     "AC21 H-1B portability job change 180 days I-485 pending",

    # ── Scenario patterns — common situations across visa types ──────────────
    r"\bunemploy\w*\b":           "unemployment days cap 90 150 OPT STEM authorized period F-1",
    r"\blose\b|\blost\b|\blaid off\b|\bterminated\b|\bfired\b":
                                  "termination layoff grace period 60 days status H-1B F-1",
    r"\bgrace period\b":          "grace period 60 days authorized stay F-1 H-1B program end",
    r"\bcap gap\b":               "cap gap H-1B OPT extension status bridge F-1",
    r"\bchange employer\b|\btransfer\b|\bnew employer\b":
                                  "employer change transfer portability H-1B AC21 I-129 OPT SEVIS",
    r"\bextend\b|\bextension\b":  "extension renewal petition I-539 I-129 OPT STEM EAD duration",
    r"\brenewal\b":               "renewal extension EAD OPT I-765 I-539 status",
    r"\btravel\b|\bre-?enter\b":  "travel re-entry admission port of entry visa stamp I-94 document",
    r"\badvance parole\b":        "advance parole travel document I-131 adjustment of status pending",
    r"\bport of entry\b":         "CBP port of entry admission inspection I-94 officer",
    r"\boverstay\b":              "overstay visa expiration status violation unlawful presence bar",
    r"\bout of status\b":         "status violation out of status reinstatement I-539",
    r"\bmaintain status\b":       "maintain status full-time enrollment authorized employment SEVIS",
    r"\bpriority date\b":         "priority date visa bulletin current cutoff EB adjustment",
    r"\bprevailing wage\b":       "prevailing wage LCA H-1B employer DOL requirement",
    r"\bcap exempt\b":            "cap exempt H-1B nonprofit university research institution",
    r"\blottery\b":               "H-1B cap lottery registration selection random",
    r"\bvolunteer\b":             "volunteer work OPT authorized employment relationship",
    r"\bspouse\b|\bwife\b|\bhusband\b":
                                  "dependent spouse H-4 F-2 J-2 L-2 EAD work authorization",
    r"\bdependent\b":             "dependent spouse child H-4 F-2 J-2 L-2 nonimmigrant status",
    r"\bage.?out\b|\bage cutou\w*":"dependent child age out 21 status change protection",
    r"\bwork permit\b":           "Employment Authorization Document EAD I-765 work authorization",
    r"\bhow long\b":              "duration days period timeline processing authorized",
    r"\bdeadline\b":              "filing deadline days period within before after grace",
    r"\bapply for\b|\bhow to apply\b":
                                  "petition application form USCIS filing process steps",
    r"\bprocessing time\b":       "USCIS processing time days months current timeline",
}

# ─────────────────────────────────────────────────────────────────────────────
# OUT OF SCOPE
# ─────────────────────────────────────────────────────────────────────────────
OUT_OF_SCOPE: list[tuple[str, str]] = [
    (r"\bdaca\b",                   "DACA"),
    (r"\bdreamer\b",                "DACA/Dreamer"),
    (r"\basylum\b",                 "asylum"),
    (r"\brefugee\b",                "refugee status"),
    (r"\bremoval\b",                "removal proceedings"),
    (r"\bdeportation\b",            "deportation"),
    (r"\bimmigration court\b",      "immigration court"),
    (r"\bcriminal\b",               "criminal immigration consequences"),
    (r"\bfelony\b",                 "criminal immigration consequences"),
    (r"\bdetention\b",              "immigration detention"),
    (r"\bu.?visa\b",                "U visa"),
    (r"\bt.?visa\b",                "T visa"),
    (r"\bvawa\b",                   "VAWA"),
    (r"\bwithholding of removal\b", "withholding of removal"),
]

# ─────────────────────────────────────────────────────────────────────────────
# ATTORNEY FLAG PATTERNS
# ─────────────────────────────────────────────────────────────────────────────
ATTORNEY_PATTERNS: list[str] = [
    r"should i\b",
    r"what should i do",
    r"my (case|application|petition|status|situation)",
    r"will i (get|be|receive|qualify)",
    r"am i eligible",
    r"can i still (apply|file|get|stay|work)",
    r"i (overstayed|violated|lost|was denied|got an rfe)",
    r"my employer (fired|terminated|withdrew|revoked)",
    r"chances of (approval|getting|winning)",
    r"my (priority date|pd) is",
]

# Intent constants
INTENT_DEFINITION    = "definition"
INTENT_PROCEDURAL    = "procedural"
INTENT_ELIGIBILITY   = "eligibility"
INTENT_COMPARISON    = "comparison"
INTENT_FORM          = "form"
INTENT_TIMELINE      = "timeline"
INTENT_CASE_SPECIFIC = "case_specific"
INTENT_OUT_OF_SCOPE  = "out_of_scope"
INTENT_GENERAL       = "general"


@dataclass
class ProcessedQuery:
    original: str
    expanded: str
    detected_visa_types: list[str]
    detected_agencies: list[str]
    detected_forms: list[str]
    detected_process_terms: list[str]
    intent: str
    is_out_of_scope: bool
    out_of_scope_reason: str | None
    requires_attorney_flag: bool
    metadata_filters: dict


class QueryProcessor:

    def process(
        self,
        question: str,
        conversation_history: list[dict] | None = None,
        visa_type_filter: list[str] | None = None,
        official_sources_only: bool = False,
    ) -> ProcessedQuery:
        # Step 1: normalize typos + shorthand aliases
        q = _normalize_query(question.strip())
        q_lower = q.lower()

        # Step 2: guardrail checks
        scope_reason  = self._check_out_of_scope(q_lower)
        is_oos        = scope_reason is not None
        attorney_flag = self._check_attorney_flag(q_lower)

        # Step 3: detect visa types from normalized query
        detected = self._extract_visa_types(q_lower)
        agencies = self._extract_agencies(q_lower)
        forms = self._extract_forms(q)
        forms = self._infer_forms_from_query(q_lower, forms)
        detected = self._expand_detected_visa_types_from_forms(detected, forms)
        detected = self._apply_query_scenario_inference(q_lower, detected, forms)
        agencies = self._apply_query_agency_inference(q_lower, agencies)
        process_terms = self._extract_process_terms(q)

        # Step 4: rewrite with conversation context + expand abbreviations
        rewritten = self._rewrite_with_context(q, conversation_history)
        expanded  = self._expand_abbreviations(rewritten)

        # Step 5: classify intent
        intent = (
            INTENT_OUT_OF_SCOPE  if is_oos else
            INTENT_CASE_SPECIFIC if attorney_flag else
            self._classify_intent(q_lower)
        )

        # Step 6: build metadata filter
        filters = self._build_filters(
            visa_types=visa_type_filter or detected,
            forms=forms,
            official_only=official_sources_only,
        )

        return ProcessedQuery(
            original=q,
            expanded=expanded,
            detected_visa_types=detected,
            detected_agencies=agencies,
            detected_forms=forms,
            detected_process_terms=process_terms,
            intent=intent,
            is_out_of_scope=is_oos,
            out_of_scope_reason=scope_reason,
            requires_attorney_flag=attorney_flag,
            metadata_filters=filters,
        )

    def _check_out_of_scope(self, q: str) -> str | None:
        for pattern, label in OUT_OF_SCOPE:
            if re.search(pattern, q):
                return label
        return None

    def _check_attorney_flag(self, q: str) -> bool:
        return any(re.search(p, q) for p in ATTORNEY_PATTERNS)

    def _extract_visa_types(self, q: str) -> list[str]:
        detected = []
        for tag, keywords in VISA_KEYWORDS.items():
            if any(kw in q for kw in keywords):
                detected.append(tag)
        return list(dict.fromkeys(detected))

    def _extract_agencies(self, q: str) -> list[str]:
        detected = []
        for agency, keywords in AGENCY_KEYWORDS.items():
            if any(keyword.lower() in q for keyword in keywords):
                detected.append(agency)
        return detected

    def _apply_query_agency_inference(self, q_lower: str, agencies: list[str]) -> list[str]:
        inferred = list(agencies)
        if any(term in q_lower for term in ["visa bulletin", "department of state", "consular", "priority date"]):
            if "DOS" not in inferred:
                inferred.append("DOS")
        if any(term in q_lower for term in ["perm", "labor certification", "lca", "prevailing wage"]):
            if "DOL" not in inferred:
                inferred.append("DOL")
        if any(term in q_lower for term in ["f-1", "opt", "stem opt", "cpt", "sevis"]):
            if "DHS" not in inferred:
                inferred.append("DHS")
        if any(term in q_lower for term in ["i-94", "port of entry", "re-enter", "reentry"]):
            if "CBP" not in inferred:
                inferred.append("CBP")
        return inferred

    def _extract_forms(self, q: str) -> list[str]:
        forms = re.findall(r"\b(?:I|DS)(?:[\s-]?\d{2,4})\b", q, flags=re.IGNORECASE)
        normalized = []
        for form in forms:
            form = form.upper().replace(" ", "")
            if "-" not in form:
                prefix = "DS" if form.startswith("DS") else "I"
                suffix = form[len(prefix):]
                form = f"{prefix}-{suffix}"
            normalized.append(form)
        return list(dict.fromkeys(normalized))

    def _infer_forms_from_query(self, q_lower: str, forms: list[str]) -> list[str]:
        inferred = list(forms)
        scenario_forms = {
            "advance parole": ["I-131"],
            "travel document": ["I-131"],
            "adjustment of status": ["I-485"],
            "adjustment of status based": ["I-485"],
            "work permit": ["I-765"],
        }
        for phrase, mapped_forms in scenario_forms.items():
            if phrase in q_lower:
                for form in mapped_forms:
                    if form not in inferred:
                        inferred.append(form)
        if "ead" in q_lower and "pending" in q_lower and ("travel" in q_lower or "advance parole" in q_lower):
            for form in ["I-765", "I-131", "I-485"]:
                if form not in inferred:
                    inferred.append(form)
        return inferred

    def _expand_detected_visa_types_from_forms(
        self,
        detected: list[str],
        forms: list[str],
    ) -> list[str]:
        inferred: list[str] = list(detected)
        form_to_tags = {
            "I-129": ["H-1B", "O-1", "L-1"],
            "I-140": ["green_card"],
            "I-485": ["green_card"],
            "I-131": ["travel", "green_card"],
            "I-765": ["EAD"],
            "I-539": ["COS"],
            "I-983": ["STEM_OPT"],
            "I-20": ["F-1", "SEVIS"],
            "I-94": ["travel"],
            "DS-2019": ["J-1"],
            "DS-160": ["travel"],
        }
        for form in forms:
            for tag in form_to_tags.get(form, []):
                if tag not in inferred:
                    inferred.append(tag)
        return inferred

    def _apply_query_scenario_inference(
        self,
        q_lower: str,
        detected: list[str],
        forms: list[str],
    ) -> list[str]:
        inferred = list(detected)
        if (
            ("ead" in q_lower or "i-765" in q_lower)
            and "pending" in q_lower
            and ("travel" in q_lower or "advance parole" in q_lower or "i-131" in q_lower)
        ):
            for tag in ["EAD", "travel", "green_card"]:
                if tag not in inferred:
                    inferred.append(tag)
        if "I-140" in forms and "green_card" not in inferred:
            inferred.append("green_card")
        if "h-4 spouse" in q_lower or "h-4" in q_lower:
            for tag in ["H-4", "EAD"]:
                if tag not in inferred:
                    inferred.append(tag)
        if "stem opt" in q_lower and "opt" not in q_lower:
            if "OPT" not in inferred:
                inferred.append("OPT")
        if "cpt" in q_lower and "i-20" in q_lower and "SEVIS" not in inferred:
            inferred.append("SEVIS")
        if "visa bulletin" in q_lower and "green_card" not in inferred:
            inferred.append("green_card")
        return inferred

    def _extract_process_terms(self, q: str) -> list[str]:
        lowered = q.lower()
        detected = [term for term in PROCESS_KEYWORDS if term.lower() in lowered]
        for canonical, patterns in PROCESS_PATTERNS.items():
            if any(re.search(pattern, lowered) for pattern in patterns):
                detected.append(canonical)
        return list(dict.fromkeys(detected))

    def _expand_abbreviations(self, text: str) -> str:
        for pattern, replacement in EXPANSIONS.items():
            try:
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            except re.error:
                pass  # skip malformed patterns gracefully
        return text

    def _rewrite_with_context(self, question: str, history: list[dict] | None) -> str:
        if not history or len(question.split()) >= 8:
            return question
        last_user = next(
            (h["content"] for h in reversed(history) if h["role"] == "user"), None
        )
        if last_user:
            return f"In the context of: '{last_user}' — {question}"
        return question

    def _classify_intent(self, q: str) -> str:
        if re.search(r"\bwhat is\b|\bdefine\b|\bmeaning of\b|\bexplain\b|\bdefinition\b", q):
            return INTENT_DEFINITION
        if re.search(r"\bhow (do|can|to)\b|\bsteps\b|\bprocess\b|\bapply\b|\bfile\b", q):
            return INTENT_PROCEDURAL
        if re.search(r"\bcan i\b|\bam i allowed\b|\beligible\b|\ballowed to\b", q):
            return INTENT_ELIGIBILITY
        if re.search(r"\bdifference\b|\bvs\.?\b|\bcompare\b|\bversus\b", q):
            return INTENT_COMPARISON
        if re.search(r"\bform\b|\bi-\d{2,4}\b|\bds-\d{3,4}\b", q):
            return INTENT_FORM
        if re.search(r"\bhow long\b|\btimeline\b|\bprocessing time\b|\bwhen\b|\bdeadline\b", q):
            return INTENT_TIMELINE
        return INTENT_GENERAL

    def _build_filters(self, visa_types: list[str], forms: list[str], official_only: bool) -> dict:
        conditions = []
        if official_only:
            conditions.append({"tier": {"$eq": 1}})
        retrieval_signals = []
        if visa_types:
            tag_conditions = [{"visa_tags": {"$contains": tag}} for tag in visa_types]
            retrieval_signals.extend(tag_conditions)
        if forms:
            form_conditions = [{"form_numbers": {"$contains": form}} for form in forms]
            retrieval_signals.extend(form_conditions)
        if retrieval_signals:
            if len(retrieval_signals) == 1:
                conditions.append(retrieval_signals[0])
            else:
                conditions.append({"$or": retrieval_signals})
        if not conditions:
            return {}
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}


def rewrite_query_with_llm(question: str) -> str:
    """Optional LLM-based query rewriter (disabled by default). Falls back gracefully."""
    return question

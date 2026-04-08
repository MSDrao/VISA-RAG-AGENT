"""Post-generation validation and safety checks."""

import logging
import re
from generation.response_model import VisaAnswer
from generation.prompt_builder import STANDARD_DISCLAIMER

logger = logging.getLogger(__name__)

HALLUCINATION_RISK_PHRASES = [
    "as of my knowledge cutoff",
    "as of my training data",
    "based on my training",
    "i believe the current rule",
    "i think the policy",
    "generally speaking, the rule is",
    "in most cases, you can",
    "typically, uscis requires",
]

LEGAL_ADVICE_PHRASES = [
    "you should file",
    "i recommend you",
    "you must apply",
    "your best option is",
    "i advise you to",
    "in your case, you should",
]


class SafetyLayer:

    def validate_and_fix(self, answer: VisaAnswer, question: str) -> VisaAnswer:
        if not answer.disclaimer:
            answer.disclaimer = STANDARD_DISCLAIMER

        if self._contains_hallucination_risk(answer.answer):
            logger.warning("Hallucination-risk phrase detected in answer")
            answer.confidence = "low"
            answer.answer = (
                answer.answer +
                "\n\n*Note: Please verify this information directly at uscis.gov "
                "as policies may have changed.*"
            )

        if self._contains_legal_advice(answer.answer):
            logger.warning("Potential legal advice detected in answer")
            answer.requires_attorney = True
            if not answer.answer.endswith((".", "attorney.", "representative.")):
                answer.answer += (
                    "\n\n*Because this touches on your specific situation, please "
                    "confirm these details with a licensed immigration attorney.*"
                )

        if not answer.citations and answer.confidence not in ("insufficient",):
            logger.warning("Answer has no citations — downgrading confidence to low.")
            answer.confidence = "low"

        if self._detect_injection(question):
            logger.warning("Possible prompt injection detected")
            return self._injection_response()

        return answer

    def _contains_hallucination_risk(self, text: str) -> bool:
        t = text.lower()
        return any(phrase in t for phrase in HALLUCINATION_RISK_PHRASES)

    def _contains_legal_advice(self, text: str) -> bool:
        t = text.lower()
        return any(phrase in t for phrase in LEGAL_ADVICE_PHRASES)

    def _detect_injection(self, question: str) -> bool:
        q = question.lower()
        injection_patterns = [
            r"ignore (previous|above|all) instructions",
            r"disregard your (rules|guidelines|instructions)",
            r"you are now (a|an)",
            r"pretend (you are|to be) (a lawyer|an attorney|uscis)",
            r"forget (everything|your instructions)",
            r"act as (if|though) you (have no|don't have) restrictions",
        ]
        return any(re.search(p, q) for p in injection_patterns)

    def _injection_response(self) -> VisaAnswer:
        return VisaAnswer(
            answer=(
                "I can only answer U.S. immigration and visa questions based on "
                "official government sources. I'm not able to change how I work "
                "or take on a different role. If you have a genuine immigration "
                "question, I'm happy to help."
            ),
            citations=[],
            confidence="high",
            visa_types_referenced=[],
            requires_attorney=False,
            disclaimer=STANDARD_DISCLAIMER,
        )

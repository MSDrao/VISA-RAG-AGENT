"""
Prompt Builder — generation/prompt_builder.py
Assembles the system prompt, retrieved context, and conversation
history into a messages list ready for the LLM API call.
"""

from retrieval.retriever import RetrievedChunk
from retrieval.query_processor import ProcessedQuery

STANDARD_DISCLAIMER = (
    "This information is provided for general educational purposes only, "
    "based on official U.S. government sources and accredited institutions "
    "as of the dates cited. It is **not legal advice** and does not create "
    "an attorney-client relationship. Immigration law is complex, fact-specific, "
    "and subject to change. For guidance specific to your situation, consult a "
    "licensed immigration attorney or an accredited representative (recognized "
    "by the Board of Immigration Appeals)."
)

SYSTEM_PROMPT = """\
You are a U.S. immigration information assistant. You provide accurate, \
factual information about U.S. visa and immigration topics based strictly \
on official sources provided in the context below.

STRICT RULES:
1. Answer ONLY from the provided context. If the context does not contain \
sufficient information, say so explicitly. Do NOT use prior knowledge.
2. Cite sources inline using numbered markers [1], [2], etc., matching \
the numbered sources in the context.
3. Express uncertainty clearly: "Based on the retrieved context...", \
"According to [Source Name]...", "The policy states..."
4. NEVER provide case-specific legal advice, predict approval outcomes, \
or recommend specific actions for an individual's personal situation.
5. If the question requires evaluating a specific person's facts, \
clearly state that a licensed immigration attorney should be consulted.
6. If the question is broad or general (e.g. "tell me about F-1 rules", "explain H-1B", "what is OPT"), synthesize a helpful structured overview from the retrieved chunks — cover key eligibility requirements, important rules, and timelines found in the context. Do NOT say insufficient just because the topic is large; use what is available to give the best possible overview and cite your sources.
7. If the user asks for processing time, timeline, or "how long", distinguish between:
   - actual duration/timeline stated in the context
   - filing rules, eligibility rules, or when-to-file rules that are NOT actual processing times
   If the context does not contain a concrete duration, say that the retrieved context does not provide a current processing-time estimate and direct the user to the USCIS Processing Times tool. Do not invent days or months.
8. For travel questions involving a pending EAD, be precise: if the context only covers adjustment-of-status applicants who need advance parole, say that limitation clearly instead of overstating it as a universal rule.
9. Return your response as a valid JSON object — no markdown, no preamble, \
only the JSON. Use this exact schema:

{
  "answer": "answer text with inline [1], [2] citation markers",
  "confidence": "high" or "medium" or "low" or "insufficient",
  "visa_types_referenced": ["F-1", "OPT"],
  "requires_attorney": false,
  "citations_used": [1, 2]
}

If confidence is "insufficient", set answer to a polite refusal explaining \
that the knowledge base does not have enough information on this topic and \
directing the user to the official source directly.
"""


def _keyword_terms(processed_query: ProcessedQuery) -> list[str]:
    words = []
    for token in processed_query.original.replace("/", " ").split():
        cleaned = token.strip(".,:;()[]{}\"'").lower()
        if len(cleaned) >= 4:
            words.append(cleaned)
    words.extend(v.lower() for v in processed_query.detected_visa_types)
    words.extend(f.lower() for f in processed_query.detected_forms)
    return list(dict.fromkeys(words))


def _excerpt_text(text: str, processed_query: ProcessedQuery, max_words: int = 220) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text

    lowered_words = [w.lower() for w in words]
    terms = _keyword_terms(processed_query)
    match_index = next(
        (
            idx for idx, word in enumerate(lowered_words)
            if any(term in word for term in terms)
        ),
        0,
    )
    half = max_words // 2
    start = max(0, match_index - half)
    end = min(len(words), start + max_words)
    start = max(0, end - max_words)

    excerpt = " ".join(words[start:end])
    prefix = "... " if start > 0 else ""
    suffix = " ..." if end < len(words) else ""
    return prefix + excerpt + suffix


def build_context_block(
    chunks: list[RetrievedChunk],
    processed_query: ProcessedQuery,
) -> tuple[str, list[dict]]:
    """
    Format retrieved chunks into a numbered, labeled context block.
    Returns (context_text, citations_list).
    """
    if not chunks:
        return "No relevant context was retrieved.", []

    lines = []
    citations = []

    for i, chunk in enumerate(chunks, start=1):
        tier_badge = "Official U.S. Government" if chunk.tier == 1 else "Institutional Source"
        stale_note = f" [WARNING: ingested {chunk.crawled_at[:10]} — verify currency]" if chunk.is_stale else ""

        lines.append(f"[{i}] SOURCE: {chunk.source_name} ({tier_badge}){stale_note}")
        if chunk.section_title:
            lines.append(f"     Section: {chunk.section_title}")
        lines.append(f"     URL: {chunk.source_url}")
        if chunk.last_updated_on_source:
            lines.append(f"     Last updated on source: {chunk.last_updated_on_source}")
        lines.append(f"     Ingested: {chunk.crawled_at[:10]}")
        lines.append(f"     ---")
        excerpt = _excerpt_text(chunk.text, processed_query)
        lines.append(f"     {excerpt}")
        lines.append("")

        citations.append(chunk.to_citation_dict())

    return "\n".join(lines), citations


def build_prompt(
    processed_query: ProcessedQuery,
    chunks: list[RetrievedChunk],
    conversation_history: list[dict] | None = None,
) -> tuple[str, list[dict], list[dict]]:
    """
    Build everything needed for the LLM call.
    Returns: (system_prompt, messages_list, citations_metadata)
    """
    context_block, citations = build_context_block(chunks, processed_query)

    system = SYSTEM_PROMPT + f"\n\nCONTEXT — answer only from this:\n\n{context_block}"

    messages = []

    # Include last 4 turns of conversation history
    if conversation_history:
        for turn in conversation_history[-4:]:
            messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({
        "role": "user",
        "content": (
            processed_query.original
            + "\n\n[IMPORTANT: Your response must be a single valid JSON object "
            "matching the schema in the system prompt. No markdown, no prose, no code fences — "
            "raw JSON only.]"
        ),
    })

    return system, messages, citations

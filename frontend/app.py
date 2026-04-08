"""Streamlit frontend."""

import os
import uuid

import requests
import streamlit as st

API_BASE = os.getenv("VISA_RAG_API_BASE", "http://localhost:8000/api/v1")

EXAMPLE_QUESTIONS = [
    "What is the 60-day grace period for F-1 students?",
    "How many days can I be unemployed on OPT?",
    "Can I work on STEM OPT while my H-1B is pending?",
    "What happens if I lose my H-1B job?",
    "Can an H-4 spouse work in the United States?",
    "How do I apply for a STEM OPT extension?",
    "Can I travel outside the US while my EAD is pending?",
    "What is cap gap and how does it work?",
]


st.set_page_config(
    page_title="U.S. Immigration Assistant",
    page_icon="🇺🇸",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://www.uscis.gov",
        "Report a bug": None,
        "About": "U.S. Immigration Assistant — powered by official government sources.",
    },
)


if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "official_only" not in st.session_state:
    st.session_state.official_only = True

is_dark = st.session_state.theme == "dark"


T = {
    "dark": dict(
        page_bg="#08111f",
        surface="#0d1728",
        surface_2="#111f35",
        surface_3="#16253d",
        border="#21324b",
        line="#2a3c59",
        text="#e8eef8",
        text_soft="#a6b6cc",
        text_faint="#74859d",
        accent="#d99b2b",
        accent_soft="#f5dfb0",
        accent_2="#5ea3ff",
        success="#48c27d",
        warn="#f0a44b",
        danger="#eb6b62",
        chip_bg="#10253e",
        input_bg="#0d1728",
        sidebar_bg="#0a1322",
        card_shadow="0 18px 60px rgba(0,0,0,0.28)",
        toggle_icon="☀",
        toggle_label="Light",
    ),
    "light": dict(
        page_bg="#f4efe7",
        surface="#fffdf8",
        surface_2="#fbf6ee",
        surface_3="#f5ede2",
        border="#e5d7c5",
        line="#ddceb9",
        text="#1d2838",
        text_soft="#59697d",
        text_faint="#7b8b9e",
        accent="#b36b00",
        accent_soft="#f7e4c1",
        accent_2="#245db7",
        success="#198754",
        warn="#b36b00",
        danger="#c84a3f",
        chip_bg="#f4ebdc",
        input_bg="#fffdf8",
        sidebar_bg="#efe5d7",
        card_shadow="0 20px 45px rgba(104,78,38,0.10)",
        toggle_icon="☾",
        toggle_label="Dark",
    ),
}[st.session_state.theme]


st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {{
  --page-bg: {T["page_bg"]};
  --surface: {T["surface"]};
  --surface-2: {T["surface_2"]};
  --surface-3: {T["surface_3"]};
  --border: {T["border"]};
  --line: {T["line"]};
  --text: {T["text"]};
  --text-soft: {T["text_soft"]};
  --text-faint: {T["text_faint"]};
  --accent: {T["accent"]};
  --accent-soft: {T["accent_soft"]};
  --accent-2: {T["accent_2"]};
  --success: {T["success"]};
  --warn: {T["warn"]};
  --danger: {T["danger"]};
  --chip-bg: {T["chip_bg"]};
  --input-bg: {T["input_bg"]};
  --sidebar-bg: {T["sidebar_bg"]};
  --shadow: {T["card_shadow"]};
}}

html, body, [class*="css"] {{
  font-family: 'IBM Plex Sans', sans-serif;
}}

body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
  background:
    radial-gradient(circle at top left, rgba(94,163,255,0.14), transparent 26%),
    radial-gradient(circle at top right, rgba(217,155,43,0.12), transparent 20%),
    linear-gradient(180deg, var(--page-bg) 0%, var(--page-bg) 100%) !important;
  color: var(--text) !important;
}}

#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] {{
  display: none !important;
}}

[data-testid="stHeader"] {{
  background: transparent !important;
  height: 0 !important;
  min-height: 0 !important;
  padding: 0 !important;
}}

[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"] {{
  display: none !important;
}}

.main .block-container {{
  max-width: 1180px;
  padding-top: 1.25rem;
  padding-bottom: 7rem;
}}

[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, var(--sidebar-bg) 0%, var(--surface) 100%) !important;
  border-right: 1px solid var(--border) !important;
}}

[data-testid="stSidebar"] > div:first-child {{
  padding: 1.25rem 1rem 1.5rem 1rem;
}}

[data-testid="stSidebar"] hr, [data-testid="stMain"] hr {{
  border-color: var(--line) !important;
}}

[data-testid="stChatMessage"] {{
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
  margin-bottom: 1.1rem !important;
}}

[data-testid="chatAvatarIcon-user"] {{
  background: var(--accent) !important;
  color: white !important;
}}

[data-testid="chatAvatarIcon-assistant"] {{
  background: var(--accent-2) !important;
  color: white !important;
}}

[data-testid="stBottomBlockContainer"] {{
  background:
    linear-gradient(180deg, rgba(0,0,0,0) 0%, var(--page-bg) 24%, var(--page-bg) 100%) !important;
  border-top: none !important;
  padding-top: 1rem !important;
  padding-bottom: 1rem !important;
}}

[data-testid="stChatInput"] {{
  background: var(--input-bg) !important;
  border: 1px solid var(--border) !important;
  border-radius: 22px !important;
  box-shadow: var(--shadow) !important;
  max-width: 880px !important;
  margin: 0 auto !important;
}}

[data-testid="stChatInput"]:focus-within {{
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 4px rgba(217,155,43,0.12), var(--shadow) !important;
}}

[data-testid="stChatInput"] textarea {{
  color: var(--text) !important;
  background: transparent !important;
  font-size: 0.95rem !important;
}}

[data-testid="stChatInput"] textarea::placeholder {{
  color: var(--text-faint) !important;
}}

[data-testid="stChatInput"] button {{
  background: var(--accent) !important;
  color: white !important;
  border-radius: 12px !important;
  border: none !important;
}}

[data-testid="stExpander"] {{
  background: var(--surface-2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 16px !important;
  overflow: hidden !important;
}}

[data-testid="stExpander"] summary {{
  color: var(--text-soft) !important;
  font-size: 0.82rem !important;
  font-weight: 600 !important;
}}

[data-testid="stAlert"] {{
  border-radius: 16px !important;
  border: 1px solid var(--border) !important;
  background: var(--surface-2) !important;
}}

.stButton > button {{
  border-radius: 14px !important;
  border: 1px solid var(--border) !important;
  box-shadow: none !important;
}}

.sidebar-card {{
  background: linear-gradient(180deg, var(--surface-2) 0%, var(--surface) 100%);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 1rem;
  box-shadow: var(--shadow);
}}

.shell {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) 320px;
  gap: 1rem;
  align-items: start;
}}

.hero {{
  background:
    linear-gradient(135deg, rgba(217,155,43,0.16), rgba(94,163,255,0.12)),
    linear-gradient(180deg, var(--surface-2), var(--surface));
  border: 1px solid var(--border);
  border-radius: 28px;
  padding: 1.4rem 1.45rem 1.2rem 1.45rem;
  box-shadow: var(--shadow);
  margin-bottom: 1rem;
}}

.hero h1 {{
  font-family: 'Manrope', sans-serif;
  font-size: 2rem;
  line-height: 1.02;
  letter-spacing: -0.04em;
  margin: 0;
  color: var(--text);
}}

.hero p {{
  margin: 0.6rem 0 0 0;
  color: var(--text-soft) !important;
  font-size: 0.98rem;
  line-height: 1.6;
}}

.hero-strip {{
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-top: 1rem;
}}

.chip {{
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.38rem 0.72rem;
  border-radius: 999px;
  background: var(--chip-bg);
  border: 1px solid var(--border);
  color: var(--text-soft);
  font-size: 0.75rem;
  font-weight: 600;
}}

.answer-card {{
  background: linear-gradient(180deg, var(--surface-2) 0%, var(--surface) 100%);
  border: 1px solid var(--border);
  border-radius: 24px;
  padding: 1rem 1rem 0.9rem 1rem;
  box-shadow: var(--shadow);
}}

.user-card {{
  background: linear-gradient(180deg, var(--surface-3) 0%, var(--surface-2) 100%);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 0.9rem 1rem;
  color: var(--text);
}}

.question-label {{
  color: var(--text-faint);
  text-transform: uppercase;
  letter-spacing: 0.09em;
  font-size: 0.68rem;
  font-weight: 700;
  margin-bottom: 0.45rem;
}}

.answer-text {{
  color: var(--text);
  font-size: 0.95rem;
  line-height: 1.78;
  margin-top: 0.8rem;
}}

.utility-card {{
  background: linear-gradient(180deg, var(--surface-2) 0%, var(--surface) 100%);
  border: 1px solid var(--border);
  border-radius: 22px;
  padding: 1rem;
  box-shadow: var(--shadow);
  margin-bottom: 1rem;
}}

.utility-card h3 {{
  font-family: 'Manrope', sans-serif;
  margin: 0 0 0.4rem 0;
  font-size: 1rem;
  color: var(--text);
}}

.utility-card p {{
  margin: 0;
  color: var(--text-soft) !important;
  font-size: 0.82rem;
  line-height: 1.55;
}}

.prompt-button .stButton > button,
[data-testid="stSidebar"] .stButton > button {{
  background: linear-gradient(180deg, var(--surface-3) 0%, var(--surface-2) 100%) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: 16px !important;
  text-align: left !important;
  padding: 0.82rem 0.9rem !important;
  width: 100% !important;
  min-height: 0 !important;
  line-height: 1.45 !important;
}}

.prompt-button .stButton > button:hover,
[data-testid="stSidebar"] .stButton > button:hover {{
  transform: translateY(-1px);
  border-color: var(--accent) !important;
}}

.feedback-row .stButton > button {{
  background: transparent !important;
  color: var(--text-soft) !important;
  border: 1px solid var(--border) !important;
}}

.feedback-row .stButton > button:hover {{
  background: var(--surface-3) !important;
  color: var(--text) !important;
}}

.header-actions .stButton > button {{
  min-width: 120px !important;
  background: linear-gradient(180deg, var(--surface-3) 0%, var(--surface-2) 100%) !important;
  color: var(--text) !important;
}}

@media (max-width: 1100px) {{
  .shell {{
    grid-template-columns: 1fr;
  }}
}}
</style>
""",
    unsafe_allow_html=True,
)


def _confidence_badge(conf: str) -> str:
    styles = {
        "high": ("background:rgba(72,194,125,0.14);color:#4fd089;border:1px solid rgba(72,194,125,0.32);", "High confidence"),
        "medium": ("background:rgba(240,164,75,0.14);color:#f0b35f;border:1px solid rgba(240,164,75,0.3);", "Moderate confidence"),
        "low": ("background:rgba(94,163,255,0.14);color:#7ab2ff;border:1px solid rgba(94,163,255,0.3);", "Low confidence"),
        "insufficient": ("background:rgba(123,139,158,0.14);color:#9eb0c4;border:1px solid rgba(123,139,158,0.28);", "Insufficient context"),
    }
    style, label = styles.get(conf, styles["low"])
    return (
        f'<span style="display:inline-flex;align-items:center;gap:0.42rem;'
        f'padding:0.38rem 0.72rem;border-radius:999px;font-size:0.74rem;'
        f'font-weight:700;letter-spacing:0.01em;{style}">{label}</span>'
    )


def _send_feedback(question: str, answer: str, rating: str):
    try:
        requests.post(
            f"{API_BASE}/feedback",
            json={
                "session_id": st.session_state.session_id,
                "question": question,
                "answer": answer,
                "rating": rating,
            },
            timeout=5,
        )
        st.toast("Feedback recorded", icon="✅")
    except Exception:
        pass


def _render_answer(data: dict, meta: dict, msg_idx: int, question: str):
    st.markdown('<div class="answer-card">', unsafe_allow_html=True)
    st.markdown(_confidence_badge(data.get("confidence", "low")), unsafe_allow_html=True)
    st.markdown(
        f'<div class="answer-text">{data.get("answer", "")}</div>',
        unsafe_allow_html=True,
    )

    if data.get("requires_attorney"):
        st.warning(
            "This question may require a licensed immigration attorney or accredited representative for case-specific guidance.",
            icon="⚖️",
        )

    if data.get("freshness_warning"):
        st.warning(data["freshness_warning"], icon="⏰")

    citations = data.get("citations", [])
    if citations:
        with st.expander(f"Sources ({len(citations)})", expanded=False):
            for i, c in enumerate(citations, 1):
                st.markdown(
                    f"""
<div style="padding:0.45rem 0 {'0.75rem' if i < len(citations) else '0.1rem'} 0;
border-bottom:{'1px solid var(--border)' if i < len(citations) else 'none'}">
  <div style="font-weight:700;color:var(--text);font-size:0.82rem;">[{i}] {c.get('source_name', '')}</div>
  <div style="font-size:0.76rem;color:var(--text-soft);margin-top:0.18rem;">{c.get('section_title') or 'General'}</div>
  <div style="font-size:0.76rem;margin-top:0.22rem;">
    <a href="{c.get('source_url','')}" target="_blank" style="color:var(--accent-2);text-decoration:none;word-break:break-all;">
      {c.get('source_url','')}
    </a>
  </div>
</div>
""",
                    unsafe_allow_html=True,
                )

    with st.expander("Disclaimer", expanded=False):
        st.markdown(
            f'<div style="color:var(--text-soft);font-size:0.78rem;line-height:1.65;">{data.get("disclaimer", "")}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="feedback-row">', unsafe_allow_html=True)
    c1, c2, _ = st.columns([1, 1.2, 5])
    with c1:
        if st.button("Helpful", key=f"up_{msg_idx}"):
            _send_feedback(question, data.get("answer", ""), "helpful")
    with c2:
        if st.button("Not helpful", key=f"down_{msg_idx}"):
            _send_feedback(question, data.get("answer", ""), "not_helpful")
    st.markdown("</div></div>", unsafe_allow_html=True)


with st.sidebar:
    st.markdown(
        """
<div class="sidebar-card">
  <div style="display:flex;align-items:center;justify-content:space-between;gap:0.8rem;">
    <div>
      <div style="font-family:'Manrope',sans-serif;font-size:1.15rem;font-weight:800;color:var(--text);">Immigration Desk</div>
      <div style="font-size:0.78rem;color:var(--text-soft);margin-top:0.22rem;">Grounded answers from official sources and selected institutional guidance.</div>
    </div>
    <div style="font-size:1.5rem;">🇺🇸</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="utility-card"><h3>Search Scope</h3><p>Use Tier 1 government-only retrieval for stricter answers, or include Tier 2 institutional guidance for practical student-status questions.</p></div>', unsafe_allow_html=True)
    st.session_state.official_only = st.toggle(
        "Official sources only",
        value=st.session_state.official_only,
        help="On: Tier 1 government sources only. Off: Tier 1 + Tier 2 sources.",
    )

    st.markdown('<div class="utility-card"><h3>Try These</h3><p>These prompts are useful for checking retrieval coverage across student, worker, and travel flows.</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="prompt-button">', unsafe_allow_html=True)
    for eq in EXAMPLE_QUESTIONS:
        if st.button(eq, key=f"ex_{eq[:24]}", use_container_width=True):
            st.session_state.pending_question = eq
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Clear conversation", use_container_width=True):
        try:
            requests.delete(f"{API_BASE}/session/{st.session_state.session_id}", timeout=5)
        except Exception:
            pass
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        """
<div class="utility-card">
  <h3>Boundaries</h3>
  <p>Not legal advice. For case-specific decisions, filings, or risk analysis, use a licensed immigration attorney or accredited representative.</p>
</div>
""",
        unsafe_allow_html=True,
    )


right_col, side_col = st.columns([4.2, 1.45], gap="large")

with right_col:
    header_left, header_right = st.columns([6, 1.25])
    with header_right:
        st.markdown('<div class="header-actions">', unsafe_allow_html=True)
        if st.button(f"{T['toggle_icon']} {T['toggle_label']}", use_container_width=True):
            st.session_state.theme = "light" if is_dark else "dark"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        f"""
<div class="hero">
  <div>
    <h1>Clearer Answers For Visa, Status, And Travel Questions</h1>
    <p>Ask about F-1, OPT, STEM OPT, CPT, H-1B, EAD, B-1/B-2, travel, re-entry, and employment-based filings. Answers are grounded in official U.S. immigration sources, with optional Tier 2 support for practical institutional guidance.</p>
  </div>
  <div class="hero-strip">
    <span class="chip">Tier {'1 only' if st.session_state.official_only else '1 + 2'}</span>
    <span class="chip">Chroma-backed retrieval</span>
    <span class="chip">Cited responses</span>
    <span class="chip">Rate-limited API</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="user-card"><div class="question-label">Question</div>{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                _render_answer(data=msg["data"], meta=msg, msg_idx=msg["idx"], question=msg["question"])

with side_col:
    st.markdown(
        """
<div class="utility-card">
  <h3>Retrieval Notes</h3>
  <p><strong>Tier 1</strong> uses official government sources such as USCIS, DOS, CBP, DHS/ICE-SEVP, and DOL. <strong>Tier 2</strong> adds selected university international-office guidance, which is often useful for clearer F-1 and OPT explanations.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div class="utility-card">
  <h3>Best Query Style</h3>
  <p>Questions are most accurate when they include the <strong>visa type</strong>, <strong>form number</strong>, and <strong>process stage</strong>. For example: <strong>H-1B layoff grace period</strong> or <strong>I-140 premium processing time</strong>.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div class="utility-card">
  <h3>Use Cases</h3>
  <p>This assistant is best for policy lookup, process guidance, and finding the right official sources. It is less reliable for individualized legal strategy or case-specific risk decisions.</p>
</div>
""",
        unsafe_allow_html=True,
    )


pending = st.session_state.pop("pending_question", None) if "pending_question" in st.session_state else None
prompt = st.chat_input("Ask an immigration question with visa type, form, or process stage...") or pending

if prompt:
    with right_col:
        with st.chat_message("user"):
            st.markdown(
                f'<div class="user-card"><div class="question-label">Question</div>{prompt}</div>',
                unsafe_allow_html=True,
            )

        with st.chat_message("assistant"):
            with st.spinner("Searching immigration sources..."):
                try:
                    resp = requests.post(
                        f"{API_BASE}/query",
                        json={
                            "question": prompt,
                            "session_id": st.session_state.session_id,
                            "official_sources_only": st.session_state.official_only,
                        },
                        timeout=60,
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    data = result.get("answer_data", {})
                    meta = {
                        "processing_time_ms": result.get("processing_time_ms", 0),
                        "retrieved_chunk_count": result.get("retrieved_chunk_count", 0),
                    }
                    idx = len(st.session_state.messages)
                    _render_answer(data=data, meta=meta, msg_idx=idx, question=prompt)

                    st.session_state.messages.append({"role": "user", "content": prompt})
                    st.session_state.messages.append({
                        "role": "assistant",
                        "data": data,
                        "question": prompt,
                        "idx": idx,
                        **meta,
                    })

                except requests.exceptions.ConnectionError:
                    st.error(
                        "Cannot connect to the API. Start it with:\n"
                        "```\nuvicorn api.main:app --host 127.0.0.1 --port 8000\n```"
                    )
                except requests.exceptions.Timeout:
                    st.error("The request timed out. Try a narrower question.")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

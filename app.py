"""
Nagarik Mitra — Streamlit UI.

Tabs (per spec section 6/9):
  1. Classify   — paste a complaint, hit Classify, see language/domain/
                   department/priority/SLA/confidence/needs_human + the
                   drafted acknowledgment reply (template, optionally
                   LLM-polished) and similar past tickets.
  2. Inbox      — officer view of stored tickets with override controls
                   (writes an audit row via nagrikmitra.store.override_ticket
                   and recomputes due_at).
  3. CSV        — bulk-upload a CSV of complaints, run predict() over all
                   rows, download results.
  4. Analytics  — volume by domain/priority/language, SLA breach counts.
  5. Metrics    — renders reports/metrics.json (accuracy/F1, keyword
                   baseline, language slices, high-priority recall,
                   memorization disclaimer) — never overstate as MuRIL.

Visual identity: navy #1a2744 / #111c33, saffron #E8751A, civic green
#1B7A6E, bilingual (Hindi + English) copy throughout.

Reload the page between test tickets; tab out of the textarea before
hitting Classify (Streamlit state quirk noted in the acceptance tests).
"""
import json
from datetime import datetime

import pandas as pd
import streamlit as st

from nagrikmitra import store
from nagrikmitra.config import METRICS_JSON_PATH
from nagrikmitra.llm import polish
from nagrikmitra.predict import predict
from nagrikmitra.reply import render_template
from nagrikmitra.similar import similar_tickets
from nagrikmitra.taxonomy import DOMAINS, PRIORITIES

NAVY = "#1a2744"
NAVY_DARK = "#111c33"
SAFFRON = "#E8751A"
GREEN = "#1B7A6E"

st.set_page_config(page_title="Nagarik Mitra | नागरिक मित्र", page_icon="🏛️", layout="wide")

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {NAVY_DARK}; }}
    section[data-testid="stSidebar"] {{ background-color: {NAVY}; }}
    h1, h2, h3 {{ color: {SAFFRON}; }}
    .nm-badge {{
        display: inline-block; padding: 2px 10px; border-radius: 12px;
        color: white; font-size: 0.85em; margin-right: 6px;
    }}
    .nm-high {{ background-color: #C0392B; }}
    .nm-medium {{ background-color: {SAFFRON}; }}
    .nm-low {{ background-color: {GREEN}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("नागरिक मित्र — Nagarik Mitra")
st.caption(
    "Civic grievance routing engine — MuRIL-style pipeline (v1 = TF-IDF; MuRIL planned). "
    "B.Tech CSE (AI/ML) demo project, not a CPGRAMS replacement."
)

tabs = st.tabs(["✍️ Classify", "📥 Inbox", "📄 CSV", "📊 Analytics", "📈 Metrics"])


def _priority_badge(priority: str) -> str:
    cls = {"High": "nm-high", "Medium": "nm-medium", "Low": "nm-low"}.get(priority, "nm-medium")
    return f'<span class="nm-badge {cls}">{priority}</span>'


# ---------------------------------------------------------------- Classify
with tabs[0]:
    st.subheader("Classify a complaint / शिकायत दर्ज करें")
    text = st.text_area(
        "Complaint text / शिकायत का विवरण",
        height=120,
        placeholder="e.g. bijli transformer se sparking ho rahi hai, khatarnak hai...",
        key="classify_text",
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        channel = st.selectbox("Channel", ["web", "app", "whatsapp", "phone"], key="classify_channel")
    with col2:
        use_llm = st.checkbox("Polish reply with LLM (optional)", value=False, key="classify_use_llm")
    with col3:
        persist = st.checkbox("Save to inbox", value=True, key="classify_persist")

    if st.button("Classify / वर्गीकृत करें", type="primary", key="classify_button"):
        if not text or not text.strip():
            st.warning("Please enter complaint text.")
        else:
            result = predict(text, channel=channel)

            template_reply = render_template(
                language=result["language"],
                domain=result["domain"],
                priority=result["priority"],
                sla_hours=result["sla_hours"],
                department=result["department"],
            )
            suggested_reply = template_reply
            llm_used = False
            llm_backend = None
            llm_error = None
            if use_llm:
                suggested_reply, llm_used, llm_backend, llm_error = polish(
                    template_reply, result["sla_hours"], result["language"]
                )

            if persist:
                store.save_ticket(
                    text=text,
                    language=result["language"],
                    domain=result["domain"],
                    department=result["department"],
                    priority=result["priority"],
                    confidence=result["confidence"],
                    needs_human=result["needs_human"],
                    sla_hours=result["sla_hours"],
                    source="ui",
                    channel=channel,
                )

            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Language", result["language"])
            c2.metric("Domain", result["domain"])
            c3.metric("Department", result["department"])
            c4.metric("Confidence", f"{result['confidence']:.2f}")

            st.markdown(
                f"Priority: {_priority_badge(result['priority'])} &nbsp;&nbsp; "
                f"SLA: **{result['sla_hours']} hours** &nbsp;&nbsp; "
                f"Needs human review: **{'Yes' if result['needs_human'] else 'No'}**",
                unsafe_allow_html=True,
            )

            st.markdown("#### Drafted acknowledgment / मसौदा प्रतिक्रिया")
            st.info(suggested_reply)
            if use_llm:
                if llm_used:
                    st.caption(f"✅ LLM-polished via {llm_backend}")
                else:
                    st.caption(f"⚠️ LLM polish unavailable ({llm_error}); using template reply.")

            with st.expander("Why this classification? / व्याख्या"):
                st.write("Domain top-3:", result["domain_top3"])
                st.write("Priority top-3:", result["priority_top3"])

            with st.expander("Similar past tickets / समान शिकायतें"):
                for item in similar_tickets(text, top_k=5):
                    st.write(f"[{item['domain']}] ({item['score']:.2f}) {item['text']}")

# -------------------------------------------------------------------- Inbox
with tabs[1]:
    st.subheader("Officer Inbox / अधिकारी इनबॉक्स")

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        filter_domain = st.selectbox("Domain filter", ["(all)"] + DOMAINS, key="inbox_domain")
    with f2:
        filter_priority = st.selectbox("Priority filter", ["(all)"] + PRIORITIES, key="inbox_priority")
    with f3:
        filter_needs_human = st.selectbox("Needs human?", ["(all)", "Yes", "No"], key="inbox_needs_human")
    with f4:
        if st.button("Refresh", key="inbox_refresh"):
            st.rerun()

    filters = {}
    if filter_domain != "(all)":
        filters["domain"] = filter_domain
    if filter_priority != "(all)":
        filters["priority"] = filter_priority
    if filter_needs_human != "(all)":
        filters["needs_human"] = filter_needs_human == "Yes"

    tickets = store.list_tickets(filters)
    if not tickets:
        st.info("No tickets yet. Classify a complaint with 'Save to inbox' checked.")
    for t in tickets:
        with st.expander(f"#{t['id']} — {t['domain']} / {t['priority']} — {t['text'][:60]}"):
            st.write(t["text"])
            st.write(
                f"Language: {t['language']} | Department: {t['department']} | "
                f"SLA: {t['sla_hours']}h | Due: {t['due_at']} | "
                f"Needs human: {'Yes' if t['needs_human'] else 'No'} | Status: {t['status']}"
            )
            oc1, oc2, oc3 = st.columns(3)
            with oc1:
                new_domain = st.selectbox(
                    "New domain", DOMAINS, index=DOMAINS.index(t["domain"]) if t["domain"] in DOMAINS else 0,
                    key=f"override_domain_{t['id']}",
                )
            with oc2:
                new_priority = st.selectbox(
                    "New priority", PRIORITIES,
                    index=PRIORITIES.index(t["priority"]) if t["priority"] in PRIORITIES else 0,
                    key=f"override_priority_{t['id']}",
                )
            with oc3:
                actor = st.text_input("Officer name", value="officer", key=f"override_actor_{t['id']}")
            reason = st.text_input("Override reason", key=f"override_reason_{t['id']}")
            if st.button("Apply override", key=f"override_apply_{t['id']}"):
                store.override_ticket(t["id"], new_domain, new_priority, reason, actor)
                st.success("Override saved.")
                st.rerun()

# ---------------------------------------------------------------------- CSV
with tabs[2]:
    st.subheader("Bulk classify from CSV / थोक अपलोड")
    st.caption("CSV must have a 'text' column. Optional 'channel' column.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="csv_uploader")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        if "text" not in df.columns:
            st.error("CSV must contain a 'text' column.")
        else:
            if st.button("Run classification on all rows", key="csv_run"):
                results = []
                progress = st.progress(0)
                for i, row in df.iterrows():
                    channel = row.get("channel", "web") if "channel" in df.columns else "web"
                    r = predict(str(row["text"]), channel=channel)
                    results.append(
                        {
                            "text": row["text"],
                            "language": r["language"],
                            "domain": r["domain"],
                            "department": r["department"],
                            "priority": r["priority"],
                            "confidence": r["confidence"],
                            "needs_human": r["needs_human"],
                            "sla_hours": r["sla_hours"],
                        }
                    )
                    progress.progress((i + 1) / len(df))
                result_df = pd.DataFrame(results)
                st.dataframe(result_df)
                st.download_button(
                    "Download results CSV",
                    result_df.to_csv(index=False).encode("utf-8"),
                    file_name="nagrikmitra_results.csv",
                    mime="text/csv",
                )

# ---------------------------------------------------------------- Analytics
with tabs[3]:
    st.subheader("Analytics / विश्लेषण")
    tickets = store.list_tickets({})
    if not tickets:
        st.info("No tickets yet.")
    else:
        df = pd.DataFrame(tickets)
        a1, a2, a3 = st.columns(3)
        with a1:
            st.write("Volume by domain")
            st.bar_chart(df["domain"].value_counts())
        with a2:
            st.write("Volume by priority")
            st.bar_chart(df["priority"].value_counts())
        with a3:
            st.write("Volume by language")
            st.bar_chart(df["language"].value_counts())

        now = datetime.utcnow()
        df["due_at_dt"] = pd.to_datetime(df["due_at"])
        breaches = df[(df["status"] != "resolved") & (df["due_at_dt"] < now)]
        st.metric("SLA breaches (open + overdue)", len(breaches))
        st.metric("Total tickets", len(df))

# ------------------------------------------------------------------ Metrics
with tabs[4]:
    st.subheader("Model evaluation / मॉडल मूल्यांकन")
    st.caption(
        "v1's encoder is TF-IDF + sklearn, not MuRIL. Fine-tuning MuRIL is "
        "planned Week 2+ work — never presented as 'powered by MuRIL'."
    )
    if not METRICS_JSON_PATH.exists():
        st.warning("reports/metrics.json not found. Run `python -m nagrikmitra.train` first.")
    else:
        with open(METRICS_JSON_PATH, "r", encoding="utf-8") as fh:
            metrics = json.load(fh)

        m1, m2 = st.columns(2)
        with m1:
            st.metric("Domain accuracy", metrics["domain_head"]["test_accuracy"])
            st.metric("Domain macro-F1", metrics["domain_head"]["test_macro_f1"])
        with m2:
            st.metric("Priority accuracy", metrics["priority_head"]["test_accuracy"])
            st.metric("Priority macro-F1", metrics["priority_head"]["test_macro_f1"])

        st.metric("Keyword baseline (domain)", metrics["keyword_baseline"]["domain_accuracy"])
        st.metric("High-priority recall", metrics["high_priority_recall"])

        st.write("Language slices:")
        st.json(metrics["language_slices"])

        st.warning(metrics["disclaimer"])

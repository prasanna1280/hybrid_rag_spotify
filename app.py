import json
from pathlib import Path

import streamlit as st

from config import EVAL_THRESHOLD, EVAL_MAX_CASES, DEEPEVAL_MODEL
from hybrid_rag import HybridRAG


st.set_page_config(
    page_title="Architecture Intelligence Assistant",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- Styling ----------
st.markdown(
    """
<style>
:root { --accent: #1ed760; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
.hero {
  padding: 1.35rem 1.5rem;
  border: 1px solid rgba(128,128,128,.25);
  border-radius: 18px;
  background: linear-gradient(135deg, rgba(30,215,96,.13), rgba(80,90,120,.12));
  margin-bottom: 1rem;
}
.hero h1 { margin: 0; font-size: 2.05rem; }
.hero p { margin: .35rem 0 0; opacity: .78; }
.metric-card {
  padding: 1rem;
  border: 1px solid rgba(128,128,128,.22);
  border-radius: 14px;
  min-height: 105px;
}
.metric-label { font-size: .78rem; opacity: .68; text-transform: uppercase; letter-spacing: .06em; }
.metric-value { font-size: 1.55rem; font-weight: 700; margin-top: .25rem; }
.answer-card {
  padding: 1.25rem 1.35rem;
  border-left: 4px solid #1ed760;
  border-radius: 12px;
  background: rgba(128,128,128,.08);
  margin: .5rem 0 1rem;
}
.status {
  display: inline-block;
  padding: .25rem .65rem;
  border-radius: 999px;
  font-size: .78rem;
  font-weight: 700;
}
.status-pass { background: rgba(30,215,96,.16); color: #159447; }
.status-warn { background: rgba(255,180,0,.16); color: #a66a00; }
.status-fail { background: rgba(230,70,70,.16); color: #b32f2f; }
.small-note { font-size: .82rem; opacity: .68; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_rag():
    return HybridRAG()


def load_latest_eval():
    path = Path("data/eval_latest.json")
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def pct(value):
    return f"{value * 100:.1f}%"


def status_class(value):
    if value >= EVAL_THRESHOLD:
        return "status-pass"
    if value >= 0.60:
        return "status-warn"
    return "status-fail"


def status_text(value):
    return "PASS" if value >= EVAL_THRESHOLD else "BELOW TARGET"


rag = load_rag()
latest = load_latest_eval()

# ---------- Sidebar ----------
st.sidebar.markdown("## 🎧 Architecture AI")
st.sidebar.caption("Hybrid RAG • Neo4j • FAISS • Self-RAG • DeepEval")
st.sidebar.divider()

st.sidebar.markdown("### Knowledge Base")
st.sidebar.success("Spotify Architecture PDF loaded")
st.sidebar.caption("Page-aware chunks + vector index + Neo4j graph")

st.sidebar.markdown("### Evaluation Target")
st.sidebar.metric("Pass threshold", pct(EVAL_THRESHOLD))
if latest:
    score = latest.get("scores", {}).get("overall_score", 0)
    st.sidebar.metric("Last overall score", pct(score), status_text(score))
else:
    st.sidebar.info("No DeepEval report yet. Run it from the Deep Evaluation tab.")

st.sidebar.markdown("### Runtime")
st.sidebar.caption(f"Chat model: {DEEPEVAL_MODEL if latest else 'configured in .env'}")
st.sidebar.caption(f"Default eval cases: {EVAL_MAX_CASES}")

# ---------- Hero ----------
st.markdown(
    """
<div class="hero">
  <h1>🎧 Architecture Intelligence Assistant</h1>
  <p>Ask grounded questions about the Spotify-like web application architecture and inspect how FAISS, Neo4j, Hybrid Retrieval, Self-RAG and DeepEval work together.</p>
</div>
""",
    unsafe_allow_html=True,
)

# ---------- Tabs ----------
tab_ask, tab_graph, tab_eval, tab_about = st.tabs(
    ["💬 Ask AI", "🕸️ Knowledge Graph", "📊 Deep Evaluation", "ℹ️ Architecture"]
)

with tab_ask:
    examples = [
        "Which database stores users, profiles, subscriptions and playlists?",
        "Which technologies are used for full-text search and autocomplete?",
        "How does the Media Processor Service work?",
        "How are media files moved from processing to delivery?",
    ]

    st.markdown("#### Try an architecture question")
    selected = st.selectbox("Examples", ["— Select an example —"] + examples, label_visibility="collapsed")
    default_question = "" if selected.startswith("—") else selected
    question = st.text_area(
        "Question",
        value=default_question,
        height=105,
        placeholder="e.g. Which services use PostgreSQL, and what information do they manage?",
        label_visibility="collapsed",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        ask_clicked = st.button("🔎 Ask AI", type="primary", use_container_width=True)
    with col2:
        st.caption("Answers are restricted to the supplied PDF. Prompt-injection requests are blocked.")

    if ask_clicked and question.strip():
        with st.spinner("Hybrid retrieval → context fusion → Self-RAG verification..."):
            result = rag.ask(question)
        st.session_state["last_result"] = result
        st.session_state["last_question"] = question

    result = st.session_state.get("last_result")
    if result:
        st.markdown("### Answer")
        st.markdown(f'<div class="answer-card">{result["answer"]}</div>', unsafe_allow_html=True)

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric("Retrieved chunks", len(result.get("sources", [])))
        with s2:
            pages = sorted({s.get("page") for s in result.get("sources", []) if s.get("page") is not None})
            st.metric("Evidence pages", len(pages))
        with s3:
            review = result.get("self_rag_review", {})
            st.metric("Self-RAG", "Grounded" if review.get("grounded") else "Needs review")
        with s4:
            st.metric("Revised", "Yes" if result.get("revised_once") else "No")

        c1, c2 = st.columns(2)
        with c1:
            with st.expander("📚 Retrieved evidence", expanded=False):
                for i, source in enumerate(result.get("sources", []), 1):
                    st.markdown(f"**Source {i} — Page {source.get('page')} — {source.get('chunk_id')}**")
                    st.caption(
                        f"Vector: {source.get('vector_score', 0):.6f}  |  "
                        f"Graph: {source.get('graph_score', 0):.6f}  |  "
                        f"Fusion: {source.get('fusion_score', 0):.6f}"
                    )
                    st.write(source.get("text", ""))
                    st.divider()
        with c2:
            with st.expander("🧠 Self-RAG review", expanded=False):
                st.json(result.get("self_rag_review", {}))
            with st.expander("🛡️ Guardrail / retrieval payload", expanded=False):
                st.json({
                    "blocked": result.get("blocked"),
                    "citations": result.get("citations"),
                    "retrieved_pages": [s.get("page") for s in result.get("sources", [])],
                })

with tab_graph:
    st.markdown("### Knowledge Graph Explorer")
    st.caption("Neo4j stores document chunks, extracted entities and explicit relationships.")
    try:
        stats = rag.graph.stats()
        if stats:
            cols = st.columns(min(len(stats), 4))
            for idx, item in enumerate(stats[:4]):
                with cols[idx]:
                    st.metric(item.get("label") or "Node", item.get("count", 0))
            st.dataframe(stats, use_container_width=True, hide_index=True)
        else:
            st.info("Neo4j is reachable, but the graph currently has no nodes.")
    except Exception as exc:
        st.error(f"Neo4j graph status could not be loaded: {exc}")

with tab_eval:
    st.markdown("### 📊 Deep Evaluation Dashboard")
    st.caption(
        "The 70% line is a measurable pass/target threshold. Scores shown here are computed by DeepEval; they are not hard-coded."
    )

    eval_cases = st.slider("Evaluation cases", min_value=5, max_value=24, value=min(EVAL_MAX_CASES, 20), step=1)
    run_eval = st.button("▶ Run DeepEval", type="primary")

    if run_eval:
        progress = st.progress(0, text="Starting evaluation...")
        status = st.empty()
        try:
            from deep_eval_runner import run_evaluation

            def on_progress(done, total, row):
                progress.progress(done / total, text=f"Evaluating {done}/{total}: {row['category']}")
                status.info(f"Current case: {row['question']}")

            report = run_evaluation(eval_cases, on_progress)
            progress.progress(1.0, text="Evaluation complete")
            status.empty()
            st.session_state["latest_eval"] = report
            latest = report
            st.success("DeepEval run completed and saved to data/eval_latest.json")
        except Exception as exc:
            progress.empty()
            st.error(f"DeepEval failed: {exc}")
            st.stop()

    latest = st.session_state.get("latest_eval") or load_latest_eval()
    if latest:
        scores = latest.get("scores", {})
        overall = scores.get("overall_score", 0)
        retrieval = scores.get("retrieval_score", 0)
        generation = scores.get("generation_score", 0)
        faithfulness = scores.get("faithfulness", 0)

        cards = st.columns(4)
        for col, label, value in [
            (cards[0], "Overall RAG", overall),
            (cards[1], "Retrieval Quality", retrieval),
            (cards[2], "Generation Quality", generation),
            (cards[3], "Faithfulness", faithfulness),
        ]:
            with col:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">{label}</div>'
                    f'<div class="metric-value">{pct(value)}</div>'
                    f'<span class="status {status_class(value)}">{status_text(value)}</span></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("#### Metric breakdown")
        metric_rows = [
            ("Answer Relevancy", scores.get("answer_relevancy", 0)),
            ("Faithfulness", scores.get("faithfulness", 0)),
            ("Contextual Relevancy", scores.get("contextual_relevancy", 0)),
            ("Contextual Precision", scores.get("contextual_precision", 0)),
            ("Contextual Recall", scores.get("contextual_recall", 0)),
            ("Retrieval Page Hit Rate", scores.get("page_hit_rate", 0)),
            ("Case Pass Rate", scores.get("case_pass_rate", 0)),
        ]
        for label, value in metric_rows:
            c1, c2, c3 = st.columns([2.2, 5, 1])
            with c1:
                st.write(label)
            with c2:
                st.progress(min(max(value, 0), 1.0))
            with c3:
                st.write(pct(value))

        st.markdown("#### Target interpretation")
        if scores.get("overall_score", 0) >= EVAL_THRESHOLD and scores.get("faithfulness", 0) >= EVAL_THRESHOLD:
            st.success("Overall score and faithfulness meet the configured 70% target.")
        else:
            st.warning("The target is not yet met. Use the case-level breakdown to identify retrieval or generation weaknesses.")

        st.markdown("#### Case-level results")
        rows = []
        for case in latest.get("cases", []):
            m = case.get("metrics", {})
            rows.append({
                "ID": case.get("id"),
                "Category": case.get("category"),
                "Question": case.get("question"),
                "Answer Relevancy": m.get("AnswerRelevancyMetric"),
                "Faithfulness": m.get("FaithfulnessMetric"),
                "Contextual Relevancy": m.get("ContextualRelevancyMetric"),
                "Contextual Precision": m.get("ContextualPrecisionMetric"),
                "Contextual Recall": m.get("ContextualRecallMetric"),
                "Page Hit": case.get("page_hit"),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No evaluation report yet. Click Run DeepEval to create the first measured baseline.")

with tab_about:
    st.markdown("### Architecture")
    st.code(
        "PDF\n"
        "  ↓\n"
        "Page-aware extraction + section chunking\n"
        "  ↓\n"
        " ┌───────────────────────┬──────────────────────┐\n"
        " │ Vector Pipeline       │ Knowledge Graph       │\n"
        " │ Sentence Transformers │ Neo4j                 │\n"
        " │ FAISS                 │ Entities/relationships│\n"
        " └───────────────────────┴──────────────────────┘\n"
        "                 ↓\n"
        "          Hybrid Retrieval\n"
        "                 ↓\n"
        "           Context Fusion\n"
        "                 ↓\n"
        "             LLM Answer\n"
        "                 ↓\n"
        "             Self-RAG\n"
        "                 ↓\n"
        "          Output Guardrail\n"
        "                 ↓\n"
        "              Answer\n"
        "\n"
        "Offline/QA path: DeepEval → RAG metrics → 70% target → regression report",
        language="text",
    )
    st.markdown("### Evaluation model")
    st.write(
        "Retrieval Quality = average(Contextual Relevancy, Contextual Precision, Contextual Recall). "
        "Generation Quality = average(Answer Relevancy, Faithfulness). "
        "Overall RAG Score = 50% Retrieval + 50% Generation. Faithfulness is also treated as a grounding gate."
    )
    st.caption("DeepEval is used for evaluation, not as the runtime answer generator.")

import streamlit as st

from hybrid_rag import HybridRAG

st.set_page_config(
    page_title="Hybrid RAG - Spotify Architecture",
    page_icon="🔎",
    layout="wide",
)

st.title("Hybrid RAG — Spotify Architecture PDF")
st.caption("Vector Search + Knowledge Graph + Self-RAG + Guardrails")

@st.cache_resource
def load_rag():
    return HybridRAG()

rag = load_rag()

question = st.text_area(
    "Ask a question about the supplied architecture PDF",
    placeholder="Example: Which service manages playback state?",
)

if st.button("Ask") and question.strip():
    with st.spinner("Retrieving from FAISS + Neo4j and validating with Self-RAG..."):
        result = rag.ask(question)

    st.subheader("Answer")
    st.write(result["answer"])

    if result.get("self_rag_review"):
        with st.expander("Self-RAG review"):
            st.json(result["self_rag_review"])

    with st.expander("Retrieved sources"):
        st.json(result["sources"])

import streamlit as st
from datetime import datetime

from utils.loader import load_urls
from utils.qa_chain import build_vectorstore, load_qa_chain, clear_vectorstore

# ---------------- Page Config ----------------
st.set_page_config(page_title="URL Research Tool", layout="wide")

# ---------------- Styles ----------------
st.markdown("""
<style>
.main { padding: 2rem 3rem; }
h1, h2, h3 { font-weight: 600; }

.card {
    background-color: #111827;
    padding: 1.5rem;
    border-radius: 12px;
    margin-bottom: 1rem;
}

.answer {
    line-height: 1.7;
    font-size: 1.05rem;
}

.source {
    font-size: 0.9rem;
    margin-bottom: 0.3rem;
}

.top-source {
    color: #facc15;
    font-weight: 600;
}

.snippet {
    background-color: #020617;
    border-left: 4px solid #38bdf8;
    padding: 0.75rem;
    margin: 0.5rem 0;
    font-size: 0.9rem;
    line-height: 1.5;
}

.timeline {
    border-left: 2px solid #1f2937;
    padding-left: 1rem;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- State ----------------
if "urls" not in st.session_state:
    st.session_state.urls = [""]

if "history" not in st.session_state:
    st.session_state.history = []

# Normalize older history entries (IMPORTANT FIX)
for item in st.session_state.history:
    if "time" not in item:
        item["time"] = "Earlier"

# ---------------- Title ----------------
st.title("🔎 URL Research Tool")
st.caption(
    "Ask questions across multiple web pages. "
    "Answers are generated only from the provided URLs."
)

# ---------------- Sidebar ----------------
st.sidebar.header("Enter URLs")

for i, url in enumerate(st.session_state.urls):
    st.session_state.urls[i] = st.sidebar.text_input(
        f"URL {i + 1}",
        value=url,
        key=f"url_{i}"
    )

c1, c2 = st.sidebar.columns(2)

with c1:
    if st.sidebar.button("➕ Add URL"):
        st.session_state.urls.append("")

with c2:
    if st.sidebar.button("➖ Remove URL") and len(st.session_state.urls) > 1:
        st.session_state.urls.pop()

if st.sidebar.button("Process URLs"):
    with st.spinner("Indexing content..."):
        docs = load_urls([u for u in st.session_state.urls if u.strip()])
        build_vectorstore(docs)
    st.sidebar.success("URLs processed successfully")

st.sidebar.divider()

if st.sidebar.button("🧹 Clear History"):
    st.session_state.history.clear()
    st.sidebar.success("History cleared")

if st.sidebar.button("🗑 Clear Index"):
    clear_vectorstore()
    st.session_state.history.clear()
    st.sidebar.success("Index cleared")

# ---------------- Main ----------------
st.divider()
query = st.text_input("Ask a question about the content")

if query:
    loaded = load_qa_chain()
    if not loaded:
        st.warning("Please process URLs first.")
    else:
        chain, vectorstore = loaded

        with st.spinner("Thinking..."):
            answer = chain.invoke(query)

            docs_scores = vectorstore.similarity_search_with_score(query, k=6)

            source_map = {}
            for doc, score in docs_scores:
                src = doc.metadata.get("source", "Unknown source")
                source_map.setdefault(src, []).append({
                    "score": score,
                    "text": doc.page_content.strip()
                })

        st.session_state.history.insert(
            0,
            {
                "time": datetime.now().strftime("%d %b %Y · %H:%M"),
                "question": query,
                "answer": answer,
                "sources": source_map
            }
        )

# ---------------- Display Latest Answer ----------------
if st.session_state.history:
    latest = st.session_state.history[0]

    st.subheader("Answer")
    st.markdown(
        f"<div class='card answer'>{latest['answer']}</div>",
        unsafe_allow_html=True
    )

    st.subheader("Sources & Highlighted Snippets")
    first = True
    for src, snippets in latest["sources"].items():
        label = "⭐ Most Relevant Source" if first else "Source"
        css = "top-source" if first else "source"
        st.markdown(f"**🔗 {src} — {label}**")
        for s in snippets[:2]:
            st.markdown(
                f"<div class='snippet'>{s['text'][:400]}...</div>",
                unsafe_allow_html=True
            )
        first = False

# ---------------- Timeline ----------------
if len(st.session_state.history) > 1:
    st.divider()
    st.subheader("Question Timeline")

    st.markdown("<div class='timeline'>", unsafe_allow_html=True)
    for item in st.session_state.history:
        timestamp = item.get("time", "Earlier")
        with st.expander(f"{timestamp} — {item['question']}"):
            st.markdown(
                f"<div class='card answer'>{item['answer']}</div>",
                unsafe_allow_html=True
            )
    st.markdown("</div>", unsafe_allow_html=True)

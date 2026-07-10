import streamlit as st
import time
import os
import shutil
from ingestion.clone_repo import clone_repo
from ingestion.file_reader import read_code_files
from indexing.chunker import chunk_files
from indexing.embeddings import get_embeddings
from indexing.vector_store import build_index, save_index
from query.search import search_index, LOW_CONFIDENCE_THRESHOLD
from query.answer import synthesize_answer, rewrite_query_for_search

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="CodeNavigator", page_icon="🔍", layout="wide")
st.title("🔍 CodeNavigator — AI-Powered GitHub Code Search")
st.caption("Ask natural-language questions about any GitHub repository.")

# ── Session state defaults ────────────────────────────────────────────────────
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "index" not in st.session_state:
    st.session_state.index = None
if "clone_path" not in st.session_state:
    st.session_state.clone_path = None

# ── Sidebar: Repository Input ─────────────────────────────────────────────────
with st.sidebar:
    st.header("Repository Setup")
    repo_url = st.text_input("GitHub Repository URL",
                             placeholder="https://github.com/user/repo")
    index_btn = st.button("Clone & Index Repository", type="primary")

    st.divider()
    st.header("AI Answer (optional)")
    groq_api_key = st.text_input(
        "Groq API Key",
        value=os.environ.get("GROQ_API_KEY", ""),
        type="password",
        help="Get a free key at console.groq.com. Leave blank to only see "
             "raw retrieved snippets without a synthesized answer.",
    )

    if st.session_state.indexed:

        st.success(f" Indexed {len(st.session_state.chunks)} chunks")

        if st.button(" Clear & Reset"):

            st.session_state.indexed = False
            st.session_state.chunks = []
            st.session_state.index = None

            # Remove the cloned repo and saved FAISS files safely
            try:
                if st.session_state.clone_path and os.path.exists(st.session_state.clone_path):
                    shutil.rmtree(st.session_state.clone_path, ignore_errors=True)

                if os.path.exists("data/index.faiss"):
                    os.remove("data/index.faiss")

                if os.path.exists("data/chunks.pkl"):
                    os.remove("data/chunks.pkl")

            except Exception as e:
                st.warning(f"Cleanup warning: {e}")

            st.session_state.clone_path = None
            st.rerun()

# ── Indexing Pipeline ─────────────────────────────────────────────────────────
if index_btn and repo_url:
    if not repo_url.startswith("https://github.com/"):
        st.sidebar.error("Please enter a valid GitHub URL.")
    else:
        with st.spinner("Step 1/5 — Cloning repository..."):

            os.makedirs("data", exist_ok=True)

            # Remove the previously cloned repo so disk usage doesn't grow
            # unboundedly across repeated "Clone & Index" clicks.
            if st.session_state.clone_path and os.path.exists(st.session_state.clone_path):
                shutil.rmtree(st.session_state.clone_path, ignore_errors=True)

            clone_path = f"data/repo_{int(time.time())}"

            success, msg = clone_repo(repo_url, clone_path)
            if not success:
                st.error(f"Clone failed: {msg}")
                st.stop()

        with st.spinner("Step 2/5 — Reading code files..."):
            files = read_code_files(clone_path)
            st.sidebar.info(f"Found {len(files)} code files.")

        if not files:
            st.error(
                "No supported code files were found in this repository. "
                "Nothing to index."
            )
            st.stop()

        with st.spinner("Step 3/5 — Chunking code..."):
            chunks = chunk_files(files)
            st.sidebar.info(f"Created {len(chunks)} chunks.")

        with st.spinner("Step 4/5 — Generating embeddings..."):
            embeddings = get_embeddings([c["text"] for c in chunks])

        with st.spinner("Step 5/5 — Building FAISS index..."):
            index = build_index(embeddings)
            save_index(index, chunks)
            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.clone_path = clone_path
            st.session_state.indexed = True

        st.sidebar.success(" Repository indexed successfully!")
        st.rerun()

# ── Query Interface ───────────────────────────────────────────────────────────
if st.session_state.indexed:
    st.subheader(" Ask a Question About the Code")
    query = st.text_input("Your question",
                          placeholder="Where is authentication implemented?")
    top_k = st.slider("Number of results", 1, 10, 5)

    if st.button("🔎 Search", type="primary") and query:
        # The embedding model is trained on terse, docstring-style code-search
        # queries — conversational phrasing like "Where is X implemented?"
        # ranks noticeably worse than a keyword-dense rewrite. Rewriting is
        # only possible with a Groq key; without one, search on the raw query.
        search_query = query
        if groq_api_key:
            search_query = rewrite_query_for_search(query, groq_api_key)

        results = search_index(
            search_query, st.session_state.index, st.session_state.chunks, top_k=top_k
        )

        if results and groq_api_key:
            st.subheader(" AI Answer")
            with st.spinner("Synthesizing answer..."):
                try:
                    answer = synthesize_answer(query, results, groq_api_key)
                    st.markdown(answer)
                except Exception as e:
                    st.warning(f"Couldn't generate an AI answer: {e}")

        st.subheader(" Results")
        if not results:
            st.warning(
                "No code was found in the indexed repository at all — "
                "did indexing complete successfully?"
            )
        elif results[0]["score"] < LOW_CONFIDENCE_THRESHOLD:
            st.warning(
                "These results scored low — the repo may not actually "
                "contain anything matching this query. Treat them as "
                "guesses, not confirmed answers."
            )
        for i, res in enumerate(results, 1):
            with st.expander(
                f"**#{i} — {res['file']}  (lines {res['start_line']}–{res['end_line']})**",
                expanded=(i == 1)
            ):
                st.caption(f" `{res['file']}`  |  Score: `{res['score']:.4f}`")
                st.code(res["text"], language=res.get("language", "python"))
else:
    st.info(" Enter a GitHub URL in the sidebar and click **Clone & Index Repository** to begin.")
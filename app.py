
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import time
import os
import shutil
from ingestion.clone_repo import clone_repo
from ingestion.file_reader import read_code_files
from indexing.chunker import chunk_files
from indexing.embeddings import get_embeddings
from indexing.vector_store import build_index, save_index, load_index
from query.search import search_index

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="CodeNavigator", page_icon="🔍", layout="wide")
st.title("🔍 CodeNavigator — AI-Powered GitHub Code Search")
st.caption("Ask natural-language questions about any GitHub repository.")

# ── Session state defaults ────────────────────────────────────────────────────
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "chunks" not in st.session_state:
    st.session_state.chunks = []

# ── Sidebar: Repository Input ─────────────────────────────────────────────────
with st.sidebar:
    st.header("Repository Setup")
    repo_url = st.text_input("GitHub Repository URL",
                             placeholder="https://github.com/user/repo")
    index_btn = st.button("Clone & Index Repository", type="primary")

    # if st.session_state.indexed:
    #     st.success(f" Indexed {len(st.session_state.chunks)} chunks")
    #     if st.button(" Clear & Reset"):
    #         st.session_state.indexed = False
    #         st.session_state.chunks = []
    #         # if os.path.exists("data/repo"):
    #         #     shutil.rmtree("data/repo")


    #         if os.path.exists("data"):
    #             shutil.rmtree("data")

    #         if os.path.exists("data/index.faiss"):
    #             os.remove("data/index.faiss")
    #         if os.path.exists("data/chunks.pkl"):
    #             os.remove("data/chunks.pkl")
    #         st.rerun()


    if st.session_state.indexed:

        st.success(f" Indexed {len(st.session_state.chunks)} chunks")

        if st.button(" Clear & Reset"):

            st.session_state.indexed = False
            st.session_state.chunks = []

            # Remove saved FAISS files safely
            try:
                if os.path.exists("data/index.faiss"):
                    os.remove("data/index.faiss")

                if os.path.exists("data/chunks.pkl"):
                    os.remove("data/chunks.pkl")

            except Exception as e:
                st.warning(f"Cleanup warning: {e}")

            st.rerun()

# ── Indexing Pipeline ─────────────────────────────────────────────────────────
if index_btn and repo_url:
    if not repo_url.startswith("https://github.com/"):
        st.sidebar.error("Please enter a valid GitHub URL.")
    else:
        with st.spinner("Step 1/5 — Cloning repository..."):

            os.makedirs("data", exist_ok=True)

            clone_path = f"data/repo_{int(time.time())}"



            # clone_path = "data/repo"

            # # Ensure parent folder exists
            # os.makedirs("data", exist_ok=True)

            # # Remove old repo if present
            # if os.path.exists(clone_path):
            #     shutil.rmtree(clone_path)



            success, msg = clone_repo(repo_url, clone_path)
            if not success:
                st.error(f"Clone failed: {msg}")
                st.stop()

        with st.spinner("Step 2/5 — Reading code files..."):
            files = read_code_files(clone_path)
            st.sidebar.info(f"Found {len(files)} code files.")

        with st.spinner("Step 3/5 — Chunking code..."):
            chunks = chunk_files(files)
            st.sidebar.info(f"Created {len(chunks)} chunks.")

        with st.spinner("Step 4/5 — Generating embeddings..."):
            embeddings = get_embeddings([c["text"] for c in chunks])

        with st.spinner("Step 5/5 — Building FAISS index..."):
            index = build_index(embeddings)
            save_index(index, chunks)
            st.session_state.chunks = chunks
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
        index, chunks = load_index()
        results = search_index(query, index, chunks, top_k=top_k)

        st.subheader(" Results")
        for i, res in enumerate(results, 1):
            with st.expander(
                f"**#{i} — {res['file']}  (lines {res['start_line']}–{res['end_line']})**",
                expanded=(i == 1)
            ):
                st.caption(f" `{res['file']}`  |  Score: `{res['score']:.4f}`")
                st.code(res["text"], language=res.get("language", "python"))
else:
    st.info(" Enter a GitHub URL in the sidebar and click **Clone & Index Repository** to begin.")
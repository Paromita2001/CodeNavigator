# 🔍 CodeNavigator
AI-Powered GitHub Repository Intelligence Tool

CodeNavigator is an AI-powered semantic code search system that allows users to ingest any public GitHub repository and ask natural-language questions about the codebase.

The system uses embeddings and vector similarity search to retrieve relevant code snippets, file paths, and line numbers from repositories.

---

# 🚀 Features

- Clone any public GitHub repository automatically
- Read and process multiple programming languages
- Semantic code search using embeddings
- Natural language query interface
- FAISS-based vector similarity search
- File path and line-number retrieval
- Streamlit-based interactive UI

---

# 🧠 Thought Process & Approach

## Initial Understanding

The core requirement of the assignment was to build an intelligent repository search system capable of understanding code semantically rather than performing simple keyword matching.

The system needed to:
- ingest repositories automatically,
- process source code,
- understand user questions,
- and retrieve precise code snippets.

The assignment emphasized:
- repository understanding,
- semantic navigation,
- and engineering thought process.

---

## Approaches Considered

### 1. Keyword-Based Search
Initially, a simple keyword search approach was considered using regex and file scanning.

#### Problems:
- Could not understand semantic meaning
- Failed when query wording differed from code wording
- Produced many irrelevant results

Example:
A query like:
"Where is authentication implemented?"

might fail if the code uses:
`verify_user()` instead of `authentication`.

---

### 2. Embedding-Based Semantic Search (Chosen Approach)

The final approach uses:
- Sentence Transformers for embeddings
- FAISS for vector similarity search

Why this approach was selected:
- Better semantic understanding
- More flexible natural-language querying
- Faster retrieval using vector indexing
- Closer to modern AI retrieval systems

---

## Tradeoffs

| Decision | Benefit | Limitation |
|---|---|---|
| Small embedding model | Faster indexing/search | Slightly lower semantic accuracy |
| Chunk-based indexing | Better retrieval granularity | Context fragmentation |
| Streamlit UI | Fast development | Limited frontend customization |
| FAISS local index | Lightweight and fast | No distributed scaling |

---

---

### Package & Environment Issues
Several dependency and interpreter issues occurred during setup, especially involving:
- sentence-transformers
- torch
- torchvision
- faiss

These were resolved using:
- virtual environments
- proper requirements.txt management
- interpreter configuration

---

# 🏗️ Architecture Diagram

```text
User Input (GitHub URL)
        ↓
Repository Cloner
        ↓
Code File Reader
        ↓
Code Chunking
        ↓
Embedding Generator
        ↓
FAISS Vector Store
        ↓
Natural Language Query
        ↓
Semantic Similarity Search
        ↓
Ranked Code Results

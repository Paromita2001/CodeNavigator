import faiss
import numpy as np
from indexing.embeddings import embed_query


def search_index(
    query: str,
    index: faiss.IndexFlatIP,
    chunks: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Converts the natural-language query to an embedding, runs a FAISS
    nearest-neighbour search, and returns the top-k matching chunks
    with their similarity scores.

    Returns a list of result dicts:
        {
            "file":       relative file path (str),
            "text":       code snippet (str),
            "start_line": int,
            "end_line":   int,
            "language":   str,
            "score":      cosine similarity (float, 0–1),
        }
    """
    query_vec = embed_query(query)               # shape (1, dim)

    # FAISS returns distances and indices arrays of shape (1, top_k)
    distances, indices = index.search(query_vec, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:           # FAISS returns -1 for empty slots
            continue
        chunk = chunks[idx].copy()
        chunk["score"] = float(dist)
        results.append(chunk)

    return results
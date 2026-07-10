import faiss
import numpy as np
from indexing.embeddings import embed_query

# Below this cosine similarity, a chunk is treated as unrelated to the query
# rather than a low-confidence match, so irrelevant results aren't shown as
# if they were real hits. Calibrated against st-codesearch-distilroberta-base,
# which separates relevant (~0.39-0.70) from irrelevant (~0.16-0.27) queries
# on real repos — 0.32 sits in that gap.
MIN_SCORE = 0.32


def search_index(
    query: str,
    index: faiss.IndexFlatIP,
    chunks: list[dict],
    top_k: int = 5,
    min_score: float = MIN_SCORE,
) -> list[dict]:
    """
    Converts the natural-language query to an embedding, runs a FAISS
    nearest-neighbour search, and returns the top-k matching chunks
    with their similarity scores. Chunks scoring below `min_score` are
    dropped instead of being returned as false-confidence matches.

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
        score = float(dist)
        if score < min_score:
            continue
        chunk = chunks[idx].copy()
        chunk["score"] = score
        results.append(chunk)

    return results
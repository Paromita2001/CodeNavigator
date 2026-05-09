import faiss
import pickle
import numpy as np
import os

INDEX_PATH  = "data/index.faiss"
CHUNKS_PATH = "data/chunks.pkl"


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Builds a FAISS IndexFlatIP (inner-product / cosine similarity) index.
    Since embeddings are L2-normalised, inner product == cosine similarity.

    Args:
        embeddings: float32 array of shape (n_chunks, dim)

    Returns:
        A populated faiss.IndexFlatIP ready for search.
    """
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_index(index: faiss.IndexFlatIP, chunks: list[dict]) -> None:
    """Persists the FAISS index and chunk metadata to disk."""
    os.makedirs("data", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)


def load_index() -> tuple[faiss.IndexFlatIP, list[dict]]:
    """Loads the FAISS index and chunk metadata from disk."""
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks
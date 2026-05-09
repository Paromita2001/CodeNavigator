from sentence_transformers import SentenceTransformer
import numpy as np

# Loaded once; subsequent calls reuse the same model object.
# "all-MiniLM-L6-v2" is small (80 MB), fast, and good for code search.
_model = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def get_embeddings(texts: list[str]) -> np.ndarray:
    """
    Converts a list of text strings into a 2-D numpy array of float32 embeddings.
    Shape: (len(texts), embedding_dim)  — 384 dims for MiniLM-L6-v2.

    Uses batched encoding for speed; show_progress_bar gives a terminal log.
    """
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine similarity via dot product
    )
    return embeddings.astype("float32")


def embed_query(query: str) -> np.ndarray:
    """
    Embeds a single query string.
    Returns shape (1, embedding_dim) float32 array.
    """
    model = _get_model()
    vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vec.astype("float32")
from sentence_transformers import SentenceTransformer
import numpy as np

# Loaded once; subsequent calls reuse the same model object.
# "st-codesearch-distilroberta-base" is fine-tuned on CodeSearchNet
# (natural-language query -> code snippet pairs), which is exactly this
# app's retrieval task — unlike all-MiniLM-L6-v2 (general prose similarity),
# it was trained to link words like "authentication" to identifiers like
# verifyToken/session.user. No trust_remote_code needed: it's a stock
# sentence-transformers model.
MODEL_NAME = "flax-sentence-embeddings/st-codesearch-distilroberta-base"

_model = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def get_embeddings(texts: list[str]) -> np.ndarray:
    """
    Converts a list of text strings into a 2-D numpy array of float32 embeddings.
    Shape: (len(texts), embedding_dim) — 768 dims for st-codesearch-distilroberta-base.

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
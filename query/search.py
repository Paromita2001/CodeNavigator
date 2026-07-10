import re
import faiss
import numpy as np
from indexing.embeddings import embed_query

# Cosine similarity baseline varies a lot between repos/corpora — on one repo
# an unrelated query topped out at 0.27, on another an unrelated query scored
# 0.41 (higher than a real, correct match on that same repo). This is used
# only to flag low confidence in the UI, never to hide results — hiding by
# absolute score risks dropping the one correct (but modestly-scored) answer.
LOW_CONFIDENCE_THRESHOLD = 0.32

# Semantic similarity alone is a noisy signal for "where is X" style
# queries — a literal word match between the query and a file's path/name
# (e.g. "auth" in the query matching "auth_service.py") is much more
# precise for code navigation, so it's blended in as a re-ranking factor
# rather than relied on exclusively.
SEMANTIC_WEIGHT = 0.65
PATH_MATCH_WEIGHT = 0.25
CONTENT_MATCH_WEIGHT = 0.10

# Prose documentation (README/markdown) is written in the same natural
# language as the user's question, so it semantically out-scores actual
# code almost by construction — code rarely spells out a word like
# "authentication" the way a doc does. Since this tool's whole purpose is
# finding CODE (not docs), markdown is down-weighted so it only wins when
# it's overwhelmingly the best match, not just because it "reads" more
# like the question.
PROSE_LANGUAGE_PENALTY = {"markdown": 0.4}

# Test files repeat the identifiers/behaviour they're testing (e.g. a
# TestLoginUser function literally says "login" and "token" repeatedly),
# so they compete with — and often beat — the actual implementation for
# "where is X implemented?" queries despite not being the implementation.
# Down-weighted for the same reason as markdown above.
TEST_FILE_PENALTY = 0.5
TEST_FILE_PATTERN = re.compile(
    r"(^|[\\/])(tests?|__tests__|spec)[\\/]"   # a tests/ or __tests__/ directory
    r"|[_.](test|spec)\.\w+$"                  # foo_test.go, foo.test.js, foo.spec.ts
    r"|(^|[\\/])test_[^\\/]+\.\w+$",           # test_foo.py
    re.IGNORECASE,
)

# Common English filler words in "where is X implemented?"-style questions
# that would otherwise dilute the keyword-overlap signal.
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "where", "what", "how", "why", "when", "which", "who",
    "does", "do", "did", "can", "could", "should", "would",
    "in", "on", "of", "to", "for", "with", "and", "or", "this", "that",
    "code", "implemented", "implement", "implementation", "implements",
    "file", "files", "function", "functions", "handled", "handle",
    "handles", "located", "location", "find", "show", "me", "please",
}

# How many candidates to pull from FAISS before re-ranking with the
# lexical signal — wider than top_k so a strong keyword match that FAISS
# ranked outside the naive top-k still has a chance to surface.
CANDIDATE_POOL_MULTIPLIER = 6
MIN_CANDIDATE_POOL = 40


def _tokenize(text: str) -> set:
    """Splits text (including snake_case/camelCase/path separators) into
    lowercase word tokens for keyword-overlap comparison."""
    text = re.sub(r"[_\-/\\.]", " ", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return {w.lower() for w in re.findall(r"[a-zA-Z0-9]+", text) if len(w) > 1}


def _query_tokens(query: str) -> set:
    return _tokenize(query) - STOPWORDS


def _tokens_match(a: str, b: str) -> bool:
    if a == b:
        return True
    # Allow prefix/substring relation to catch stem variations that exact
    # matching misses — e.g. query "authentication" vs path token "auth",
    # or "login" vs "logged". Length-gated so short tokens (e.g. "is"/"db")
    # don't spuriously substring-match unrelated words.
    if len(a) >= 4 and len(b) >= 4:
        return a.startswith(b) or b.startswith(a)
    return False


def _lexical_overlap(query_tokens: set, tokens: set) -> float:
    if not query_tokens:
        return 0.0
    matched = sum(1 for qt in query_tokens if any(_tokens_match(qt, t) for t in tokens))
    return matched / len(query_tokens)


def search_index(
    query: str,
    index: faiss.IndexFlatIP,
    chunks: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Retrieves candidate chunks via FAISS semantic search, then re-ranks a
    wider candidate pool by blending in a literal keyword-overlap score
    against each chunk's file path and content — see SEMANTIC_WEIGHT /
    PATH_MATCH_WEIGHT / CONTENT_MATCH_WEIGHT. Always returns up to top_k
    chunks; callers wanting a relevance signal should check "score" against
    LOW_CONFIDENCE_THRESHOLD rather than expect this to filter anything.

    Returns a list of result dicts:
        {
            "file":       relative file path (str),
            "text":       code snippet (str),
            "start_line": int,
            "end_line":   int,
            "language":   str,
            "score":      combined relevance score (float, 0-1ish),
        }
    """
    query_vec = embed_query(query)               # shape (1, dim)
    query_tokens = _query_tokens(query)

    pool_size = min(len(chunks), max(top_k * CANDIDATE_POOL_MULTIPLIER, MIN_CANDIDATE_POOL))
    distances, indices = index.search(query_vec, pool_size)

    candidates = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:           # FAISS returns -1 for empty slots
            continue
        chunk = chunks[idx].copy()
        semantic_score = float(dist)
        path_overlap = _lexical_overlap(query_tokens, _tokenize(chunk["file"]))
        content_overlap = _lexical_overlap(query_tokens, _tokenize(chunk["text"]))

        score = (
            SEMANTIC_WEIGHT * semantic_score
            + PATH_MATCH_WEIGHT * path_overlap
            + CONTENT_MATCH_WEIGHT * content_overlap
        )
        score *= PROSE_LANGUAGE_PENALTY.get(chunk.get("language"), 1.0)
        if TEST_FILE_PATTERN.search(chunk["file"]):
            score *= TEST_FILE_PENALTY

        chunk["score"] = score
        candidates.append(chunk)

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:top_k]

from typing import List, Dict, Tuple

# Hard cap: an oversized single block (a huge function, a minified file)
# gets sliced with the sliding-window fallback instead of becoming one
# huge chunk.
MAX_CHUNK_LINES = 100

# A block smaller than this is considered "too small to stand alone" (e.g.
# a lone import line, a two-line getter) and gets merged with its
# neighbor. Once accumulated content reaches this size, packing stops —
# it does NOT keep absorbing further blocks just because MAX_CHUNK_LINES
# has room left. Otherwise a "utils.go" grab-bag file merges unrelated
# functions (a JWT helper next to a random-string helper) into one chunk,
# diluting the embedding signal for whichever function a query is
# actually about.
MIN_CHUNK_LINES = 12

FALLBACK_CHUNK_SIZE = 80
FALLBACK_OVERLAP = 5

# Per-language line-comment prefixes, used to detect chunks that are
# entirely disabled/dead code left in place (e.g. an old duplicate
# implementation someone commented out instead of deleting). Such chunks
# are never useful as "here's the implementation" — they're noise that
# competes with the real, live code in search results — so they're
# excluded rather than just down-ranked.
COMMENT_PREFIXES = {
    "python": ("#",), "ruby": ("#",), "bash": ("#",), "yaml": ("#",),
    "javascript": ("//",), "typescript": ("//",), "java": ("//",),
    "csharp": ("//",), "cpp": ("//",), "c": ("//",), "go": ("//",),
    "rust": ("//",), "kotlin": ("//",), "swift": ("//",), "scala": ("//",),
    "dart": ("//",), "php": ("//", "#"), "sql": ("--",),
}
DEAD_CODE_COMMENT_RATIO = 0.8  # chunk is "dead" if >=80% of its non-blank lines are commented out
DEAD_CODE_MIN_LINES = 3        # too short to judge reliably below this — a short doc-comment isn't dead code


def _is_commented_out(text: str, language: str) -> bool:
    prefixes = COMMENT_PREFIXES.get(language)
    if not prefixes:
        return False
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < DEAD_CODE_MIN_LINES:
        return False
    commented = sum(1 for l in lines if l.startswith(prefixes))
    return commented / len(lines) >= DEAD_CODE_COMMENT_RATIO


def _split_into_blocks(lines: List[str]) -> List[Tuple[int, int]]:
    """
    Splits a file's lines into logical top-level blocks: a new block starts
    at a non-blank, non-indented line that follows a blank line (or is the
    first line). This keeps a function/class body — which is indented —
    together with its signature, instead of slicing it at an arbitrary
    fixed line count the way naive windowed chunking would.

    Returns a list of (start_idx, end_idx) pairs, 0-indexed, end exclusive.
    """
    boundaries = [0]
    for i, line in enumerate(lines):
        if i == 0:
            continue
        if not line.strip():
            continue
        is_top_level = line[0] not in (" ", "\t")
        prev_blank = not lines[i - 1].strip()
        if is_top_level and prev_blank:
            boundaries.append(i)
    boundaries.append(len(lines))

    return [(a, b) for a, b in zip(boundaries, boundaries[1:]) if b > a]


def _pack_blocks(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Merges only small consecutive blocks (below MIN_CHUNK_LINES) so tiny
    fragments (a single import line, a two-line getter) aren't embedded
    alone. As soon as the accumulated chunk reaches MIN_CHUNK_LINES, it's
    flushed as its own unit — it does not keep absorbing further blocks,
    so two substantial-but-unrelated functions never get merged into one
    chunk just because there's room under MAX_CHUNK_LINES. A block that's
    already larger than MAX_CHUNK_LINES on its own (a huge function, a
    minified file) is split with the sliding-window fallback.
    """
    chunks = []
    cur_start = cur_end = None

    def flush():
        if cur_start is not None:
            chunks.append((cur_start, cur_end))

    for start, end in ranges:
        if end - start > MAX_CHUNK_LINES:
            flush()
            cur_start = cur_end = None
            pos = start
            while pos < end:
                stop = min(pos + FALLBACK_CHUNK_SIZE, end)
                chunks.append((pos, stop))
                if stop >= end:
                    break
                pos += FALLBACK_CHUNK_SIZE - FALLBACK_OVERLAP
            continue

        if cur_start is None:
            cur_start, cur_end = start, end
        elif cur_end - cur_start < MIN_CHUNK_LINES and end - cur_start <= MAX_CHUNK_LINES:
            cur_end = end
        else:
            flush()
            cur_start, cur_end = start, end

    flush()
    return chunks


def chunk_files(files: List[Dict]) -> List[Dict]:
    """
    Splits files into chunks aligned to logical top-level blocks (function/
    class boundaries, via an indentation heuristic) rather than arbitrary
    fixed-size line windows, so each chunk is a coherent unit instead of a
    fragment sliced mid-function. Oversized single blocks still fall back
    to a sliding window so no chunk becomes unreasonably large.
    """
    chunks = []

    for file in files:
        lines = file["content"].splitlines()
        if not lines:
            continue

        for start, end in _pack_blocks(_split_into_blocks(lines)):
            chunk_text = "\n".join(lines[start:end])
            if not chunk_text.strip():
                continue
            if _is_commented_out(chunk_text, file["language"]):
                continue

            chunks.append({
                "file": file["path"],
                "text": chunk_text,
                "start_line": start + 1,
                "end_line": end,
                "language": file["language"],
            })

    return chunks

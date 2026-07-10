
from typing import List, Dict

CHUNK_SIZE = 80
CHUNK_OVERLAP = 5


def chunk_files(files: List[Dict]) -> List[Dict]:
    """
    Splits files into overlapping chunks for semantic search.
    """

    chunks = []

    for file in files:
        lines = file["content"].splitlines()

        start = 0

        while start < len(lines):
            end = min(start + CHUNK_SIZE, len(lines))

            chunk_text = "\n".join(lines[start:end])

            chunks.append({
                "file": file["path"],
                "text": chunk_text,
                "start_line": start + 1,
                "end_line": end,
                "language": file["language"],
            })

            start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks
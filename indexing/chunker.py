# import os

# # Supported file extensions → language label for syntax highlighting
# SUPPORTED_EXTENSIONS = {
#     ".py":   "python",
#     ".js":   "javascript",
#     ".ts":   "typescript",
#     ".java": "java",
#     ".cpp":  "cpp",
#     ".c":    "c",
#     ".go":   "go",
#     ".rb":   "ruby",
#     ".rs":   "rust",
#     ".php":  "php",
#     ".cs":   "csharp",
#     ".html": "html",
#     ".css":  "css",
#     ".sh":   "bash",
#     ".md":   "markdown",
#     ".json": "json",
#     ".yaml": "yaml",
#     ".yml":  "yaml",
# }

# # Folders to always skip
# SKIP_DIRS = {
#     ".git", "__pycache__", "node_modules", ".venv", "venv",
#     "env", "dist", "build", ".idea", ".vscode", "vendor",
# }

# MAX_FILE_SIZE_BYTES = 200_000   # skip files larger than 200 KB


# def read_code_files(root: str) -> list[dict]:
#     """
#     Walks the cloned repo directory and reads all supported code files.

#     Returns a list of dicts:
#         {
#             "path":     relative file path (str),
#             "content":  file content (str),
#             "language": language label (str),
#             "lines":    total line count (int),
#         }
#     """
#     results = []

#     for dirpath, dirnames, filenames in os.walk(root):
#         # Prune skipped directories in-place so os.walk doesn't recurse into them
#         dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

#         for fname in filenames:
#             ext = os.path.splitext(fname)[1].lower()
#             if ext not in SUPPORTED_EXTENSIONS:
#                 continue

#             full_path = os.path.join(dirpath, fname)

#             # Skip very large files
#             try:
#                 if os.path.getsize(full_path) > MAX_FILE_SIZE_BYTES:
#                     continue
#             except OSError:
#                 continue

#             # Read file content
#             try:
#                 with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
#                     content = f.read()
#             except Exception:
#                 continue

#             if not content.strip():
#                 continue   # skip empty files

#             rel_path = os.path.relpath(full_path, root)

#             results.append({
#                 "path":     rel_path,
#                 "content":  content,
#                 "language": SUPPORTED_EXTENSIONS[ext],
#                 "lines":    content.count("\n") + 1,
#             })

#     return results





from typing import List, Dict

CHUNK_SIZE = 120
CHUNK_OVERLAP = 20


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
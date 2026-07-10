import os
import re

# Supported file extensions → language label for syntax highlighting
SUPPORTED_EXTENSIONS = {
    ".py":   "python",
    ".js":   "javascript",
    ".jsx":  "javascript",
    ".mjs":  "javascript",
    ".cjs":  "javascript",
    ".ts":   "typescript",
    ".tsx":  "typescript",
    ".java": "java",
    ".kt":   "kotlin",
    ".kts":  "kotlin",
    ".swift": "swift",
    ".dart": "dart",
    ".cpp":  "cpp",
    ".cc":   "cpp",
    ".h":    "cpp",
    ".hpp":  "cpp",
    ".c":    "c",
    ".go":   "go",
    ".rb":   "ruby",
    ".rs":   "rust",
    ".php":  "php",
    ".cs":   "csharp",
    ".scala": "scala",
    ".vue":  "vue",
    ".svelte": "svelte",
    ".html": "html",
    ".css":  "css",
    ".scss": "scss",
    ".sql":  "sql",
    ".graphql": "graphql",
    ".sh":   "bash",
    ".md":   "markdown",
    ".json": "json",
    ".yaml": "yaml",
    ".yml":  "yaml",
}

# Files with no extension that are still worth indexing (matched by exact
# filename since they have none).
EXTENSIONLESS_FILES = {
    "Dockerfile": "dockerfile",
    "Makefile": "makefile",
}

# Filename patterns for build output that's dense, unreadable, and never
# what a "where is X" query actually wants to see as a snippet.
NOISE_FILENAME_PATTERN = re.compile(
    r"\.(min|bundle|chunk)\.(js|css)$", re.IGNORECASE
)

# Folders to always skip
SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "env", "dist", "build", ".idea", ".vscode", "vendor",
}

# Auto-generated dependency manifests. These are dense with package names
# like "jsonwebtoken"/"passport-jwt"/"bcryptjs" that spuriously match
# semantic searches (e.g. "where is authentication?") despite containing no
# actual implementation code, so they're excluded regardless of extension.
SKIP_FILES = {
    "package-lock.json", "npm-shrinkwrap.json", "yarn.lock", "pnpm-lock.yaml",
    "composer.lock", "Gemfile.lock", "poetry.lock", "Pipfile.lock",
    "Cargo.lock", "go.sum", "mix.lock",
}

MAX_FILE_SIZE_BYTES = 200_000   # skip files larger than 200 KB

# JSON data dumps (Postman collections, OpenAPI specs, i18n catalogs,
# fixtures) are dense with domain keywords that spuriously match semantic
# searches — same problem as lockfiles — but unlike lockfiles they don't
# have one canonical filename to blacklist. Hand-written config JSON
# (package.json, tsconfig.json) is reliably small; large .json is reliably
# a generated data dump, so a tighter size cap on just this extension
# filters the noise without needing to enumerate every tool's filename.
JSON_MAX_SIZE_BYTES = 30_000


def read_code_files(root: str) -> list[dict]:
    """
    Walks the cloned repo directory and reads all supported code files.

    Returns a list of dicts:
        {
            "path":     relative file path (str),
            "content":  file content (str),
            "language": language label (str),
            "lines":    total line count (int),
        }
    """
    results = []

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skipped directories in-place so os.walk doesn't recurse into them
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for fname in filenames:
            if fname in SKIP_FILES or NOISE_FILENAME_PATTERN.search(fname):
                continue

            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                language = SUPPORTED_EXTENSIONS[ext]
            elif fname in EXTENSIONLESS_FILES:
                language = EXTENSIONLESS_FILES[fname]
            else:
                continue

            full_path = os.path.join(dirpath, fname)

            # Skip very large files (tighter cap for .json — see JSON_MAX_SIZE_BYTES)
            size_cap = JSON_MAX_SIZE_BYTES if ext == ".json" else MAX_FILE_SIZE_BYTES
            try:
                if os.path.getsize(full_path) > size_cap:
                    continue
            except OSError:
                continue

            # Read file content
            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception:
                continue

            if not content.strip():
                continue   # skip empty files

            rel_path = os.path.relpath(full_path, root)

            results.append({
                "path":     rel_path,
                "content":  content,
                "language": language,
                "lines":    content.count("\n") + 1,
            })

    return results
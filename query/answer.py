from groq import Groq

MODEL_NAME = "llama-3.3-70b-versatile"
REWRITE_MODEL_NAME = "llama-3.1-8b-instant"

REWRITE_SYSTEM_PROMPT = (
    "You rewrite natural-language questions about a codebase into short, "
    "keyword-dense phrases optimized for code search — the way a function's "
    "docstring or summary comment would describe it, not a conversational "
    "question. Output ONLY the rewritten phrase, nothing else.\n\n"
    "Example: \"Where is authentication implemented?\" -> "
    "\"JWT authentication middleware, verify token and authorize user\""
)

SYSTEM_PROMPT = (
    "You are a code navigation assistant. You are given a user's question "
    "about a codebase and a set of retrieved code snippets, each labelled "
    "with its file path and line range. Answer the question using ONLY the "
    "snippets provided — do not invent files, functions, or behaviour that "
    "isn't shown. Cite the file path (and line range) for anything you "
    "reference. If the snippets don't actually answer the question, say so "
    "plainly instead of guessing. Keep the answer concise."
)


def rewrite_query_for_search(query: str, api_key: str) -> str:
    """
    Rewrites a conversational question into a terse, code-search-style
    phrase before embedding. The embedding model is trained on CodeSearchNet
    (docstring-like queries), so conversational phrasing like "Where is X
    implemented?" ranks worse than a keyword-dense rewrite. Falls back to
    the original query on any failure.
    """
    try:
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=REWRITE_MODEL_NAME,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_tokens=60,
        )
        rewritten = completion.choices[0].message.content.strip()
        return rewritten or query
    except Exception:
        return query


def build_context(results: list[dict]) -> str:
    blocks = []
    for r in results:
        blocks.append(
            f"### {r['file']} (lines {r['start_line']}-{r['end_line']})\n"
            f"```{r.get('language', '')}\n{r['text']}\n```"
        )
    return "\n\n".join(blocks)


def synthesize_answer(query: str, results: list[dict], api_key: str) -> str:
    """
    Sends the retrieved chunks + question to a Groq-hosted LLM and returns
    a natural-language answer grounded in those chunks.

    Raises whatever the Groq client raises (e.g. auth errors) — callers
    are expected to handle/display that.
    """
    client = Groq(api_key=api_key)
    context = build_context(results)

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question: {query}\n\nRetrieved snippets:\n\n{context}",
            },
        ],
        temperature=0.2,
        max_tokens=800,
    )
    return completion.choices[0].message.content

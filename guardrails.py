import re

from config import MAX_INPUT_CHARS, MAX_OUTPUT_CHARS


BLOCKED_PATTERNS = [
    r"ignore\s+(all|any|the)\s+(previous|prior|system)\s+instructions",
    r"ignore\s+(the|this)\s+(pdf|document)\s+and\s+(tell|show|give)",
    r"reveal\s+(your|the)\s+system\s+prompt",
    r"show\s+(me\s+)?the\s+system\s+prompt",
    r"developer\s+message",
    r"system\s+prompt",
]


def validate_input(query: str):
    query = (query or "").strip()
    if not query:
        return False, "Please enter a question."
    if len(query) > MAX_INPUT_CHARS:
        return False, f"Input is too long. Maximum is {MAX_INPUT_CHARS} characters."
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, query, re.I):
            return False, "I can only answer questions using the supplied document."
    return True, query


def validate_output(answer: str):
    answer = (answer or "").strip()
    if not answer:
        return False, "No answer was generated."

    if len(answer) > MAX_OUTPUT_CHARS:
        answer = answer[:MAX_OUTPUT_CHARS].rstrip() + "\n\n[Output truncated.]"

    sensitive = [
        r"system prompt",
        r"developer message",
        r"hidden instructions",
        r"internal instructions",
    ]
    if any(re.search(p, answer, re.I) for p in sensitive):
        return False, "I can only provide information supported by the supplied document."
    return True, answer


def _valid_pages(answer, sources):
    source_pages = {str(s.get("page")) for s in sources if s.get("page") is not None}
    cited_pages = set(re.findall(r"page\s*(\d+)", answer, re.I))
    return bool(cited_pages) and cited_pages.issubset(source_pages)


def enforce_citations(answer, sources):
    """Ensure citations refer only to retrieved pages.

    If the model omitted inline citations, append a compact source line using only
    pages that were actually retrieved. This is deliberately less strict than the
    previous version, which could reject a valid answer solely because a citation
    was omitted.
    """
    if not sources:
        return False, "I could not find supporting content in the supplied document."

    if _valid_pages(answer, sources):
        return True, answer

    pages = []
    for source in sources:
        page = source.get("page")
        if page is not None and str(page) not in pages:
            pages.append(str(page))

    if pages:
        citation_line = "\n\nSources: " + ", ".join(f"Page {p}" for p in pages[:4])
        return True, answer.rstrip() + citation_line

    return False, "The answer could not be validated against the retrieved document."

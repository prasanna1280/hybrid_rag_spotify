import re

from config import MAX_INPUT_CHARS, MAX_OUTPUT_CHARS


BLOCKED_PATTERNS = [
    r"ignore\s+(all|any|the)\s+(previous|prior)\s+instructions",
    r"reveal\s+(your|the)\s+system\s+prompt",
    r"show\s+(me\s+)?the\s+system\s+prompt",
    r"developer\s+message",
]


def validate_input(query: str):
    query = (query or "").strip()

    if not query:
        return False, "Please enter a question."

    if len(query) > MAX_INPUT_CHARS:
        return False, f"Input is too long. Maximum is {MAX_INPUT_CHARS} characters."

    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, query, re.I):
            return False, "The request contains an instruction-injection pattern."

    return True, query


def validate_output(answer: str):
    answer = (answer or "").strip()

    if not answer:
        return False, "No answer was generated."

    if len(answer) > MAX_OUTPUT_CHARS:
        answer = answer[:MAX_OUTPUT_CHARS].rstrip() + "\n\n[Output truncated.]"

    return True, answer


def enforce_citations(answer, sources):
    """Simple output guardrail: require source/page references for RAG answers."""
    if not sources:
        return False, "I could not find supporting content in the supplied document."

    if not re.search(r"(page\s*\d+|source\s*\d+)", answer, re.I):
        return False, (
            "The answer could not be validated against the retrieved document. "
            "Please ask a more specific question."
        )

    return True, answer

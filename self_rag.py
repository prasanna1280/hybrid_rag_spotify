from llm import json_chat, chat


SELF_RAG_SYSTEM = """
You are a Self-RAG controller. Answer only from the supplied retrieved context.
Do not invent facts. If context is insufficient, say so.

Return JSON with:
{
  "supported": true/false,
  "missing_information": "string",
  "answer": "string",
  "citations": ["Page 1", "Page 2"]
}
"""


def generate_answer(question, context):
    prompt = f"""
QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

Produce a grounded answer. Every substantive claim must be traceable to the
retrieved context. Cite page numbers in the answer.
"""
    return json_chat(SELF_RAG_SYSTEM, prompt)


def critique(question, answer, context):
    system = """
You are a strict RAG evaluator.
Check whether every important claim in the answer is supported by the context.
Return JSON:
{
  "grounded": true/false,
  "issues": ["..."],
  "needs_revision": true/false
}
"""

    prompt = f"""
QUESTION:
{question}

ANSWER:
{answer}

CONTEXT:
{context}
"""
    return json_chat(system, prompt)


def revise(question, answer, context, critique_result):
    system = """
Rewrite the answer using only the supplied context.
Remove unsupported claims. Keep page citations.
If the context cannot answer the question, explicitly state that.
"""

    prompt = f"""
QUESTION:
{question}

DRAFT:
{answer}

CRITIQUE:
{critique_result}

CONTEXT:
{context}
"""
    return chat(system, prompt)

from llm import json_chat


SELF_RAG_SYSTEM = """
You are a practical grounded Self-RAG answerer for a single technical architecture PDF.
Use ONLY the supplied retrieved context. Never use outside knowledge.

Answer directly when the context supports the question. It is acceptable for the answer
 to combine multiple passages. If only part is supported, answer the supported part and
state the limitation instead of refusing the whole question.

Citations must use only page numbers present in the supplied context, e.g. (Page 1).
Do not cite pages that are not present.

Return JSON only:
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

Instructions:
1. Answer only from the retrieved context.
2. Do not invent or supplement facts from general knowledge.
3. Use multiple retrieved passages when needed.
4. If only part is supported, provide that part and state the limitation.
5. Keep the answer concise but complete for the question.
6. Include page citations in the answer itself, e.g. (Page 1).
7. Return valid JSON matching the requested schema.
"""
    return json_chat(SELF_RAG_SYSTEM, prompt)


def critique(question, answer, context):
    system = """
You are a practical Self-RAG grounding evaluator.
Evaluate the answer against the retrieved context as a whole.

A response is grounded when its important factual claims are directly supported by one
or more supplied passages. Do NOT reject an answer merely because:
- it combines multiple passages;
- a citation was omitted (the output guardrail handles citations separately);
- the wording is a concise paraphrase of the source.

Set grounded=false only when an important claim is contradicted or unsupported by the
retrieved context. Minor wording issues should not cause rejection.

Return JSON only:
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

Judge whether the important claims are supported by the context as a whole.
If supported, use grounded=true and needs_revision=false.
If unsupported claims exist, identify them and request revision.
"""
    return json_chat(system, prompt)


def revise(question, answer, context, critique_result):
    system = """
Rewrite the answer using only the supplied context.
Preserve supported claims and remove unsupported claims.
If the context supports a partial answer, provide the partial answer rather than refusing.
Use only page numbers that occur in the supplied context.
Return JSON only:
{
  "answer": "string",
  "citations": ["Page 1", "Page 2"]
}
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
    return json_chat(system, prompt)

import json
from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_CHAT_MODEL, OPENAI_EMBEDDING_MODEL


client = OpenAI(api_key=OPENAI_API_KEY)


def chat(system, user, temperature=0):
    response = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content.strip()


def json_chat(system, user):
    response = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return json.loads(response.choices[0].message.content)


def embed_texts(texts):
    response = client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=texts,
    )
    return [x.embedding for x in response.data]

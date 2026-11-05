import math
from typing import List

from litellm import embedding

from ingestion import DocumentChunk

# Local embedding model served by Ollama (run: ollama pull nomic-embed-text).
# LiteLLM routes the "ollama/" prefix to the local Ollama server.
MODEL_NAME = "ollama/nomic-embed-text"
API_BASE = "http://localhost:11434"


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of strings via LiteLLM. Vectors line up with the input order."""
    response = embedding(model=MODEL_NAME, input=texts, api_base=API_BASE)
    # The response format mimics OpenAI's standard API
    return [item["embedding"] for item in response.data]


def embed_query(query: str) -> List[float]:
    """Embed a single user query."""
    return embed_texts([query])[0]


def embed_chunks(chunks: List[DocumentChunk]) -> List[List[float]]:
    """Embed DocumentChunks; the returned vectors line up with the input order."""
    return embed_texts([chunk.page_content for chunk in chunks])


def get_embedding(text: str) -> List[float]:
    """Converts a single text into a vector using LiteLLM."""
    return embed_query(text)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


def main():
    text1 = "The cat sat on the mat."
    text2 = "A cat is sitting on a mat."
    text3 = "I love programming in Python."

    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)
    emb3 = get_embedding(text3)

    print("Embedding length:", len(emb1))
    print("First 5 values:", emb1[:5])

    print("\nCosine similarities:")
    print("text1 vs text2:", cosine_similarity(emb1, emb2))
    print("text1 vs text3:", cosine_similarity(emb1, emb3))


if __name__ == "__main__":
    main()

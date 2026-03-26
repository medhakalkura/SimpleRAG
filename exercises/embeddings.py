"""
Exercise 1: Embeddings — get a vector for text and compare similarity.

TODO: Implement the functions below. Run with: python exercises/01_embeddings.py
"""

import os
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
import numpy as np

# load HF model once
_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

class HFSentenceTransformerEmbeddings(Embeddings):
    def __init__(self):
        # Load the HF model once
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Many texts -> many vectors
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        # One text -> one vector
        v = self.model.encode([text], normalize_embeddings=True)[0]
        return v.tolist()


# one global embedding engine
_emb = HFSentenceTransformerEmbeddings()


def get_embedding(text: str) -> list[float]:
    return _emb.embed_query(text)


def compare_texts(text_a: str, text_b: str) -> float:
    emb_a = get_embedding(text_a)
    emb_b = get_embedding(text_b)
    return float(
        np.dot(emb_a, emb_b) /
        (np.linalg.norm(emb_a) * np.linalg.norm(emb_b))
    )


def main():
    a = "A cozy coffee shop in the rain"
    b = "Warm café with rainy atmosphere"
    c = "The capital of France is Paris"

    emb_a = get_embedding(a)
    print(f"Embedding dimensions: {len(emb_a)}")

    sim_ab = compare_texts(a, b)
    sim_ac = compare_texts(a, c)
    print(f"Similarity (a,b): {sim_ab:.3f}")  # should be high
    print(f"Similarity (a,c): {sim_ac:.3f}")  # should be lower


if __name__ == "__main__":
    main()
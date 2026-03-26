"""
Exercise 4: Query the vector store and retrieve relevant chunks.

TODO: Implement load_store and retrieve.
Run with: python exercises/query_retrieval.py "your query here"
"""

import sys
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_core.documents import Document
from embeddings import HFSentenceTransformerEmbeddings


def load_store(persist_dir: str | Path):
    """Load the Chroma vector store from persist_dir."""
    emb = HFSentenceTransformerEmbeddings()
    return Chroma(
        persist_directory=str(persist_dir),
        embedding_function=emb,
        collection_name="prompt_guide",
    )


def retrieve(vector_store, query: str, k: int = 3) -> list[str]:
    """
    Return the top-k chunk texts most similar to the query.
    Hint: vector_store.similarity_search(query, k=k)
    """
    return vector_store.similarity_search(query, k=k)


def main():
    if len(sys.argv) < 2:
        print("Usage: python exercises/query_retrieval.py 'your query'")
        return

    query = " ".join(sys.argv[1:])
    project_root = Path(__file__).resolve().parent.parent
    persist_dir = project_root / "chroma_db"

    vs = load_store(persist_dir)
    chunks = retrieve(vs, query, k=1)
    print(f"Query: {query}\n")
    for i, c in enumerate(chunks, 1):
        print(f"--- Result {i} ---\n{c}\n")


if __name__ == "__main__":
    main()

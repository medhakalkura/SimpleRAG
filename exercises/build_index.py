"""
Exercise 3: Build a vector store from documents.

TODO: Implement build_index. Load .txt files, chunk, embed, store in Chroma.
Run with: python exercises/build_index.py
"""

from pathlib import Path
import os
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_core.documents import Document
from chunking import chunk_text
from embeddings import HFSentenceTransformerEmbeddings

def build_index(data_dir: str | Path, persist_dir: str | Path):
    """
    - Load all .txt files from data_dir
    - Split each into chunks (use RecursiveCharacterTextSplitter or your chunker)
    - Get embeddings for each chunk
    - Create a Chroma vector store and add the chunks
    - Persist to persist_dir
    - Return the vector store

    Hints:
    - from langchain_community.document_loaders import TextLoader
    - from langchain_text_splitters import RecursiveCharacterTextSplitter
    - from langchain_chroma import Chroma
    - Chroma.from_documents(documents=..., embedding=..., persist_directory=...)
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} does not exist")
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)
    #Load all .txt files from data_dir
    documents = []
    for file in os.listdir(data_dir):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(data_dir, file))
            documents.extend(loader.load())
            print(f"Loaded {file}")

    #Split each into chunks using chunker that we implemented)
    full_text = "\n\n".join(doc.page_content for doc in documents)
    chunks = chunk_text(full_text, chunk_size=200, chunk_overlap=50)  # now chunks is list[str]
    print(f"Split {len(documents)} document into {len(chunks)} chunks")

    # Create the embedding object (not a list of vectors)
    emb = HFSentenceTransformerEmbeddings()

    # Build Chroma index from texts
    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=emb,
        persist_directory=str(persist_dir),
        collection_name="prompt_guide",
    )
    return vector_store


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    persist_dir = project_root / "chroma_db"

    vs = build_index(data_dir, persist_dir)
    print(f"Index built. Persisted to {persist_dir}")
    # Quick sanity check: similarity_search with a random query
    docs = vs.similarity_search("Cozy and rustic", k=3)
    
    for i, doc in enumerate(docs, 1):
        print(f"--- Result {i} ---")
        print(doc.page_content)
        print()


if __name__ == "__main__":
    main()
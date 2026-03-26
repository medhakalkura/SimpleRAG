# RAG coding exercises

Work through these **in order**. Each file has **TODOs** (some may already be filled in as you iterate) — implement and understand the logic.

---

## Exercise 1: `embeddings.py`

**Goal:** Understand embeddings by calling a local embedding model.

- Load sample strings and get embedding vectors.
- Compare two strings: are their embeddings similar?
- Print the vector size (dimensions).

**Hints:** This repo uses `sentence_transformers` / a small LangChain-compatible wrapper in the same file. Alternatively you can experiment with `langchain_openai.OpenAIEmbeddings` or `langchain_community.embeddings.HuggingFaceEmbeddings` and set `OPENAI_API_KEY` if you choose that path.

**Run:** `python exercises/embeddings.py`

---

## Exercise 2: `chunking.py`

**Goal:** Split a document into chunks.

- Load `data/prompt_guide.txt`.
- Implement overlapping chunks: `chunk_size` and `chunk_overlap`.
- Return a list of strings (plain Python is fine).

**Example:** `chunk_text("hello world", chunk_size=5, chunk_overlap=2)` → something like `["hello", "lo wo", "world"]` depending on your rules.

**Run:** `python exercises/chunking.py`

---

## Exercise 3: `build_index.py`

**Goal:** Build a vector store from your documents.

- Load all `.txt` files from `data/`.
- Split into chunks (your chunker or LangChain’s `RecursiveCharacterTextSplitter`).
- Embed each chunk and store in **Chroma**, persisted under `chroma_db/`.

**Implement / verify:** `build_index(data_dir, persist_dir)` returns the vector store.

**Run:** `python exercises/build_index.py`

---

## Exercise 4: `query_retrieval.py`

**Goal:** Query the vector store and inspect retrieved chunks.

- Load the vector store from `chroma_db/`.
- Given a query string, return the top‑k most similar chunks.
- Print results to verify they make sense.

**API to aim for:** `retrieve(vector_store, query, k=3)` (see file for exact signature).

**Run:** `python exercises/query_retrieval.py "your query"`

---

## Exercise 5: `rag_chain.py`

**Goal:** Wire retrieval + LLM into a short pipeline.

- Retrieve context for the user query.
- Build a prompt that includes context + user idea.
- Call **Claude** and return an improved image prompt.
- Optionally call **`image_gen.py`** (needs `NVIDIA_API_KEY`).

**Run:** `python exercises/rag_chain.py "cozy coffee shop in the rain"`

---

## Environment variables

Set these in your shell or a **`.env`** file (never commit secrets):

| Variable | Used by |
|----------|---------|
| `ANTHROPIC_API_KEY` | `rag_chain.py` (Claude) |
| `NVIDIA_API_KEY` | `image_gen.py` / full chain in `rag_chain.py` |

Optional: `OPENAI_API_KEY` if you switch embeddings to OpenAI.

See **`.env.example`** in the project root for a template.

---

## How to run (from project root)

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env         # then edit .env with your keys

python exercises/embeddings.py
python exercises/chunking.py
python exercises/build_index.py
python exercises/query_retrieval.py "test query"
python exercises/rag_chain.py "your image idea"
```

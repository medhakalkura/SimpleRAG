# RAG Coding Exercises

Work through these in order. Each file has **TODOs** — you implement the logic.  
Use the docstrings and hints; avoid peeking at `src/` until you're stuck.

---

## Exercise 1: `01_embeddings.py`
**Goal:** Understand embeddings by calling an embedding model.

- Load a sample string and get its embedding vector.
- Compare two strings: are their embeddings similar?
- Print the shape of the vector (how many dimensions?).

**Hints:** Use `langchain_openai.OpenAIEmbeddings` or `langchain_community.embeddings.HuggingFaceEmbeddings`. Call `.embed_query("text")` and `.embed_documents(["a", "b"])`.

---

## Exercise 2: `02_chunking.py`
**Goal:** Split a document into chunks.

- Load `data/prompt_guide.txt`.
- Implement a simple chunker: split by `chunk_size` chars with `chunk_overlap` overlap.
- Return a list of strings. No LangChain yet — just plain Python.

**Example:** `chunk("hello world", size=5, overlap=2)` → `["hello", "lo wo", "world"]`

---

## Exercise 3: `03_build_index.py`
**Goal:** Build a vector store from your documents.

- Load all `.txt` files from `data/`.
- Split each into chunks (use your chunker or LangChain's `RecursiveCharacterTextSplitter`).
- Embed each chunk.
- Store in Chroma. Persist to `chroma_db/`.

**Implement:** `build_index(data_dir, persist_dir)` — returns the vector store.

---

## Exercise 4: `04_query_retrieval.py`
**Goal:** Query the vector store and see retrieved chunks.

- Load the vector store from `chroma_db/`.
- Given a user query string, return the top-k most similar chunks.
- Print them so you can verify they make sense.

**Implement:** `retrieve(query: str, k: int = 3) -> list[str]`

---

## Exercise 5: `05_rag_chain.py`
**Goal:** Wire retrieval + LLM into a simple chain.

- Retrieve context for the user query.
- Build a prompt: `"Context: {context}\n\nUser: {query}\n\nWrite an improved image prompt:"`
- Call Claude with that prompt.
- Return the enhanced prompt.

**Implement:** `enhance_prompt(user_query: str) -> str`

---

## How to run

From project root:
```bash
cd /Users/medhak/Projects/RAG
source venv/bin/activate
pip install -r requirements.txt
python exercises/01_embeddings.py
```

Set env vars: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` (see `.env.example`).

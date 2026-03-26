# Simple RAG: From Embeddings to Image Prompts

Hands-on exercises that build a small **retrieval-augmented** pipeline: chunk and index a local prompt guide, retrieve relevant context, use **Claude** to refine a user idea into a stronger image prompt, then call an image API. The runnable code lives under **`exercises/`**; this README summarizes concepts and how to run things.

---

## What’s in this repo

| Path | Role |
|------|------|
| **`data/`** | Text sources for RAG (e.g. `prompt_guide.txt`). |
| **`exercises/`** | Numbered learning scripts — see [exercises/README.md](exercises/README.md). |
| **`chroma_db/`** | Local Chroma persistence (created when you build the index). Safe to delete to rebuild; usually not committed to git. |
| **`requirements.txt`** | Python dependencies. |

There is **no** `src/` package in this snapshot: the workflow is implemented as the exercise modules and a short chain in `rag_chain.py`.

---

## Quick start

1. **Python 3.10+** recommended.

2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. **Environment variables** (never commit real keys). Copy the template and edit:

```bash
cp .env.example .env
```

- **`ANTHROPIC_API_KEY`** — required for `rag_chain.py` (Claude).
- **`NVIDIA_API_KEY`** — required if you run image generation in `image_gen.py` / `rag_chain.py` ([NVIDIA Build](https://build.nvidia.com/)).

Optional: if you switch embeddings to OpenAI per the exercise hints, set **`OPENAI_API_KEY`**.

4. **Run the exercises in order** (from the project root):

```bash
python exercises/embeddings.py
python exercises/chunking.py
python exercises/build_index.py
python exercises/query_retrieval.py "your search query"
python exercises/rag_chain.py "your idea for an image"
```

`rag_chain.py` loads the index from `chroma_db/`, retrieves context, calls Claude, then generates an image (if `NVIDIA_API_KEY` is set). On macOS it may open the saved image with the system viewer.

---

## Exercise map

| Order | File | You practice |
|------|------|----------------|
| 1 | `embeddings.py` | Embeddings, similarity between strings. |
| 2 | `chunking.py` | Splitting text into overlapping chunks. |
| 3 | `build_index.py` | Load `data/*.txt`, chunk, embed, persist Chroma to `chroma_db/`. |
| 4 | `query_retrieval.py` | Load the store and run similarity search. |
| 5 | `rag_chain.py` | Retrieve → prompt Claude → optional image generation. |

`image_gen.py` is used by the chain; it reads **`NVIDIA_API_KEY`** from the environment only (no keys in source).

More detail and hints: **[exercises/README.md](exercises/README.md)**.

---

## Concepts (short)

### RAG (Retrieval-Augmented Generation)

LLMs do not know your private notes or style guides. **RAG** means: **retrieve** relevant snippets from your documents, **augment** the model input with that text, then **generate** (here: a better image prompt).

### Embeddings and vector stores

- **Embedding:** a numeric vector capturing meaning; similar texts tend to have similar vectors.
- **Vector store:** stores vectors (and metadata) so you can fetch the top‑k chunks closest to a query embedding.

This project uses **Chroma** with **local Hugging Face** sentence embeddings (see `embeddings.py`).

### LangChain here

**LangChain** is used for document loading, Chroma integration, and the **Anthropic** chat model. The exercises are intentionally small scripts rather than a full LangGraph agent graph; you can extend toward **LangGraph** (nodes for retrieve / enhance / generate) once the linear pipeline is clear.

---

## Public repo checklist

- Use **`.env`** for secrets; keep **`.env`** out of git (see `.env.example` for variable names only).
- Do not commit API keys, tokens, or private datasets you do not intend to share.
- `chroma_db/` is machine-generated; add it to **`.gitignore`** if you do not want the index in the remote.

---

## Suggested learning path

1. Read the concept section above and skim [exercises/README.md](exercises/README.md).
2. Run exercises 1–4 until retrieval results look sensible.
3. Run `rag_chain.py` with your keys set; tweak the system prompt and `data/prompt_guide.txt`.
4. Optionally refactor retrieve → enhance → image into a **LangGraph** workflow or add a small UI.

Good luck.

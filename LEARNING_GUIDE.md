# Learning Guide: Text-to-Image RAG Application

This guide walks you through the concepts and steps to build your LangChain-based image generation application. Read each section before coding that part.

---

## Part 1: Core Concepts

### 1.1 What is RAG (Retrieval Augmented Generation)?

**The problem:** LLMs like Claude are trained on data up to a cutoff date. They don’t know your private docs, latest styles, or domain-specific “how to describe images” rules. So raw prompts can be vague or off-style.

**The idea:**  
**RAG = Retrieve** relevant information from your own documents, then **Augment** the LLM’s prompt with that context so it **Generates** better answers (or in our case, better image prompts).

```
User: "Draw something like our hero style"
         ↓
   [Retrieve] → Search your "style guide" docs → "Hero: bold colors, 16:9, cinematic"
         ↓
   [Augment] → Add that text to the prompt
         ↓
   [Generate] → LLM gets: "Draw X. Style: bold colors, 16:9, cinematic" → better image prompt
```

**In this project:** We use RAG to store **prompt-improvement knowledge** (e.g. “how to describe art styles”, “good DALL·E/Flux prompt patterns”). When the user says “make it pop” we retrieve similar examples and inject them into the prompt so the final image prompt is clearer and more consistent.

---

### 1.2 Embeddings and Vector Stores (the “Retrieve” in RAG)

- **Embedding:** A list of numbers that represents the *meaning* of a piece of text. Similar meanings → similar vectors. So we can search by *semantics*, not just keywords.
- **Vector store:** A database that stores these vectors and lets you do “find the top‑k most similar texts to this query.”

Flow:

1. **Indexing:** Split your docs into chunks → get an embedding for each chunk → save (chunk text + vector) in a vector store.
2. **Retrieval:** User query → embed query → search vector store for nearest vectors → return the corresponding text chunks.

So “semantic search” = search by embedding similarity (e.g. cosine similarity).

---

### 1.3 What is LangChain?

**LangChain** is a library that gives you building blocks for LLM apps:

- **Models:** Connect to Claude, OpenAI, etc. with a common interface.
- **Prompts:** Templates (e.g. “You are an expert. Context: {context}. User: {question}”).
- **Chains:** Wire model + prompt + optional tools (e.g. “retrieve → then generate”) in a simple pipeline.
- **Vector stores:** Integrations with Chroma, FAISS, etc., and helpers to create them from documents.
- **Document loaders & splitters:** Load PDFs, text, etc. and split into chunks for RAG.

You use it to: load docs → embed → store in a vector store → run a chain that retrieves and then calls Claude.

---

### 1.4 What is LangGraph?

**LangGraph** extends LangChain with **graphs** and **state**:

- **Nodes:** Steps (e.g. “retrieve context”, “call LLM”, “call image API”).
- **Edges:** What runs after what (linear or conditional).
- **State:** A shared dictionary that flows through the graph; each node can read and update it.

So you can build **agents** that:

- Decide which tool to call (e.g. “search knowledge base” vs “generate image”).
- Loop (e.g. “if prompt is not good enough, refine and try again”).
- Keep context in state (user message, retrieved docs, enhanced prompt, image URL).

**In this project:** We use LangGraph to define a small workflow: **retrieve** from RAG → **enhance prompt** with Claude → **generate image** (and optionally **evaluate** and loop). Claude acts as the “agent” that uses the retrieved context to improve the prompt.

---

### 1.5 MCP (Model Context Protocol) and Agents — For Later

- **MCP** is a protocol so that an AI model (e.g. Claude in Cursor) can call **tools** exposed by **MCP servers** (e.g. “search my notes”, “run a build”).
- **Agents** are systems that use an LLM to decide *which* tool to call and with *what* arguments, then act on the result.

For this project we focus on **one agent (Claude)** inside our app using LangGraph; you can add MCP later so that the same Claude (e.g. in Cursor) can trigger your image pipeline via an MCP server. The learning path stays: RAG → LangChain → LangGraph → then MCP.

---

## Part 2: Application Architecture

High-level flow:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  User: "A cozy coffee shop in the rain"                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 1: RAG (LangChain)                                                │
│  • Embed user query                                                     │
│  • Search vector store for similar "prompt tips" / style docs           │
│  • Return top-k chunks as context                                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 2: Prompt enhancement (LangGraph + Claude)                         │
│  • Node: inject retrieved context + user prompt into a system message   │
│  • Claude: output a refined, detailed image generation prompt           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 3: Image generation                                               │
│  • Call image API (e.g. Anthropic images, or OpenAI DALL·E, etc.)       │
│  • Return image to user                                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Part 3: Step-by-Step Build Plan

Build in this order so each step teaches one concept.

| Step | What you build | What you learn |
|------|----------------|----------------|
| **Step 1** | Load a small “prompt guide” (text file), embed it, put it in a vector store (Chroma). Query it from the command line. | RAG basics: embeddings, chunking, vector store, retrieval. |
| **Step 2** | A LangGraph graph: “retrieve” node → “enhance prompt” node (Claude). Input: user text. Output: enhanced prompt. | LangChain prompts, LangGraph nodes/state, using Claude as the “agent”. |
| **Step 3** | Add an “image generation” node (e.g. Anthropic Images API or another provider). Wire it to Step 2. Add a simple CLI or script to run the full pipeline. | End-to-end pipeline: RAG → enhanced prompt → image. |
| **Step 4** (optional) | Add a simple UI (e.g. Streamlit or FastAPI + HTML) and/or an MCP server so Cursor/Claude can call your app. | Integration and deployment. |

---

## Part 4: What You’ll Have in This Repo

- **`data/`** – Sample text file(s) with “prompt improvement” tips (your RAG content).
- **`src/`** – Python package:
  - **`retrieval.py`** – Build vector store from docs, query it (Step 1).
  - **`graph.py`** – LangGraph workflow: retrieve → enhance (→ generate) (Steps 2–3).
  - **`image_gen.py`** – Call the image API (Step 3).
- **`scripts/`** – Scripts to build the index and run the pipeline.
- **`requirements.txt`** – Dependencies (langchain, langgraph, anthropic, chromadb, etc.).

---

## Part 5: Order of Learning (Recommended)

1. Read **Part 1** (concepts) and **Part 2** (architecture) above.
2. Implement **Step 1** (retrieval only). Run queries and see retrieved chunks. Understand embeddings and vector store.
3. Implement **Step 2** (graph with retrieve + enhance). See how context from RAG changes Claude’s output.
4. Implement **Step 3** (image generation). Run the full pipeline.
5. Experiment: add more documents to RAG, change the system prompt, try different image APIs.
6. (Later) Add MCP server or a simple UI.

Use the comments in the code and this guide together: each file will point back to which “step” and which concept it implements. Good luck building.

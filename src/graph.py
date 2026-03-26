"""
Step 2 (and 3): LangGraph workflow — retrieve → enhance prompt → generate image.

Concepts:
- State: a dict that flows through the graph (user_prompt, context, enhanced_prompt, image_url, etc.).
- Nodes: functions that take state, do work, return state updates.
- Edges: which node runs next (here: linear chain).

We use TypedDict for state so the graph knows the schema.
"""

import os
from typing import TypedDict

from langgraph.graph import StateGraph, END, START
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

# Local retrieval (Step 1) and image generation (Step 3)
try:
    from src.retrieval import load_vector_store, retrieve_context
    from src.image_gen import generate_image
except ImportError:
    from retrieval import load_vector_store, retrieve_context
    from image_gen import generate_image


# ----- State schema -----
# Every node receives the full state and can return a partial update (merged into state).
class PromptImageState(TypedDict):
    user_prompt: str
    retrieved_context: str
    enhanced_prompt: str
    image_url: str
    error: str


def _get_llm():
    """Claude for prompt enhancement. Requires ANTHROPIC_API_KEY."""
    return ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=1024,
    )


def retrieve_node(state: PromptImageState) -> dict:
    """
    Node 1: RAG retrieval.
    Reads user_prompt from state, queries the vector store, returns retrieved_context.
    """
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
    try:
        vs = load_vector_store(persist_dir)
        context = retrieve_context(vs, state["user_prompt"], k=3)
    except Exception as e:
        context = f"(Retrieval failed: {e}. Proceeding without RAG context.)"
    return {"retrieved_context": context}


def enhance_prompt_node(state: PromptImageState) -> dict:
    """
    Node 2: Use Claude to turn the user prompt + retrieved context into a single,
    detailed image-generation prompt (no code, just the improved prompt text).
    """
    llm = _get_llm()
    system = """You are an expert at writing prompts for text-to-image models (e.g. DALL·E, Stable Diffusion).
Given the user's idea and the retrieved guidance below, output ONE improved, detailed prompt that describes
the image clearly: style, composition, lighting, mood, and any key details. Output only the prompt, no explanation."""

    user_content = f"""Retrieved guidance (use to improve the prompt):

{state["retrieved_context"]}

User's idea: {state["user_prompt"]}

Write the enhanced image prompt (one paragraph, no code):"""

    messages = [
        SystemMessage(content=system),
        HumanMessage(content=user_content),
    ]
    try:
        response = llm.invoke(messages)
        enhanced = response.content.strip() if hasattr(response, "content") else str(response)
    except Exception as e:
        enhanced = state["user_prompt"]
        return {"enhanced_prompt": enhanced, "error": str(e)}
    return {"enhanced_prompt": enhanced, "error": ""}


def image_generation_node(state: PromptImageState) -> dict:
    """
    Node 3: Call the image API with the enhanced prompt.
    Returns image_url (or error in state).
    """
    prompt = state.get("enhanced_prompt") or state["user_prompt"]
    try:
        url = generate_image(prompt)
        return {"image_url": url, "error": ""}
    except Exception as e:
        return {"image_url": "", "error": str(e)}


def build_graph():
    """
    Build the LangGraph: START → retrieve → enhance → image_gen → END.
    """
    graph = StateGraph(PromptImageState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("enhance", enhance_prompt_node)
    graph.add_node("generate_image", image_generation_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "enhance")
    graph.add_edge("enhance", "generate_image")
    graph.add_edge("generate_image", END)

    return graph.compile()


def run_pipeline(user_prompt: str, persist_dir: str | None = None) -> dict:
    """
    Run the full pipeline: RAG → enhance prompt → generate image.
    Initial state only has user_prompt; other fields are filled by nodes.
    """
    if persist_dir:
        os.environ["CHROMA_PERSIST_DIR"] = persist_dir
    compiled = build_graph()
    initial: PromptImageState = {
        "user_prompt": user_prompt,
        "retrieved_context": "",
        "enhanced_prompt": "",
        "image_url": "",
        "error": "",
    }
    result = compiled.invoke(initial)
    return result

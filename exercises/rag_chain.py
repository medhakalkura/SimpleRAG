"""
Exercise 5: RAG chain — retrieve context, then call Claude to enhance the prompt.

TODO: Implement enhance_prompt. This is the core RAG flow.
Run with: python exercises/05_rag_chain.py "cozy coffee shop in the rain"
"""

import sys
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from image_gen import generate_image
from query_retrieval import load_store, retrieve

load_dotenv()

def enhance_prompt(user_query: str) -> str:
    # 1. Load vector store
    project_root = Path(__file__).resolve().parent.parent
    persist_dir = project_root / "chroma_db"
    vs = load_store(persist_dir)

    # 2. Retrieve context (top-3 chunks)
    chunks = retrieve(vs, user_query, k=3)
    context_text = "\n\n---\n\n".join(doc.page_content for doc in chunks)

    # 3. Build messages for Claude
    system = (
        "You are an expert at writing prompts for text-to-image models. "
        "Use the context to improve the user's idea into a detailed, clear image prompt."
    )
    user = (
        f"Context from prompt guide:\n{context_text}\n\n"
        f"User idea: {user_query}\n\n"
        "Write one improved image prompt."
    )

    llm = ChatAnthropic(
        #model="claude-3-5-sonnet-20241022",
        model="claude-3-haiku-20240307",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=512,
    )

    messages = [
        SystemMessage(content=system),
        HumanMessage(content=user),
    ]

    response = llm.invoke(messages)
    return response.content.strip()

def main():
    if len(sys.argv) < 2:
        print("Usage: python exercises/rag_chain.py 'your idea'")
        return

    query = " ".join(sys.argv[1:])
    project_root = Path(__file__).resolve().parent.parent
    persist_dir = project_root / "chroma_db"

    enhanced = enhance_prompt(query)
    print("Enhanced prompt:", enhanced)

    image_path = generate_image(enhanced)   # enhanced prompt is the input
    print(f"Image saved to: {image_path}") 

    subprocess.run(["open", image_path])  # Opens in Preview on macOS 


if __name__ == "__main__":
    main()

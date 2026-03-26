"""
Exercise 2: Chunking — split a document into overlapping chunks.

TODO: Implement chunk_text. Use only standard library + pathlib.
Run with: python exercises/02_chunking.py
"""

from pathlib import Path


def chunk_text(text: str, chunk_size: int = 200, chunk_overlap: int = 50) -> list[str]:
    """
    Split text into chunks of at most chunk_size chars, with chunk_overlap
    characters overlapping between consecutive chunks.

    Example: chunk_text("hello world", size=5, overlap=2)
             -> ["hello", "lo wo", "world"]  (or similar, depending on your logic)
    """
    chunks = []
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks


def load_and_chunk(filepath: Path) -> list[str]:
    """Load file and return list of chunks."""
    text = filepath.read_text(encoding="utf-8")
    return chunk_text(text, chunk_size=300, chunk_overlap=50)


def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    guide = data_dir / "prompt_guide.txt"
    if not guide.exists():
        print("Create data/prompt_guide.txt first")
        return

    chunks = load_and_chunk(guide)
    print(f"Got {len(chunks)} chunks")
    for i, c in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} (len={len(c)}) ---\n{c[:150]}...")

if __name__ == "__main__":
    main()
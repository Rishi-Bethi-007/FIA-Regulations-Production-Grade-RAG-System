from typing import List

def chunk(text: str, chunk_size: int=400, overlap: int=100) -> List[str]:
    """
    Fixed-size chunking with overlap (by characters).

    Overlap helps prevent boundary losses:
    - Higher recall
    - More chunks (higher cost)
    """
    text = text.strip()  # Remove leading/trailing whitespace
    if not text:
        return []  # Return empty list for empty input
    
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")
    
    step = chunk_size - overlap # Calculate step size based on overlap

    chunks: List[str] = []
    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks
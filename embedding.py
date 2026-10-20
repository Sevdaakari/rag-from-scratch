import ollama
from chunking import load_documents, chunk_text_improved

def embed_text(text):
    """
    Convert a single piece of text into an embedding vector.

    Args:
        text (str): the text to embed

    Returns:
        list[float]: the embedding vector representing this text
    """
    response = ollama.embeddings(model='nomic-embed-text', prompt=text)
    return response['embedding']


if __name__ == "__main__":
    docs = load_documents("docs")

    all_chunks = []
    for filename, text in docs.items():
        for sentence in chunk_text_improved(text):
            all_chunks.append({"source": filename, "text": sentence})

    # Embed every chunk, attaching the vector to its own dict
    for chunk in all_chunks:
        chunk['embedding'] = embed_text(chunk['text'])

    print(f"Embedded {len(all_chunks)} chunks.")
    print(f"Example chunk keys: {list(all_chunks[0].keys())}")
    print(f"Vector length check: {len(all_chunks[0]['embedding'])}")
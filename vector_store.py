import numpy as np
from chunking import load_documents, chunk_text_improved
from embedding import embed_text

def cosine_similarity(vec_a, vec_b):
    """
    Compute the cosine similarity between two vectors.

    Measures the angle between two vectors rather than raw distance,
    so it captures similarity in meaning/direction, not magnitude.

    Args:
        vec_a (list[float]): first vector
        vec_b (list[float]): second vector

    Returns:
        float: similarity score between -1 and 1 (1 = identical direction)
    """
    a = np.array(vec_a)
    b = np.array(vec_b)

    dot_product = np.dot(a, b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)

    return dot_product / (magnitude_a * magnitude_b)


def search(query, all_chunks, top_k=3):
    """
    Find the most similar chunks to a query, by cosine similarity.

    Args:
        query (str): the user's question
        all_chunks (list[dict]): chunks, each with a precomputed 'embedding'
        top_k (int): how many top matches to return

    Returns:
        list[dict]: the top_k chunks, sorted by similarity (highest first)
    """
    query_vector = embed_text(query)

    # Score every chunk against the query
    for chunk in all_chunks:
        chunk['score'] = cosine_similarity(query_vector, chunk['embedding'])

    # Sort by score, highest first, and take the top_k
    sorted_chunks = sorted(all_chunks, key=lambda c: c['score'], reverse=True)
    return sorted_chunks[:top_k]


if __name__ == "__main__":
    docs = load_documents("docs")

    all_chunks = []
    for filename, text in docs.items():
        for sentence in chunk_text_improved(text):
            all_chunks.append({"source": filename, "text": sentence})

    for chunk in all_chunks:
        chunk['embedding'] = embed_text(chunk['text'])

    query = "How often should I change the oil on a Caterpillar excavator?"
    results = search(query, all_chunks, top_k=3)

    print(f"Query: {query}\n")
    for r in results:
        print(f"Score: {r['score']:.4f} | [{r['source']}] {r['text']}")
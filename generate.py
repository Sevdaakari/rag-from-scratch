import ollama
from chunking import load_documents, chunk_text
from embedding import embed_text
from vector_store import search

def build_prompt(query, retrieved_chunks):
    """
    Construct a prompt that grounds the LLM's answer in retrieved context.

    Args:
        query (str): the user's original question
        retrieved_chunks (list[dict]): top chunks from search(), each with 'text'

    Returns:
        str: the full prompt to send to the LLM
    """
    context = "\n".join([f"- {c['text']}" for c in retrieved_chunks])

    prompt = f"""Answer the question using ONLY the context below. If the context doesn't contain the answer, say you don't know.

Context:
{context}

Question: {query}

Answer:"""
    return prompt


if __name__ == "__main__":
    docs = load_documents("docs")

    all_chunks = []
    for filename, text in docs.items():
        for sentence in chunk_text(text):
            all_chunks.append({"source": filename, "text": sentence})

    for chunk in all_chunks:
        chunk['embedding'] = embed_text(chunk['text'])

    query = "How often should I change the oil on a Caterpillar excavator?"
    retrieved = search(query, all_chunks, top_k=3)

    prompt = build_prompt(query, retrieved)
    print("--- PROMPT SENT TO LLM ---")
    print(prompt)
    print("---------------------------\n")

    response = ollama.chat(model='qwen2:7b', messages=[
        {'role': 'user', 'content': prompt}
    ])

    print("ANSWER:")
    print(response['message']['content'])

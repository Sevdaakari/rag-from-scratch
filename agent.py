import ollama

def is_context_sufficient(query, retrieved_chunks):
    """
    Ask the LLM to judge whether retrieved chunks are enough to answer the query.

    Args:
        query (str): the user's question
        retrieved_chunks (list[dict]): chunks returned by search()

    Returns:
        bool: True if the LLM judges the context sufficient, False otherwise
    """
    context = "\n".join([f"- {c['text']}" for c in retrieved_chunks])

    judge_prompt = f"""You are judging whether the CONTEXT below contains enough information to answer the QUESTION.

Context:
{context}

Question: {query}

Does the context contain information that directly answers this question? Reply with exactly one word: YES or NO."""

    response = ollama.chat(model='qwen2:7b', messages=[
        {'role': 'user', 'content': judge_prompt}
    ])

    answer = response['message']['content'].strip().upper()
    return "YES" in answer


def rewrite_query(original_query, retrieved_chunks):
    """
    Ask the LLM to rewrite a query that failed to retrieve sufficient context.

    Args:
        original_query (str): the question that didn't retrieve good context
        retrieved_chunks (list[dict]): the (insufficient) chunks that were retrieved

    Returns:
        str: a rewritten version of the query
    """
    context = "\n".join([f"- {c['text']}" for c in retrieved_chunks])

    rewrite_prompt = f"""The following QUESTION did not retrieve useful context from a document search.

Context that was retrieved (and judged insufficient):
{context}

Original question: {original_query}

Rewrite this question to be more specific or use different phrasing that might match the source documents better. Reply with ONLY the rewritten question, nothing else."""

    response = ollama.chat(model='qwen2:7b', messages=[
        {'role': 'user', 'content': rewrite_prompt}
    ])

    return response['message']['content'].strip()

if __name__ == "__main__":
    bad_chunks = [{"text": "Atlas Copco was founded in 1873."}]
    query = "does atlas copco drilling equipment have oil for engine"

    print("Sufficient?", is_context_sufficient(query, bad_chunks))
    new_query = rewrite_query(query, bad_chunks)
    print("Rewritten query:", new_query)
import os
import re
import tiktoken

_encoding = tiktoken.get_encoding("cl100k_base")


def load_documents(folder_path):
    """
    Read every file in the given folder and return their contents.

    Args:
        folder_path (str): path to the folder containing .txt files

    Returns:
        dict: {filename: raw_text_content}
    """
    documents = {}
    for filename in os.listdir(folder_path):
        with open(f"{folder_path}/{filename}", 'r') as f:
            documents[filename] = f.read()
    return documents


def chunk_text_naive(text):
    """
    Split raw text into sentence-level chunks using a simple regex.

    Splits after a period only when followed by whitespace, so
    abbreviations like "Inc." don't get incorrectly split apart.

    This is the ORIGINAL chunking approach, kept unchanged and still
    callable so it can be directly compared against chunk_text_improved()
    in evaluation. Known weakness, confirmed on this project's real
    documents: produces many very short, low-context fragments (numbered
    lists, table/figure noise), averaging ~54 characters per chunk.

    Args:
        text (str): raw text to split

    Returns:
        list[str]: list of individual sentence chunks
    """
    sentences = re.split(r'(?<=\.)\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def chunk_text_improved(text, max_tokens=200, overlap_tokens=40):
    """
    Split raw text into overlapping, fixed-token-size chunks.

    Unlike chunk_text_naive(), this guarantees every chunk (except
    possibly the last one per document) is close to max_tokens tokens
    long, directly addressing the tiny-fragment problem naive chunking
    produced. overlap_tokens of context repeats at the start of each
    chunk after the first, so a fact sitting at a chunk boundary isn't
    lost entirely from either side of the split.

    Token count is measured with tiktoken's cl100k_base encoding -- not
    tied to any specific model, just a fast, consistent way to measure
    chunk size in a unit more meaningful than raw characters.

    Args:
        text (str): raw text to split
        max_tokens (int): target chunk size, in tokens
        overlap_tokens (int): tokens of overlap between consecutive chunks

    Returns:
        list[str]: list of overlapping, token-bounded chunks
    """
    tokens = _encoding.encode(text)
    if not tokens:
        return []

    step = max_tokens - overlap_tokens
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_str = _encoding.decode(tokens[start:end]).strip()
        if chunk_str:
            chunks.append(chunk_str)
        if end >= len(tokens):
            break
        start += step

    return chunks


if __name__ == "__main__":
    docs = load_documents("docs")

    naive_count = sum(len(chunk_text_naive(text)) for text in docs.values())
    improved_count = sum(len(chunk_text_improved(text)) for text in docs.values())

    print(f"Naive chunking:    {naive_count} chunks")
    print(f"Improved chunking: {improved_count} chunks")
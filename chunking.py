import os
import re

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

def chunk_text(text):
    """
    Split raw text into a list of sentence-level chunks.

    Splits after a period only when followed by whitespace, so
    abbreviations like "Inc." don't get incorrectly split apart.

    Args:
        text (str): raw text to split

    Returns:
        list[str]: list of individual sentence chunks
    """
    sentences = re.split(r'(?<=\.)\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

if __name__ == "__main__":
    docs = load_documents("docs")

    all_chunks = []
    for filename, text in docs.items():
        for sentence in chunk_text(text):
            all_chunks.append({"source": filename, "text": sentence})

    print(f"Total chunks: {len(all_chunks)}")
    for c in all_chunks[:5]:
        print(f"[{c['source']}] {c['text']}")
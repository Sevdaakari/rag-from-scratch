# RAG Agent — Learning Template

A minimal, from-scratch Retrieval-Augmented Generation (RAG) agent, built without any 
framework, to understand exactly how each piece works before adopting heavier tools.

Runs fully locally using [Ollama](https://ollama.com) — no API keys, no cloud calls.

## How it works

1. **Chunking** (`chunking.py`) — splits `.txt` documents into sentence-level chunks
2. **Embedding** (`embedding.py`) — converts each chunk into a vector using `nomic-embed-text`
3. **Vector store** (`vector_store.py`) — cosine similarity search over chunks, implemented from scratch with numpy
4. **Generation** (`generate.py`) — builds a context-grounded prompt and calls `qwen2:7b`
5. **Agent loop** (`agent.py`) — judges whether retrieved context is sufficient; if not, rewrites 
   the query and retries (up to a limit) before falling back to "I don't know"
6. **Entry point** (`main.py`) — wires everything together into an interactive chat loop with memory

## Setup

1. Install Ollama: https://ollama.com/download
2. Pull the required models:
```bash
   ollama pull qwen2:7b
   ollama pull nomic-embed-text
```
3. Create and activate a virtual environment:
```bash
   python3 -m venv .venv
   source .venv/bin/activate
```
4. Install dependencies:
```bash
   pip install -r requirements.txt
```
5. (Optional) Verify Ollama is reachable:
```bash
   python3 scripts/check_ollama_connection.py
```

## Usage

```bash
python3 main.py
```

Ask questions about the sample documents in `docs/` (heavy equipment specs, maintenance 
schedules, and company info for Caterpillar / Atlas Copco). Type `quit` to exit.

## Known limitations (intentional, for this learning template)

- Sentence chunking is naive (`re.split` on periods) — breaks on abbreviations like "Inc."
- No persistence — conversation history and embeddings are recomputed every run
- Agentic loop can rewrite queries but cannot retrieve facts that aren't in `docs/`
- `max_attempts` is hardcoded to 2 in `main.py`

## Next steps

- Add persistence (save embeddings/history to disk)
- Add a proper vector database for scale (FAISS, Chroma) instead of the from-scratch numpy version
- Expand the knowledge base
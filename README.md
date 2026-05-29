# PDF Question Answering RAG

A Retrieval-Augmented Generation (RAG) application built using:

- Python
- LangChain
- OpenAI Embeddings
- FAISS
- PyPDF
- Streamlit

## Features

- Upload PDF
- Chunk documents
- Generate embeddings
- Store vectors in FAISS
- Semantic search
- Question answering using retrieved context

## Architecture

PDF
→ Loader
→ Chunking
→ Embeddings
→ FAISS
→ Retrieval
→ LLM
→ Answer

## Setup

```bash
pip install -r requirements.txt
```

Create `.env`

```env
OPENAI_API_KEY=your_api_key
```

Run:

```bash
python app.py
```

# News Research Tool

This project is a local-first news research and question answering system.

Users provide multiple news article URLs. The system extracts the content, builds semantic embeddings, and allows users to ask questions that are answered strictly using the provided sources.

## Features
- URL-based article ingestion
- Semantic search using FAISS
- Local LLM-powered question answering
- Source-aware responses
- No paid APIs

## Tech Stack
- Python
- Streamlit
- LangChain
- FAISS
- Ollama (Local LLM)
- Sentence Transformers

## How to Run

1. Install dependencies
pip install -r requirements.txt

markdown
Copy code

2. Install and start Ollama
ollama pull llama3

markdown
Copy code

3. Run the app
streamlit run app.py

sql
Copy code

## Notes
- All embeddings and indexes are generated locally
- No external paid services are used
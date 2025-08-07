# HackRx 6.0 - LLM Document Processing System

## Overview
An intelligent query-retrieval system that processes natural language queries against large documents using LLMs and semantic search.

## Features
- PDF, DOCX, and email document processing
- Natural language query understanding
- Semantic search with Pinecone/FAISS
- Explainable AI decisions
- Real-time API responses

## Setup

1. Clone the repository
2. Copy `.env.example` to `.env` and fill in your API keys
3. Install dependencies: `pip install -r requirements.txt`
4. Run the application: `uvicorn app.main:app --reload`

## API Usage
import requests
response = requests.post("http://localhost:8000/hackrx/run", json={
"documents": "document_url_here",
"questions": ["Your question here"]
})


## Architecture
- FastAPI backend
- OpenAI GPT-4 for reasoning
- Pinecone for vector storage
- Sentence transformers for embeddings



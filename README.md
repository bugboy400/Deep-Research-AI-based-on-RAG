# Deep-Research-AI-based-on-RAG

# Deep Research AI (RAG-Based System)

Deep Research AI is a Retrieval-Augmented Generation (RAG) system that combines web search, document processing, and large language models to generate accurate, context-aware answers.

This project is designed to demonstrate how modern AI systems integrate retrieval and reasoning to answer real-world questions.

---

## Overview

The system takes a user query, gathers relevant information from multiple sources, and generates a response grounded in retrieved context instead of relying only on the model’s internal knowledge.

It supports both web-based information and user-provided documents.

---

## Tech Stack

### Backend

* FastAPI
* LangChain

### Frontend

* HTML
* CSS
* JavaScript

### Models

* Groq (Qwen-32B) – language model
* Sentence Transformers – embeddings
* CrossEncoder – reranking

### Data & Retrieval

* ChromaDB – vector database
* SERP API – web search
* Newspaper3k – article extraction

---

## Pipeline

```
User Query
   ↓
Query Expansion (LLM)
   ↓
Web Search (SERP API)
   ↓
Parallel Web Scraping
   ↓
Document Cleaning
   ↓
Text Chunking
   ↓
Embedding Generation
   ↓
Vector Storage (ChromaDB)
   ↓
Similarity Search
   ↓
Reranking (CrossEncoder)
   ↓
Final Answer Generation (LLM)
```

---

## Features

* Retrieval-Augmented Generation (RAG)
* Combines web data and local files
* Parallel scraping for improved speed
* Persistent vector database
* Reranking for better relevance
* Basic caching for repeated queries
* Error handling across pipeline stages

---

## Supported Inputs

* PDF documents
* Image files (OCR)
* Text files
* Web content

---

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the backend:

```bash
uvicorn main:app --reload
```

Open the frontend in your browser (HTML file) and connect it to the API endpoint.

---

## Example

```python
run_rag("What is Retrieval Augmented Generation?")
```

---

## What I Learned

* How RAG systems work end-to-end
* Vector databases and embeddings
* Information retrieval and reranking
* Integrating LLMs into real applications
* Building full-stack AI systems with FastAPI and frontend

---

## Future Improvements

* Add chat history (multi-turn conversation)
* Improve UI/UX
* Add streaming responses
* Use Redis for caching
* Deploy the system on cloud

---

## Author

Built as part of my learning journey in applied AI and Retrieval-Augmented Generation systems.

# ================== IMPORTS ==================
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from serpapi import GoogleSearch
from newspaper import Article

from concurrent.futures import ThreadPoolExecutor

import hashlib
import re
import os
from dotenv import load_dotenv

# ================== SETUP ==================
load_dotenv()

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

llm = ChatGroq(
    model="qwen/qwen3-32b",
    api_key=os.getenv("GROQ_API_KEY")
)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

SERP_API_KEY = os.getenv("serpapi_API_KEY")

# Simple cache
CACHE = {}

# ================== HELPERS ==================

def get_cache_key(question):
    return hashlib.md5(question.encode()).hexdigest()


def extract_text(file_path):
    if not file_path:
        return ""

    if file_path.endswith(".pdf"):
        docs = PyPDFLoader(file_path).load()
        text = "\n".join([d.page_content for d in docs])

    elif file_path.endswith((".png", ".jpg", ".jpeg")):
        import pytesseract
        from PIL import Image
        text = pytesseract.image_to_string(Image.open(file_path))

    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        return ""

    return re.sub(r'\s+', ' ', text).strip()


def clean_and_validate_chunks(chunks):
    valid_chunks = []
    for chunk in chunks:
        content = chunk.page_content.strip()

        if not content:
            continue
        if len(content) < 20:
            continue
        if not re.search(r'[a-zA-Z0-9]', content):
            continue

        chunk.page_content = content
        valid_chunks.append(chunk)

    return valid_chunks


# ================== SCRAPING ==================

def fetch_article(url):
    try:
        article = Article(url)
        article.download()
        article.parse()

        text = re.sub(r'\s+', ' ', article.text).strip()

        if len(text) > 100:
            return Document(page_content=text, metadata={"source": url})
    except:
        return None


def scrape_urls_parallel(urls):
    documents = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = executor.map(fetch_article, urls)

    for doc in results:
        if doc:
            documents.append(doc)

    return documents


# ================== VECTOR DB ==================

def get_vector_store():
    return Chroma(
        persist_directory="db",
        embedding_function=embedding_model
    )


# ================== MAIN FUNCTION ==================

def run_rag(question, file_path=None):

    # ===== CACHE =====
    cache_key = get_cache_key(question)
    if cache_key in CACHE:
        return CACHE[cache_key]

    # ===== VALIDATION =====
    if not question.strip():
        return "Question cannot be empty."

    # ===== FILE TEXT =====
    text = extract_text(file_path) if file_path else ""

    # ===== QUERY EXPANSION (1 LLM CALL) =====
    try:
        response = llm.invoke(f"""
        Break this into 2 short search queries:
        {question}
        """)

        clean = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL)
        sub_questions = re.findall(r"^\d+\.\s(.*)", clean, re.MULTILINE)

    except:
        sub_questions = []

    queries = list(set([question] + sub_questions))

    # ===== WEB SEARCH =====
    urls = []

    for q in queries:
        try:
            params = {
                "q": q,
                "engine": "google",
                "api_key": SERP_API_KEY
            }

            results = GoogleSearch(params).get_dict()

            urls += [
                item.get("link")
                for item in results.get("organic_results", [])[:2]
                if item.get("link")
            ]
        except:
            continue

    urls = list(set(urls))

    # ===== SCRAPE =====
    documents = scrape_urls_parallel(urls)

    # Add file content
    if text:
        documents.append(Document(page_content=text, metadata={"source": "file"}))

    if not documents:
        return "No data found."

    # ===== CHUNKING =====
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)
    chunks = clean_and_validate_chunks(chunks)

    if not chunks:
        return "No valid chunks."

    # ===== VECTOR STORE =====
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)

    # ===== RETRIEVAL =====
    results = vector_store.similarity_search(question, k=8)

    if not results:
        return "No relevant info found."

    # ===== RERANK =====
    pairs = [(question, doc.page_content[:500]) for doc in results]
    scores = reranker.predict(pairs)

    ranked_docs = [
        doc for _, doc in sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
    ]

    top_docs = ranked_docs[:3]

    context = "\n\n".join([doc.page_content[:1000] for doc in top_docs])

    # ===== FINAL ANSWER =====
    prompt = f"""
    You are a precise AI assistant.

    Use ONLY the context below.
    If answer is not present, say:
    "I don't have enough information."

    Context:
    {context}

    Question:
    {question}

    Answer clearly and concisely.
    """

    response = llm.invoke(prompt)
    answer = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL)

    # ===== CACHE SAVE =====
    CACHE[cache_key] = answer

    return answer
import os
import time
import requests
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document
import re
import hashlib
import json

def clean_text(text: str) -> str:
    lines = text.splitlines()
    cleaned_lines = [line for line in lines if not re.match(r'^[_\W\s]{5,}$', line.strip())]
    return "\n".join(cleaned_lines).strip()

def generate_text_hash(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()[:8]

# === CONFIGURATION ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "deepseek-r1-distill-llama-70b"
PDF_PATH = "book2.pdf"

# === Step 1: Enhanced PDF Loading & Chunking with Page Tracking ===
print("Loading and chunking PDF with page tracking...")
loader = PyPDFLoader(PDF_PATH)
raw_pages = loader.load()  # Get all pages as separate documents

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)

chunks = []
for page_num, page_doc in enumerate(raw_pages, start=1):
    page_text = clean_text(page_doc.page_content)
    page_chunks = text_splitter.split_text(page_text)
    
    for chunk_num, chunk_text in enumerate(page_chunks, start=1):
        start_pos = page_text.find(chunk_text)
        end_pos = start_pos + len(chunk_text)
        
        chunks.append(Document(
            page_content=chunk_text,
            metadata={
                "source": PDF_PATH,
                "page": page_num,
                "chunk_id": f"p{page_num}c{chunk_num}",
                "position": {
                    "start": start_pos,
                    "end": end_pos,
                    "length": len(chunk_text)
                },
                "preview": chunk_text[:50] + ("..." if len(chunk_text) > 50 else ""),
                "text_hash": generate_text_hash(chunk_text),
                "page_hash": generate_text_hash(page_text)
            }
        ))

print(f"Created {len(chunks)} text chunks from {len(raw_pages)} pages")

# === Step 2: Embed Chunks Using Gemini Embedding Model ===
print("Creating embeddings and vector store...")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(chunks, embedding_model)
print(f"Vector store created with {vectorstore.index.ntotal} embeddings")
vectorstore.save_local("book_vectorstore")
print("Vector store saved locally")

# === Step 3: LLM Interaction ===
def call_groq_llm(prompt):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant. Your work is to answer the Question given in prompt by strictly taking help of provided Context. Your solution should provide accurate solution"},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Groq LLM error: {response.status_code} - {response.text}")
    
    return response.json()["choices"][0]["message"]["content"]

def call_groq_to_get_context(prompt):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant. answer the query in bookish language, the language which i can find in books"},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Groq LLM error: {response.status_code} - {response.text}")
    
    return response.json()["choices"][0]["message"]["content"]

# === Step 4: Expand Query for Embedding ===
def expand_query_with_llm(query):
    prompt = f"""You are an expert assistant. The user query below is too short for accurate search.
So please you answer that query in 10 lines 

Query: {query}

Expanded version:"""
    return call_groq_to_get_context(prompt)

# === Step 5: Enhanced Answer Generation with Page References ===
def answer_question(question):
    # Step 1: Expand the query
    expanded_query = expand_query_with_llm(question)
    
    # Step 2: Semantic search on expanded query
    similar_docs = vectorstore.max_marginal_relevance_search(
        query=expanded_query, 
        k=5, 
        fetch_k=25
    )

    if not similar_docs:
        return json.dumps({
            "answer": "No relevant context found.",
            "references": [],
            "thinking_process": ""
        }, indent=2)

    # Prepare context for LLM
    full_context = "\n\n".join([doc.page_content for doc in similar_docs])

    # Generate answer with thinking process
    prompt = f"""Analyze the question and provide:
1. Your thinking process (marked with <thinking> tags)
2. A detailed answer based strictly on the context
3. Key points from each relevant chunk
4. Be as detailed as possible

Question: {question}

Context:
{full_context}

Format your response as:
<thinking>Your analytical process here</thinking>
<answer>Your structured answer here</answer>"""
    
    llm_response = call_groq_llm(prompt)
    
    # Extract thinking and answer parts
    thinking_process = llm_response.split("<thinking>")[1].split("</thinking>")[0].strip()
    answer = llm_response.split("<answer>")[1].split("</answer>")[0].strip()

    # Prepare structured response
    response = {
        "question": question,
        "expanded_query": expanded_query,
        "thinking_process": thinking_process,
        "answer": answer,
        "references": [
            {
                "page": doc.metadata["page"],
                "chunk_id": doc.metadata["chunk_id"],
                "position": doc.metadata["position"],
                "text": doc.page_content,
                "preview": doc.metadata["preview"],
                "page_hash": doc.metadata["page_hash"],
                "text_hash": doc.metadata["text_hash"]
            } for doc in similar_docs
        ],
        "context_hash": generate_text_hash(full_context)
    }

    return json.dumps(response, indent=2)

# === Run Example Query ===
question = "what is Modeling Latency: Dynamic Congestion Window"
print("\nGenerating answer using Groq LLM...")
result = answer_question(question)
print(result)
import os
import time
import requests
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document
from youtube_transcript_api import YouTubeTranscriptApi
import re
import hashlib
import json

def clean_text(text: str) -> str:
    lines = text.splitlines()
    cleaned_lines = [line for line in lines if not re.match(r'^[_\W\s]{5,}$', line.strip())]
    return "\n".join(cleaned_lines).strip()

def generate_text_hash(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()[:8]

def extract_video_id(video_url: str) -> str:
    if "v=" in video_url:
        return video_url.split("v=")[1].split("&")[0]
    return video_url  # Assume it's just the ID if no URL structure

def format_timestamp_url(video_url: str, timestamp: float) -> str:
    """Format video URL with timestamp in seconds in YouTube-compatible format"""
    video_id = extract_video_id(video_url)
    # Clean the URL to just the base video URL
    base_url = f"https://www.youtube.com/watch?v={video_id}"
    # YouTube accepts both 't=Xs' and 't=XmYs' formats - we'll use seconds for simplicity
    return f"{base_url}&t={int(timestamp)}s"

# === CONFIGURATION ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "deepseek-r1-distill-llama-70b"

# === Step 1: YouTube Transcript Loading & Chunking ===
def load_youtube_transcript(video_url: str, languages: list = ['en']):
    """Load and process YouTube transcript into chunks with timestamps"""
    video_id = extract_video_id(video_url)
    print(f"\nProcessing YouTube video: {video_url}")
    
    # Try to get transcript in each language until successful
    transcript = None
    for lang in languages:
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
            print(f"Found transcript in language: {lang}")
            break
        except:
            continue
    
    if not transcript:
        raise Exception("No transcript available for the video in the specified languages")
    
    # Combine text with timestamps
    full_text = " ".join([entry['text'] for entry in transcript])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    
    # First split the full text into chunks
    text_chunks = text_splitter.split_text(clean_text(full_text))
    
    # Then map these chunks back to timestamp ranges
    chunks = []
    for chunk_num, chunk_text in enumerate(text_chunks, start=1):
        # Find the position of this chunk in the full text
        start_pos = full_text.find(chunk_text)
        end_pos = start_pos + len(chunk_text)
        
        # Find corresponding timestamps
        start_time = 0
        end_time = 0
        current_pos = 0
        matched_entries = []
        
        for entry in transcript:
            entry_end = current_pos + len(entry['text']) + 1  # +1 for space
            if current_pos <= end_pos and entry_end >= start_pos:
                matched_entries.append(entry)
            current_pos = entry_end
        
        if matched_entries:
            start_time = matched_entries[0]['start']
            end_time = matched_entries[-1]['start'] + matched_entries[-1]['duration']
        
        chunks.append(Document(
            page_content=chunk_text,
            metadata={
                "source": format_timestamp_url(video_url, start_time),  # Updated to include timestamp
                "chunk_id": f"c{chunk_num}",
                "timestamp": {
                    "start": start_time,
                    "end": end_time,
                    "length": end_time - start_time
                },
                "preview": chunk_text[:50] + ("..." if len(chunk_text) > 50 else ""),
                "text_hash": generate_text_hash(chunk_text),
                "video_hash": generate_text_hash(full_text)
            }
        ))
    
    print(f"Created {len(chunks)} text chunks from YouTube video")
    return chunks

def create_vector_store(chunks, store_name="youtube_vectorstore"):
    print("Creating embeddings and vector store...")
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    print(f"Vector store created with {vectorstore.index.ntotal} embeddings")
    vectorstore.save_local(store_name)
    print("Vector store saved locally")
    return vectorstore

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


def answer_question(vectorstore, question):
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
1. Your thinking process (clearly marked with "THINKING PROCESS:" at the beginning)
2. A detailed answer based strictly on the context
3. Key points from each relevant chunk
4. Include timestamps where this information appears in the video
5. Be as detailed as possible

Question: {question}

Context:
{full_context}

Please format your response with "THINKING PROCESS:" at the beginning of your analysis,
followed by your answer."""
    
    llm_response = call_groq_llm(prompt)
    
    # Extract thinking and answer parts with more flexible parsing
    thinking_process = ""
    answer = llm_response
    
    # Try to split the response if it contains "THINKING PROCESS:"
    if "THINKING PROCESS:" in llm_response:
        parts = llm_response.split("THINKING PROCESS:", 1)
        thinking_process = parts[1].strip()
        answer = parts[0].strip() if parts[0].strip() else parts[1].strip()
    elif "thinking>" in llm_response.lower():  # Fallback for if they use tags anyway
        try:
            thinking_process = llm_response.split("<thinking>")[1].split("</thinking>")[0].strip()
            answer = llm_response.split("<answer>")[1].split("</answer>")[0].strip()
        except:
            thinking_process = "The model did not provide a separate thinking process."
            answer = llm_response

    # Prepare structured response
    response = {
        "question": question,
        "expanded_query": expanded_query,
        "thinking_process": thinking_process,
        "answer": answer,
        "references": [
            {
                "source": doc.metadata["source"],  # This now includes the timestamp
                "chunk_id": doc.metadata["chunk_id"],
                "timestamp": doc.metadata["timestamp"],
                "text": doc.page_content,
                "preview": doc.metadata["preview"],
                "text_hash": doc.metadata["text_hash"]
            } for doc in similar_docs
        ],
        "context_hash": generate_text_hash(full_context)
    }

    return json.dumps(response, indent=2)

def youtube_qa_workflow(video_url: str, question: str):
    # Step 1: Load and chunk the transcript
    chunks = load_youtube_transcript(video_url)
    
    # Step 2: Create vector store
    vectorstore = create_vector_store(chunks)
    
    # Step 3: Answer question
    print("\nGenerating answer using Groq LLM...")
    result = answer_question(vectorstore, question)
    return result

# === Example Usage ===
if __name__ == "__main__":
    video_url = "https://www.youtube.com/watch?v=vJOGC8QJZJQ"
    # question = "What are the main topics discussed in this video?"
    question = "What is difference between langgraph & langchain?"
    
    result = youtube_qa_workflow(video_url, question)
    print(result)
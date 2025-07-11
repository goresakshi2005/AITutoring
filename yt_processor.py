import os
import re
import json
import hashlib
import logging
from typing import List, Dict, Optional, Union
from yt_dlp import YoutubeDL
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
import requests

# Disable yt-dlp logger to suppress ffmpeg warnings
logging.getLogger('yt_dlp').setLevel(logging.ERROR)

class YouTubeProcessor:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_model = "deepseek-r1-distill-llama-70b"
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.supported_languages = ['en', 'hi']  # English and Hindi (English first)

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean text by removing empty or whitespace-only lines"""
        lines = text.splitlines()
        cleaned_lines = [line for line in lines if not re.match(r'^[_\W\s]{5,}$', line.strip())]
        return "\n".join(cleaned_lines).strip()

    @staticmethod
    def generate_text_hash(text: str) -> str:
        """Generate a short hash for text content"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()[:8]

    @staticmethod
    def extract_video_id(video_url: str) -> str:
        """Extract YouTube video ID from URL"""
        if "v=" in video_url:
            return video_url.split("v=")[1].split("&")[0]
        return video_url  # Assume it's just the ID if no URL structure

    @staticmethod
    def get_youtube_video_info(video_url: str) -> Dict:
        """Get YouTube video metadata using yt-dlp"""
        ydl_opts = {
            'quiet': True,
            'skip_download': True,
            'extract_flat': True,
        }
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                return {
                    "title": info.get('title', ''),
                    "description": info.get('description', ''),
                    "thumbnail": info.get('thumbnail', ''),
                    "duration": info.get('duration', 0),
                    "view_count": info.get('view_count', 0),
                    "upload_date": info.get('upload_date', '')
                }
        except Exception as e:
            print(f"Warning: Couldn't get video info - {str(e)}")
            return {
                "title": "",
                "description": "",
                "thumbnail": ""
            }

    @staticmethod
    def format_timestamp_url(video_url: str, timestamp: float) -> str:
        """Format URL with timestamp parameter"""
        video_id = YouTubeProcessor.extract_video_id(video_url)
        return f"https://www.youtube.com/watch?v={video_id}&t={int(timestamp)}s"

    def get_transcript(self, video_id: str) -> Union[List[Dict], None]:
        """Get transcript with preference for English, then Hindi (including auto-generated), then None"""
        # First try to get English transcript (manual or auto-generated)
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            print("Found English transcript")
            return transcript, 'en'
        except Exception as e:
            print(f"Couldn't find English transcript: {str(e)}")
        
        # If English not available, try Hindi (including auto-generated)
        try:
            # First try manual Hindi transcript
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['hi'])
            print("Found manual Hindi transcript")
            return transcript, 'hi'
        except NoTranscriptFound:
            # If no manual Hindi transcript, try auto-generated Hindi
            try:
                transcript = YouTubeTranscriptApi.list_transcripts(video_id)
                for t in transcript:
                    if t.language_code == 'hi' and t.is_generated:
                        print("Found auto-generated Hindi transcript")
                        return t.fetch(), 'hi'
            except Exception as e:
                print(f"Couldn't find Hindi transcript (manual or auto-generated): {str(e)}")
        except Exception as e:
            print(f"Error while fetching Hindi transcript: {str(e)}")
        
        # If neither is available, return None
        return None, None

    def load_youtube_transcript(self, video_url: str) -> List[Document]:
        """Load and process YouTube transcript into chunks with timestamps"""
        video_id = self.extract_video_id(video_url)
        print(f"\nProcessing YouTube video: {video_url}")
        
        # Get video metadata
        video_info = self.get_youtube_video_info(video_url)
        if video_info.get('title'):
            print(f"Video Title: {video_info['title']}")
        
        # Get transcript with language preference
        transcript, transcript_lang = self.get_transcript(video_id)
        if not transcript:
            raise Exception(f"No transcript available in supported languages: {self.supported_languages}")
        
        # Process transcript into chunks with metadata
        full_text = " ".join([entry['text'] for entry in transcript])
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        text_chunks = text_splitter.split_text(self.clean_text(full_text))
        
        chunks = []
        for chunk_num, chunk_text in enumerate(text_chunks, start=1):
            # Map chunk to timestamp range
            start_pos = full_text.find(chunk_text)
            end_pos = start_pos + len(chunk_text)
            
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
                    "source": self.format_timestamp_url(video_url, start_time),
                    "thumbnail": video_info.get('thumbnail', ''),
                    "chunk_id": f"c{chunk_num}",
                    "timestamp": {
                        "start": start_time,
                        "end": end_time,
                        "length": end_time - start_time
                    },
                    "preview": chunk_text[:50] + ("..." if len(chunk_text) > 50 else ""),
                    "text_hash": self.generate_text_hash(chunk_text),
                    "video_hash": self.generate_text_hash(full_text),
                    "video_title": video_info.get('title', 'Unknown'),
                    "video_id": video_id,
                    "language": transcript_lang  # Add language metadata
                }
            ))
        
        print(f"Created {len(chunks)} text chunks from YouTube video")
        return chunks

    def create_vector_store(self, chunks: List[Document], store_name: str) -> FAISS:
        """Create and save FAISS vector store from document chunks"""
        print("Creating embeddings and vector store...")
        vectorstore = FAISS.from_documents(chunks, self.embedding_model)
        print(f"Vector store created with {vectorstore.index.ntotal} embeddings")
        
        # Save to specified path
        store_path = os.path.join("vectorstores", store_name)
        vectorstore.save_local(store_path)
        print(f"Vector store saved at {store_path}")
        return vectorstore

    def load_vector_store(self, store_name: str) -> FAISS:
        """Load existing vector store from disk"""
        store_path = os.path.join("vectorstores", store_name)
        return FAISS.load_local(
            store_path,
            self.embedding_model,
            allow_dangerous_deserialization=True
        )

    def call_groq_llm(self, prompt: str, language: str = 'en') -> str:
        """Call Groq LLM API with the given prompt"""
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        
        system_message = {
            "en": "You are a helpful AI assistant. Answer questions using the provided context.",
            "hi": "आप एक सहायक AI सहायक हैं। प्रदान किए गए संदर्भ का उपयोग करके प्रश्नों का उत्तर दें।"
        }.get(language, "en")
        
        payload = {
            "model": self.groq_model,
            "messages": [
                {
                    "role": "system", 
                    "content": system_message
                },
                {"role": "user", "content": prompt}
            ]
        }

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions", 
            json=payload, 
            headers=headers
        )
        if response.status_code != 200:
            raise Exception(f"Groq LLM error: {response.status_code} - {response.text}")
        
        return response.json()["choices"][0]["message"]["content"]

    def expand_query_with_llm(self, query: str, language: str = 'en') -> str:
        """Expand short queries for better semantic search"""
        prompt_templates = {
            'en': """You are an expert assistant. The user query below is too short for accurate search.
Please expand it into a more detailed version while preserving the original intent.

Original Query: {query}

Expanded Version:""",
            'hi': """आप एक विशेषज्ञ सहायक हैं। नीचे दिया गया उपयोगकर्ता प्रश्न सटीक खोज के लिए बहुत छोटा है।
कृपया इसे मूल इरादे को संरक्षित करते हुए अधिक विस्तृत संस्करण में विस्तारित करें।

मूल प्रश्न: {query}

विस्तारित संस्करण:"""
        }
        
        prompt = prompt_templates.get(language, 'en').format(query=query)
        return self.call_groq_llm(prompt, language)

    def answer_question(self, vectorstore: FAISS, question: str) -> Dict:
        """Answer question using vector store context. Always returns answer in English."""
        # Detect language of the question (but we'll always answer in English)
        question_lang = 'hi' if any('\u0900' <= char <= '\u097F' for char in question) else 'en'
        
        # Step 1: Expand the query (can be in original language)
        expanded_query = self.expand_query_with_llm(question, question_lang)
        
        # Step 2: Semantic search on expanded query
        similar_docs = vectorstore.max_marginal_relevance_search(
            query=expanded_query, 
            k=5, 
            fetch_k=25
        )

        if not similar_docs:
            return {
                "answer": "No relevant context found in the video.",
                "references": [],
                "thinking_process": ""
            }

        # Prepare context for LLM (can be in any language)
        full_context = "\n\n".join([doc.page_content for doc in similar_docs])
        context_lang = similar_docs[0].metadata.get('language', 'en')

        # Generate answer - always in English regardless of context language
        prompt_template = """Analyze the question and provide:
1. Your thinking process (marked with <thinking> tags)
2. A detailed answer in English based strictly on the context
3. Key points from each relevant chunk
4. Include timestamps where this information appears in the video

Question: {question}

Context:
{context}

IMPORTANT: Your answer must be in English, even if the context is in another language.

Format your response as:
<thinking>Your analytical process here</thinking>
<answer>Your structured answer in English here</answer>"""
        
        prompt = prompt_template.format(
            question=question,
            context=full_context
        )
        
        # Force English response by setting language to 'en'
        llm_response = self.call_groq_llm(prompt, 'en')
        
        # Extract thinking and answer parts
        thinking_process = ""
        answer = ""
        try:
            thinking_process = llm_response.split("<thinking>")[1].split("</thinking>")[0].strip()
            answer = llm_response.split("<answer>")[1].split("</answer>")[0].strip()
        except:
            thinking_process = "The model did not provide a separate thinking process."
            answer = llm_response

        return {
            "question": question,
            "expanded_query": expanded_query,
            "thinking_process": thinking_process,
            "answer": answer,
            "references": [
                {
                    "source": doc.metadata["source"],
                    "thumbnail": doc.metadata["thumbnail"],
                    "chunk_id": doc.metadata["chunk_id"],
                    "timestamp": doc.metadata["timestamp"],
                    "text": doc.page_content,
                    "preview": doc.metadata["preview"],
                    "video_title": doc.metadata.get("video_title", "Unknown"),
                    "language": doc.metadata.get("language", "en")
                } for doc in similar_docs
            ],
            "context_hash": self.generate_text_hash(full_context),
            "language": "en"  # Always return English as the response language
        }

    def process_video(self, video_url: str, store_name: str) -> Dict:
        """Full processing pipeline for a YouTube video"""
        chunks = self.load_youtube_transcript(video_url)
        vectorstore = self.create_vector_store(chunks, store_name)
        video_info = self.get_youtube_video_info(video_url)
        
        return {
            "vectorstore": vectorstore,
            "video_info": video_info,
            "chunks": chunks,
            "store_name": store_name
        }

def main():
    # Initialize the YouTube processor
    processor = YouTubeProcessor()
    
    # Get YouTube URL from user
    video_url = input("Enter YouTube video URL: ").strip()
    
    # Create a store name based on video ID
    video_id = processor.extract_video_id(video_url)
    store_name = f"youtube_{video_id}"
    
    try:
        # Process the video
        print("\nProcessing video...")
        result = processor.process_video(video_url, store_name)
        print(f"\nSuccessfully processed video and created vector store: {store_name}")
        
        # Show video info
        print("\nVideo Information:")
        print(f"Title: {result['video_info'].get('title', 'Unknown')}")
        print(f"Duration: {result['video_info'].get('duration', 0)} seconds")
        print(f"Views: {result['video_info'].get('view_count', 0)}")
        
        # Interactive Q&A loop
        print("\nYou can now ask questions about the video (type 'quit' to exit):")
        while True:
            question = input("\nYour question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
                
            if not question:
                print("Please enter a valid question.")
                continue
                
            try:
                answer = processor.answer_question(result['vectorstore'], question)
                print("\nAnswer:")
                print(answer['answer'])
                
                print("\nReferences:")
                for ref in answer['references']:
                    print(f"- [{ref['chunk_id']}] {ref['preview']} (at {int(ref['timestamp']['start'])}s)")
                    print(f"  Source: {ref['source']}")
                
            except Exception as e:
                print(f"Error answering question: {str(e)}")
                
    except Exception as e:
        print(f"\nError processing video: {str(e)}")

if __name__ == "__main__":
    main()
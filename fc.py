import os
import json
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from groq import Groq

class YouTubeProcessor:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key)
        self.model = "llama3-70b-8192"  # Updated to use Llama 3 70B model

    def extract_video_id(self, url):
        query = urlparse(url)
        if query.hostname == "youtu.be":
            return query.path[1:]
        if query.hostname in ("www.youtube.com", "youtube.com"):
            if query.path == "/watch":
                return parse_qs(query.query).get("v", [None])[0]
        return None

    def load_youtube_transcript(self, video_url):
        video_id = self.extract_video_id(video_url)
        if not video_id:
            raise ValueError("Invalid YouTube URL")

        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
        except Exception as e:
            print(f"❌ Error fetching transcript: {e}")
            return []

        # Combine all transcript text
        full_text = " ".join([item["text"] for item in transcript])

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = splitter.create_documents([full_text])

        # Return chunks as Document objects
        return [Document(page_content=doc.page_content, metadata={"language": "en"}) for doc in docs]

    def call_groq_llm(self, prompt, language="en"):
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.3,
                max_tokens=4000  # Added to ensure we get complete responses
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Groq API Error: {str(e)}")
            raise


def generate_10_flashcards(video_url: str, output_path: str = "flashcards.json"):
    processor = YouTubeProcessor()
    chunks = processor.load_youtube_transcript(video_url)

    if not chunks:
        print("❌ No transcript chunks found.")
        return

    print(f"\n🔍 Extracting key points from {len(chunks)} transcript chunks...")

    key_points = []

    for idx, doc in enumerate(chunks):
        prompt = f"""
You are an expert assistant. Extract the most important topics (2-3 max) from the transcript chunk below. 
Return only bullet points — no extra text.

Transcript:
\"\"\"
{doc.page_content}
\"\"\"
"""
        try:
            response = processor.call_groq_llm(prompt, language=doc.metadata.get("language", "en"))
            key_points.append(response.strip())
            print(f"✅ Chunk {idx + 1}/{len(chunks)} - key points extracted")
        except Exception as e:
            print(f"❌ Error in chunk {idx + 1}: {str(e)}")

    combined_points = "\n".join(key_points)

    final_prompt = f"""
From the following important topics, generate exactly 6 high-quality flashcards.
Each flashcard should follow this format:
Front: <Clear, specific topic> and should not be too similar to existing ones. also 5-6 words long. and should be meaningful.
Back: <Comprehensive 2 sentence explanation that covers key aspects of the topic, including:
- Core concept definition
- Key characteristics or components
- Practical applications or significance
- Important relationships with other concepts>

Important Topics:
\"\"\"
{combined_points}
\"\"\"

Return only the flashcards. No extra text or numbering.
Ensure each back side provides substantial, well-structured knowledge that would help someone truly understand the topic.
"""
    print("\n🧠 Generating final 10 flashcards...")

    try:
        response = processor.call_groq_llm(final_prompt)

        flashcards = []
        for block in response.strip().split("Front:")[1:]:
            parts = block.strip().split("Back:")
            if len(parts) == 2:
                front = parts[0].strip()
                back = parts[1].strip()
                flashcards.append({
                    "front": front,
                    "back": back
                })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(flashcards, f, indent=4, ensure_ascii=False)

        print(f"\n✅ 10 Flashcards saved to {output_path}")
    except Exception as e:
        print(f"❌ Error during final flashcard generation: {str(e)}")


if __name__ == "__main__":
    video_url = input("Enter YouTube video URL: ").strip()
    generate_10_flashcards(video_url)
import os
from typing import List, Dict
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
import hashlib

class FlashcardGenerator:
    def __init__(self):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.generated_fronts = set()  # Track already used front topics
    
    def load_vector_store(self, store_path: str) -> FAISS:
        """Load existing FAISS vector store"""
        return FAISS.load_local(
            store_path,
            self.embedding_model,
            allow_dangerous_deserialization=True
        )
    
    @staticmethod
    def generate_content_hash(content: str) -> str:
        """Generate a short hash for content"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
    
    def generate_flashcards(self, store_path: str, num_flashcards: int = 2) -> List[Dict]:
        """
        Generate flashcards from FAISS vector store
        :param store_path: Path to the FAISS store directory (contains index.faiss)
        :param num_flashcards: Number of flashcards to generate
        :return: List of flashcards with metadata
        """
        # Load the vector store
        vectorstore = self.load_vector_store(store_path)
        
        # Get the most representative documents (centroids)
        # We'll use the mean embedding to find central concepts
        mean_embedding = vectorstore.index.reconstruct_batch(range(1))[0]
        
        # Find similar documents to the mean embedding
        similar_docs = vectorstore.similarity_search_by_vector(
            embedding=mean_embedding,
            k=num_flashcards * 2  # Get extra docs to ensure unique fronts
        )
        
        # Process each document into concise flashcards
        flashcards = []
        for doc in similar_docs:
            if len(flashcards) >= num_flashcards:
                break
                
            # Use Gemini to summarize the content into a flashcard
            prompt = f"""Convert this video transcript into a unique, high-quality flashcard.
Requirements:
1. Front must be a specific, unique topic not about general comparisons
2. Back must contain 3-4 specific facts about the front topic
3. No repetitive or similar topics to these already used: {', '.join(self.generated_fronts) if self.generated_fronts else 'none'}
4. Format exactly like this:
Front: [Specific Unique Topic]
Back: * [Fact 1]
* [Fact 2]
* [Fact 3]

Content:
{doc.page_content}"""
            
            response = self.model.generate_content(prompt)
            flashcard_content = response.text
            
            # Split into front and back if the format was followed
            if "Front:" in flashcard_content and "Back:" in flashcard_content:
                front = flashcard_content.split("Front:")[1].split("Back:")[0].strip()
                back = flashcard_content.split("Back:")[1].strip()
                
                # Skip if front is too similar to existing ones
                if any(existing.lower() in front.lower() for existing in self.generated_fronts):
                    continue
                    
                self.generated_fronts.add(front)
                flashcards.append({
                    "id": self.generate_content_hash(front + back),
                    "front": front,
                    "back": back,
                    "source": doc.metadata.get("source", ""),
                    "timestamp": doc.metadata.get("timestamp", {}),
                    "video_title": doc.metadata.get("video_title", "Unknown"),
                })
        
        return flashcards
    
    def generate_grouped_flashcards(self, store_path: str, num_groups: int = 5) -> Dict[str, List]:
        """
        Generate flashcards grouped by topic
        :param store_path: Path to the FAISS store directory
        :param num_groups: Number of topic groups to create
        :return: Dictionary of topics with their flashcards
        """
        vectorstore = self.load_vector_store(store_path)
        
        # Get all documents (we'll sample a reasonable number)
        all_docs = vectorstore.similarity_search("", k=100)
        
        # Cluster documents into topics
        prompt = f"""Analyze these video transcript chunks and create {num_groups} topic groups.
For each topic, generate 3-5 unique flashcards with these requirements:
1. Each Front must be completely unique and specific (no variations of same concept)
2. Back must contain 3-5 specific facts about the front topic
3. No repetitive or similar topics across groups
4. Format exactly like this:
5. Do not markdown 
Front: [Specific Unique Topic]
Back: * [Fact 1]
* [Fact 2]
* [Fact 3]

Content Chunks:
{"\n\n".join([doc.page_content[:500] for doc in all_docs])}"""
        
        response = self.model.generate_content(prompt)
        return self._parse_grouped_response(response.text)
    
    def _parse_grouped_response(self, response_text: str) -> Dict[str, List]:
        """Parse the LLM response into structured groups"""
        groups = {}
        current_topic = None
        all_fronts = set()
        
        for line in response_text.split('\n'):
            line = line.strip()
            if line.startswith('Topic '):
                current_topic = line.split(': ')[1].strip()
                groups[current_topic] = []
            elif line.startswith('Front: '):
                front = line.split('Front: ')[1].strip()
                # Skip if front is too similar to existing ones
                if any(existing.lower() in front.lower() for existing in all_fronts):
                    continue
                all_fronts.add(front)
            elif line.startswith('Back: '):
                back = line.split('Back: ')[1].strip()
                if current_topic and front:  # Ensure we have both topic and front
                    groups[current_topic].append({
                        "id": self.generate_content_hash(front + back),
                        "front": front,
                        "back": back
                    })
                    front = None  # Reset front to avoid duplicate additions
        
        return groups

# Example usage
if __name__ == "__main__":
    generator = FlashcardGenerator()
    
    # Path to your FAISS store (directory containing index.faiss)
    store_path = r"D:\Documents\AITutoring\vectorstores\youtube_p4pHsuEf4Ms"  # Replace with your actual path
    
    # Generate simple flashcards
    print("Generating basic flashcards...")
    flashcards = generator.generate_flashcards(store_path)
    for i, card in enumerate(flashcards, 1):
        print(f"\nFlashcard {i}:")
        print(f"Front: {card['front']}")
        print(f"Back: {card['back']}")
        print(f"Source: {card['source']}")
    
    # Generate grouped flashcards by topic
    print("\nGenerating grouped flashcards by topic...")
    grouped_flashcards = generator.generate_grouped_flashcards(store_path)
    for topic, cards in grouped_flashcards.items():
        print(f"\nTopic: {topic}")
        for card in cards:
            print(f"\nFront: {card['front']}")
            print(f"Back: {card['back']}")
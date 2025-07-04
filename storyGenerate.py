import os
from typing import List, Dict, TypedDict
from youtube_search import YoutubeSearch
from tavily import TavilyClient
import google.generativeai as genai

# Initialize Gemini and Tavily
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Define type hints
class VideoResource(TypedDict):
    title: str
    url: str
    channel: str
    duration: str

class WebResource(TypedDict):
    title: str
    url: str
    source: str

class ChapterOutput(TypedDict):
    name: str
    youtube_videos: List[VideoResource]
    web_resources: List[WebResource]

def generate_chapter_names(topic: str, grade: str) -> List[str]:
    prompt = f"""
        Generate exactly 10-12 comprehensive chapter names for studying {topic} 
        at {grade} level following these strict guidelines:

        1. Progression Structure:
        - Chapters 1-3: Foundational concepts
        - Chapters 4-6: Core techniques/methods  
        - Chapters 7-8: Advanced applications
        - Chapters 9-10: Cutting-edge extensions

        2. Naming Requirements:
        - Each 5-8 words
        - Include 2-3 key subtopics when possible
        - Use appropriate technical terms for the level
        - Distinct concepts (no overlap)
        - Progress logically

        3. Style:
        - Clear and concise
        - Avoid vague terms like "introduction to"
        - Action-oriented where applicable

        4. Format:
        - ONLY output numbered list
        - No explanations
        - No section headers
        - No additional text

        Example for "Machine Learning (Undergrad)":
        1. Supervised Learning: Regression, Classification, Loss Functions  
        2. Neural Networks: Architectures, Backpropagation, Activation Functions
        ...
        10. Federated Learning: Distributed Training, Privacy Preservation

        Now generate for {topic} at {grade} level:
        1. 
        2. 
        ...
        10.
        """
    
    response = model.generate_content(prompt)
    chapters = []
    
    for line in response.text.split('\n'):
        line = line.strip()
        if line and line[0].isdigit():
            chapter_name = line.split('.', 1)[1].strip()
            chapters.append(chapter_name)
            if len(chapters) == 10:
                break
    
    return chapters

def get_video_resources(topic: str, grade: str, chapter_name: str) -> List[VideoResource]:
    query = f"{topic} {chapter_name} tutorial for {grade} grade"
    results = YoutubeSearch(query, max_results=5).to_dict()
    
    videos = []
    for result in results[:4]:
        videos.append({
            "title": result["title"],
            "url": f"https://youtube.com{result['url_suffix']}",
            "channel": result["channel"],
            "duration": result["duration"]
        })
    
    return videos

def get_web_resources(topic: str, grade: str, chapter_name: str) -> List[WebResource]:
    query = f"{topic} {chapter_name} tutorial OR guide for {grade} grade"
    search_results = tavily.search(query=query, include_raw_content=False, max_results=5)
    
    resources = []
    for result in search_results.get('results', [])[:4]:
        resources.append({
            "title": result.get('title', 'No title available'),
            "url": result.get('url', '#'),
            "source": result.get('url', '').split('/')[2] if '/' in result.get('url', '') else 'Unknown'
        })
    
    return resources

def display_chapters(chapter_names: List[str]):
    print("\nGenerated Chapters:")
    for i, name in enumerate(chapter_names, 1):
        print(f"{i}. {name}")

def display_single_chapter_resources(chapter: ChapterOutput):
    print(f"\nCHAPTER: {chapter['name']}")
    
    print("\nYouTube Videos:")
    for video in chapter["youtube_videos"]:
        print(f"- {video['title']} ({video['duration']})")
        print(f"  URL: {video['url']}")
        print(f"  Channel: {video['channel']}")
    
    print("\nWeb Resources:")
    for resource in chapter["web_resources"]:
        print(f"- {resource['title']}")
        print(f"  URL: {resource['url']}")
        print(f"  Source: {resource['source']}")

if __name__ == "__main__":
    print("Study Resource Generator")
    topic = input("Enter your study topic: ").strip() or "Python Programming"
    grade = input("Enter grade/standard level: ").strip() or "high school"
    
    try:
        # First generate all chapter names
        chapter_names = generate_chapter_names(topic, grade)
        display_chapters(chapter_names)
        
        # Ask user which chapter they want resources for
        while True:
            try:
                chapter_num = input("\nEnter chapter number to generate resources for (1-10) or 'q' to quit: ").strip()
                if chapter_num.lower() == 'q':
                    break
                
                chapter_num = int(chapter_num)
                if 1 <= chapter_num <= 10:
                    selected_chapter = chapter_names[chapter_num - 1]
                    print(f"\nGenerating resources for Chapter {chapter_num}: {selected_chapter}...")
                    
                    # Generate resources only for the selected chapter
                    videos = get_video_resources(topic, grade, selected_chapter)
                    websites = get_web_resources(topic, grade, selected_chapter)
                    
                    chapter_output = {
                        "name": selected_chapter,
                        "youtube_videos": videos,
                        "web_resources": websites
                    }
                    
                    display_single_chapter_resources(chapter_output)
                else:
                    print("Please enter a number between 1 and 10.")
            except ValueError:
                print("Please enter a valid number or 'q' to quit.")
                
    except Exception as e:
        print(f"Error: {e}")
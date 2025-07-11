import os
from typing import List, Dict, TypedDict
from langgraph.graph import StateGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
from youtube_search import YoutubeSearch
from tavily import TavilyClient
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Initialize Gemini and Tavily
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Define type hints
class Chapter(TypedDict):
    title: str
    description: str
    learning_objectives: List[str]

class VideoResource(TypedDict):
    title: str
    url: str
    channel: str
    duration: str

class WebResource(TypedDict):
    title: str
    url: str
    source: str
    description: str

class StudyPlan(TypedDict):
    topic: str
    introduction: str
    chapters: List[Chapter]
    video_resources: Dict[str, List[VideoResource]]
    web_resources: Dict[str, List[WebResource]]

# Define the state
class AgentState(TypedDict):
    topic: str
    introduction: str
    chapters: List[Chapter]
    video_resources: Dict[str, List[VideoResource]]
    web_resources: Dict[str, List[WebResource]]

# Introduction Generator Agent
def introduction_generator_agent(state: AgentState) -> dict:
    prompt = f"""
    Provide a comprehensive but concise introduction to the topic: {state['topic']}
    
    The introduction should:
    - Explain what the topic is about in simple terms
    - Mention its importance and real-world applications
    - Outline the key concepts that will be covered
    - Be written in clear, beginner-friendly language
    - Be approximately 200-300 words
    
    Return just the introduction text, no JSON formatting needed.
    """
    
    response = model.generate_content(prompt)
    return {"introduction": response.text}

# Chapter Generator Agent
def chapter_generator_agent(state: AgentState) -> dict:
    prompt = f"""
    Create a detailed 10-chapter study plan for: {state['topic']}
    
    For each chapter provide:
    - title (5-7 words, clear and descriptive)
    - description (3-4 sentences explaining what will be covered)
    - 3-4 specific learning objectives (start with action verbs like "Understand", "Learn", "Apply")
    
    Present the chapters in this format:
    
    1. [Chapter Title]
    Description: [Chapter description]
    Learning Objectives:
    - [Objective 1]
    - [Objective 2]
    - [Objective 3]
    
    2. [Chapter Title]
    ...
    
    Ensure the chapters:
    1. Progress from basic to advanced concepts
    2. Cover both theoretical and practical aspects
    3. Include real-world applications where relevant
    4. Are comprehensive yet not overwhelming for beginners
    """
    
    response = model.generate_content(prompt)
    chapters = []
    
    # Parse the response into chapters
    current_chapter = None
    for line in response.text.split('\n'):
        line = line.strip()
        if line.startswith(tuple(str(i) + '.' for i in range(1, 11))):
            if current_chapter:
                chapters.append(current_chapter)
            title = line.split('.', 1)[1].strip()
            current_chapter = {
                "title": title,
                "description": "",
                "learning_objectives": []
            }
        elif line.startswith("Description:"):
            if current_chapter:
                current_chapter["description"] = line.split("Description:", 1)[1].strip()
        elif line.startswith("- "):
            if current_chapter and "learning_objectives" in current_chapter:
                current_chapter["learning_objectives"].append(line[2:].strip())
    
    if current_chapter:
        chapters.append(current_chapter)
    
    # Ensure we have 10 chapters
    while len(chapters) < 10:
        chapters.append({
            "title": f"Advanced {state['topic']} Concepts {len(chapters)+1}",
            "description": f"Deep dive into advanced aspects of {state['topic']}",
            "learning_objectives": [
                "Master advanced techniques",
                "Solve complex problems",
                "Apply knowledge to real-world scenarios"
            ]
        })
    
    return {"chapters": chapters[:10]}

# Video Finder Agent (for specific chapter)
def video_finder_agent(state: AgentState, chapter_title: str) -> dict:
    query = f"{state['topic']} {chapter_title} tutorial"
    results = YoutubeSearch(query, max_results=5).to_dict()
    
    videos = []
    for result in results:
        videos.append({
            "title": result["title"],
            "url": f"https://youtube.com{result['url_suffix']}",
            "channel": result["channel"],
            "duration": result["duration"]
        })
    
    return {"video_resources": {chapter_title: videos}}

# Web Resource Finder Agent (for specific chapter)
def web_resource_finder_agent(state: AgentState, chapter_title: str) -> dict:
    query = f"{state['topic']} {chapter_title} tutorial OR guide OR documentation"
    search_results = tavily.search(query=query, include_raw_content=False, max_results=5)
    
    resources = []
    for result in search_results.get('results', [])[:5]:
        resources.append({
            "title": result.get('title', 'No title available'),
            "url": result.get('url', '#'),
            "source": result.get('url', '').split('/')[2] if '/' in result.get('url', '') else 'Unknown',
            "description": result.get('content', 'No description available')[:200] + '...'
        })
    
    return {"web_resources": {chapter_title: resources}}

# Create workflow for initial study plan (without resources)
def create_initial_workflow():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("introduction_generator", introduction_generator_agent)
    workflow.add_node("chapter_generator", chapter_generator_agent)
    
    workflow.set_entry_point("introduction_generator")
    workflow.add_edge("introduction_generator", "chapter_generator")
    
    return workflow.compile()

# Generate initial study plan (without resources)
def generate_initial_study_plan(topic: str) -> StudyPlan:
    app = create_initial_workflow()
    final_state = app.invoke({"topic": topic})
    
    return {
        "topic": topic,
        "introduction": final_state["introduction"],
        "chapters": final_state["chapters"],
        "video_resources": {},
        "web_resources": {}
    }

# Display chapter list
def print_chapter_list(study_plan: StudyPlan):
    print(f"\nStudy Plan: {study_plan['topic']}\n{'='*50}")
    print(f"\nIntroduction:\n{study_plan['introduction']}\n")
    
    print("\nChapters:")
    for i, chapter in enumerate(study_plan["chapters"], 1):
        print(f"{i}. {chapter['title']}")
        print(f"   {chapter['description']}\n")

# Display resources for a specific chapter
def print_chapter_resources(study_plan: StudyPlan, chapter_index: int):
    chapter = study_plan["chapters"][chapter_index]
    print(f"\nChapter {chapter_index + 1}: {chapter['title']}")
    print(f"\n{chapter['description']}")
    print("\nLearning Objectives:")
    for obj in chapter["learning_objectives"]:
        print(f"- {obj}")
    
    print("\nRecommended Videos:")
    for video in study_plan["video_resources"].get(chapter["title"], []):
        print(f"- {video['title']} ({video['duration']})")
        print(f"  {video['url']}")
        print(f"  Channel: {video['channel']}")
    
    print("\nAdditional Reading (Web Resources):")
    for resource in study_plan["web_resources"].get(chapter["title"], []):
        print(f"- {resource['title']}")
        print(f"  {resource['description']}")
        print(f"  URL: {resource['url']}")
        print(f"  Source: {resource['source']}")

# Main execution
if __name__ == "__main__":
    print("AI Tutor - Comprehensive Study Plan Generator")
    topic = input("Enter your study topic: ").strip() or "Python Programming"
    
    try:
        print("\nGenerating study plan...")
        study_plan = generate_initial_study_plan(topic)
        print_chapter_list(study_plan)
        
        while True:
            chapter_choice = input("\nEnter chapter number to get resources (1-10) or 'q' to quit: ").strip()
            if chapter_choice.lower() == 'q':
                break
            
            try:
                chapter_index = int(chapter_choice) - 1
                if 0 <= chapter_index < 10:
                    chapter_title = study_plan["chapters"][chapter_index]["title"]
                    
                    print(f"\nFetching resources for: {chapter_title}...")
                    
                    # Get video resources
                    video_resources = video_finder_agent(study_plan, chapter_title)
                    study_plan["video_resources"].update(video_resources["video_resources"])
                    
                    # Get web resources
                    web_resources = web_resource_finder_agent(study_plan, chapter_title)
                    study_plan["web_resources"].update(web_resources["web_resources"])
                    
                    print_chapter_resources(study_plan, chapter_index)
                else:
                    print("Please enter a number between 1 and 10.")
            except ValueError:
                print("Please enter a valid number or 'q' to quit.")
                
    except Exception as e:
        print(f"Error generating study plan: {e}")
import os
from typing import List, Dict, TypedDict, Optional
from langgraph.graph import StateGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
from youtube_search import YoutubeSearch
from youtube_transcript_api import YouTubeTranscriptApi
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

class Question(TypedDict):
    question: str
    answer: str
    question_type: str  # "short" or "descriptive"
    marks: int

class StudyPlan(TypedDict):
    topic: str
    introduction: str
    chapters: List[Chapter]
    video_resources: Dict[str, List[VideoResource]]
    web_resources: Dict[str, List[WebResource]]
    chapter_questions: Dict[str, List[Question]]  # New field for questions

# Define the state
class AgentState(TypedDict):
    topic: str
    introduction: str
    chapters: List[Chapter]
    video_resources: Dict[str, List[VideoResource]]
    web_resources: Dict[str, List[WebResource]]
    chapter_questions: Dict[str, List[Question]]
    

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

# YouTube Transcript Fetcher
def get_youtube_transcript(video_url: str) -> Optional[str]:
    try:
        video_id = video_url.split('v=')[1].split('&')[0]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return ' '.join([entry['text'] for entry in transcript])
    except Exception as e:
        print(f"Error fetching transcript: {e}")
        return None

# Video Question Generator Agent
def video_question_generator_agent(state: AgentState, chapter_title: str) -> dict:
    questions = []
    videos = state["video_resources"].get(chapter_title, [])
    
    for video in videos[:3]:  # Limit to 3 videos to avoid too many questions
        transcript = get_youtube_transcript(video["url"])
        if not transcript:
            continue
            
        prompt = f"""
        Generate 5 test questions (30 marks total) based on the following YouTube video transcript about {state['topic']} - {chapter_title}.
        The video title is: {video['title']}
        
        Requirements:
        - Include both short answer (1-2 words) and descriptive questions (2-3 sentences)
        - Total marks should be 30 (distribute marks appropriately)
        - Questions should test key concepts from the video
        - Provide clear, concise answers
        - Format as JSON with question, answer, question_type, and marks
        
        Transcript:
        {transcript[:10000]}  # Limiting to first 10k chars to avoid token limits
        
        Return only the JSON array of questions, nothing else.
        Example format:
        [
            {{
                "question": "What is the main topic of this video?",
                "answer": "The main topic is...",
                "question_type": "descriptive",
                "marks": 5
            }},
            {{
                "question": "What does the acronym XYZ stand for?",
                "answer": "Extended Yield Zone",
                "question_type": "short",
                "marks": 2
            }}
        ]
        """
        
        try:
            response = model.generate_content(prompt)
            # Parse the JSON response
            if response.text.startswith('```json'):
                json_str = response.text.split('```json')[1].split('```')[0].strip()
            elif response.text.startswith('```'):
                json_str = response.text.split('```')[1].split('```')[0].strip()
            else:
                json_str = response.text.strip()
                
            video_questions = eval(json_str)  # Using eval for simplicity (be careful with untrusted input)
            questions.extend(video_questions)
        except Exception as e:
            print(f"Error generating questions from video: {e}")
    
    return {"chapter_questions": {chapter_title: questions[:10]}}  # Limit to 10 questions per chapter

# Web Resource Question Generator Agent
def web_question_generator_agent(state: AgentState, chapter_title: str) -> dict:
    questions = []
    resources = state["web_resources"].get(chapter_title, [])
    
    for resource in resources[:3]:  # Limit to 3 resources to avoid too many questions
        prompt = f"""
        Generate 5 test questions (30 marks total) based on the following web resource about {state['topic']} - {chapter_title}.
        The resource title is: {resource['title']}
        Source: {resource['source']}
        
        Requirements:
        - Include both short answer (1-2 words) and descriptive questions (2-3 sentences)
        - Total marks should be 30 (distribute marks appropriately)
        - Questions should test key concepts from the resource
        - Provide clear, concise answers
        - Format as JSON with question, answer, question_type, and marks
        
        Resource Description:
        {resource['description']}
        
        Return only the JSON array of questions, nothing else.
        Example format:
        [
            {{
                "question": "What is the main topic of this resource?",
                "answer": "The main topic is...",
                "question_type": "descriptive",
                "marks": 5
            }},
            {{
                "question": "What does the acronym XYZ stand for?",
                "answer": "Extended Yield Zone",
                "question_type": "short",
                "marks": 2
            }}
        ]
        """
        
        try:
            response = model.generate_content(prompt)
            # Parse the JSON response
            if response.text.startswith('```json'):
                json_str = response.text.split('```json')[1].split('```')[0].strip()
            elif response.text.startswith('```'):
                json_str = response.text.split('```')[1].split('```')[0].strip()
            else:
                json_str = response.text.strip()
                
            web_questions = eval(json_str)  # Using eval for simplicity (be careful with untrusted input)
            questions.extend(web_questions)
        except Exception as e:
            print(f"Error generating questions from web resource: {e}")
    
    return {"chapter_questions": {chapter_title: questions[:10]}}  # Limit to 10 questions per chapter

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
        "web_resources": {},
        "chapter_questions": {}
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

# Display questions for a specific chapter
def print_chapter_questions(study_plan: StudyPlan, chapter_index: int):
    chapter = study_plan["chapters"][chapter_index]
    questions = study_plan["chapter_questions"].get(chapter["title"], [])
    
    if not questions:
        print("\nNo questions generated yet for this chapter.")
        return
    
    print(f"\nChapter Test: {chapter['title']} (Total Marks: 30)")
    print("\nAnswer all questions:")
    
    for i, question in enumerate(questions, 1):
        print(f"\nQ{i}. {question['question']} ({question['marks']} marks)")
        if question['question_type'] == 'descriptive':
            print("   (Answer in 2-3 sentences)")
        else:
            print("   (Short answer)")
    
    print("\n\nAnswers:")
    for i, question in enumerate(questions, 1):
        print(f"\nA{i}. {question['answer']}")

# Main execution
if __name__ == "__main__":
    print("AI Tutor - Comprehensive Study Plan Generator")
    topic = input("Enter your study topic: ").strip() or "Python Programming"
    
    try:
        print("\nGenerating study plan...")
        study_plan = generate_initial_study_plan(topic)
        print_chapter_list(study_plan)
        
        while True:
            chapter_choice = input("\nEnter chapter number to get resources (1-10), 't' for test, or 'q' to quit: ").strip()
            if chapter_choice.lower() == 'q':
                break
            elif chapter_choice.lower() == 't':
                test_chapter = input("Enter chapter number to generate test for (1-10): ").strip()
                try:
                    chapter_index = int(test_chapter) - 1
                    if 0 <= chapter_index < 10:
                        chapter_title = study_plan["chapters"][chapter_index]["title"]
                        
                        print(f"\nGenerating test questions for: {chapter_title}...")
                        
                        # Get video questions
                        if chapter_title not in study_plan["chapter_questions"]:
                            video_questions = video_question_generator_agent(study_plan, chapter_title)
                            study_plan["chapter_questions"].update(video_questions["chapter_questions"])
                            
                            # Get web questions
                            web_questions = web_question_generator_agent(study_plan, chapter_title)
                            # Merge questions if they exist
                            if chapter_title in study_plan["chapter_questions"]:
                                existing = study_plan["chapter_questions"][chapter_title]
                                new = web_questions["chapter_questions"].get(chapter_title, [])
                                study_plan["chapter_questions"][chapter_title] = (existing + new)[:10]  # Keep top 10
                            else:
                                study_plan["chapter_questions"].update(web_questions["chapter_questions"])
                        
                        print_chapter_questions(study_plan, chapter_index)
                    else:
                        print("Please enter a number between 1 and 10.")
                except ValueError:
                    print("Please enter a valid number.")
            else:
                try:
                    chapter_index = int(chapter_choice) - 1
                    if 0 <= chapter_index < 10:
                        chapter_title = study_plan["chapters"][chapter_index]["title"]
                        
                        print(f"\nFetching resources for: {chapter_title}...")
                        
                        # Get video resources if not already fetched
                        if chapter_title not in study_plan["video_resources"]:
                            video_resources = video_finder_agent(study_plan, chapter_title)
                            study_plan["video_resources"].update(video_resources["video_resources"])
                        
                        # Get web resources if not already fetched
                        if chapter_title not in study_plan["web_resources"]:
                            web_resources = web_resource_finder_agent(study_plan, chapter_title)
                            study_plan["web_resources"].update(web_resources["web_resources"])
                        
                        print_chapter_resources(study_plan, chapter_index)
                    else:
                        print("Please enter a number between 1 and 10.")
                except ValueError:
                    print("Please enter a valid number or 'q' to quit.")
                
    except Exception as e:
        print(f"Error generating study plan: {e}")
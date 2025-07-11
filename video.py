import os
from youtube_transcript_api import YouTubeTranscriptApi
import google.generativeai as genai
import time

def youtube_transcriber(
    video_url: str,
    output_format: str = "text",
    gemini_model: str = "gemini-1.5-flash",
    summarize: bool = False,
    google_api_key: str = None
):
    """
    Transcribe YouTube videos using Google Gemini API directly.
    
    Args:
        video_url (str): YouTube video URL or ID
        output_format (str): "text" or "srt"
        gemini_model (str): Gemini model name
        summarize (bool): Whether to generate summary
        google_api_key (str): Your Google API key
    """
    # Configure Gemini
    if not google_api_key:
        google_api_key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=google_api_key)
    
    # Extract video ID from URL
    video_id = video_url.split("v=")[1].split("&")[0]
    
    print(f"\nProcessing YouTube video: {video_url}")
    start_time = time.time()
    
    try:
        # Get transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Process based on output format
        if output_format.lower() == "srt":
            result = ""
            for i, entry in enumerate(transcript, start=1):
                result += f"{i}\n"
                result += f"{entry['start']:.3f} --> {entry['start'] + entry['duration']:.3f}\n"
                result += f"{entry['text']}\n\n"
        else:
            result = " ".join([entry['text'] for entry in transcript])
        
        # Generate summary if requested
        if summarize:
            model = genai.GenerativeModel(gemini_model)
            summary = model.generate_content(
                f"Summarize this YouTube transcript in 3-5 key points:\n\n{result}"
            )
            result += "\n\n=== SUMMARY ===\n" + summary.text
        
        processing_time = time.time() - start_time
        
        print("\n" + "="*50)
        print(f"RESULTS (Format: {output_format.upper()})")
        if summarize:
            print("INCLUDES SUMMARY")
        print("="*50)
        print(result)
        print("\n" + "-"*50)
        print(f"Completed in {processing_time:.2f} seconds using {gemini_model}")
        print("-"*50)
        
        return result
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    # Set your API key here or as environment variable
    API_KEY = "YOUR_GOOGLE_API_KEY"  # Replace with your actual key
    
    # Example usage
    video_url = "https://www.youtube.com/watch?v=p4pHsuEf4Ms"
    
    # Basic transcript
    print("\nBasic Transcript:")
    youtube_transcriber(video_url, google_api_key=API_KEY)
    
    # With timestamps
    print("\nTranscript with Timestamps:")
    youtube_transcriber(video_url, output_format="srt", google_api_key=API_KEY)
    
    # With summary
    print("\nTranscript with Summary:")
    youtube_transcriber(video_url, summarize=True, google_api_key=API_KEY)
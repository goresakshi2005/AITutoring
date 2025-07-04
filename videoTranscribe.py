import os
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import SRTFormatter
import time

def youtube_transcriber(
    video_url: str,
    output_format: str = "srt",
    target_lines: int = 3,  # Target number of lines per chunk
    max_chars: int = 250,   # Maximum characters per chunk
    languages: list = ['en']  # List of language codes to try
):
    """
    Transcribe YouTube videos with timestamps grouped into larger chunks.
    
    Args:
        video_url (str): YouTube video URL or ID
        output_format (str): "text" or "srt" (default is "srt")
        target_lines (int): Target number of lines to group (default 3)
        max_chars (int): Maximum characters per chunk (default 250)
        languages (list): List of language codes to try (default ['en'])
    """
    # Extract video ID from URL
    if "v=" in video_url:
        video_id = video_url.split("v=")[1].split("&")[0]
    else:
        video_id = video_url  # Assume it's just the ID if no URL structure
    
    print(f"\nProcessing YouTube video: {video_url}")
    start_time = time.time()
    
    try:
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
        
        # Process based on output format
        if output_format.lower() == "srt":
            result = ""
            current_group = []
            current_char_count = 0
            group_number = 1
            
            for entry in transcript:
                # If adding this entry would exceed our limits, finalize current group
                if (current_group and 
                    (len(current_group) >= target_lines or 
                     current_char_count + len(entry['text']) > max_chars)):
                    # Format the group
                    group_text = ' '.join([e['text'] for e in current_group])
                    start_time = current_group[0]['start']
                    end_time = current_group[-1]['start'] + current_group[-1]['duration']
                    
                    # Add to result
                    result += f"{group_number}\n"
                    result += f"{format_time(start_time)} --> {format_time(end_time)}\n"
                    result += f"{group_text}\n\n"
                    
                    # Start new group
                    group_number += 1
                    current_group = [entry]
                    current_char_count = len(entry['text'])
                else:
                    # Add entry to current group
                    current_group.append(entry)
                    current_char_count += len(entry['text'])
            
            # Add the last group if it exists
            if current_group:
                group_text = ' '.join([e['text'] for e in current_group])
                start_time = current_group[0]['start']
                end_time = current_group[-1]['start'] + current_group[-1]['duration']
                result += f"{group_number}\n"
                result += f"{format_time(start_time)} --> {format_time(end_time)}\n"
                result += f"{group_text}\n\n"
        else:
            # For plain text, just join all the text entries
            result = " ".join([entry['text'] for entry in transcript])
        
        processing_time = time.time() - start_time
        
        print("\n" + "="*50)
        print(f"RESULTS (Format: {output_format.upper()})")
        print("="*50)
        print(result[:1000] + "..." if len(result) > 1000 else result)  # Print preview if long
        print("\n" + "-"*50)
        print(f"Completed in {processing_time:.2f} seconds")
        print(f"Total transcript segments: {len(transcript)}")
        print(f"Grouped into {group_number if output_format.lower() == 'srt' else 1} chunks")
        print("-"*50)
        
        return result
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def format_time(seconds: float) -> str:
    """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace('.', ',')

if __name__ == "__main__":
    # Example usage
    video_url = "https://www.youtube.com/watch?v=CnXdddeZ4tQ"
    
    # Transcript with grouped timestamps (3-4 lines at a time)
    print("\nTranscript with Grouped Timestamps:")
    full_transcript = youtube_transcriber(
        video_url, 
        output_format="srt", 
        target_lines=3, 
        max_chars=250,
        languages=['en', 'en-US', 'en-GB']  # Try multiple English variants
    )
    
    # Save to file
    if full_transcript:
        with open("youtube_transcript.srt", "w", encoding="utf-8") as f:
            f.write(full_transcript)
        print("\nTranscript saved to 'youtube_transcript.srt'")
    
    # Or basic text transcript without timestamps
    print("\nBasic Transcript:")
    youtube_transcriber(video_url, output_format="text")
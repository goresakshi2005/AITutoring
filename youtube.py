import os
from youtube_transcript_api import YouTubeTranscriptApi
import time

def youtube_transcriber(
    video_url: str,
    output_format: str = "srt",
    google_api_key: str = None,
    target_lines: int = 3,  # Target number of lines per chunk
    max_chars: int = 250   # Maximum characters per chunk
):
    """
    Transcribe YouTube videos with timestamps grouped into larger chunks.
    
    Args:
        video_url (str): YouTube video URL or ID
        output_format (str): "text" or "srt" (default is "srt")
        target_lines (int): Target number of lines to group (default 3)
        max_chars (int): Maximum characters per chunk (default 250)
    """
    # Extract video ID from URL
    if "v=" in video_url:
        video_id = video_url.split("v=")[1].split("&")[0]
    else:
        video_id = video_url  # Assume it's just the ID if no URL structure
    
    print(f"\nProcessing YouTube video: {video_url}")
    start_time = time.time()
    
    try:
        # Get transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Process based on output format
        if output_format.lower() == "srt":
            result = ""
            current_group = []
            current_char_count = 0
            
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
                    result += f"{len(result.split('\n\n')) + 1}\n"
                    result += f"{start_time/60:.3f} --> {end_time/60:.3f}\n"
                    result += f"{group_text}\n\n"
                    
                    # Start new group
                    current_group = []
                    current_char_count = 0
                
                # Add entry to current group
                current_group.append(entry)
                current_char_count += len(entry['text'])
            
            # Add the last group if it exists
            if current_group:
                group_text = ' '.join([e['text'] for e in current_group])
                start_time = current_group[0]['start']
                end_time = current_group[-1]['start'] + current_group[-1]['duration']
                result += f"{len(result.split('\n\n')) + 1}\n"
                result += f"{start_time/60:.3f} --> {end_time/60:.3f}\n"
                result += f"{group_text}\n\n"
        else:
            result = " ".join([entry['text'] for entry in transcript])
        
        processing_time = time.time() - start_time
        
        print("\n" + "="*50)
        print(f"RESULTS (Format: {output_format.upper()})")
        print("="*50)
        print(result)
        print("\n" + "-"*50)
        print(f"Completed in {processing_time:.2f} seconds")
        print("-"*50)
        
        return result
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    # video_url = "https://www.youtube.com/watch?v=p4pHsuEf4Ms"
    video_url = "https://www.youtube.com/watch?v=CnXdddeZ4tQ"
    
    # Transcript with grouped timestamps (3-4 lines at a time)
    print("\nTranscript with Grouped Timestamps:")
    youtube_transcriber(video_url, output_format="srt", target_lines=8, max_chars=600)
    
    # Or basic text transcript without timestamps
    print("\nBasic Transcript:")
    youtube_transcriber(video_url, output_format="text")
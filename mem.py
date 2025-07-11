import cv2
from deepface import DeepFace
import vlc
import time
import random
import requests
from PIL import Image, ImageTk
import tkinter as tk
import threading
import sys
import re
from youtube_transcript_api import YouTubeTranscriptApi
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from urllib.parse import urlparse, parse_qs

# Download NLTK stopwords if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Configuration
CHECK_INTERVAL = 10  # Check emotion every 10 seconds

# Expanded meme database with more categories
MEME_DB = {
    "learning": [
        {"url": "https://media.giphy.com/media/3o7TKSjRrfIPjeiVyM/giphy.gif", "caption": "When you finally understand a difficult concept!"},
        {"url": "https://media.giphy.com/media/xT5LMHxhOfscxPfIfm/giphy.gif", "caption": "Knowledge is power!"},
        {"url": "https://media.giphy.com/media/l0HU7JIWpxh3z8vU4/giphy.gif", "caption": "Keep pushing through!"},
    ],
    "default": [
        {"url": "https://media.giphy.com/media/3o6vY1vESVsac/giphy.gif", "caption": "Learning can be fun!"},
        {"url": "https://media.giphy.com/media/LmNwrBhejkK9EFP504/giphy.gif", "caption": "When your hard work pays off!"},
        {"url": "https://media.giphy.com/media/3o6Zt481isNVuQI1l6/giphy.gif", "caption": "Keep going! Learning is a journey!"}
    ],
    "frustration": [
        {"url": "https://media.giphy.com/media/l2JehQ2GitHGdVG9y/giphy.gif", "caption": "Take a deep breath and try again!"},
        {"url": "https://media.giphy.com/media/3o7TKsQ8gqVlz8kYyY/giphy.gif", "caption": "Struggling means you're learning!"},
        {"url": "https://media.giphy.com/media/3o7qE1YN7aBOFPRw8E/giphy.gif", "caption": "Every expert was once a beginner!"}
    ],
    "boredom": [
        {"url": "https://media.giphy.com/media/3o6Zt6ML6BklcajjsA/giphy.gif", "caption": "Let's make this more interesting!"},
        {"url": "https://media.giphy.com/media/3o7qE1YN7aBOFPRw8E/giphy.gif", "caption": "Wake up your brain with this!"},
        {"url": "https://media.giphy.com/media/3o7TKsQ8gqVlz8kYyY/giphy.gif", "caption": "Time for a quick break!"}
    ]
}

class EIESystem:
    def __init__(self):
        self.current_emotion = "neutral"
        self.current_topic = "learning"  # Default general topic
        self.video_playing = False
        self.stop_detection = False
        self.youtube_url = ""
        
        # Initialize video player
        self.instance = None
        self.player = None
        self.media = None
        
        # Initialize GUI
        self.root = tk.Tk()
        self.root.title("AI Tutoring with EIE System")
        
        # URL Entry Frame
        self.url_frame = tk.Frame(self.root)
        self.url_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.url_label = tk.Label(self.url_frame, text="YouTube URL:")
        self.url_label.pack(side=tk.LEFT)
        
        self.url_entry = tk.Entry(self.url_frame, width=40)
        self.url_entry.pack(side=tk.LEFT, expand=True, fill=tk.X)
        
        self.load_btn = tk.Button(self.url_frame, text="Load Video", command=self.load_video_from_url)
        self.load_btn.pack(side=tk.RIGHT)
        
        # Video frame
        self.video_panel = tk.Frame(self.root)
        self.video_panel.pack(fill=tk.BOTH, expand=1)
        
        # Canvas for video
        self.canvas = tk.Canvas(self.video_panel)
        self.canvas.pack(fill=tk.BOTH, expand=1)
        
        # Webcam frame
        self.webcam_label = tk.Label(self.root)
        self.webcam_label.pack()
        
        # Status label
        self.status_label = tk.Label(self.root, text="Status: Waiting for YouTube URL...")
        self.status_label.pack()
        
        # Emotion label
        self.emotion_label = tk.Label(self.root, text="Detected Emotion: None")
        self.emotion_label.pack()
        
        # Topic label
        self.topic_label = tk.Label(self.root, text="Detected Topic: None")
        self.topic_label.pack()
        
        # Meme popup (initially hidden)
        self.meme_window = None
        
    def extract_video_id(self, url):
        """Extract YouTube video ID from URL using more robust method"""
        # Examples:
        # - http://youtu.be/SA2iWivDJiE
        # - http://www.youtube.com/watch?v=_oPAwA_Udwc&feature=feedu
        # - http://www.youtube.com/embed/SA2iWivDJiE
        # - http://www.youtube.com/v/SA2iWivDJiE?version=3&amp;hl=en_US
        
        query = urlparse(url)
        if query.hostname == 'youtu.be':
            return query.path[1:]
        if query.hostname in ('www.youtube.com', 'youtube.com'):
            if query.path == '/watch':
                p = parse_qs(query.query)
                return p['v'][0]
            if query.path[:7] == '/embed/':
                return query.path.split('/')[2]
            if query.path[:3] == '/v/':
                return query.path.split('/')[2]
        return None
    
    def analyze_video_topic(self, video_id):
        """Analyze video transcript to determine important topics using TF-IDF"""
        try:
            # Get video transcript
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            transcript = None
            
            # Try to get English transcript first
            for t in transcript_list:
                if t.language_code == 'en':
                    transcript = t.fetch()
                    break
            
            # If no English transcript, try any available
            if not transcript:
                transcript = transcript_list[0].fetch()
            
            text = " ".join([entry['text'] for entry in transcript]).lower()
            
            if not text.strip():
                self.current_topic = "learning"
                return self.current_topic
            
            # Preprocess text
            stop_words = set(stopwords.words('english'))
            words = [word for word in text.split() if word.isalpha() and word not in stop_words]
            processed_text = " ".join(words)
            
            # Use TF-IDF to find important terms
            vectorizer = TfidfVectorizer(max_features=5)
            tfidf_matrix = vectorizer.fit_transform([processed_text])
            feature_names = vectorizer.get_feature_names_out()
            
            # Get top 3 most important terms
            top_terms = feature_names[:3]
            self.current_topic = ", ".join(top_terms) if top_terms else "learning"
            
            # Update topic label
            self.topic_label.config(text=f"Detected Topic: {self.current_topic}")
            
            return self.current_topic
                
        except Exception as e:
            print(f"Error analyzing video: {e}")
            self.current_topic = "learning"
            self.topic_label.config(text=f"Detected Topic: {self.current_topic} (default)")
            return self.current_topic
    
    def load_video_from_url(self):
        """Load and play video from YouTube URL"""
        self.youtube_url = self.url_entry.get().strip()
        if not self.youtube_url:
            self.status_label.config(text="Status: Please enter a YouTube URL")
            return
            
        video_id = self.extract_video_id(self.youtube_url)
        if not video_id:
            self.status_label.config(text="Status: Invalid YouTube URL")
            return
            
        # Analyze video topic
        self.status_label.config(text="Status: Analyzing video content...")
        self.root.update()  # Force UI update
        
        try:
            self.current_topic = self.analyze_video_topic(video_id)
            self.status_label.config(text=f"Status: Loading video - Detected topics: {self.current_topic}")
            
            # Start playing the video
            self.play_youtube_vlc(self.youtube_url)
            
            # Start emotion detection if not already running
            if not hasattr(self, 'cap'):
                self.start_emotion_detection()
                
        except Exception as e:
            self.status_label.config(text=f"Status: Error loading video - {str(e)}")
            print(f"Error loading video: {e}")
    
    def play_youtube_vlc(self, youtube_url):
        """Play YouTube video using VLC with youtube-dl support"""
        # Stop current playback if any
        if self.player:
            self.player.stop()
            time.sleep(1)  # Give time to release resources
        
        try:
            # Create new media player with YouTube support
            self.instance = vlc.Instance("--no-xlib --quiet")
            self.player = self.instance.media_player_new()
            
            # Set up media with YouTube URL
            self.media = self.instance.media_new(youtube_url)
            self.media.get_mrl()  # This helps with some URL parsing issues
            self.player.set_media(self.media)
            
            # Set video output to canvas
            if sys.platform.startswith('win'):
                self.player.set_hwnd(self.canvas.winfo_id())
            elif sys.platform.startswith('linux'):
                self.player.set_xwindow(self.canvas.winfo_id())
            elif sys.platform.startswith('darwin'):
                self.player.set_nsobject(int(self.canvas.winfo_id()))
            
            # Play the video
            self.player.play()
            
            # Wait for video to start (up to 10 seconds)
            start_time = time.time()
            while not self.player.is_playing() and (time.time() - start_time) < 10:
                time.sleep(0.1)
            
            if self.player.is_playing():
                self.video_playing = True
                self.status_label.config(text=f"Status: Playing video - Topic: {self.current_topic}")
            else:
                self.status_label.config(text="Status: Failed to play video")
                self.video_playing = False
                
        except Exception as e:
            self.status_label.config(text=f"Status: Error playing video - {str(e)}")
            print(f"Error playing video: {e}")
            self.video_playing = False
    
    def start_emotion_detection(self):
        """Start the emotion detection thread"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Could not open webcam")
                
            self.stop_detection = False
            threading.Thread(target=self.detect_emotion_loop, daemon=True).start()
            threading.Thread(target=self.monitor_engagement, daemon=True).start()
            self.status_label.config(text="Status: Emotion detection started")
        except Exception as e:
            self.status_label.config(text=f"Status: Webcam error - {str(e)}")
            print(f"Webcam error: {e}")
    
    def detect_emotion_loop(self):
        """Continuously detect emotions from webcam"""
        while not self.stop_detection:
            ret, frame = self.cap.read()
            if ret:
                try:
                    # Convert frame to RGB and analyze
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)
                    dominant_emotion = result[0]['dominant_emotion']
                    
                    # Map to our categories
                    if dominant_emotion in ['sad', 'angry', 'fear']:
                        self.current_emotion = "frustration"
                    elif dominant_emotion == 'neutral':
                        self.current_emotion = "boredom"
                    else:
                        self.current_emotion = "engaged"
                    
                    # Update emotion label
                    self.emotion_label.config(text=f"Detected Emotion: {self.current_emotion} ({dominant_emotion})")
                    
                    # Update webcam feed in GUI
                    img = cv2.resize(frame, (320, 240))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(img)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.webcam_label.imgtk = imgtk
                    self.webcam_label.config(image=imgtk)
                    
                except Exception as e:
                    print(f"Emotion detection error: {e}")
                    
            time.sleep(0.1)  # Prevent high CPU usage
    
    def monitor_engagement(self):
        """Check emotion periodically and intervene if needed"""
        while not self.stop_detection:
            if self.video_playing and self.current_emotion in ["boredom", "frustration"]:
                self.intervene()
                
            time.sleep(CHECK_INTERVAL)
    
    def intervene(self):
        """Pause video and show educational meme"""
        try:
            if not self.video_playing:
                return
                
            self.player.pause()
            self.video_playing = False
            self.status_label.config(text=f"Status: Detected {self.current_emotion} - showing intervention")
            
            # Select appropriate meme category
            meme_category = self.current_emotion if self.current_emotion in MEME_DB else "default"
            memes = MEME_DB.get(meme_category, MEME_DB["default"])
            
            # Select a random meme
            selected_meme = random.choice(memes)
            
            # Show meme in popup window
            self.show_meme_popup(selected_meme)
            
        except Exception as e:
            print(f"Intervention error: {e}")
            self.resume_video()
    
    def show_meme_popup(self, meme):
        """Display meme in a popup window"""
        if self.meme_window:
            self.meme_window.destroy()
            
        self.meme_window = tk.Toplevel(self.root)
        self.meme_window.title("Quick Break!")
        
        try:
            # Download meme image
            response = requests.get(meme["url"], stream=True)
            response.raise_for_status()
            
            img = Image.open(response.raw)
            img = img.resize((400, 300), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            # Display image
            label = tk.Label(self.meme_window, image=photo)
            label.image = photo  # Keep reference
            label.pack()
            
            # Add caption
            caption = tk.Label(self.meme_window, text=meme["caption"], wraplength=380)
            caption.pack()
            
            # Add continue button
            continue_btn = tk.Button(
                self.meme_window, 
                text="Continue Learning", 
                command=self.resume_video
            )
            continue_btn.pack()
            
        except Exception as e:
            print(f"Error showing meme: {e}")
            self.resume_video()
    
    def resume_video(self):
        """Resume video playback"""
        try:
            if self.meme_window:
                self.meme_window.destroy()
                self.meme_window = None
                
            if self.player:
                self.player.play()
                self.video_playing = True
                self.current_emotion = "neutral"
                self.emotion_label.config(text="Detected Emotion: neutral")
                self.status_label.config(text=f"Status: Video resumed - Topic: {self.current_topic}")
                
        except Exception as e:
            print(f"Error resuming video: {e}")
            self.status_label.config(text="Status: Error resuming video")
    
    def run(self):
        """Run the main application loop"""
        try:
            self.root.mainloop()
        finally:
            self.stop_detection = True
            if hasattr(self, 'cap'):
                self.cap.release()
            if self.player:
                self.player.stop()
    
if __name__ == "__main__":
    eie_system = EIESystem()
    eie_system.run()
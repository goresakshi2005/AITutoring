import os
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AITutor:
    def __init__(self):
        """Initialize the AI tutor with Gemini"""
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in .env file")
            
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            print("AI Tutor initialized successfully!")
        except Exception as e:
            print(f"Failed to initialize AI Tutor: {e}")
            exit(1)

    def _generate_answer(self, question, image=None):
        """Generate answer using Gemini with an optimized tutoring prompt"""
        try:
            prompt = (
                "You are an expert tutor specializing in personalized education. "
                "When responding to the student's question, please follow these guidelines:\n\n"
                "1. First, analyze the question to understand the student's current level and needs\n"
                "2. Provide a clear, concise explanation using simple language\n"
                "3. Break down complex concepts into step-by-step instructions\n"
                "4. Include relevant examples or analogies to aid understanding\n"
                "5. Where appropriate, offer visual descriptions or diagrams in text format\n"
                "6. Suggest practice exercises or thought questions to reinforce learning\n"
                "7. Identify common misconceptions related to the topic\n"
                "8. Adapt your explanation based on the apparent complexity of the question\n\n"
                "If the question is unclear, ask for clarification before proceeding.\n\n"
                "Student's question:\n"
                f"{question}\n\n"
                "Please provide your expert explanation:"
            )
            
            if image:
                # For image-based questions
                response = self.model.generate_content([prompt, image])
            else:
                # For text-based questions
                response = self.model.generate_content(prompt)
                
            return response.text
        except Exception as e:
            return f"Error generating answer: {e}"

    def solve_text_doubt(self, text):
        """Solve text-based doubts"""
        return self._generate_answer(text)

    def solve_image_doubt(self, image_path):
        """Solve doubts from images using Gemini's multimodal capabilities"""
        try:
            print(f"Processing image: {image_path}")
            img = Image.open(image_path)
            
            # No need to extract text - send image directly to Gemini
            answer = self._generate_answer("Explain the content of this image and answer any questions it might contain.", img)
            return answer
        except Exception as e:
            return f"Error processing image: {e}"

def main():
    # Initialize AI Tutor
    tutor = AITutor()
    
    print("\nWelcome to AI Tutor!")
    print("1. Text question")
    print("2. Image question")
    print("3. Exit")
    
    while True:
        choice = input("\nChoose option (1-3): ").strip()
        
        if choice == "1":
            question = input("\nEnter your question: ")
            answer = tutor.solve_text_doubt(question)
            print(f"\nAnswer:\n{answer}")
        elif choice == "2":
            image_path = input("Enter image path: ").strip()
            answer = tutor.solve_image_doubt(image_path)
            print(f"\nAnswer:\n{answer}")
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
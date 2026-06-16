import os
import re
import sys
from dotenv import load_dotenv
from groq import Groq
from gtts import gTTS
import pygame
load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

print("--- Step 1: Generating Text (Llama 3.3 70B) ---")

try:
    text_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile", 
        messages=[{"role": "user", "content": "Explain how a Directed Acyclic Graph works in two simple sentences."}]
    )
    raw_output = text_response.choices[0].message.content
    spoken_text = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL).strip()
    print(f"Final Answer: {spoken_text}\n")

except Exception as e:
    print(f"Text Generation Failed: {e}")
    sys.exit(1)


# --- Step 2: Generating Audio (Groq Orpheus TTS) ---
# --- Step 2: Generating Audio (Groq Orpheus TTS) ---
try:
    audio_response = client.audio.speech.create(
        model="canopylabs/orpheus-v1-english",
        voice="austin",  
        input=spoken_text,
        response_format="wav"  
    )
    
    audio_file = "groq_audio_output.wav"  
    # FIXED: Use Groq's specific binary write method
    audio_response.write_to_file(audio_file)
    print("Audio generation complete.\n")

except Exception as e:
    print(f"\n[API BLOCKED] Audio Generation Failed: {e}")
    sys.exit(1)

print("--- Step 3: Playback ---")

try:
    pygame.mixer.init()
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    print("\nPipeline Execution Complete.")
except Exception as e:
    print(f"Audio Playback Failed: {e}")

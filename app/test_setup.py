# test_setup.py
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
print(f"API Key found: {'✅ Yes' if api_key else '❌ No'}")
print(f"API Key length: {len(api_key) if api_key else 0}")

if api_key:
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        print("✅ Gemini client created successfully!")
    except Exception as e:
        print(f"❌ Error creating client: {e}")
else:
    print("❌ Please add GEMINI_API_KEY to your .env file")

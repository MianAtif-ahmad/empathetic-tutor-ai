 #test_openai.py - Quick test of your OpenAI setup
import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Check if key exists
api_key = os.getenv('OPENAI_API_KEY')
print(f"API Key loaded: {bool(api_key)}")
if api_key:
    print(f"Key starts with: {api_key[:10]}...")
else:
    print("❌ No API key found!")
    exit(1)

# Test OpenAI import and basic call
try:
    import openai
    print("✅ OpenAI library imported successfully")
    
    client = openai.OpenAI(api_key=api_key)
    print("✅ OpenAI client created")
    
    # Test a simple call
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "What's wrong with this code: print{hello}"}
        ],
        max_tokens=100
    )
    
    print("✅ OpenAI API call successful!")
    print(f"Response: {response.choices[0].message.content}")
    
except ImportError as e:
    print(f"❌ OpenAI library not installed: {e}")
    print("Run: pip install openai")
    
except Exception as e:
    print(f"❌ OpenAI API call failed: {e}")
    print("Check your API key and internet connection")
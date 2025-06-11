import requests
import json
import time

print("🎭 EMPATHETIC TUTOR AI - LIVE DEMO")
print("=" * 60)

API_URL = "http://localhost:8000"

# Demo messages showing progression
demo_messages = [
    ("Hi, can you explain what a variable is?", "calm_student"),
    ("I'm a bit confused about how loops work", "learning_student"),
    ("Why isn't my code working? This is frustrating!", "frustrated_student"),
    ("I HATE THIS! NOTHING MAKES SENSE ANYMORE!!!", "very_frustrated_student"),
    ("I've been stuck for hours and I want to give up", "exhausted_student"),
]

print("\nDemonstrating frustration detection across different emotional states:\n")

for i, (message, student_id) in enumerate(demo_messages, 1):
    print(f"{i}. Student '{student_id}':")
    print(f"   Message: \"{message}\"")
    
    # Send to API
    response = requests.post(
        f"{API_URL}/analyze",
        json={"message": message, "student_id": student_id}
    )
    
    if response.status_code == 200:
        data = response.json()
        
        # Show scores
        print(f"   Simple Score: {data['frustration_score']:.1f}")
        print(f"   ML Score: {data['ml_score']:.1f} {'⚠️' if data['ml_score'] > 7 else ''}")
        print(f"   Response: \"{data['response'][:60]}...\"")
        
        # Visual frustration meter
        level = int(data['ml_score'])
        meter = "🟢" * max(0, 3-level) + "🟡" * max(0, min(3, 7-level)) + "🔴" * max(0, level-6)
        print(f"   Frustration: [{meter:<10}] {data['ml_score']:.1f}/10")
        
    print()
    time.sleep(0.5)

print("=" * 60)
print("Notice how ML scoring captures emotional nuance that keywords miss!")
print("This enables more appropriate and empathetic responses.")

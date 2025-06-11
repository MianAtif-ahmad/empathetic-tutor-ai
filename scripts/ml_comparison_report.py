import sqlite3
import json

conn = sqlite3.connect('empathetic_tutor.db')
cursor = conn.cursor()

print("🤖 ML vs Simple Scoring Comparison")
print("=" * 70)

cursor.execute("""
    SELECT 
        message,
        frustration_score as ml_score,
        json_extract(additional_data, '$.simple_score') as simple_score,
        features
    FROM interactions
    WHERE additional_data LIKE '%ml_score%'
    ORDER BY created_at DESC
    LIMIT 10
""")

for row in cursor.fetchall():
    message, ml_score, simple_score, features_json = row
    features = json.loads(features_json) if features_json else {}
    
    print(f"\nMessage: '{message[:60]}...'")
    print(f"Simple Score: {simple_score or 0:.1f} | ML Score: {ml_score:.1f}")
    print(f"Key factors: sentiment={features.get('sentiment', 0):.2f}, "
          f"keywords={features.get('keywords', 0)}, "
          f"emotional={features.get('emotional_intensity', 0)}")
    
    if abs((ml_score or 0) - (simple_score or 0)) > 3:
        print("⚠️  Large difference - ML detected nuance missed by keywords!")

conn.close()

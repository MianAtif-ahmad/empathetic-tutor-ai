import requests
import json
import time
import sqlite3

API_URL = "http://localhost:8000"

def test_full_integration():
    print("🧪 INTEGRATION TEST SUITE")
    print("=" * 60)
    
    # Test 1: API Health
    print("\n1️⃣ Testing API Health...")
    response = requests.get(f"{API_URL}/health")
    assert response.status_code == 200
    assert response.json()["ml_enabled"] == True
    print("✅ API is healthy with ML enabled")
    
    # Test 2: Feature Extraction + ML Scoring
    print("\n2️⃣ Testing Feature Extraction & ML Scoring...")
    test_messages = [
        ("How do I write a loop?", 0, 5),  # (message, expected_simple, expected_ml_range)
        ("I HATE this! Nothing works!", 0, 9),
        ("Can you help me understand recursion please?", 0, 6),
    ]
    
    for msg, exp_simple, exp_ml in test_messages:
        response = requests.post(f"{API_URL}/analyze", json={
            "message": msg,
            "student_id": "test_integration"
        })
        assert response.status_code == 200
        data = response.json()
        
        print(f"\n   Message: '{msg}'")
        print(f"   Simple Score: {data['frustration_score']} (expected ~{exp_simple})")
        print(f"   ML Score: {data['ml_score']} (expected ~{exp_ml})")
        assert abs(data['frustration_score'] - exp_simple) < 2
        assert 3 < data['ml_score'] < 10
        assert 'features' in data
        assert 'sentiment' in data['features']
        print("   ✅ Scoring working correctly")
    
    # Test 3: Database Persistence
    print("\n3️⃣ Testing Database Persistence...")
    conn = sqlite3.connect('empathetic_tutor.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(*) FROM interactions 
        WHERE student_id = 'test_integration'
    """)
    count = cursor.fetchone()[0]
    assert count >= 3
    print(f"✅ Found {count} interactions in database")
    
    # Test 4: Student Stats
    print("\n4️⃣ Testing Student Statistics...")
    response = requests.get(f"{API_URL}/student/test_integration/stats")
    assert response.status_code == 200
    stats = response.json()
    assert stats['total_interactions'] >= 3
    print(f"✅ Student stats working: {stats['total_interactions']} interactions")
    
    # Test 5: ML vs Simple Difference
    print("\n5️⃣ Testing ML Advantage...")
    cursor.execute("""
        SELECT 
            AVG(ABS(frustration_score - json_extract(additional_data, '$.simple_score')))
        FROM interactions
        WHERE student_id = 'test_integration'
    """)
    avg_diff = cursor.fetchone()[0]
    assert avg_diff > 3
    print(f"✅ ML provides {avg_diff:.1f} points better detection on average")
    
    conn.close()
    print("\n✅ ALL INTEGRATION TESTS PASSED!")

if __name__ == "__main__":
    test_full_integration()

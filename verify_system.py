import os
import sys
import subprocess
import sqlite3

print("🔍 SYSTEM VERIFICATION")
print("=" * 60)

# Check files exist
print("\n📁 Checking core files...")
files_to_check = [
    ("simple_api_enhanced.py", "Main API"),
    ("backend/app/services/nlp/feature_extractor.py", "Feature Extractor"),
    ("backend/app/services/ml/frustration_estimator_simple.py", "ML Estimator"),
    ("empathetic_tutor.db", "Database"),
]

for file, desc in files_to_check:
    if os.path.exists(file):
        print(f"✅ {desc}: {file}")
    else:
        print(f"❌ {desc}: {file} NOT FOUND")

# Check API is running
print("\n🌐 Checking API...")
try:
    import requests
    response = requests.get("http://localhost:8000/health")
    if response.status_code == 200:
        print("✅ API is running on port 8000")
        print(f"   Response: {response.json()}")
    else:
        print("❌ API returned error")
except:
    print("❌ API is not reachable")

# Check database
print("\n💾 Checking database...")
try:
    conn = sqlite3.connect('empathetic_tutor.db')
    cursor = conn.cursor()
    
    # Check tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [t[0] for t in cursor.fetchall()]
    print(f"✅ Tables found: {', '.join(tables)}")
    
    # Check data
    cursor.execute("SELECT COUNT(*) FROM interactions")
    count = cursor.fetchone()[0]
    print(f"✅ Interactions stored: {count}")
    
    # Check ML scoring
    cursor.execute("""
        SELECT COUNT(*) FROM interactions 
        WHERE json_extract(additional_data, '$.ml_score') IS NOT NULL
    """)
    ml_count = cursor.fetchone()[0]
    print(f"✅ ML-scored interactions: {ml_count}")
    
    conn.close()
except Exception as e:
    print(f"❌ Database error: {e}")

# Check Python packages
print("\n📦 Checking Python packages...")
required_packages = ['fastapi', 'nltk', 'numpy', 'sqlalchemy']
import pkg_resources
for package in required_packages:
    try:
        version = pkg_resources.get_distribution(package).version
        print(f"✅ {package}: {version}")
    except:
        print(f"❌ {package}: NOT INSTALLED")

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")

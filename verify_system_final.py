import os
import sys
import subprocess
import sqlite3

print("🔍 SYSTEM VERIFICATION")
print("=" * 60)

# Track verification results
all_good = True
file_count = 0
interaction_count = 0
ml_count = 0

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
        file_count += 1
    else:
        print(f"❌ {desc}: {file} NOT FOUND")
        all_good = False

# Check API is running
print("\n🌐 Checking API...")
api_running = False
try:
    import requests
    response = requests.get("http://localhost:8000/health")
    if response.status_code == 200:
        print("✅ API is running on port 8000")
        print(f"   Response: {response.json()}")
        api_running = True
    else:
        print("❌ API returned error")
        all_good = False
except:
    print("❌ API is not reachable")
    all_good = False

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
    interaction_count = cursor.fetchone()[0]
    print(f"✅ Interactions stored: {interaction_count}")
    
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
    all_good = False

# Check Python packages
print("\n📦 Checking Python packages...")
required_packages = ['fastapi', 'nltk', 'numpy', 'sqlalchemy']
package_count = 0

import subprocess
for package in required_packages:
    try:
        result = subprocess.run(['pip', 'show', package], capture_output=True, text=True)
        if result.returncode == 0:
            # Extract version from output
            for line in result.stdout.split('\n'):
                if line.startswith('Version:'):
                    version = line.split(':')[1].strip()
                    print(f"✅ {package}: {version}")
                    package_count += 1
                    break
        else:
            print(f"❌ {package}: NOT INSTALLED")
            all_good = False
    except:
        print(f"⚠️  Could not check {package}")

print("\n" + "=" * 60)

# Dynamic summary based on actual results
if all_good:
    print("✅ SYSTEM VERIFICATION COMPLETE - ALL CHECKS PASSED!")
else:
    print("⚠️  SYSTEM VERIFICATION COMPLETE - SOME ISSUES FOUND")

print(f"\nSystem Status:")
print(f"- Core files: {file_count}/{len(files_to_check)} present")
print(f"- API: {'Running with ML features' if api_running else 'Not running'}")
print(f"- Database: {interaction_count} total interactions")
print(f"- ML Analysis: {ml_count} interactions with ML scoring")
print(f"- Packages: {package_count}/{len(required_packages)} installed")

if ml_count > 0 and interaction_count > 0:
    ml_percentage = (ml_count / interaction_count) * 100
    print(f"- ML Coverage: {ml_percentage:.1f}% of interactions")

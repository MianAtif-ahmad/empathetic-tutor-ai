import subprocess
import sqlite3
import requests
import os

def check_system():
    print("🔍 SYSTEM STATUS CHECK")
    print("=" * 50)
    
    # 1. Check database
    print("📊 Database Status:")
    if os.path.exists('empathetic_tutor.db'):
        print("  ✅ Database file exists")
        conn = sqlite3.connect('empathetic_tutor.db')
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"  ✅ Tables: {[t[0] for t in tables]}")
        cursor.execute("SELECT COUNT(*) FROM interactions")
        count = cursor.fetchone()[0]
        print(f"  ✅ Interactions stored: {count}")
        conn.close()
    else:
        print("  ❌ Database not found!")
    
    # 2. Check API
    print("\n🌐 API Status:")
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            print("  ✅ API is running on port 8000")
            print(f"  ✅ Health check: {response.json()}")
        else:
            print("  ⚠️  API returned status:", response.status_code)
    except requests.exceptions.RequestException:
        print("  ❌ API is not responding on port 8000")
        print("  Run: python3 simple_api.py")
    
    # 3. Check virtual environment
    print("\n🐍 Environment Status:")
    if os.environ.get('VIRTUAL_ENV'):
        print(f"  ✅ Virtual environment active: {os.environ['VIRTUAL_ENV']}")
    else:
        print("  ⚠️  Virtual environment not active")
        print("  Run: source venv/bin/activate")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    check_system()

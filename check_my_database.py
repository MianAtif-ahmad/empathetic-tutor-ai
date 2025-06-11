# check_my_database.py
import sqlite3
import json

def analyze_database():
    conn = sqlite3.connect('empathetic_tutor.db')
    cursor = conn.cursor()
    
    print("🔍 YOUR DATABASE ANALYSIS")
    print("=" * 50)
    
    # Check table structure
    cursor.execute("PRAGMA table_info(interactions);")
    columns = cursor.fetchall()
    print(f"\n📋 INTERACTIONS TABLE COLUMNS:")
    for col in columns:
        print(f"   {col[1]} ({col[2]})")
    
    # Check sample data
    cursor.execute("SELECT * FROM interactions ORDER BY created_at DESC LIMIT 1")
    sample = cursor.fetchone()
    
    if sample:
        print(f"\n📊 SAMPLE INTERACTION:")
        column_names = [col[1] for col in columns]
        for i, value in enumerate(sample):
            if column_names[i] == 'additional_data' and value:
                try:
                    parsed = json.loads(value)
                    print(f"   {column_names[i]}: {parsed}")
                except:
                    print(f"   {column_names[i]}: {value}")
            else:
                print(f"   {column_names[i]}: {value}")
    
    # Check for feedback table
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='feedback';")
    feedback_exists = cursor.fetchone()
    print(f"\n💬 FEEDBACK TABLE: {'✅ Exists' if feedback_exists else '❌ Not created yet'}")
    
    # Summary stats
    cursor.execute("SELECT COUNT(*), AVG(frustration_score), MIN(created_at), MAX(created_at) FROM interactions")
    stats = cursor.fetchone()
    print(f"\n📈 SUMMARY:")
    print(f"   Total interactions: {stats[0]}")
    print(f"   Average frustration: {stats[1]:.2f}")
    print(f"   Date range: {stats[2]} to {stats[3]}")
    
    conn.close()

if __name__ == "__main__":
    analyze_database()
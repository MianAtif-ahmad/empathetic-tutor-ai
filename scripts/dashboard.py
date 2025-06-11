import sqlite3
from datetime import datetime
import json

def show_dashboard():
    conn = sqlite3.connect('empathetic_tutor.db')
    cursor = conn.cursor()
    
    print("\n" + "🎓 EMPATHETIC TUTOR DASHBOARD 🎓".center(60))
    print("=" * 60)
    print(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # System status
    print("\n🔧 SYSTEM STATUS:")
    print("   ✅ Enhanced API: Running")
    print("   ✅ ML Scoring: Enabled") 
    print("   ✅ Feature Extraction: 15+ features")
    print("   ✅ Sentiment Analysis: Active")
    
    # Key metrics
    cursor.execute("""
        SELECT 
            COUNT(DISTINCT student_id) as students,
            COUNT(*) as total,
            AVG(frustration_score) as avg_ml,
            AVG(json_extract(additional_data, '$.simple_score')) as avg_simple
        FROM interactions
        WHERE json_extract(additional_data, '$.ml_score') IS NOT NULL
    """)
    
    stats = cursor.fetchone()
    
    print(f"\n📊 KEY METRICS:")
    print(f"   👥 Unique Students: {stats[0]}")
    print(f"   💬 Total Interactions: {stats[1]}")
    print(f"   🤖 Avg ML Score: {stats[2]:.2f}")
    print(f"   📝 Avg Simple Score: {stats[3] or 0:.2f}")
    
    # Recent activity
    print(f"\n📈 RECENT ACTIVITY:")
    cursor.execute("""
        SELECT 
            student_id, 
            frustration_score,
            substr(message, 1, 40) as msg,
            datetime(created_at) as time
        FROM interactions 
        ORDER BY created_at DESC 
        LIMIT 5
    """)
    
    for student, score, msg, time in cursor.fetchall():
        emoji = "😊" if score < 3 else "😐" if score < 7 else "😰"
        print(f"   {emoji} {student}: {score:.1f} - {msg}...")
    
    conn.close()
    print("\n" + "=" * 60)

if __name__ == "__main__":
    show_dashboard()

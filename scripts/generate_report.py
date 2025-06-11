import sqlite3
import json
from datetime import datetime

def generate_report():
    conn = sqlite3.connect('empathetic_tutor.db')
    cursor = conn.cursor()
    
    print("=" * 60)
    print("EMPATHETIC TUTOR SYSTEM REPORT")
    print("=" * 60)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Total interactions
    cursor.execute("SELECT COUNT(*) FROM interactions")
    total = cursor.fetchone()[0]
    print(f"Total Interactions: {total}")
    
    # Average frustration
    cursor.execute("SELECT AVG(frustration_score) FROM interactions")
    avg_frustration = cursor.fetchone()[0] or 0
    print(f"Average Frustration Score: {avg_frustration:.2f}")
    
    # Student summary
    print("\nSTUDENT SUMMARY:")
    print("-" * 40)
    cursor.execute("""
        SELECT 
            student_id,
            COUNT(*) as interactions,
            ROUND(AVG(frustration_score), 2) as avg_frustration
        FROM interactions
        GROUP BY student_id
        ORDER BY avg_frustration DESC
    """)
    
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]} interactions, avg frustration: {row[2]}")
    
    # Concept frequency
    print("\nCONCEPT FREQUENCY:")
    print("-" * 40)
    cursor.execute("""
        SELECT concepts, COUNT(*) as freq
        FROM interactions
        WHERE concepts != '[]'
        GROUP BY concepts
    """)
    
    concept_counts = {}
    for row in cursor.fetchall():
        concepts = json.loads(row[0])
        for concept in concepts:
            concept_counts[concept] = concept_counts.get(concept, 0) + 1
    
    for concept, count in sorted(concept_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {concept}: {count}")
    
    # Frustration distribution
    print("\nFRUSTRATION DISTRIBUTION:")
    print("-" * 40)
    cursor.execute("""
        SELECT 
            CASE 
                WHEN frustration_score < 3 THEN 'Low (0-3)'
                WHEN frustration_score < 7 THEN 'Medium (3-7)'
                ELSE 'High (7-10)'
            END as level,
            COUNT(*) as count
        FROM interactions
        GROUP BY level
    """)
    
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]} interactions")
    
    conn.close()
    print("\n" + "=" * 60)

if __name__ == "__main__":
    generate_report()

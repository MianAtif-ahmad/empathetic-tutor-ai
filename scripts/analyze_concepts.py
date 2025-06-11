import sqlite3
import json

conn = sqlite3.connect('empathetic_tutor.db')
cursor = conn.cursor()

# Get all concepts
cursor.execute("SELECT concepts FROM interactions WHERE concepts != '[]'")

concept_count = {}
for row in cursor.fetchall():
    concepts = json.loads(row[0])
    for concept in concepts:
        concept_count[concept] = concept_count.get(concept, 0) + 1

print("Concept Frequency Analysis:")
print("-" * 30)
for concept, count in sorted(concept_count.items(), key=lambda x: x[1], reverse=True):
    print(f"{concept}: {count}")

conn.close()

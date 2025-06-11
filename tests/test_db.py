import sqlite3

# Connect to database
conn = sqlite3.connect('student_frustration.db')
cursor = conn.cursor()

# Check tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables created:", tables)

# Check students table structure
cursor.execute("PRAGMA table_info(students);")
print("\nStudents table structure:")
for column in cursor.fetchall():
    print(f"  {column}")

conn.close()

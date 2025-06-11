import sqlite3

# Connect to the correct database
conn = sqlite3.connect('empathetic_tutor.db')
cursor = conn.cursor()

# Check tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables in empathetic_tutor.db:", tables)

# Check each table structure
for table in tables:
    table_name = table[0]
    print(f"\n{table_name} table structure:")
    cursor.execute(f"PRAGMA table_info({table_name});")
    for column in cursor.fetchall():
        print(f"  {column}")

conn.close()

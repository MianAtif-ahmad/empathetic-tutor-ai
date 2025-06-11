#!/usr/bin/env python3
"""
Railway startup script for Empathetic AI Tutor
Initializes database and directories
"""
import os
import sqlite3
from pathlib import Path

def setup_directories():
    """Create required directories"""
    directories = ['logs', 'config', 'data', 'backups', 'static']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Directory ready: {directory}")

def setup_database():
    """Initialize SQLite database for Railway"""
    db_path = os.getenv('DATABASE_PATH', './data/empathetic_tutor.db')
    
    # Ensure data directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(db_path):
        print(f"🗃️  Initializing database: {db_path}")
        conn = sqlite3.connect(db_path)
        
        # Configure for Railway environment
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA synchronous=NORMAL')
        conn.execute('PRAGMA cache_size=10000')
        conn.execute('PRAGMA temp_store=memory')
        
        conn.close()
        print("✅ Database initialized for Railway")
    else:
        print("✅ Database already exists")

def main():
    """Main startup function"""
    print("🚀 Empathetic AI Tutor - Railway Startup")
    print("=" * 50)
    
    setup_directories()
    setup_database()
    
    print("✅ Railway startup completed successfully!")
    print("🎓 AI Tutor ready for students!")

if __name__ == "__main__":
    main()

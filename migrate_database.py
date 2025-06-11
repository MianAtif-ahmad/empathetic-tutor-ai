#!/usr/bin/env python3
# migrate_database.py - Database Migration for Critical Fixes
import sqlite3
import os
from datetime import datetime

def migrate_database(db_path: str):
    """Apply database migrations for critical fixes"""
    print(f"🔄 Starting database migration for {db_path}")
    
    # Backup database first
    backup_path = f"{db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if os.path.exists(db_path):
        import shutil
        shutil.copy2(db_path, backup_path)
        print(f"💾 Database backed up to {backup_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Enable WAL mode for better concurrency
        cursor.execute("PRAGMA journal_mode=WAL")
        print("✅ Enabled WAL mode")
        
        # Add indexes for better performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_interactions_student_id ON interactions(student_id)",
            "CREATE INDEX IF NOT EXISTS idx_interactions_created_at ON interactions(created_at)", 
            "CREATE INDEX IF NOT EXISTS idx_learning_feedback_interaction_id ON learning_feedback(interaction_id)",
            "CREATE INDEX IF NOT EXISTS idx_learning_feedback_created_at ON learning_feedback(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_discovered_patterns_status ON discovered_patterns(status)",
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
            print(f"✅ Created index: {index_sql.split()[-1]}")
        
        # Add missing columns if they don't exist
        new_columns = [
            ("interactions", "processing_time_ms", "REAL"),
            ("interactions", "memory_usage_mb", "REAL"),
            ("learning_feedback", "validation_score", "REAL"),
        ]
        
        for table, column, data_type in new_columns:
            try:
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {data_type}")
                print(f"✅ Added column {table}.{column}")
            except sqlite3.OperationalError:
                # Column already exists
                pass
        
        conn.commit()
        print("✅ Database migration completed successfully")
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Migration failed: {e}")
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python migrate_database.py <database_path>")
        sys.exit(1)
    
    migrate_database(sys.argv[1])

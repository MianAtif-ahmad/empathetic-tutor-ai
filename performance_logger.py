# performance_logger.py
import time
import json
from datetime import datetime
from typing import Dict, Any

class PerformanceLogger:
    def __init__(self, db_path: str = "empathetic_tutor.db"):
        self.db_path = db_path
        self._init_performance_table()
    
    def _init_performance_table(self):
        """Create performance logging table"""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_logs (
                id TEXT PRIMARY KEY,
                interaction_id TEXT,
                ai_provider TEXT,
                response_time_ms INTEGER,
                tokens_used INTEGER,
                prompt_length INTEGER,
                response_length INTEGER,
                success BOOLEAN,
                error_message TEXT,
                created_at TIMESTAMP,
                FOREIGN KEY (interaction_id) REFERENCES interactions (id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def log_performance(
        self, 
        interaction_id: str, 
        ai_provider: str,
        response_time: float,
        prompt_length: int,
        response_length: int,
        success: bool,
        error_message: str = None,
        tokens_used: int = None
    ):
        """Log performance metrics"""
        import sqlite3
        import uuid
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO performance_logs 
                (id, interaction_id, ai_provider, response_time_ms, tokens_used,
                 prompt_length, response_length, success, error_message, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(uuid.uuid4()),
                interaction_id,
                ai_provider,
                int(response_time * 1000),  # Convert to milliseconds
                tokens_used,
                prompt_length,
                response_length,
                success,
                error_message,
                datetime.now()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Performance logging failed: {e}")

# Global performance logger
performance_logger = PerformanceLogger()
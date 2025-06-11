# system_logger.py
import sqlite3
import uuid
import json
from datetime import datetime
from typing import Dict, Any, Optional

class SystemLogger:
    def __init__(self, db_path: str = "empathetic_tutor.db"):
        self.db_path = db_path
        self._init_system_logs_table()
    
    def _init_system_logs_table(self):
        """Create system events logging table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_logs (
                id TEXT PRIMARY KEY,
                event_type TEXT,
                event_data TEXT,
                user_id TEXT,
                ip_address TEXT,
                user_agent TEXT,
                success BOOLEAN,
                error_message TEXT,
                duration_ms INTEGER,
                created_at TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def log_event(
        self,
        event_type: str,
        event_data: Dict[str, Any] = None,
        user_id: str = None,
        success: bool = True,
        error_message: str = None,
        duration_ms: int = None,
        request = None
    ):
        """Log system events"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Extract request info if available
            ip_address = None
            user_agent = None
            if request:
                ip_address = getattr(request.client, 'host', None) if hasattr(request, 'client') else None
                user_agent = request.headers.get('user-agent', '') if hasattr(request, 'headers') else None
            
            cursor.execute("""
                INSERT INTO system_logs 
                (id, event_type, event_data, user_id, ip_address, user_agent,
                 success, error_message, duration_ms, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(uuid.uuid4()),
                event_type,
                json.dumps(event_data) if event_data else None,
                user_id,
                ip_address,
                user_agent,
                success,
                error_message,
                duration_ms,
                datetime.now()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"System logging failed: {e}")

# Global system logger
system_logger = SystemLogger()
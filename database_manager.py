
# database_manager.py - Centralized Database Connection Management
import sqlite3
import threading
from contextlib import contextmanager
from typing import Generator
import time

class DatabaseManager:
    """
    Centralized database connection manager with connection pooling
    and proper resource management
    """
    
    def __init__(self, db_path: str, max_connections: int = 10):
        self.db_path = db_path
        self.max_connections = max_connections
        self._local = threading.local()
        self._connection_count = 0
        self._lock = threading.Lock()
    
    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection with proper resource management"""
        conn = None
        try:
            conn = self._get_thread_connection()
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise
        else:
            if conn:
                conn.commit()
        finally:
            # Connection stays open for thread reuse
            pass
    
    def _get_thread_connection(self) -> sqlite3.Connection:
        """Get or create connection for current thread"""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            with self._lock:
                if self._connection_count >= self.max_connections:
                    time.sleep(0.1)  # Brief wait if at limit
                
                self._local.conn = sqlite3.connect(
                    self.db_path,
                    timeout=30.0,  # 30 second timeout
                    check_same_thread=False
                )
                self._local.conn.execute("PRAGMA journal_mode=WAL")  # Better concurrency
                self._local.conn.execute("PRAGMA synchronous=NORMAL")  # Better performance
                self._connection_count += 1
        
        return self._local.conn
    
    def execute_query(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a query with proper connection management"""
        with self.get_connection() as conn:
            return conn.execute(query, params)
    
    def execute_many(self, query: str, params_list: list) -> None:
        """Execute multiple queries with proper connection management"""
        with self.get_connection() as conn:
            conn.executemany(query, params_list)
    
    def close_all_connections(self):
        """Close all connections (for shutdown)"""
        with self._lock:
            if hasattr(self._local, 'conn') and self._local.conn:
                self._local.conn.close()
                self._local.conn = None
                self._connection_count = max(0, self._connection_count - 1)

# Global database manager instance
db_manager = None

def initialize_database_manager(db_path: str) -> DatabaseManager:
    """Initialize global database manager"""
    global db_manager
    db_manager = DatabaseManager(db_path)
    return db_manager

def get_db_manager() -> DatabaseManager:
    """Get global database manager instance"""
    if db_manager is None:
        raise RuntimeError("Database manager not initialized. Call initialize_database_manager() first.")
    return db_manager

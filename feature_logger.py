#feature_logger.py
import sqlite3
import json
import uuid
from datetime import datetime
from typing import Dict, Any

class FeatureLogger:
    def __init__(self, db_path: str = "empathetic_tutor.db"):
        self.db_path = db_path
        self._init_features_table()
    
    def _init_features_table(self):
        """Create detailed feature logging table"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detailed_features (
                id TEXT PRIMARY KEY,
                interaction_id TEXT,
                student_id TEXT,
                
                -- Emotional Analysis
                emotional_state TEXT,
                confidence_level TEXT,
                learning_style TEXT,
                frustration_pattern TEXT,
                
                -- Text Features
                message_length INTEGER,
                word_count INTEGER,
                exclamation_count INTEGER,
                question_count INTEGER,
                caps_ratio REAL,
                
                -- Keyword Analysis
                high_frustration_keywords INTEGER,
                medium_frustration_keywords INTEGER,
                mild_frustration_keywords INTEGER,
                error_keywords INTEGER,
                help_seeking_keywords INTEGER,
                
                -- Advanced Features
                sentiment_score REAL,
                empathy_level TEXT,
                intervention_type TEXT,
                
                -- Context
                previous_interactions_count INTEGER,
                time_since_last_interaction INTEGER,
                
                created_at TIMESTAMP,
                FOREIGN KEY (interaction_id) REFERENCES interactions (id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def log_detailed_features(
        self,
        interaction_id: str,
        student_id: str,
        features: Dict[str, Any]
    ):
        """Log detailed feature analysis"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO detailed_features 
                (id, interaction_id, student_id, emotional_state, confidence_level,
                 learning_style, frustration_pattern, message_length, word_count,
                 exclamation_count, question_count, caps_ratio, high_frustration_keywords,
                 medium_frustration_keywords, mild_frustration_keywords, error_keywords,
                 help_seeking_keywords, sentiment_score, empathy_level, intervention_type,
                 previous_interactions_count, time_since_last_interaction, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(uuid.uuid4()),
                interaction_id,
                student_id,
                features.get('emotional_state'),
                features.get('confidence_level'),
                features.get('learning_style'),
                features.get('frustration_pattern'),
                features.get('message_length', 0),
                features.get('word_count', 0),
                features.get('exclamation_count', 0),
                features.get('question_count', 0),
                features.get('caps_ratio', 0.0),
                features.get('high_frustration_keywords', 0),
                features.get('medium_frustration_keywords', 0),
                features.get('mild_frustration_keywords', 0),
                features.get('error_keywords', 0),
                features.get('help_seeking_keywords', 0),
                features.get('sentiment_score', 0.0),
                features.get('empathy_level'),
                features.get('intervention_type'),
                features.get('previous_interactions_count', 0),
                features.get('time_since_last_interaction', 0),
                datetime.now()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Detailed feature logging failed: {e}")

# Global feature logger
feature_logger = FeatureLogger()
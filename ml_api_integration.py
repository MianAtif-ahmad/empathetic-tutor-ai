ml_api_integration.py - API Integration for ML Learning System
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
import sqlite3
import json
import uuid
from datetime import datetime

# Import the ML learning components
from ml_learning_system import (
    HybridFrustrationAnalyzer, 
    LearningFeedback, 
    PatternDiscoveryEngine
)

# Enhanced Pydantic Models
class MLStudentMessage(BaseModel):
    message: str
    student_id: str = "default_student"
    context: Optional[Dict] = None

class DetailedFeedbackRequest(BaseModel):
    interaction_id: str
    student_id: str
    helpful: bool
    frustration_before: int  # 1-10
    frustration_after: int   # 1-10
    response_quality: int    # 1-5
    empathy_level: str       # "too_low", "just_right", "too_high"
    response_time_seconds: Optional[float] = None
    follow_up_needed: Optional[bool] = None
    missing_concepts: Optional[List[str]] = None
    new_emotional_words: Optional[List[str]] = None
    additional_comments: Optional[str] = None

class PatternApprovalRequest(BaseModel):
    pattern_ids: List[str]
    action: str  # "approve", "reject", "modify"
    modifications: Optional[Dict[str, Dict]] = None

class MLAnalyzeResponse(BaseModel):
    interaction_id: str
    frustration_score: float
    confidence: float
    empathy_level: str
    concepts: List[str]
    rule_contribution: float
    ml_contribution: float
    learning_opportunity: bool
    response: str
    ai_provider: Optional[str] = None
    debug_info: Optional[List[str]] = None

# Database Migration Functions
def migrate_database_for_ml(db_path: str):
    """Add ML learning tables to existing database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Learning feedback table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS learning_feedback (
            id TEXT PRIMARY KEY,
            interaction_id TEXT,
            student_id TEXT,
            prediction_error REAL,
            actual_frustration REAL,
            predicted_frustration REAL,
            feature_importance TEXT,  -- JSON
            response_helpful INTEGER,
            empathy_appropriate INTEGER,
            response_time_seconds REAL,
            follow_up_needed INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (interaction_id) REFERENCES interactions (id)
        )
    """)
    
    # Discovered patterns table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS discovered_patterns (
            id TEXT PRIMARY KEY,
            pattern_type TEXT,  -- 'emotional_pattern', 'keyword', 'concept_keyword'
            pattern_text TEXT,
            concept_category TEXT,  -- For concept keywords
            weight REAL,
            confidence_score REAL,
            usage_count INTEGER DEFAULT 0,
            effectiveness_score REAL DEFAULT 0.5,
            status TEXT DEFAULT 'candidate',  -- 'candidate', 'approved', 'active', 'deprecated'
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            approved_at TIMESTAMP,
            last_used TIMESTAMP
        )
    """)
    
    # Student learning profiles table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS student_learning_profiles (
            student_id TEXT PRIMARY KEY,
            personalized_weights TEXT,  -- JSON
            learning_style TEXT,  -- JSON
            adaptation_rate REAL DEFAULT 0.01,
            total_interactions INTEGER DEFAULT 0,
            avg_prediction_error REAL DEFAULT 0.0,
            last_weight_update TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Global weights history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS global_weights (
            id TEXT PRIMARY KEY,
            weights TEXT,  -- JSON
            version INTEGER,
            performance_metrics TEXT,  -- JSON
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Pattern performance tracking
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pattern_performance (
            id TEXT PRIMARY KEY,
            pattern_id TEXT,
            usage_date DATE,
            prediction_improvement REAL,
            student_satisfaction REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (pattern_id) REFERENCES discovered_patterns (id)
        )
    """)
    
    # Add ML features column to interactions table if not exists
    try:
        cursor.execute("ALTER TABLE interactions ADD COLUMN ml_features TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    try:
        cursor.execute("ALTER TABLE interactions ADD COLUMN confidence_score REAL")
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    conn.commit()
    conn.close()
    print("✅ Database migrated for ML learning system")

# API Integration Class
class MLTutoringAPI:
    """Enhanced API with ML learning capabilities"""
    
    def __init__(self, app: FastAPI, db_path: str, existing_analyzer, ai_generator=None):
        self.app = app
        self.db_path = db_path
        self.ai_generator = ai_generator
        
        # Migrate database
        migrate_database_for_ml(db_path)
        
        # Initialize ML system
        self.ml_analyzer = HybridFrustrationAnalyzer(db_path, existing_analyzer)
        self.pattern_discovery = PatternDiscoveryEngine(db_path)
        
        # Add new routes
        self._add_ml_routes()
        
    def _add_ml_routes(self):
        """Add ML-specific API routes"""
        
        @self.app.post("/analyze/ml", response_model=MLAnalyzeResponse)
        async def ml_analyze_message(request: MLStudentMessage):
            """Enhanced analysis with ML learning"""
            
            # Analyze with hybrid system
            analysis = self.ml_analyzer.analyze_with_learning(
                request.message, 
                request.student_id
            )
            
            # Generate response using existing AI system
            ai_provider_used = None
            if self.ai_generator:
                try:
                    response = await self.ai_generator.generate_response(
                        message=request.message,
                        frustration_score=analysis["frustration_score"],
                        concepts=analysis["concepts"],
                        student_id=request.student_id
                    )
                    ai_provider_used = "ai_system"
                except Exception as e:
                    print(f"❌ AI generation failed: {e}")
                    response = self._generate_fallback_response(
                        analysis["frustration_score"], 
                        analysis["concepts"], 
                        request.message
                    )
                    ai_provider_used = "template_fallback"
            else:
                response = self._generate_fallback_response(
                    analysis["frustration_score"], 
                    analysis["concepts"], 
                    request.message
                )
                ai_provider_used = "template_only"
            
            # Store interaction with ML features
            interaction_id = str(uuid.uuid4())
            self._store_ml_interaction(
                interaction_id, request, analysis, response, ai_provider_used
            )
            
            return MLAnalyzeResponse(
                interaction_id=interaction_id,
                frustration_score=analysis["frustration_score"],
                confidence=analysis["confidence"],
                empathy_level=analysis["empathy_level"],
                concepts=analysis["concepts"],
                rule_contribution=analysis["rule_contribution"],
                ml_contribution=analysis["ml_contribution"],
                learning_opportunity=analysis["learning_opportunity"],
                response=response,
                ai_provider=ai_provider_used,
                debug_info=analysis.get("debug_info", [])
            )
        
        @self.app.post("/feedback/detailed")
        async def submit_detailed_feedback(request: DetailedFeedbackRequest):
            """Submit detailed feedback for ML learning"""
            
            # Create learning feedback object
            prediction_error = request.frustration_after - self._get_predicted_frustration(request.interaction_id)
            
            feedback = LearningFeedback(
                interaction_id=request.interaction_id,
                student_id=request.student_id,
                prediction_error=prediction_error,
                actual_frustration=float(request.frustration_after),
                predicted_frustration=self._get_predicted_frustration(request.interaction_id),
                feature_importance={},  # Will be calculated
                response_helpful=request.helpful,
                empathy_appropriate=request.empathy_level == "just_right"
            )
            
            # Process learning
            self.ml_analyzer.learn_from_feedback(request.interaction_id, feedback)
            
            # Store detailed feedback
            self._store_detailed_feedback(request, feedback)
            
            # Add new patterns if provided
            if request.new_emotional_words:
                self._add_suggested_patterns(request.new_emotional_words, request.student_id)
            
            return {
                "feedback_id": str(uuid.uuid4()),
                "message": "Detailed feedback processed for learning",
                "learning_triggered": True,
                "prediction_error": prediction_error
            }
        
        @self.app.get("/learning/patterns")
        async def get_discovered_patterns(status: str = "candidate", limit: int = 50):
            """Get discovered patterns for review"""
            return self._get_patterns_by_status(status, limit)
        
        @self.app.post("/learning/patterns/approve")
        async def approve_patterns(request: PatternApprovalRequest):
            """Approve or reject discovered patterns"""
            return self._process_pattern_approval(request)
        
        @self.app.post("/learning/discover")
        async def trigger_pattern_discovery(lookback_days: int = 30):
            """Manually trigger pattern discovery"""
            patterns = self.pattern_discovery.discover_patterns(lookback_days)
            return {
                "discovered_patterns": len(patterns),
                "patterns": patterns,
                "message": f"Discovered {len(patterns)} new patterns from last {lookback_days} days"
            }
        
        @self.app.get("/learning/student/{student_id}/profile")
        async def get_student_learning_profile(student_id: str):
            """Get student's learning profile and personalized weights"""
            return self._get_student_profile(student_id)
        
        @self.app.get("/learning/analytics")
        async def get_learning_analytics(days: int = 30):
            """Get learning system analytics"""
            return self._get_learning_analytics(days)
        
        @self.app.post("/learning/weights/reset")
        async def reset_student_weights(student_id: str):
            """Reset student weights to global defaults"""
            return self._reset_student_weights(student_id)
    
    def _store_ml_interaction(self, interaction_id: str, request: MLStudentMessage, 
                            analysis: Dict, response: str, ai_provider: str):
        """Store interaction with ML features"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO interactions 
                (id, student_id, message, response, frustration_score, intervention_level,
                 features, concepts, additional_data, ml_features, confidence_score, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                interaction_id,
                request.student_id,
                request.message,
                response,
                analysis["frustration_score"],
                1 if analysis["frustration_score"] < 3 else 2 if analysis["frustration_score"] < 7 else 3,
                json.dumps({
                    "empathy_level": analysis["empathy_level"],
                    "learning_opportunity": analysis["learning_opportunity"]
                }),
                json.dumps(analysis["concepts"]),
                json.dumps({
                    "ai_provider": ai_provider,
                    "rule_contribution": analysis["rule_contribution"],
                    "ml_contribution": analysis["ml_contribution"],
                    "version": "4.0.0_ml"
                }),
                json.dumps(analysis["ml_features"]),
                analysis["confidence"],
                datetime.now()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error storing ML interaction: {e}")
    
    def _store_detailed_feedback(self, request: DetailedFeedbackRequest, feedback: LearningFeedback):
        """Store detailed feedback in learning_feedback table"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO learning_feedback 
                (id, interaction_id, student_id, prediction_error, actual_frustration,
                 predicted_frustration, response_helpful, empathy_appropriate,
                 response_time_seconds, follow_up_needed, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(uuid.uuid4()),
                request.interaction_id,
                request.student_id,
                feedback.prediction_error,
                feedback.actual_frustration,
                feedback.predicted_frustration,
                1 if request.helpful else 0,
                1 if request.empathy_level == "just_right" else 0,
                request.response_time_seconds,
                1 if request.follow_up_needed else 0,
                datetime.now()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error storing detailed feedback: {e}")
    
    def _get_predicted_frustration(self, interaction_id: str) -> float:
        """Get the predicted frustration score for an interaction"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT frustration_score FROM interactions WHERE id = ?", 
                (interaction_id,)
            )
            row = cursor.fetchone()
            conn.close()
            
            return row[0] if row else 0.0
            
        except Exception:
            return 0.0
    
    def _get_patterns_by_status(self, status: str, limit: int) -> Dict:
        """Get discovered patterns by status"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, pattern_type, pattern_text, weight, confidence_score, 
                       usage_count, effectiveness_score, created_at
                FROM discovered_patterns 
                WHERE status = ? 
                ORDER BY confidence_score DESC, created_at DESC 
                LIMIT ?
            """, (status, limit))
            
            patterns = []
            for row in cursor.fetchall():
                patterns.append({
                    "id": row[0],
                    "type": row[1],
                    "pattern": row[2],
                    "weight": row[3],
                    "confidence": row[4],
                    "usage_count": row[5],
                    "effectiveness": row[6],
                    "created_at": row[7]
                })
            
            conn.close()
            return {"patterns": patterns, "count": len(patterns)}
            
        except Exception as e:
            return {"patterns": [], "count": 0, "error": str(e)}
    
    def _process_pattern_approval(self, request: PatternApprovalRequest) -> Dict:
        """Process pattern approval/rejection"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            approved_count = 0
            
            for pattern_id in request.pattern_ids:
                if request.action == "approve":
                    cursor.execute("""
                        UPDATE discovered_patterns 
                        SET status = 'approved', approved_at = ?
                        WHERE id = ?
                    """, (datetime.now(), pattern_id))
                    
                    # Add to active configuration
                    self._add_pattern_to_config(pattern_id)
                    approved_count += 1
                    
                elif request.action == "reject":
                    cursor.execute("""
                        UPDATE discovered_patterns 
                        SET status = 'rejected'
                        WHERE id = ?
                    """, (pattern_id,))
            
            conn.commit()
            conn.close()
            
            return {
                "processed": len(request.pattern_ids),
                "approved": approved_count,
                "action": request.action,
                "message": f"Processed {len(request.pattern_ids)} patterns"
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _add_pattern_to_config(self, pattern_id: str):
        """Add approved pattern to active configuration"""
        # This would update the JSON config files
        # Implementation depends on how you want to handle config updates
        pass
    
    def _generate_fallback_response(self, frustration_score: float, concepts: List[str], message: str) -> str:
        """Generate fallback response when AI is not available"""
        if frustration_score > 7:
            return "I can sense you're really struggling with this, and that's completely understandable. Let's break this down step by step and work through it together."
        elif frustration_score > 4:
            concept_text = " and ".join(concepts) if concepts else "programming"
            return f"I understand that {concept_text} can be challenging. Let me help you work through this systematically."
        else:
            return "Great question! Let me help you understand this concept better."

# Usage in main API file
def integrate_ml_system(app: FastAPI, db_path: str, existing_analyzer, ai_generator=None):
    """Integrate ML learning system with existing API"""
    ml_api = MLTutoringAPI(app, db_path, existing_analyzer, ai_generator)
    print("✅ ML learning system integrated with API")
    return ml_api
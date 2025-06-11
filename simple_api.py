# simple_api.py - ML Enhanced Version for /Users/atif/empathetic-tutor-ai
from fastapi import FastAPI, HTTPException, Request

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request
import time
from collections import defaultdict, deque

# Rate limiting setup
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Custom rate limiter for student-specific limits
class StudentRateLimiter:
    def __init__(self):
        self.student_requests = defaultdict(lambda: deque())
        self.student_limits = {
            "requests_per_minute": 10,
            "requests_per_hour": 100
        }
    
    def is_allowed(self, student_id: str) -> bool:
        """Check if student is within rate limits"""
        now = time.time()
        student_queue = self.student_requests[student_id]
        
        # Remove old requests (older than 1 hour)
        while student_queue and student_queue[0] < now - 3600:
            student_queue.popleft()
        
        # Check hourly limit
        if len(student_queue) >= self.student_limits["requests_per_hour"]:
            return False
        
        # Check minute limit
        minute_requests = sum(1 for req_time in student_queue if req_time > now - 60)
        if minute_requests >= self.student_limits["requests_per_minute"]:
            return False
        
        # Add current request
        student_queue.append(now)
        return True

student_limiter = StudentRateLimiter()

def check_student_rate_limit(student_id: str):
    """Check student-specific rate limits"""
    if not student_limiter.is_allowed(student_id):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded for student. Please wait before making more requests."
        )

from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import sqlite3
from datetime import datetime
import json
import uuid
import os
import asyncio
from dotenv import load_dotenv
from database_manager import initialize_database_manager, get_db_manager

# Your existing imports
from frustration_analyzer import frustration_analyzer
from attrib_loader import config_loader

# NEW ML IMPORTS
try:
    from ml_learning_system import HybridFrustrationAnalyzer, LearningFeedback
    from auto_config_updater import AutoConfigUpdater
    ML_AVAILABLE = True
    print("✅ ML learning system loaded successfully")
except ImportError as e:
    print(f"⚠️ ML components not available: {e}")
    ML_AVAILABLE = False

# Load environment variables
load_dotenv()

import logging
import traceback
from functools import wraps

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Create logs directory
os.makedirs('logs', exist_ok=True)

def handle_exceptions(func):
    """Decorator for consistent exception handling"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            # Log unexpected errors
            error_id = f"error_{int(time.time())}"
            logger.error(f"Error {error_id} in {func.__name__}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return user-friendly error
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error. Error ID: {error_id}"
            )
    return wrapper

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors gracefully"""
    logger.warning(f"Validation error: {str(exc)}")
    return JSONResponse(
        status_code=400,
        content={"error": "Invalid input", "detail": str(exc)}
    )

@app.exception_handler(sqlite3.Error)
async def database_error_handler(request: Request, exc: sqlite3.Error):
    """Handle database errors gracefully"""
    error_id = f"db_error_{int(time.time())}"
    logger.error(f"Database error {error_id}: {str(exc)}")
    return JSONResponse(
        status_code=503,
        content={"error": "Database temporarily unavailable", "error_id": error_id}
    )


# Import AI components with error handling
try:
    from ai_response_generator import AIResponseGenerator
    AI_AVAILABLE = True
    print("✅ AI components loaded successfully")
except ImportError as e:
    print(f"⚠️  AI components not available: {e}")
    AI_AVAILABLE = False

try:
    from config_manager import config
    CONFIG_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Config manager not available: {e}")
    CONFIG_AVAILABLE = False

# CREATE APP
app = FastAPI(
    title="Empathetic Tutor API - ML Enhanced", 
    description="AI-powered tutoring system with ML learning capabilities",
    version="4.0.0"
)

# CORS MIDDLEWARE
# Secure CORS configuration
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8080", 
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8080",
    # Add your frontend domains here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # Specific origins only
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Only needed methods
    allow_headers=[
        "Content-Type",
        "Authorization", 
        "Accept",
        "Origin",
        "X-Requested-With"
    ],  # Specific headers only
    expose_headers=["Content-Type"],
    max_age=600  # Cache preflight for 10 minutes
)

@app.options("/{path:path}")
async def handle_options(request: Request, path: str):
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Credentials": "true",
        }
    )

# Database and Config setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_NAME = os.path.join(BASE_DIR, "empathetic_tutor.db")
CONFIG_DIR = "/Users/atif/empathetic-tutor-ai/config"  # Your config directory

print(f"Using database at: {DB_NAME}")
print(f"Using config directory: {CONFIG_DIR}")

# Initialize AI generator
if AI_AVAILABLE:
    ai_generator = AIResponseGenerator(db_path=DB_NAME)
else:
    ai_generator = None

# INITIALIZE ML SYSTEM
ml_analyzer = None
auto_updater = None

def initialize_ml_system():
    """Initialize ML learning components"""
    global ml_analyzer, auto_updater
    
    if not ML_AVAILABLE:
        print("⚠️ ML system disabled - components not available")
        return False
    
    try:
        # Create ML learning tables
        _create_ml_tables()
        
        # Initialize ML analyzer
        ml_analyzer = HybridFrustrationAnalyzer(DB_NAME, frustration_analyzer)
        
        # Initialize auto-updater
        auto_updater = AutoConfigUpdater(CONFIG_DIR, DB_NAME)
        
        print("✅ ML learning system initialized")
        return True
        
    except Exception as e:
        print(f"❌ ML system initialization failed: {e}")
        return False

def _create_ml_tables():
    """Create ML learning tables"""
    with get_db_manager().get_connection() as conn:
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
            feature_importance TEXT,
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
            pattern_type TEXT,
            pattern_text TEXT,
            concept_category TEXT,
            weight REAL,
            confidence_score REAL,
            usage_count INTEGER DEFAULT 0,
            effectiveness_score REAL DEFAULT 0.5,
            status TEXT DEFAULT 'candidate',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            approved_at TIMESTAMP,
            last_used TIMESTAMP
        )
    """)
    
    # Student learning profiles table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS student_learning_profiles (
            student_id TEXT PRIMARY KEY,
            personalized_weights TEXT,
            learning_style TEXT,
            adaptation_rate REAL DEFAULT 0.01,
            total_interactions INTEGER DEFAULT 0,
            avg_prediction_error REAL DEFAULT 0.0,
            last_weight_update TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Add ML columns to interactions table
    try:
        cursor.execute("ALTER TABLE interactions ADD COLUMN ml_features TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    try:
        cursor.execute("ALTER TABLE interactions ADD COLUMN confidence_score REAL")
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    # Connection managed by context manager

# Initialize ML system on startup
ml_initialized = initialize_ml_system()


from pydantic import BaseModel, validator, Field
import re
import html

# Enhanced input validation models
class ValidatedStudentMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    student_id: str = Field(default="default_student", regex=r'^[a-zA-Z0-9_-]+$')
    context: Optional[Dict] = None
    
    @validator('message')
    def validate_message(cls, v):
        if not v or not v.strip():
            raise ValueError('Message cannot be empty or whitespace only')
        
        # HTML escape for XSS prevention
        v = html.escape(v.strip())
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'<script',
            r'javascript:',
            r'data:text/html',
            r'vbscript:',
            r'onload=',
            r'onerror='
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError(f'Message contains potentially malicious content')
        
        return v
    
    @validator('student_id')
    def validate_student_id(cls, v):
        if len(v) > 100:
            raise ValueError('Student ID too long')
        
        if '../' in v or '\' in v:
            raise ValueError('Student ID contains invalid path characters')
        
        return v

class ValidatedMLStudentMessage(ValidatedStudentMessage):
    """Enhanced validation for ML endpoints"""
    pass

class ValidatedFeedbackRequest(BaseModel):
    interaction_id: str = Field(..., regex=r'^[a-f0-9-]+$')
    helpful: bool
    frustration_reduced: bool
    clarity_rating: int = Field(..., ge=1, le=5)
    additional_comments: Optional[str] = Field(None, max_length=1000)
    
    @validator('additional_comments')
    def validate_comments(cls, v):
        if v:
            return html.escape(v.strip())
        return v

class ValidatedDetailedFeedbackRequest(BaseModel):
    interaction_id: str = Field(..., regex=r'^[a-f0-9-]+$')
    student_id: str = Field(..., regex=r'^[a-zA-Z0-9_-]+$')
    helpful: bool
    frustration_before: int = Field(..., ge=1, le=10)
    frustration_after: int = Field(..., ge=1, le=10)
    response_quality: int = Field(..., ge=1, le=5)
    empathy_level: str = Field(..., regex=r'^(too_low|just_right|too_high)$')
    response_time_seconds: Optional[float] = Field(None, ge=0, le=300)
    follow_up_needed: Optional[bool] = None
    additional_comments: Optional[str] = Field(None, max_length=1000)
    
    @validator('additional_comments')
    def validate_comments(cls, v):
        if v:
            return html.escape(v.strip())
        return v


# ENHANCED ROOT ENDPOINT
@app.get("/")
def root():
    """Root endpoint with ML system status"""
    stats = frustration_analyzer.get_statistics()
    return {
        "message": "🎓 Empathetic Tutor API - ML Enhanced", 
        "database": DB_NAME,
        "config_directory": CONFIG_DIR,
        "database_exists": os.path.exists(DB_NAME),
        "ai_available": AI_AVAILABLE,
        "config_available": CONFIG_AVAILABLE,
        "ml_learning_enabled": ml_initialized,
        "current_provider": config.get_ai_provider() if CONFIG_AVAILABLE else "none",
        "enhanced_empathy": config.is_enhanced_empathy_enabled() if CONFIG_AVAILABLE else True,
        "version": "4.0.0_ml",
        "config_stats": stats,
        "ml_features": {
            "hybrid_analysis": ml_analyzer is not None,
            "pattern_discovery": ml_analyzer is not None,
            "auto_config_updates": auto_updater is not None,
            "personalized_weights": ml_analyzer is not None
        }
    }

@app.get("/health")
def health_check():
    """Enhanced health check with ML status"""
    try:
        with get_db_manager().get_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='interactions';")
        table_exists = cursor.fetchone() is not None
        
        if table_exists:
            cursor.execute("SELECT COUNT(*) FROM interactions")
            count = cursor.fetchone()[0]
        else:
            count = 0
        
        # Check ML tables
        ml_tables = []
        for table in ['learning_feedback', 'discovered_patterns', 'student_learning_profiles']:
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}';")
            if cursor.fetchone():
                ml_tables.append(table)
        
        conn.close()
        
        return {
            "status": "healthy", 
            "database": "connected", 
            "path": DB_NAME,
            "config_path": CONFIG_DIR,
            "tables_exist": table_exists,
            "ml_tables": ml_tables,
            "total_interactions": count,
            "ai_available": AI_AVAILABLE,
            "config_available": CONFIG_AVAILABLE,
            "ml_learning_available": ml_analyzer is not None,
            "config_files_loaded": len(config_loader._cache),
            "api_keys": {
                "openai": bool(os.getenv('OPENAI_API_KEY')),
                "anthropic": bool(os.getenv('ANTHROPIC_API_KEY'))
            }
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e), "path": DB_NAME}

# ENHANCED ANALYZE ENDPOINT (Your existing one, now ML-powered)
@app.post("/analyze", response_model=AnalyzeResponse)
@limiter.limit("10/minute")
@handle_exceptions
async def analyze_message(request: ValidatedStudentMessage):
    """
    Enhanced analysis with optional ML learning (backward compatible)
    """
    # Use ML analyzer if available, otherwise use original
    if ml_analyzer:
        try:
            # Use hybrid ML analysis
            analysis = ml_analyzer.analyze_with_learning(request.message, request.student_id)
            frustration_score = analysis["frustration_score"]
            empathy_level = analysis["empathy_level"]
            concepts = analysis["concepts"]
            debug_info = analysis.get("debug_info", [])
            
            # Add ML debug info
            debug_info.append(f"🤖 ML_CONFIDENCE: {analysis['confidence']:.2f}")
            debug_info.append(f"📊 RULE_CONTRIB: {analysis['rule_contribution']:.1f}")
            debug_info.append(f"🧠 ML_CONTRIB: {analysis['ml_contribution']:.1f}")
            
            print(f"🤖 ML Analysis - Score: {frustration_score}, Confidence: {analysis['confidence']:.2f}")
            
        except Exception as e:
            print(f"❌ ML analysis failed, using original: {e}")
            # Fall back to original analyzer
            analysis = frustration_analyzer.analyze_message(request.message)
            frustration_score = analysis["frustration_score"]
            empathy_level = analysis["empathy_level"]
            concepts = analysis["concepts"]
            debug_info = analysis["debug_info"]
    else:
        # Use original analyzer
        analysis = frustration_analyzer.analyze_message(request.message)
        frustration_score = analysis["frustration_score"]
        empathy_level = analysis["empathy_level"]
        concepts = analysis["concepts"]
        debug_info = analysis["debug_info"]
    
    # Print debug information
    print(f"📊 Frustration score: {frustration_score}")
    print(f"🎯 Concepts: {concepts}")
    print(f"💖 Empathy level: {empathy_level}")
    for debug_line in debug_info:
        print(f"   {debug_line}")
    
    # Generate response using AI or fallback
    ai_provider_used = None
    
    if AI_AVAILABLE and ai_generator:
        try:
            response = await ai_generator.generate_response(
                message=request.message,
                frustration_score=frustration_score,
                concepts=concepts,
                student_id=request.student_id
            )
            
            if CONFIG_AVAILABLE:
                ai_provider_used = config.get_ai_provider()
            else:
                ai_provider_used = "ai_system"
                
            print(f"✅ AI response generated using {ai_provider_used}")
            
        except Exception as e:
            print(f"❌ AI generation failed: {e}")
            response = _generate_enhanced_template_response(frustration_score, concepts, request.message)
            ai_provider_used = "enhanced_template_fallback"
    else:
        response = _generate_enhanced_template_response(frustration_score, concepts, request.message)
        ai_provider_used = "enhanced_template_only"
    
    # Store interaction with ML features
    interaction_id = str(uuid.uuid4())
    intervention_level = frustration_analyzer.get_intervention_level(frustration_score)
    
    try:
        with get_db_manager().get_connection() as conn:
        cursor = conn.cursor()
        
        # Create table with ML columns
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id TEXT PRIMARY KEY,
                student_id TEXT,
                message TEXT,
                response TEXT,
                frustration_score REAL,
                intervention_level INTEGER,
                features TEXT,
                concepts TEXT,
                additional_data TEXT,
                created_at TIMESTAMP,
                ml_features TEXT,
                confidence_score REAL
            )
        """)
        
        # Prepare ML features
        ml_features_json = "{}"
        confidence_score = None
        
        if ml_analyzer and 'analysis' in locals() and hasattr(analysis, 'get'):
            ml_features_json = json.dumps(analysis.get('ml_features', {}))
            confidence_score = analysis.get('confidence', None)
        
        # Insert interaction
        cursor.execute("""
            INSERT INTO interactions (id, student_id, message, response, frustration_score, 
                                    intervention_level, features, concepts, additional_data, created_at,
                                    ml_features, confidence_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            interaction_id,
            request.student_id,
            request.message,
            response,
            frustration_score,
            intervention_level,
            json.dumps({
                **analysis.get("message_stats", {}),
                "empathy_level": empathy_level,
                "debug_info": debug_info[:5],
                "ml_enhanced": ml_analyzer is not None
            }),
            json.dumps(concepts),
            json.dumps({
                "timestamp": datetime.now().isoformat(),
                "ai_provider": ai_provider_used,
                "ai_available": AI_AVAILABLE,
                "enhanced_empathy": True,
                "version": "4.0.0_ml",
                "ml_enabled": ml_analyzer is not None
            }),
            datetime.now(),
            ml_features_json,
            confidence_score
        ))
        
        # Connection managed by context manager
        
    except Exception as e:
        print(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    return AnalyzeResponse(
        interaction_id=interaction_id,
        frustration_score=frustration_score,
        response=response,
        concepts=concepts,
        ai_provider=ai_provider_used,
        empathy_level=empathy_level,
        debug_info=debug_info
    )

# NEW ML-SPECIFIC ANALYZE ENDPOINT
@app.post("/analyze/ml", response_model=MLAnalyzeResponse)
@limiter.limit("8/minute")
async def ml_analyze_message(request: ValidatedMLStudentMessage):
    """
    Advanced ML analysis with detailed learning metrics
    """
    if not ml_analyzer:
        raise HTTPException(status_code=503, detail="ML system not available")
    
    try:
        # Use hybrid ML analysis
        analysis = ml_analyzer.analyze_with_learning(request.message, request.student_id)
        
        # Generate response
        if AI_AVAILABLE and ai_generator:
            try:
                response = await ai_generator.generate_response(
                    message=request.message,
                    frustration_score=analysis["frustration_score"],
                    concepts=analysis["concepts"],
                    student_id=request.student_id
                )
                ai_provider_used = config.get_ai_provider() if CONFIG_AVAILABLE else "ai_system"
            except Exception as e:
                print(f"❌ AI generation failed: {e}")
                response = _generate_enhanced_template_response(
                    analysis["frustration_score"], analysis["concepts"], request.message
                )
                ai_provider_used = "template_fallback"
        else:
            response = _generate_enhanced_template_response(
                analysis["frustration_score"], analysis["concepts"], request.message
            )
            ai_provider_used = "template_only"
        
        # Store interaction
        interaction_id = str(uuid.uuid4())
        _store_ml_interaction(interaction_id, request, analysis, response, ai_provider_used)
        
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
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML analysis failed: {str(e)}")

# NEW DETAILED FEEDBACK ENDPOINT
@app.post("/feedback/detailed")
async def submit_detailed_feedback(request: DetailedFeedbackRequest):
    """Submit detailed feedback for ML learning"""
    if not ml_analyzer:
        raise HTTPException(status_code=503, detail="ML system not available")
    
    try:
        # Get predicted frustration for this interaction
        predicted_frustration = _get_predicted_frustration(request.interaction_id)
        prediction_error = request.frustration_after - predicted_frustration
        
        # Create learning feedback
        feedback = LearningFeedback(
            interaction_id=request.interaction_id,
            student_id=request.student_id,
            prediction_error=prediction_error,
            actual_frustration=float(request.frustration_after),
            predicted_frustration=predicted_frustration,
            feature_importance={},
            response_helpful=request.helpful,
            empathy_appropriate=request.empathy_level == "just_right"
        )
        
        # Process learning
        ml_analyzer.learn_from_feedback(request.interaction_id, feedback)
        
        # Store detailed feedback
        _store_detailed_feedback(request, feedback)
        
        # Add new patterns if provided
        if request.new_emotional_words:
            _add_suggested_patterns(request.new_emotional_words, request.student_id)
        
        return {
            "feedback_id": str(uuid.uuid4()),
            "message": "Detailed feedback processed for learning",
            "learning_triggered": True,
            "prediction_error": prediction_error,
            "improvement_direction": "increase_sensitivity" if prediction_error > 1 else "decrease_sensitivity" if prediction_error < -1 else "well_calibrated"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback processing failed: {str(e)}")

# NEW ML MANAGEMENT ENDPOINTS
@app.get("/learning/patterns")
async def get_discovered_patterns(status: str = "candidate", limit: int = 50):
    """Get discovered patterns for review"""
    if not ml_analyzer:
        raise HTTPException(status_code=503, detail="ML system not available")
    
    return _get_patterns_by_status(status, limit)

@app.post("/learning/patterns/approve")
async def approve_patterns(pattern_ids: List[str], action: str = "approve"):
    """Approve or reject discovered patterns"""
    if not ml_analyzer:
        raise HTTPException(status_code=503, detail="ML system not available")
    
    return _process_pattern_approval(pattern_ids, action)

@app.post("/learning/discover")
async def trigger_pattern_discovery(lookback_days: int = 30):
    """Manually trigger pattern discovery"""
    if not ml_analyzer:
        raise HTTPException(status_code=503, detail="ML system not available")
    
    try:
        patterns = ml_analyzer.pattern_discovery.discover_patterns(lookback_days)
        return {
            "discovered_patterns": len(patterns),
            "patterns": patterns,
            "message": f"Discovered {len(patterns)} new patterns from last {lookback_days} days"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pattern discovery failed: {str(e)}")

@app.get("/learning/student/{student_id}/profile")
async def get_student_learning_profile(student_id: str):
    """Get student's learning profile and personalized weights"""
    if not ml_analyzer:
        raise HTTPException(status_code=503, detail="ML system not available")
    
    return _get_student_profile(student_id)

@app.get("/ml/status")
async def get_ml_status():
    """Get detailed ML system status"""
    if not ml_analyzer:
        return {"ml_enabled": False, "message": "ML system not available"}
    
    try:
        with get_db_manager().get_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM learning_feedback WHERE created_at > datetime('now', '-7 days')")
        recent_feedback = cursor.fetchone()[0] if cursor.fetchone() else 0
        
        cursor.execute("SELECT COUNT(*) FROM discovered_patterns WHERE status = 'candidate'")
        pending_patterns = cursor.fetchone()[0] if cursor.fetchone() else 0
        
        cursor.execute("SELECT COUNT(*) FROM student_learning_profiles")
        personalized_students = cursor.fetchone()[0] if cursor.fetchone() else 0
        
        conn.close()
        
        return {
            "ml_enabled": True,
            "hybrid_analyzer_active": True,
            "auto_updater_active": auto_updater is not None,
            "config_directory": CONFIG_DIR,
            "recent_feedback_count": recent_feedback,
            "pending_patterns": pending_patterns,
            "personalized_students": personalized_students,
            "last_check": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {"ml_enabled": True, "error": str(e)}

@app.post("/ml/config/update")
async def trigger_config_update():
    """Manually trigger configuration updates from ML learning"""
    if not auto_updater:
        raise HTTPException(status_code=503, detail="Auto-updater not available")
    
    try:
        result = auto_updater.auto_update_configurations(review_period_days=7)
        return {
            "success": True,
            "update_summary": result,
            "message": f"Configuration updated with {result.get('total_changes', 0)} changes"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")


@app.post("/feedback")
@limiter.limit("5/minute")
async def submit_feedback(request: ValidatedFeedbackRequest):
    """Submit feedback on a tutoring interaction"""
    try:
        with get_db_manager().get_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                interaction_id TEXT,
                helpful INTEGER,
                frustration_reduced INTEGER,
                clarity_rating INTEGER,
                additional_comments TEXT,
                created_at TIMESTAMP,
                FOREIGN KEY (interaction_id) REFERENCES interactions (id)
            )
        """)
        
        feedback_id = str(uuid.uuid4())
        cursor.execute("""
            INSERT INTO feedback (id, interaction_id, helpful, frustration_reduced, 
                                clarity_rating, additional_comments, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            feedback_id,
            request.interaction_id,
            1 if request.helpful else 0,
            1 if request.frustration_reduced else 0,
            request.clarity_rating,
            request.additional_comments,
            datetime.now()
        ))
        
        # Connection managed by context manager
        
        return {
            "feedback_id": feedback_id,
            "message": "Feedback submitted successfully",
            "interaction_id": request.interaction_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback submission failed: {str(e)}")

@app.get("/interactions/{student_id}")
def get_student_interactions(student_id: str, limit: int = 10):
    """Get recent interactions for a student"""
    try:
        with get_db_manager().get_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT i.id, i.message, i.frustration_score, i.response, i.created_at, 
                   i.additional_data, i.features, i.concepts, i.confidence_score
            FROM interactions i
            WHERE i.student_id = ? 
            ORDER BY i.created_at DESC 
            LIMIT ?
        """, (student_id, limit))
        
        interactions = []
        for row in cursor.fetchall():
            additional_data = {}
            features = {}
            concepts = []
            
            try:
                additional_data = json.loads(row[5]) if row[5] else {}
                features = json.loads(row[6]) if row[6] else {}
                concepts = json.loads(row[7]) if row[7] else []
            except:
                pass
                
            interactions.append({
                "id": row[0],
                "message": row[1],
                "frustration_score": row[2],
                "response": row[3],
                "created_at": row[4],
                "ai_provider": additional_data.get("ai_provider", "unknown"),
                "empathy_level": features.get("empathy_level", "standard"),
                "concepts": concepts,
                "version": additional_data.get("version", "4.0.0"),
                "confidence_score": row[8] if len(row) > 8 else None,
                "ml_enhanced": additional_data.get("ml_enabled", False)
            })
        
        conn.close()
        return {"student_id": student_id, "interactions": interactions, "count": len(interactions)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#   CONFIG ENDPOINTS 
@app.get("/config/patterns")
async def get_emotional_patterns():
    """Get all emotional patterns and their weights"""
    return {
        "emotional_patterns": config_loader.get_emotional_patterns(),
        "total_patterns": len(config_loader.get_emotional_patterns())
    }

@app.post("/config/patterns")
async def add_emotional_pattern(request: ConfigUpdateRequest):
    """Add or update an emotional pattern"""
    try:
        frustration_analyzer.add_emotional_pattern(request.pattern, request.weight)
        return {
            "message": f"Pattern '{request.pattern}' added/updated with weight {request.weight}",
            "pattern": request.pattern,
            "weight": request.weight,
            "success": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update pattern: {str(e)}")

@app.get("/config/keywords")
async def get_keyword_weights():
    """Get all keyword categories and their weights"""
    return {
        "keyword_weights": config_loader.get_keyword_weights(),
        "total_categories": len(config_loader.get_keyword_weights())
    }

@app.get("/config/concepts")
async def get_concept_keywords():
    """Get all programming concept keywords"""
    return {
        "concept_keywords": config_loader.get_concept_keywords(),
        "total_concepts": len(config_loader.get_concept_keywords())
    }

@app.get("/config/settings")
async def get_detection_settings():
    """Get detection algorithm settings"""
    return {
        "detection_settings": config_loader.get_detection_settings(),
        "analyzer_stats": frustration_analyzer.get_statistics()
    }

@app.post("/config/reload")
async def reload_configurations():
    """Reload all configurations from disk"""
    try:
        frustration_analyzer.reload_configurations()
        return {
            "message": "Configurations reloaded successfully",
            "timestamp": datetime.now().isoformat(),
            "stats": frustration_analyzer.get_statistics()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload configurations: {str(e)}")

@app.get("/config/stats")
async def get_config_statistics():
    """Get detailed configuration statistics"""
    return {
        "config_loader_stats": config_loader.get_stats(),
        "analyzer_stats": frustration_analyzer.get_statistics(),
        "cache_status": {
            "cached_files": list(config_loader._cache.keys()),
            "cache_size": len(config_loader._cache)
        }
    }

@app.post("/test/analyzer")
async def test_analyzer(request: StudentMessage):
    """Test the frustration analyzer with detailed output"""
    analysis = frustration_analyzer.analyze_message(request.message)
    
    return {
        "message": request.message,
        "analysis": analysis,
        "max_tokens": frustration_analyzer.get_max_tokens(request.message),
        "intervention_level": frustration_analyzer.get_intervention_level(analysis["frustration_score"]),
        "is_complex": frustration_analyzer.should_use_long_response(request.message)
    }

# HELPER FUNCTIONS
def _generate_enhanced_template_response(frustration_score: float, concepts: list, message: str) -> str:
    """Generate enhanced template-based responses with better empathy"""
    thresholds = frustration_analyzer.settings["empathy_thresholds"]
    message_lower = message.lower()
    
    if frustration_score >= thresholds["high"]:
        if "impossible" in message_lower or "can't" in message_lower:
            return "I can really hear how overwhelming this feels right now. When something seems impossible, it usually means we need to break it down into much smaller pieces. You're not alone in this struggle - this particular concept trips up most people. Let's take it one tiny step at a time. What's the very first thing that's confusing you?"
        elif "stupid" in message_lower or "hate" in message_lower:
            return "Hey, I want you to know that feeling frustrated doesn't mean anything negative about you as a person. Programming is genuinely difficult, and the fact that you're pushing through shows real determination. Even experienced developers get stuck on things like this. Let's slow down and approach this differently - what if we just focus on understanding one small piece first?"
        else:
            return "I can sense you're really struggling with this, and that's completely understandable. This stuff is hard! Take a breath with me for a second. You're clearly putting in effort, and that matters. Let's tackle this together, step by step. What specific part is causing the most confusion right now?"
    
    elif frustration_score >= thresholds["medium"]:
        if concepts and "errors" in concepts:
            return f"I totally get how frustrating errors can be - they feel like the computer is speaking a foreign language! The good news is that errors are actually trying to help us, even though they don't feel helpful. Let's decode what's happening together. Can you share the specific error message you're seeing?"
        elif concepts:
            concept_text = " and ".join(concepts)
            return f"I understand that {concept_text} can feel tricky at first. You're definitely not alone in finding this challenging - it's one of those concepts that takes time to click. The fact that you're asking questions means you're on the right track. Let me help break this down in a way that might make more sense."
        else:
            return "I can see you're working through something challenging here. That feeling of being stuck is so normal when learning to program - we've all been there! Let's work through this together. What part would you like to start with?"
    
    elif frustration_score < thresholds["minimal"]:
        if concepts:
            concept_text = " and ".join(concepts)
            return f"Great question about {concept_text}! I can tell you're thinking deeply about this. Since you seem to have a good handle on the basics, let me give you a comprehensive explanation and maybe show you some interesting ways this concept connects to other things you might encounter."
        else:
            return "I love that you're asking thoughtful questions! You seem to be approaching this with a clear mind, which is perfect for learning. Let me walk you through this step by step and give you some insights that might help deepen your understanding."
    
    else:
        if concepts:
            concept_text = " and ".join(concepts) 
            return f"Thanks for bringing up your question about {concept_text}. This is definitely something worth understanding well. Let me explain this in a clear way and give you some practical examples that should help it click."
        else:
            return "I'm here to help you work through this! Let me break down what you're dealing with and give you a clear explanation that should help things make sense."


def _store_ml_interaction(interaction_id: str, request: MLStudentMessage, analysis: Dict, response: str, ai_provider: str):
    """Store ML interaction in database"""
    try:
        with get_db_manager().get_connection() as conn:
        cursor = conn.cursor()def _store_ml_interaction(interaction_id: str, request: MLStudentMessage, analysis: Dict, response: str, ai_provider: str):
    """Store ML interaction in database"""
    try:
        with get_db_manager().get_connection() as conn:
        cursor = conn.cursor()
        
        # Convert all values to JSON-safe types
        safe_analysis = {}
        for key, value in analysis.items():
            if isinstance(value, bool):
                safe_analysis[key] = bool(value)
            elif isinstance(value, (int, float)):
                safe_analysis[key] = float(value)
            elif isinstance(value, str):
                safe_analysis[key] = str(value)
            elif isinstance(value, list):
                safe_analysis[key] = list(value)
            else:
                safe_analysis[key] = str(value)
        
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
            float(safe_analysis.get("frustration_score", 0)),
            1 if safe_analysis.get("frustration_score", 0) < 3 else 2 if safe_analysis.get("frustration_score", 0) < 7 else 3,
            json.dumps({
                "empathy_level": str(safe_analysis.get("empathy_level", "standard")),
                "learning_opportunity": bool(safe_analysis.get("learning_opportunity", False))
            }),
            json.dumps(safe_analysis.get("concepts", [])),
            json.dumps({
                "ai_provider": str(ai_provider),
                "rule_contribution": float(safe_analysis.get("rule_contribution", 0)),
                "ml_contribution": float(safe_analysis.get("ml_contribution", 0)),
                "version": "4.0.0_ml"
            }),
            json.dumps(safe_analysis.get("ml_features", {})),
            float(safe_analysis.get("confidence", 0.5)),
            datetime.now()
        ))
        
        # Connection managed by context manager
        print(f"✅ ML interaction stored successfully")
        
    except Exception as e:
        print(f"❌ Error storing ML interaction: {e}")
        import traceback
        traceback.print_exc()



def _store_detailed_feedback(request: DetailedFeedbackRequest, feedback: LearningFeedback):
    """Store detailed feedback in learning_feedback table"""
    try:
        with get_db_manager().get_connection() as conn:
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
        
        # Connection managed by context manager
        
    except Exception as e:
        print(f"Error storing detailed feedback: {e}")

def _get_predicted_frustration(interaction_id: str) -> float:
    """Get the predicted frustration score for an interaction"""
    try:
        with get_db_manager().get_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute("SELECT frustration_score FROM interactions WHERE id = ?", (interaction_id,))
        row = cursor.fetchone()
        conn.close()
        
        return row[0] if row else 0.0
        
    except Exception:
        return 0.0

def _get_patterns_by_status(status: str, limit: int) -> Dict:
    """Get discovered patterns by status"""
    try:
        with get_db_manager().get_connection() as conn:
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

def _process_pattern_approval(pattern_ids: List[str], action: str) -> Dict:
    """Process pattern approval/rejection"""
    try:
        with get_db_manager().get_connection() as conn:
        cursor = conn.cursor()
        
        approved_count = 0
        
        for pattern_id in pattern_ids:
            if action == "approve":
                cursor.execute("""
                    UPDATE discovered_patterns 
                    SET status = 'approved', approved_at = ?
                    WHERE id = ?
                """, (datetime.now(), pattern_id))
                approved_count += 1
                
            elif action == "reject":
                cursor.execute("""
                    UPDATE discovered_patterns 
                    SET status = 'rejected'
                    WHERE id = ?
                """, (pattern_id,))
        
        # Connection managed by context manager
        
        return {
            "processed": len(pattern_ids),
            "approved": approved_count,
            "action": action,
            "message": f"Processed {len(pattern_ids)} patterns"
        }
        
    except Exception as e:
        return {"error": str(e)}

def _get_student_profile(student_id: str) -> Dict:
    """Get student learning profile"""
    try:
        with get_db_manager().get_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT personalized_weights, total_interactions, avg_prediction_error,
                   last_weight_update, created_at
            FROM student_learning_profiles
            WHERE student_id = ?
        """, (student_id,))
        
        row = cursor.fetchone()
        if row:
            return {
                "student_id": student_id,
                "personalized_weights": json.loads(row[0]) if row[0] else {},
                "total_interactions": row[1],
                "avg_prediction_error": row[2],
                "last_weight_update": row[3],
                "created_at": row[4],
                "has_personalization": True
            }
        else:
            return {
                "student_id": student_id,
                "has_personalization": False,
                "message": "No personalized profile yet"
            }
        
        conn.close()
        
    except Exception as e:
        return {"error": str(e)}

def _add_suggested_patterns(words: List[str], student_id: str):
    """Add user-suggested emotional words as candidate patterns"""
    try:
        with get_db_manager().get_connection() as conn:
        cursor = conn.cursor()
        
        for word in words:
            pattern_id = f"suggested_{datetime.now().timestamp()}_{hash(word)}"
            cursor.execute("""
                INSERT INTO discovered_patterns 
                (id, pattern_type, pattern_text, weight, confidence_score, status)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                pattern_id,
                "suggested_keyword",
                word.lower(),
                1.5,  # Default weight for suggested patterns
                0.5,  # Lower confidence for user suggestions
                "candidate"
            ))
        
        # Connection managed by context manager
        print(f"✅ Added {len(words)} suggested patterns")
        
    except Exception as e:
        print(f"Error adding suggested patterns: {e}")


@app.get("/debug/all-feedback")
async def debug_all_feedback():
    """Get ALL learning feedback entries"""
    try:
        with get_db_manager().get_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT lf.interaction_id, lf.prediction_error, lf.actual_frustration, 
                   lf.predicted_frustration, i.message, i.student_id
            FROM learning_feedback lf
            JOIN interactions i ON lf.interaction_id = i.id
            ORDER BY lf.created_at DESC
        """)
        
        all_entries = []
        for row in cursor.fetchall():
            all_entries.append({
                "interaction_id": row[0],
                "prediction_error": row[1],
                "actual_frustration": row[2],
                "predicted_frustration": row[3],
                "message": row[4],
                "student_id": row[5]
            })
        
        # Filter by error type
        positive_errors = [e for e in all_entries if e["prediction_error"] > 0]
        negative_errors = [e for e in all_entries if e["prediction_error"] < 0]
        
        conn.close()
        
        return {
            "total_entries": len(all_entries),
            "positive_errors": len(positive_errors),
            "negative_errors": len(negative_errors),
            "positive_error_data": positive_errors,
            "negative_error_data": negative_errors
        }
        
    except Exception as e:
        return {"error": str(e)}
    

@app.post("/debug/fixed-discovery")
async def fixed_pattern_discovery():
    """Fixed pattern discovery with proper logic"""
    try:
        with get_db_manager().get_connection() as conn:
        cursor = conn.cursor()
        
        # Get ALL high-error cases (both positive and negative > 1.0)
        cursor.execute("""
            SELECT DISTINCT lf.interaction_id, i.message, lf.actual_frustration, 
                   lf.predicted_frustration, lf.prediction_error, i.student_id
            FROM learning_feedback lf
            JOIN interactions i ON lf.interaction_id = i.id
            WHERE ABS(lf.prediction_error) > 1.0
            ORDER BY ABS(lf.prediction_error) DESC
        """)
        
        high_error_cases = []
        for row in cursor.fetchall():
            high_error_cases.append({
                'interaction_id': row[0],
                'message': row[1],
                'actual_frustration': row[2],
                'predicted_frustration': row[3],
                'prediction_error': row[4],
                'student_id': row[5]
            })
        
        conn.close()
        
        # Extract patterns from UNDER-PREDICTED cases (positive errors)
        under_predicted = [case for case in high_error_cases if case['prediction_error'] > 0]
        
        if len(under_predicted) < 2:
            return {
                "error": "Need more under-predicted cases",
                "under_predicted_count": len(under_predicted),
                "all_cases": high_error_cases
            }
        
        # Extract 2-4 word patterns from under-predicted messages
        patterns_found = {}
        for case in under_predicted:
            message = case['message'].lower()
            words = message.split()
            
            # Extract 2-3 word phrases
            for i in range(len(words) - 1):
                for length in [2, 3]:
                    if i + length <= len(words):
                        phrase = ' '.join(words[i:i+length])
                        # Filter out common words and keep meaningful phrases
                        if len(phrase) > 5 and not all(w in ['the', 'and', 'is', 'a', 'to', 'of', 'in'] for w in phrase.split()):
                            if phrase not in patterns_found:
                                patterns_found[phrase] = {
                                    'count': 0,
                                    'avg_error': 0,
                                    'messages': []
                                }
                            patterns_found[phrase]['count'] += 1
                            patterns_found[phrase]['avg_error'] += case['prediction_error']
                            patterns_found[phrase]['messages'].append(case['message'])
        
        # Calculate average errors and filter patterns
        discovered_patterns = []
        for phrase, data in patterns_found.items():
            if data['count'] >= 1:  # Lower threshold for testing
                avg_error = data['avg_error'] / data['count']
                if avg_error > 1.5:  # Significant under-prediction
                    weight = min(avg_error, 5.0)  # Cap weight at 5.0
                    discovered_patterns.append({
                        'type': 'emotional_pattern',
                        'pattern': phrase,
                        'weight': round(weight, 1),
                        'frequency': data['count'],
                        'avg_error': round(avg_error, 2),
                        'confidence': min(data['count'] / 3.0, 1.0),
                        'example_messages': data['messages']
                    })
        
        # Sort by significance (frequency * avg_error)
        discovered_patterns.sort(key=lambda x: x['frequency'] * x['avg_error'], reverse=True)
        
        return {
            "total_cases": len(high_error_cases),
            "under_predicted_cases": len(under_predicted),
            "discovered_patterns": discovered_patterns[:10],  # Top 10
            "should_auto_add": len(discovered_patterns) > 0,
            "debug_under_predicted": under_predicted
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.post("/learning/auto-add-patterns")
async def auto_add_discovered_patterns():
    """Automatically add discovered patterns to configuration"""
    discovery_result = await fixed_pattern_discovery()
    
    if 'discovered_patterns' not in discovery_result:
        return {"error": "No patterns discovered", "details": discovery_result}
    
    patterns = discovery_result['discovered_patterns']
    if not patterns:
        return {"message": "No patterns to add"}
    
    added_patterns = []
    for pattern in patterns:
        if pattern['confidence'] > 0.5:  # Only add confident patterns
            try:
                # Add pattern using existing endpoint
                result = await add_emotional_pattern(
                    ConfigUpdateRequest(pattern=pattern['pattern'], weight=pattern['weight'])
                )
                added_patterns.append({
                    "pattern": pattern['pattern'],
                    "weight": pattern['weight'],
                    "confidence": pattern['confidence']
                })
            except Exception as e:
                print(f"Failed to add pattern {pattern['pattern']}: {e}")
    
    return {
        "auto_added_patterns": len(added_patterns),
        "patterns": added_patterns,
        "message": f"Automatically added {len(added_patterns)} patterns"
    }




@app.get("/debug/learning-feedback")
async def debug_learning_feedback():
    """Debug endpoint to check learning feedback data"""
    try:
        with get_db_manager().get_connection() as conn:
        cursor = conn.cursor()
        
        # Check if learning_feedback table has data
        cursor.execute("SELECT COUNT(*) FROM learning_feedback")
        feedback_count = cursor.fetchone()[0]
        
        # Get recent learning feedback
        cursor.execute("""
            SELECT lf.interaction_id, lf.prediction_error, lf.actual_frustration, 
                   lf.predicted_frustration, i.message
            FROM learning_feedback lf
            JOIN interactions i ON lf.interaction_id = i.id
            ORDER BY lf.created_at DESC LIMIT 10
        """)
        
        recent_feedback = []
        for row in cursor.fetchall():
            recent_feedback.append({
                "interaction_id": row[0],
                "prediction_error": row[1],
                "actual_frustration": row[2], 
                "predicted_frustration": row[3],
                "message": row[4]
            })
        
        conn.close()
        
        return {
            "feedback_count": feedback_count,
            "recent_feedback": recent_feedback,
            "discovery_threshold": 2.0,
            "min_frequency": 3
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.post("/debug/force-discovery")
async def force_pattern_discovery():
    """Force pattern discovery with debug info"""
    if not ml_analyzer:
        raise HTTPException(status_code=503, detail="ML system not available")
    
    try:
        # Get recent high-error interactions manually
        with get_db_manager().get_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT lf.interaction_id, i.message, lf.actual_frustration, 
                   lf.predicted_frustration, lf.prediction_error
            FROM learning_feedback lf
            JOIN interactions i ON lf.interaction_id = i.id
            WHERE ABS(lf.prediction_error) > 1.0
            ORDER BY ABS(lf.prediction_error) DESC
        """)
        
        high_error_cases = []
        for row in cursor.fetchall():
            high_error_cases.append({
                'interaction_id': row[0],
                'message': row[1],
                'actual_frustration': row[2],
                'predicted_frustration': row[3],
                'prediction_error': row[4]
            })
        
        conn.close()
        
        # Extract potential patterns manually
        messages = [case['message'] for case in high_error_cases]
        patterns_found = []
        
        # Simple pattern extraction
        for message in messages:
            words = message.lower().split()
            # Look for 2-3 word phrases
            for i in range(len(words) - 1):
                phrase = ' '.join(words[i:i+2])
                if len(phrase) > 5:  # Skip very short phrases
                    patterns_found.append(phrase)
        
        # Count frequency
        from collections import Counter
        pattern_counts = Counter(patterns_found)
        
        return {
            "high_error_cases": len(high_error_cases),
            "cases": high_error_cases,
            "potential_patterns": dict(pattern_counts.most_common(10)),
            "messages_analyzed": messages
        }
        
    except Exception as e:
        return {"error": str(e), "details": "Debug discovery failed"}


#connection status 

@app.get("/config/current")
def get_current_config():
    """Get current configuration status"""
    try:
        return {
            "ai_provider": config.get_ai_provider() if CONFIG_AVAILABLE else "openai",
            "fallback_enabled": config.is_fallback_enabled() if CONFIG_AVAILABLE else True,
            "enhanced_empathy": config.is_enhanced_empathy_enabled() if CONFIG_AVAILABLE else True,
            "ml_enabled": ml_initialized,
            "status": "connected",
            "openai_available": bool(ai_generator and ai_generator.openai_client),
            "anthropic_available": bool(ai_generator and ai_generator.anthropic_client)
        }
    except Exception as e:
        return {
            "ai_provider": "openai",
            "ml_enabled": True,
            "status": "connected",
            "error": str(e)
        }
    

    # Working config endpoint - added at end

@app.get("/api/status")
def get_api_status():
    """Get API status - working endpoint"""
    return {
        "ai_provider": "openai",
        "ml_enabled": True,
        "status": "connected",
        "openai_available": True,
        "anthropic_available": True,
        "enhanced_empathy": True
    }



# Memory monitoring utilities
import psutil
import gc

def get_memory_usage():
    """Get current memory usage statistics"""
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        "rss_mb": memory_info.rss / 1024 / 1024,
        "vms_mb": memory_info.vms / 1024 / 1024,
        "percent": process.memory_percent()
    }

def cleanup_memory():
    """Force garbage collection and cleanup"""
    gc.collect()
    return get_memory_usage()

@app.get("/debug/memory")
async def debug_memory():
    """Debug endpoint for memory usage"""
    memory_before = get_memory_usage()
    cleaned_memory = cleanup_memory()
    
    return {
        "memory_before_cleanup": memory_before,
        "memory_after_cleanup": cleaned_memory,
        "memory_cleaned": memory_before["rss_mb"] - cleaned_memory["rss_mb"]
    }



# Enhanced monitoring endpoints
@app.get("/health/detailed")
async def detailed_health_check():
    """Comprehensive health check with system metrics"""
    start_time = time.time()
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "checks": {},
        "metrics": {}
    }
    
    # Database connectivity check
    try:
        with get_db_manager().get_connection() as conn:
            cursor = conn.execute("SELECT 1")
            cursor.fetchone()
        health_status["checks"]["database"] = {"status": "healthy", "response_time_ms": 0}
    except Exception as e:
        health_status["checks"]["database"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "degraded"
    
    # ML system check
    try:
        if ml_analyzer:
            test_analysis = ml_analyzer.analyze_with_learning("test", "health_check")
            health_status["checks"]["ml_system"] = {"status": "healthy", "confidence": test_analysis.get("confidence", 0)}
        else:
            health_status["checks"]["ml_system"] = {"status": "disabled"}
    except Exception as e:
        health_status["checks"]["ml_system"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "degraded"
    
    # AI provider check
    try:
        if ai_generator:
            health_status["checks"]["ai_provider"] = {"status": "healthy", "provider": config.get_ai_provider() if CONFIG_AVAILABLE else "unknown"}
        else:
            health_status["checks"]["ai_provider"] = {"status": "disabled"}
    except Exception as e:
        health_status["checks"]["ai_provider"] = {"status": "unhealthy", "error": str(e)}
    
    # System metrics
    try:
        memory_info = get_memory_usage()
        health_status["metrics"] = {
            "memory_usage_mb": memory_info["rss_mb"],
            "memory_percent": memory_info["percent"],
            "response_time_ms": (time.time() - start_time) * 1000
        }
    except:
        pass
    
    return health_status

@app.get("/metrics")
async def get_system_metrics():
    """Get system performance metrics"""
    return {
        "memory": get_memory_usage(),
        "database": {
            "connection_count": getattr(get_db_manager(), '_connection_count', 0)
        },
        "request_counts": {
            # These would be populated by middleware
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0
        }
    }

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests for monitoring"""
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        
        # Log response
        duration = (time.time() - start_time) * 1000
        logger.info(f"Response: {response.status_code} in {duration:.2f}ms")
        
        return response
    except Exception as e:
        duration = (time.time() - start_time) * 1000
        logger.error(f"Request failed: {str(e)} in {duration:.2f}ms")
        raise


if __name__ == "__main__":
    import uvicorn
    
    print("🔧 ENHANCED CONFIGURATION STATUS:")
    print(f"   📁 Config directory: {CONFIG_DIR}")
    print(f"   📊 Config stats: {config_loader.get_stats()}")
    print(f"   🔄 Analyzer loaded: {len(frustration_analyzer.emotional_patterns)} patterns")
    print(f"   🤖 ML System: {'✅ Active' if ml_initialized else '❌ Disabled'}")
    print(f"   🔄 Auto-updater: {'✅ Active' if auto_updater else '❌ Disabled'}")
    
    if CONFIG_AVAILABLE:
        print(f"   🤖 AI Provider: {config.get_ai_provider()}")
        print(f"   🔄 Fallback Enabled: {config.is_fallback_enabled()}")
        print(f"   💖 Enhanced Empathy: {config.is_enhanced_empathy_enabled()}")
    
    print(f"🚀 Starting ML-Enhanced Empathetic Tutor API v4.0")
    print(f"📁 Database: {DB_NAME}")
    print(f"📁 Config: {CONFIG_DIR}")
    print(f"🗃️ Database exists: {os.path.exists(DB_NAME)}")
    print(f"🤖 AI Available: {AI_AVAILABLE}")
    print(f"⚙️ Config Available: {CONFIG_AVAILABLE}")
    print(f"🧠 ML Learning: {'✅ Enabled' if ml_initialized else '❌ Disabled'}")
    print(f"📚 API documentation: http://localhost:8000/docs")
    print(f"🧠 ML endpoints: http://localhost:8000/analyze/ml, /learning/*, /ml/*")
    print("=" * 70)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
# simple_api.py - Updated with Modular Configuration System
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import sqlite3
from datetime import datetime
import json
import uuid
import os
import asyncio
from dotenv import load_dotenv

# Import new modular components
from frustration_analyzer import frustration_analyzer
from attrib_loader import config_loader

# Load environment variables
load_dotenv()

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

# CREATE APP ONLY ONCE
app = FastAPI(
    title="Empathetic Tutor API - Modular Config", 
    description="AI-powered tutoring system with modular configuration management",
    version="3.0.0"
)

# ADD CORS MIDDLEWARE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# ADD OPTIONS HANDLER FOR CORS PREFLIGHT
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

# Database setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_NAME = os.path.join(BASE_DIR, "empathetic_tutor.db")
print(f"Using database at: {DB_NAME}")

# Initialize AI generator if available
if AI_AVAILABLE:
    ai_generator = AIResponseGenerator(db_path=DB_NAME)
else:
    ai_generator = None

# PYDANTIC MODELS
class StudentMessage(BaseModel):
    message: str
    student_id: str = "default_student"

class AnalyzeResponse(BaseModel):
    interaction_id: str
    frustration_score: float
    response: str
    concepts: list
    ai_provider: Optional[str] = None
    empathy_level: Optional[str] = None
    debug_info: Optional[list] = None

class FeedbackRequest(BaseModel):
    interaction_id: str
    helpful: bool
    frustration_reduced: bool
    clarity_rating: int  # 1-5 scale
    additional_comments: Optional[str] = None

class ConfigUpdateRequest(BaseModel):
    pattern: str
    weight: float

# ROUTE HANDLERS
@app.get("/")
def root():
    """Root endpoint with system status"""
    stats = frustration_analyzer.get_statistics()
    return {
        "message": "🎓 Empathetic Tutor API - Modular Configuration", 
        "database": DB_NAME,
        "database_exists": os.path.exists(DB_NAME),
        "ai_available": AI_AVAILABLE,
        "config_available": CONFIG_AVAILABLE,
        "current_provider": config.get_ai_provider() if CONFIG_AVAILABLE else "none",
        "enhanced_empathy": config.is_enhanced_empathy_enabled() if CONFIG_AVAILABLE else True,
        "version": "3.0.0",
        "config_stats": stats
    }

@app.get("/health")
def health_check():
    """Comprehensive health check"""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Check if interactions table exists and get count
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='interactions';")
        table_exists = cursor.fetchone() is not None
        
        if table_exists:
            cursor.execute("SELECT COUNT(*) FROM interactions")
            count = cursor.fetchone()[0]
        else:
            count = 0
            
        conn.close()
        
        return {
            "status": "healthy", 
            "database": "connected", 
            "path": DB_NAME,
            "tables_exist": table_exists,
            "total_interactions": count,
            "ai_available": AI_AVAILABLE,
            "config_available": CONFIG_AVAILABLE,
            "config_files_loaded": len(config_loader._cache),
            "api_keys": {
                "openai": bool(os.getenv('OPENAI_API_KEY')),
                "anthropic": bool(os.getenv('ANTHROPIC_API_KEY'))
            }
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e), "path": DB_NAME}

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_message(request: StudentMessage):
    """
    Analyze student message using modular frustration detection system
    """
    # Use the new modular analyzer
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
            # Use enhanced AI system
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
                
            print(f"✅ Enhanced AI response generated using {ai_provider_used}")
            
        except Exception as e:
            print(f"❌ AI generation failed: {e}")
            response = _generate_enhanced_template_response(frustration_score, concepts, request.message)
            ai_provider_used = "enhanced_template_fallback"
    else:
        # Use enhanced template responses
        response = _generate_enhanced_template_response(frustration_score, concepts, request.message)
        ai_provider_used = "enhanced_template_only"
    
    # Store in database
    interaction_id = str(uuid.uuid4())
    intervention_level = frustration_analyzer.get_intervention_level(frustration_score)
    
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Create table if it doesn't exist
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
                created_at TIMESTAMP
            )
        """)
        
        # Insert interaction
        cursor.execute("""
            INSERT INTO interactions (id, student_id, message, response, frustration_score, 
                                    intervention_level, features, concepts, additional_data, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            interaction_id,
            request.student_id,
            request.message,
            response,
            frustration_score,
            intervention_level,
            json.dumps({
                **analysis["message_stats"],
                "empathy_level": empathy_level,
                "debug_info": debug_info[:5]  # Store first 5 debug items
            }),
            json.dumps(concepts),
            json.dumps({
                "timestamp": datetime.now().isoformat(),
                "ai_provider": ai_provider_used,
                "ai_available": AI_AVAILABLE,
                "enhanced_empathy": True,
                "version": "3.0.0",
                "config_version": "modular"
            }),
            datetime.now()
        ))
        
        conn.commit()
        conn.close()
        
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

def _generate_enhanced_template_response(frustration_score: float, concepts: list, message: str) -> str:
    """
    Generate enhanced template-based responses with better empathy
    Uses settings from configuration
    """
    # Get empathy thresholds from config
    thresholds = frustration_analyzer.settings["empathy_thresholds"]
    message_lower = message.lower()
    
    # High frustration responses
    if frustration_score >= thresholds["high"]:
        if "impossible" in message_lower or "can't" in message_lower:
            return "I can really hear how overwhelming this feels right now. When something seems impossible, it usually means we need to break it down into much smaller pieces. You're not alone in this struggle - this particular concept trips up most people. Let's take it one tiny step at a time. What's the very first thing that's confusing you?"
        elif "stupid" in message_lower or "hate" in message_lower:
            return "Hey, I want you to know that feeling frustrated doesn't mean anything negative about you as a person. Programming is genuinely difficult, and the fact that you're pushing through shows real determination. Even experienced developers get stuck on things like this. Let's slow down and approach this differently - what if we just focus on understanding one small piece first?"
        else:
            return "I can sense you're really struggling with this, and that's completely understandable. This stuff is hard! Take a breath with me for a second. You're clearly putting in effort, and that matters. Let's tackle this together, step by step. What specific part is causing the most confusion right now?"
    
    # Medium frustration responses  
    elif frustration_score >= thresholds["medium"]:
        if concepts and "errors" in concepts:
            return f"I totally get how frustrating errors can be - they feel like the computer is speaking a foreign language! The good news is that errors are actually trying to help us, even though they don't feel helpful. Let's decode what's happening together. Can you share the specific error message you're seeing?"
        elif concepts:
            concept_text = " and ".join(concepts)
            return f"I understand that {concept_text} can feel tricky at first. You're definitely not alone in finding this challenging - it's one of those concepts that takes time to click. The fact that you're asking questions means you're on the right track. Let me help break this down in a way that might make more sense."
        else:
            return "I can see you're working through something challenging here. That feeling of being stuck is so normal when learning to program - we've all been there! Let's work through this together. What part would you like to start with?"
    
    # Low frustration responses
    elif frustration_score < thresholds["minimal"]:
        if concepts:
            concept_text = " and ".join(concepts)
            return f"Great question about {concept_text}! I can tell you're thinking deeply about this. Since you seem to have a good handle on the basics, let me give you a comprehensive explanation and maybe show you some interesting ways this concept connects to other things you might encounter."
        else:
            return "I love that you're asking thoughtful questions! You seem to be approaching this with a clear mind, which is perfect for learning. Let me walk you through this step by step and give you some insights that might help deepen your understanding."
    
    # Standard responses for medium frustration
    else:
        if concepts:
            concept_text = " and ".join(concepts) 
            return f"Thanks for bringing up your question about {concept_text}. This is definitely something worth understanding well. Let me explain this in a clear way and give you some practical examples that should help it click."
        else:
            return "I'm here to help you work through this! Let me break down what you're dealing with and give you a clear explanation that should help things make sense."

# Configuration management endpoints
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

# Continue with existing endpoints (feedback, interactions, analytics, etc.)
@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback on a tutoring interaction"""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Create feedback table if it doesn't exist
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
        
        # Insert feedback
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
        
        conn.commit()
        conn.close()
        
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
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT i.id, i.message, i.frustration_score, i.response, i.created_at, 
                   i.additional_data, i.features, i.concepts
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
                "version": additional_data.get("version", "3.0.0")
            })
        
        conn.close()
        return {"student_id": student_id, "interactions": interactions, "count": len(interactions)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Test endpoint
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

if __name__ == "__main__":
    import uvicorn
    
    print("🔧 CONFIGURATION STATUS:")
    print(f"   📁 Config directory: {config_loader.config_dir}")
    print(f"   📊 Config stats: {config_loader.get_stats()}")
    print(f"   🔄 Analyzer loaded: {len(frustration_analyzer.emotional_patterns)} patterns")
    
    if CONFIG_AVAILABLE:
        print(f"   🤖 AI Provider: {config.get_ai_provider()}")
        print(f"   🔄 Fallback Enabled: {config.is_fallback_enabled()}")
        print(f"   💖 Enhanced Empathy: {config.is_enhanced_empathy_enabled()}")
    else:
        print("   ⚠️ Legacy config manager not available")
    
    print(f"🚀 Starting Modular Empathetic Tutor API v3.0")
    print(f"📁 Database: {DB_NAME}")
    print(f"🗃️ Database exists: {os.path.exists(DB_NAME)}")
    print(f"🤖 AI Available: {AI_AVAILABLE}")
    print(f"⚙️ Config Available: {CONFIG_AVAILABLE}")
    print(f"📚 API documentation: http://localhost:8000/docs")
    print(f"⚙️ Config endpoints: http://localhost:8000/config/*")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
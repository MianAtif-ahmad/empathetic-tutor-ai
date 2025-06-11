# simple_api.py - Final Version with Enhanced Empathy
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import sqlite3
from datetime import datetime
import json
import uuid
import os
import asyncio
from dotenv import load_dotenv

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

app = FastAPI(
    title="Empathetic Tutor API - Enhanced", 
    description="AI-powered tutoring system with advanced empathy and personalization",
    version="2.0.0"
)

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_NAME = os.path.join(BASE_DIR, "empathetic_tutor.db")

print(f"Using database at: {DB_NAME}")

# Initialize AI generator if available
if AI_AVAILABLE:
    ai_generator = AIResponseGenerator(db_path=DB_NAME)
else:
    ai_generator = None

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

class FeedbackRequest(BaseModel):
    interaction_id: str
    helpful: bool
    frustration_reduced: bool
    clarity_rating: int  # 1-5 scale
    additional_comments: Optional[str] = None

@app.get("/")
def root():
    """Root endpoint with system status"""
    return {
        "message": "🎓 Empathetic Tutor API - Enhanced Version", 
        "database": DB_NAME,
        "database_exists": os.path.exists(DB_NAME),
        "ai_available": AI_AVAILABLE,
        "config_available": CONFIG_AVAILABLE,
        "current_provider": config.get_ai_provider() if CONFIG_AVAILABLE else "none",
        "enhanced_empathy": config.is_enhanced_empathy_enabled() if CONFIG_AVAILABLE else False,
        "version": "2.0.0"
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
    Analyze student message with enhanced empathy and generate response
    """
    # Enhanced frustration detection
    frustration_keywords = [
        # High frustration (weight 3.0)
        "impossible", "hate", "stupid", "giving up", "quit", "worst", "terrible",
        # Medium frustration (weight 2.0) 
        "stuck", "frustrated", "angry", "annoying", "difficult", "hard",
        # Mild frustration (weight 1.0)
        "confused", "don't understand", "lost", "help", "struggling", "unclear"
    ]
    
    error_keywords = ["error", "exception", "crash", "bug", "broken", "failed", "traceback"]
    
    message_lower = request.message.lower()
    
    # Calculate weighted frustration score
    frustration_score = 0.0
    
    # Count high-impact keywords
    high_impact = ["impossible", "hate", "stupid", "giving up", "quit"]
    frustration_score += sum(3.0 for keyword in high_impact if keyword in message_lower)
    
    # Count medium-impact keywords
    medium_impact = ["stuck", "frustrated", "angry", "annoying", "difficult"]
    frustration_score += sum(2.0 for keyword in medium_impact if keyword in message_lower)
    
    # Count mild frustration keywords
    mild_impact = ["confused", "don't understand", "lost", "help", "struggling"]
    frustration_score += sum(1.0 for keyword in mild_impact if keyword in message_lower)
    
    # Add error-related frustration
    frustration_score += sum(1.5 for keyword in error_keywords if keyword in message_lower)
    
    # Add punctuation-based frustration indicators
    exclamation_count = message_lower.count('!')
    question_count = message_lower.count('?')
    caps_ratio = sum(1 for c in request.message if c.isupper()) / max(len(request.message), 1)
    
    frustration_score += min(exclamation_count * 0.5, 2.0)  # Max 2 points from exclamations
    frustration_score += min(question_count * 0.3, 1.0)     # Max 1 point from questions
    frustration_score += caps_ratio * 3.0                   # Up to 3 points from caps
    
    # Normalize to 0-10 scale
    frustration_score = min(frustration_score, 10.0)
    
    # Enhanced concept detection
    concepts = []
    concept_keywords = {
        "loops": ["for", "while", "loop", "iterate", "iteration", "range"],
        "functions": ["function", "def", "return", "parameter", "argument", "call"],
        "variables": ["variable", "var", "assignment", "value", "store"],
        "conditionals": ["if", "else", "elif", "condition", "boolean", "true", "false"],
        "recursion": ["recursion", "recursive", "base case", "call itself"],
        "errors": ["error", "exception", "traceback", "bug", "debug", "syntax"],
        "classes": ["class", "object", "method", "self", "init", "instance"],
        "lists": ["list", "array", "index", "append", "element"],
        "dictionaries": ["dict", "dictionary", "key", "value", "hash"],
        "strings": ["string", "str", "text", "character", "concatenate"]
    }
    
    for concept, keywords in concept_keywords.items():
        if any(keyword in message_lower for keyword in keywords):
            concepts.append(concept)
    
    # Determine empathy level needed
    empathy_level = "standard"
    if frustration_score > 7:
        empathy_level = "high"
    elif frustration_score > 4:
        empathy_level = "medium"
    elif frustration_score < 2:
        empathy_level = "minimal"
    
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
            1 if frustration_score < 3 else 2 if frustration_score < 7 else 3,
            json.dumps({
                "exclamation_count": exclamation_count,
                "caps_ratio": caps_ratio,
                "message_length": len(request.message),
                "empathy_level": empathy_level
            }),
            json.dumps(concepts),
            json.dumps({
                "timestamp": datetime.now().isoformat(),
                "ai_provider": ai_provider_used,
                "ai_available": AI_AVAILABLE,
                "enhanced_empathy": True,
                "version": "2.0.0"
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
        empathy_level=empathy_level
    )

def _generate_enhanced_template_response(frustration_score: float, concepts: list, message: str) -> str:
    """
    Generate enhanced template-based responses with better empathy
    """
    # Detect emotional indicators for better response
    message_lower = message.lower()
    
    # High frustration responses
    if frustration_score > 7:
        if "impossible" in message_lower or "can't" in message_lower:
            return "I can really hear how overwhelming this feels right now. When something seems impossible, it usually means we need to break it down into much smaller pieces. You're not alone in this struggle - this particular concept trips up most people. Let's take it one tiny step at a time. What's the very first thing that's confusing you?"
        elif "stupid" in message_lower or "hate" in message_lower:
            return "Hey, I want you to know that feeling frustrated doesn't mean anything negative about you as a person. Programming is genuinely difficult, and the fact that you're pushing through shows real determination. Even experienced developers get stuck on things like this. Let's slow down and approach this differently - what if we just focus on understanding one small piece first?"
        else:
            return "I can sense you're really struggling with this, and that's completely understandable. This stuff is hard! Take a breath with me for a second. You're clearly putting in effort, and that matters. Let's tackle this together, step by step. What specific part is causing the most confusion right now?"
    
    # Medium frustration responses  
    elif frustration_score > 4:
        if concepts and "errors" in concepts:
            return f"I totally get how frustrating errors can be - they feel like the computer is speaking a foreign language! The good news is that errors are actually trying to help us, even though they don't feel helpful. Let's decode what's happening together. Can you share the specific error message you're seeing?"
        elif concepts:
            concept_text = " and ".join(concepts)
            return f"I understand that {concept_text} can feel tricky at first. You're definitely not alone in finding this challenging - it's one of those concepts that takes time to click. The fact that you're asking questions means you're on the right track. Let me help break this down in a way that might make more sense."
        else:
            return "I can see you're working through something challenging here. That feeling of being stuck is so normal when learning to program - we've all been there! Let's work through this together. What part would you like to start with?"
    
    # Low frustration responses
    elif frustration_score < 2:
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

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Submit feedback on a tutoring interaction
    """
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
    """
    Get recent interactions for a student with enhanced data
    """
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
                "version": additional_data.get("version", "1.0.0")
            })
        
        conn.close()
        return {"student_id": student_id, "interactions": interactions, "count": len(interactions)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Configuration endpoints (only if config is available)
if CONFIG_AVAILABLE:
    @app.post("/config/ai_provider/{provider}")
    async def change_ai_provider(provider: str):
        """Change AI provider via API"""
        try:
            config.set_ai_provider(provider)
            return {
                "message": f"AI provider changed to {provider}",
                "current_provider": config.get_ai_provider(),
                "success": True
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/config/current")
    async def get_current_config():
        """Get current configuration"""
        return {
            "ai_provider": config.get_ai_provider(),
            "fallback_provider": config.get_fallback_provider(),
            "fallback_enabled": config.is_fallback_enabled(),
            "enhanced_empathy": config.is_enhanced_empathy_enabled(),
            "available_providers": ["openai", "anthropic", "local", "disabled"],
            "ai_available": AI_AVAILABLE,
            "config_available": CONFIG_AVAILABLE
        }

    @app.get("/config/status")
    async def get_config_status():
        """Get detailed configuration status"""
        return {
            "current_config": config.config,
            "config_file_exists": os.path.exists("config.yaml"),
            "ai_available": AI_AVAILABLE,
            "enhanced_empathy_available": True,  # Always available in this version
            "environment_keys": {
                "openai_key_present": bool(os.getenv('OPENAI_API_KEY')),
                "anthropic_key_present": bool(os.getenv('ANTHROPIC_API_KEY'))
            },
            "database_status": {
                "exists": os.path.exists(DB_NAME),
                "path": DB_NAME
            }
        }

# Analytics endpoints
@app.get("/analytics/overview")
async def get_analytics_overview():
    """Get system analytics overview"""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Overall statistics
        cursor.execute("SELECT COUNT(*) FROM interactions")
        total_interactions = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT student_id) FROM interactions")
        unique_students = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(frustration_score) FROM interactions")
        avg_frustration = cursor.fetchone()[0] or 0
        
        # Frustration distribution
        cursor.execute("""
            SELECT 
                CASE 
                    WHEN frustration_score < 3 THEN 'Low'
                    WHEN frustration_score < 7 THEN 'Medium'
                    ELSE 'High'
                END as level,
                COUNT(*) as count
            FROM interactions
            GROUP BY level
        """)
        frustration_dist = {row[0]: row[1] for row in cursor.fetchall()}
        
        # AI provider usage
        cursor.execute("""
            SELECT json_extract(additional_data, '$.ai_provider') as provider, COUNT(*) as count
            FROM interactions
            WHERE additional_data IS NOT NULL
            GROUP BY provider
        """)
        provider_usage = {row[0] or 'unknown': row[1] for row in cursor.fetchall()}
        
        conn.close()
        
        return {
            "total_interactions": total_interactions,
            "unique_students": unique_students,
            "average_frustration": round(avg_frustration, 2),
            "frustration_distribution": frustration_dist,
            "ai_provider_usage": provider_usage,
            "system_version": "2.0.0"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics error: {str(e)}")

@app.get("/analytics/student/{student_id}")
async def get_student_analytics(student_id: str):
    """Get detailed analytics for a specific student"""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Student statistics
        cursor.execute("""
            SELECT COUNT(*), AVG(frustration_score), MIN(created_at), MAX(created_at)
            FROM interactions 
            WHERE student_id = ?
        """, (student_id,))
        
        result = cursor.fetchone()
        if not result or result[0] == 0:
            raise HTTPException(status_code=404, detail="Student not found")
        
        total_sessions, avg_frustration, first_session, last_session = result
        
        # Frustration trend
        cursor.execute("""
            SELECT frustration_score, created_at
            FROM interactions 
            WHERE student_id = ?
            ORDER BY created_at
        """, (student_id,))
        
        frustration_trend = [{"score": row[0], "date": row[1]} for row in cursor.fetchall()]
        
        # Concept frequency
        cursor.execute("""
            SELECT concepts
            FROM interactions 
            WHERE student_id = ? AND concepts != '[]'
        """, (student_id,))
        
        concept_counts = {}
        for row in cursor.fetchall():
            try:
                concepts = json.loads(row[0])
                for concept in concepts:
                    concept_counts[concept] = concept_counts.get(concept, 0) + 1
            except:
                pass
        
        conn.close()
        
        return {
            "student_id": student_id,
            "total_sessions": total_sessions,
            "average_frustration": round(avg_frustration, 2),
            "first_session": first_session,
            "last_session": last_session,
            "frustration_trend": frustration_trend,
            "concept_frequency": concept_counts
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Student analytics error: {str(e)}")

# Test endpoints
@app.post("/test/template")
async def test_template_response(request: StudentMessage):
    """Test enhanced template responses only"""
    # Calculate frustration score (simplified for testing)
    frustration_keywords = ["stuck", "frustrated", "confused", "help", "error", "impossible"]
    message_lower = request.message.lower()
    keyword_count = sum(1 for keyword in frustration_keywords if keyword in message_lower)
    frustration_score = min(keyword_count * 2.0, 10.0)
    
    # Detect concepts
    concepts = []
    if "function" in message_lower:
        concepts.append("functions")
    if "loop" in message_lower:
        concepts.append("loops")
    if "error" in message_lower:
        concepts.append("errors")
    
    response = _generate_enhanced_template_response(frustration_score, concepts, request.message)
    
    return {
        "message": request.message,
        "frustration_score": frustration_score,
        "concepts": concepts,
        "response": response,
        "type": "enhanced_template_only"
    }

@app.post("/test/ai/{provider}")
async def test_ai_response(provider: str, request: StudentMessage):
    """Test AI responses with specific provider"""
    if not AI_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI components not available")
    
    if not CONFIG_AVAILABLE:
        raise HTTPException(status_code=503, detail="Configuration not available")
    
    if provider not in ["openai", "anthropic"]:
        raise HTTPException(status_code=400, detail="Provider must be 'openai' or 'anthropic'")
    
    # Calculate frustration score
    frustration_keywords = ["stuck", "frustrated", "confused", "help", "error", "impossible"]
    message_lower = request.message.lower()
    keyword_count = sum(1 for keyword in frustration_keywords if keyword in message_lower)
    frustration_score = min(keyword_count * 2.0, 10.0)
    
    # Detect concepts
    concepts = []
    concept_keywords = {
        "loops": ["for", "while", "loop", "iterate"],
        "functions": ["function", "def", "return", "parameter"],
        "errors": ["error", "exception", "traceback"]
    }
    
    for concept, keywords in concept_keywords.items():
        if any(keyword in message_lower for keyword in keywords):
            concepts.append(concept)
    
    try:
        # Save current provider
        original_provider = config.get_ai_provider()
        
        # Temporarily change to requested provider
        config.set_ai_provider(provider)
        
        response = await ai_generator.generate_response(
            message=request.message,
            frustration_score=frustration_score,
            concepts=concepts,
            student_id=request.student_id
        )
        
        # Restore original provider
        config.set_ai_provider(original_provider)
        
        return {
            "message": request.message,
            "frustration_score": frustration_score,
            "concepts": concepts,
            "response": response,
            "provider_used": provider,
            "type": "ai_test"
        }
        
    except Exception as e:
        # Restore original provider even if error occurs
        if CONFIG_AVAILABLE:
            config.set_ai_provider(original_provider)
        raise HTTPException(status_code=500, detail=f"AI generation failed: {str(e)}")

# Documentation endpoint
@app.get("/docs/empathy")
async def get_empathy_documentation():
    """Get documentation about the empathy system"""
    return {
        "empathy_system": {
            "version": "2.0.0",
            "features": [
                "Enhanced emotional state detection",
                "Learning style adaptation",
                "Historical context awareness", 
                "Personalized response generation",
                "Multi-level frustration analysis"
            ],
            "emotional_states": [
                "overwhelmed", "discouraged", "impatient", 
                "curious", "analytical", "frustrated_technical"
            ],
            "learning_styles": [
                "visual", "auditory", "kinesthetic", "reading"
            ],
            "frustration_levels": {
                "minimal": "0-2",
                "standard": "2-4", 
                "medium": "4-7",
                "high": "7-10"
            }
        },
        "configuration": {
            "enhanced_empathy_enabled": config.is_enhanced_empathy_enabled() if CONFIG_AVAILABLE else "unknown",
            "current_ai_provider": config.get_ai_provider() if CONFIG_AVAILABLE else "unknown"
        }
    }

if __name__ == "__main__":
    import uvicorn
    print(f"🚀 Starting Enhanced Empathetic Tutor API")
    print(f"📁 Database: {DB_NAME}")
    print(f"🗃️ Database exists: {os.path.exists(DB_NAME)}")
    print(f"🤖 AI Available: {AI_AVAILABLE}")
    print(f"⚙️ Config Available: {CONFIG_AVAILABLE}")
    
    if CONFIG_AVAILABLE:
        print(f"🎯 Current AI Provider: {config.get_ai_provider()}")
        print(f"💖 Enhanced Empathy: {config.is_enhanced_empathy_enabled()}")
    
    print(f"📚 API documentation: http://localhost:8000/docs")
    print(f"💖 Empathy docs: http://localhost:8000/docs/empathy")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
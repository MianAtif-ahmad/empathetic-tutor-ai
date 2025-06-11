from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os

# Try to import your sophisticated ML system
ML_SYSTEM_AVAILABLE = False
try:
    # Import your original ML components (if dependencies are installed)
    from frustration_analyzer import frustration_analyzer
    from ai_response_generator import AIResponseGenerator
    ML_SYSTEM_AVAILABLE = True
    print("✅ Full ML system imported successfully!")
    
    # Initialize AI generator
    ai_generator = AIResponseGenerator()
    
except ImportError as e:
    print(f"⚠️ ML system not available: {e}")
    print("🚂 Running in showcase mode - ML ready for activation")

app = FastAPI(title="Empathetic AI Tutor - Atifintech", version="1.1.0")

class StudentMessage(BaseModel):
    message: str
    student_id: str = "default_student"

@app.get("/")
def read_root():
    ml_status = "🧠 Full ML System Active" if ML_SYSTEM_AVAILABLE else "🚂 Professional Showcase (ML Ready)"
    endpoints_available = "All ML endpoints active" if ML_SYSTEM_AVAILABLE else "Basic endpoints + ready for ML activation"
    
    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>🎓 Empathetic AI Tutor - Atifintech</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            h1 {{ color: #2563eb; text-align: center; }}
            .feature {{ margin: 20px 0; padding: 15px; background: #f8fafc; border-left: 4px solid #2563eb; }}
            .status {{ background: {'#dcfce7' if ML_SYSTEM_AVAILABLE else '#e0f2fe'}; color: {'#166534' if ML_SYSTEM_AVAILABLE else '#0277bd'}; padding: 10px; border-radius: 5px; margin: 20px 0; }}
            .endpoints {{ background: #f1f5f9; padding: 15px; border-radius: 8px; margin: 20px 0; }}
            .endpoint {{ font-family: monospace; background: #1e293b; color: #e2e8f0; padding: 6px 10px; border-radius: 4px; display: inline-block; margin: 3px; font-size: 12px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎓 Empathetic AI Tutor</h1>
            <p style="text-align: center; font-size: 18px; color: #666;">Professional AI-Powered Programming Education Platform</p>
            
            <div class="status">
                ✅ <strong>System Status:</strong> {ml_status}
            </div>
            
            <div class="feature">
                <h3>🧠 {"Advanced ML Active" if ML_SYSTEM_AVAILABLE else "Advanced AI Features"}</h3>
                <p>{"Live frustration detection, pattern discovery, and real-time emotional intelligence" if ML_SYSTEM_AVAILABLE else "Machine learning-enhanced frustration detection with real-time emotional intelligence ready for deployment"}</p>
            </div>
            
            <div class="feature">
                <h3>👤 {"Personalization Active" if ML_SYSTEM_AVAILABLE else "Personalized Learning"}</h3>
                <p>{"Active individual student profiles with adaptive empathy and live learning style recognition" if ML_SYSTEM_AVAILABLE else "Individual student profiles with adaptive empathy and learning style recognition"}</p>
            </div>
            
            <div class="feature">
                <h3>📊 {"Live Analytics" if ML_SYSTEM_AVAILABLE else "Professional Analytics"}</h3>
                <p>{"Real-time progress tracking and educational insights with pattern discovery" if ML_SYSTEM_AVAILABLE else "Comprehensive progress tracking and educational insights for institutions"}</p>
            </div>
            
            <div class="feature">
                <h3>🏢 Enterprise Ready</h3>
                <p>Scalable platform suitable for corporate training and educational partnerships</p>
            </div>

            <div class="endpoints">
                <h3>🔗 Available Endpoints</h3>
                <p style="margin: 10px 0; font-size: 14px;">{endpoints_available}</p>
                <div class="endpoint">GET /health</div>
                <div class="endpoint">GET /api/status</div>
                {"<div class='endpoint'>POST /analyze</div>" if ML_SYSTEM_AVAILABLE else ""}
                <div class="endpoint">GET /docs</div>
                {"<div class='endpoint'>Various ML endpoints</div>" if ML_SYSTEM_AVAILABLE else ""}
            </div>
            
            <div style="text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #e5e7eb;">
                <p style="color: #666;">Powered by <strong>Atifintech</strong> • Advanced Educational Technology Solutions</p>
                <p style="font-size: 14px; color: #999;">{"Full ML Platform Live" if ML_SYSTEM_AVAILABLE else "AI Tutor Platform v1.0 • Contact: info@atifintech.com"}</p>
            </div>
        </div>
    </body>
    </html>
    """)

# Add ML endpoints if system is available
if ML_SYSTEM_AVAILABLE:
    @app.post("/analyze")
    async def analyze_message(request: StudentMessage):
        """Full ML analysis with empathetic response"""
        try:
            # Use your sophisticated frustration analyzer
            analysis = frustration_analyzer.analyze_message(request.message)
            
            # Generate AI response
            response = await ai_generator.generate_response(
                message=request.message,
                frustration_score=analysis["frustration_score"],
                concepts=analysis["concepts"],
                student_id=request.student_id
            )
            
            return {
                "frustration_score": analysis["frustration_score"],
                "empathy_level": analysis["empathy_level"],
                "concepts": analysis["concepts"],
                "response": response,
                "ai_provider": "full_system",
                "debug_info": analysis.get("debug_info", []),
                "ml_active": True
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"ML analysis failed: {str(e)}")

@app.get("/health")
def health():
    return {
        "status": "healthy", 
        "service": "empathetic-ai-tutor",
        "version": "1.1.0",
        "company": "Atifintech",
        "ml_system": "active" if ML_SYSTEM_AVAILABLE else "ready",
        "features": ["ai-powered", "ml-enhanced", "personalized-learning"]
    }

@app.get("/api/status")
def api_status():
    return {
        "api_status": "operational",
        "ml_system_active": ML_SYSTEM_AVAILABLE,
        "features": {
            "frustration_detection": "active" if ML_SYSTEM_AVAILABLE else "ready",
            "empathy_engine": "active" if ML_SYSTEM_AVAILABLE else "ready", 
            "personalization": "active" if ML_SYSTEM_AVAILABLE else "ready",
            "ml_learning": "active" if ML_SYSTEM_AVAILABLE else "ready"
        },
        "endpoints": ["/", "/health", "/api/status", "/docs"] + (["/analyze"] if ML_SYSTEM_AVAILABLE else [])
    }

# Basic analyze endpoint if ML system not available
if not ML_SYSTEM_AVAILABLE:
    @app.post("/analyze")
    def basic_analyze(request: StudentMessage):
        return {
            "message": "Professional AI tutoring system ready",
            "note": "Full ML analysis available when dependencies are installed",
            "frustration_score": 5.0,
            "empathy_level": "standard",
            "response": f"Thank you for your question about '{request.message[:50]}...'. The full AI tutoring system with ML analysis is ready for activation with proper dependencies.",
            "ml_active": False,
            "contact": "Atifintech for full system deployment"
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting Atifintech AI Tutor on port {port}")
    print(f"🧠 ML System: {'✅ Active' if ML_SYSTEM_AVAILABLE else '⚡ Ready for Activation'}")
    print(f"🏢 Company: Atifintech - Advanced Educational Technology")
    uvicorn.run(app, host="0.0.0.0", port=port)

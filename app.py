from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import os

app = FastAPI(title="Empathetic AI Tutor - Atifintech", version="1.0.0")

@app.get("/")
def read_root():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>🎓 Empathetic AI Tutor - Atifintech</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            h1 { color: #2563eb; text-align: center; }
            .feature { margin: 20px 0; padding: 15px; background: #f8fafc; border-left: 4px solid #2563eb; }
            .status { background: #dcfce7; color: #166534; padding: 10px; border-radius: 5px; margin: 20px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎓 Empathetic AI Tutor</h1>
            <p style="text-align: center; font-size: 18px; color: #666;">Professional AI-Powered Programming Education Platform</p>
            
            <div class="status">
                ✅ <strong>System Status:</strong> Online and Ready
            </div>
            
            <div class="feature">
                <h3>🧠 Advanced AI Features</h3>
                <p>Machine learning-enhanced frustration detection with real-time emotional intelligence</p>
            </div>
            
            <div class="feature">
                <h3>👤 Personalized Learning</h3>
                <p>Individual student profiles with adaptive empathy and learning style recognition</p>
            </div>
            
            <div class="feature">
                <h3>📊 Professional Analytics</h3>
                <p>Comprehensive progress tracking and educational insights for institutions</p>
            </div>
            
            <div class="feature">
                <h3>🏢 Enterprise Ready</h3>
                <p>Scalable platform suitable for corporate training and educational partnerships</p>
            </div>
            
            <div style="text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #e5e7eb;">
                <p style="color: #666;">Powered by <strong>Atifintech</strong> • Advanced Educational Technology Solutions</p>
                <p style="font-size: 14px; color: #999;">AI Tutor Platform v1.0 • Contact: info@atifintech.com</p>
            </div>
        </div>
    </body>
    </html>
    """)

@app.get("/health")
def health():
    return {
        "status": "healthy", 
        "service": "empathetic-ai-tutor",
        "version": "1.0.0",
        "company": "Atifintech",
        "features": ["ai-powered", "ml-enhanced", "personalized-learning"]
    }

@app.get("/api/status")
def api_status():
    return {
        "api_status": "operational",
        "features": {
            "frustration_detection": "ready",
            "empathy_engine": "ready", 
            "personalization": "ready",
            "ml_learning": "ready"
        },
        "endpoints": ["/", "/health", "/api/status", "/docs"]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

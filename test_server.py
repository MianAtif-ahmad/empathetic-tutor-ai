# standalone_server.py - Save and run with: python3 standalone_server.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import JSONResponse

@app.options("/{full_path:path}")
async def options_handler(request, full_path: str):
    """Handle preflight OPTIONS requests"""
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

class Message(BaseModel):
    message: str
    student_id: str = "test"

@app.get("/")
def root():
    return {"status": "Server is working!", "message": "Hello from standalone server"}

@app.get("/health") 
def health():
    return {"status": "healthy"}

@app.post("/analyze")
def analyze(msg: Message):
    return {
        "interaction_id": "test-123",
        "frustration_score": 2.5,
        "response": f"You said: {msg.message}. This is a test response from the standalone server.",
        "concepts": ["test"],
        "ai_provider": "test",
        "empathy_level": "standard"
    }

if __name__ == "__main__":
    print("🚀 Starting standalone server...")
    print("📍 Server will run on http://localhost:8000")
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        print("💡 Try installing uvicorn: pip install uvicorn")
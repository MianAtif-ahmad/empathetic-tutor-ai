# super_simple_server.py - Absolute minimal test
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    message: str
    student_id: str = "test"

@app.get("/")
def root():
    return {"status": "Server is working!"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/analyze")
def analyze(msg: Message):
    return {
        "interaction_id": "test-123",
        "frustration_score": 2.5,
        "response": f"You said: {msg.message}",
        "concepts": ["test"],
        "ai_provider": "test",
        "empathy_level": "standard"
    }

# No if __name__ == "__main__" block - run with uvicorn command
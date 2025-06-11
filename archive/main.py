# backend/app/main.py

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import traceback
from contextlib import asynccontextmanager
from loguru import logger
import os
from dotenv import load_dotenv

from .api.routes import student_routes, analytics_routes, admin_routes
from .core.config import settings
from .core.logging import setup_logging
from .db.database import engine, Base
from .services.ml.model_manager import ModelManager
from .services.intervention.knowledge_graph import KnowledgeGraphManager
from .utils.monitoring import setup_monitoring

# Load environment variables
load_dotenv()

# Setup logging
setup_logging()

# Initialize managers
model_manager = ModelManager()
kg_manager = KnowledgeGraphManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    # Startup
    logger.info("🚀 Starting Empathetic Tutor AI...")
    
    # Create database tables
    Base.metadata.create_all(bind=engine)
    
    # Load ML models
    await model_manager.load_models()
    
    # Initialize knowledge graph
    await kg_manager.initialize()
    
    # Setup monitoring
    setup_monitoring()
    
    logger.info("✅ Application started successfully")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down application...")
    await model_manager.cleanup()
    await kg_manager.cleanup()
    logger.info("👋 Application shut down complete")

# Create FastAPI app
app = FastAPI(
    title="Empathetic Tutor AI",
    description="Adaptive AI tutoring system with emotional intelligence",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    request_id = request.headers.get("X-Request-ID", "no-id")
    
    # Log request
    logger.info(f"📥 Request {request_id}: {request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        logger.info(
            f"📤 Response {request_id}: {response.status_code} "
            f"(Processing time: {process_time:.3f}s)"
        )
        
        # Add custom headers
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Request-ID"] = request_id
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Request {request_id} failed: {str(e)}")
        logger.error(traceback.format_exc())
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "request_id": request_id,
                "detail": str(e) if settings.DEBUG else "An error occurred"
            }
        )

# Include routers
app.include_router(
    student_routes.router,
    prefix="/api/v1/student",
    tags=["student"]
)

app.include_router(
    analytics_routes.router,
    prefix="/api/v1/analytics",
    tags=["analytics"]
)

app.include_router(
    admin_routes.router,
    prefix="/api/v1/admin",
    tags=["admin"]
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "models_loaded": model_manager.is_ready(),
        "knowledge_graph_ready": kg_manager.is_ready()
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Empathetic Tutor AI",
        "version": "1.0.0",
        "description": "Adaptive AI tutoring system with emotional intelligence",
        "documentation": "/docs",
        "health": "/health"
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not found",
            "path": str(request.url.path)
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.DEBUG else "An error occurred"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import uvicorn
import sys
import traceback
from contextlib import asynccontextmanager

from api.endpoints import router
from config import get_settings

# Get settings
settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI app startup and shutdown events
    """
    # Startup
    print("🚀 Starting Video Subtitle Processing API...")
    print(f"📊 OpenAI API configured: {bool(settings.OPENAI_API_KEY)}")
    print(f"🎤 AssemblyAI API configured: {bool(settings.ASSEMBLYAI_API_KEY)}")
    
    # Validate required dependencies
    try:
        import cv2
        import openai
        import requests
        import numpy as np
        import sklearn
        import soundfile as sf
        print("✅ All required dependencies are available")
    except ImportError as e:
        print(f"❌ Missing required dependency: {e}")
        sys.exit(1)
    
    # Check if ffmpeg is available
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ FFmpeg is available")
        else:
            print("⚠️ FFmpeg not found - video processing may fail")
    except Exception:
        print("⚠️ FFmpeg not found - video processing may fail")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down Video Subtitle Processing API...")

# Initialize FastAPI app
app = FastAPI(
    title="Video Subtitle Processing API",
    description="""
    A comprehensive API for processing videos to generate styled subtitles.
    
    ## Features
    - Extract visual styles from reference videos
    - Transcribe audio from input videos
    - Apply AI-powered styling to subtitles
    - Generate ASS format subtitle files
    
    ## Process Flow
    1. Upload reference video (for style extraction)
    2. Upload input video (for transcription)
    3. API processes both videos through a 10-step pipeline
    4. Returns styled ASS subtitle file
    
    ## Supported Formats
    - Video: MP4, AVI, MOV, MKV, WebM
    - Output: ASS subtitle format
    
    ## Requirements
    - OpenAI API key (for vision and text processing)
    - AssemblyAI API key (for audio transcription)
    - FFmpeg (for video processing)
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1", tags=["Video Processing"])

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Video Subtitle Processing API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/api/v1/health",
        "supported_formats": "/api/v1/supported-formats",
        "processing_info": "/api/v1/processing-info"
    }

# Global exception handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Exception",
            "detail": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url)
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors"""
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "detail": exc.errors(),
            "status_code": 422,
            "path": str(request.url)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    print(f"❌ Unhandled exception: {exc}")
    traceback.print_exc()
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred. Please try again later.",
            "status_code": 500,
            "path": str(request.url)
        }
    )

# Add middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log incoming requests"""
    print(f"📨 {request.method} {request.url}")
    
    response = await call_next(request)
    
    print(f"📤 {request.method} {request.url} - {response.status_code}")
    return response

# Health check endpoint at root level
@app.get("/health", tags=["Health"])
async def health_check():
    """Simple health check endpoint"""
    return {
        "status": "healthy",
        "service": "Video Subtitle Processing API",
        "version": "1.0.0"
    }

# Development server configuration
if __name__ == "__main__":
    print("🔧 Starting in development mode...")
    print(f"📋 Environment: {settings.ENVIRONMENT}")
    print(f"🌐 Debug mode: {settings.DEBUG}")
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info" if settings.DEBUG else "warning",
        access_log=settings.DEBUG
    )
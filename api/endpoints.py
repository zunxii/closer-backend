from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import Response
import tempfile
import shutil
import os
from typing import List

from core.processor import VideoSubtitleProcessor
from config import get_settings

router = APIRouter()
settings = get_settings()

@router.post("/process-videos")
async def process_videos_direct(
    reference_video: UploadFile = File(...),
    input_video: UploadFile = File(...)
):
    """
    Process two videos and return ASS content directly without creating a file.
    
    Args:
        reference_video: Video file to extract visual styles from
        input_video: Video file to transcribe and apply styles to
        
    Returns:
        ASS subtitle file content with styled captions
    """
    # Validate API keys
    if not settings.OPENAI_API_KEY or not settings.ASSEMBLYAI_API_KEY:
        raise HTTPException(status_code=500, detail="Missing required API keys")
    
    # Validate file types
    allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    
    def validate_video_file(file: UploadFile) -> bool:
        return any(file.filename.lower().endswith(ext) for ext in allowed_extensions)
    
    if not validate_video_file(reference_video):
        raise HTTPException(
            status_code=400, 
            detail=f"Reference video must be one of: {', '.join(allowed_extensions)}"
        )
    
    if not validate_video_file(input_video):
        raise HTTPException(
            status_code=400, 
            detail=f"Input video must be one of: {', '.join(allowed_extensions)}"
        )

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Save uploaded files
        ref_video_path = os.path.join(temp_dir, f"reference_{reference_video.filename}")
        input_video_path = os.path.join(temp_dir, f"input_{input_video.filename}")
        
        # Save reference video
        with open(ref_video_path, "wb") as f:
            shutil.copyfileobj(reference_video.file, f)
        
        # Save input video
        with open(input_video_path, "wb") as f:
            shutil.copyfileobj(input_video.file, f)

        # Initialize processor
        processor = VideoSubtitleProcessor()
        
        # Process videos
        ass_content = await processor.process_videos(
            reference_video_path=ref_video_path,
            input_video_path=input_video_path,
            temp_dir=temp_dir
        )
        
        # Return ASS content as downloadable file
        return Response(
            content=ass_content,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": "attachment; filename=styled_subtitles.ass",
                "Content-Type": "application/octet-stream"
            }
        )
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
        except Exception as cleanup_error:
            print(f"⚠️ Warning: Could not clean up temp directory: {cleanup_error}")

@router.get("/health")
async def health_check():
    """Health check endpoint to verify API is running"""
    return {
        "status": "healthy",
        "service": "Video Subtitle Processing API",
        "version": "1.0.0"
    }

@router.get("/config")
async def get_config_status():
    """Get configuration status without exposing sensitive data"""
    return {
        "openai_configured": bool(settings.OPENAI_API_KEY),
        "assemblyai_configured": bool(settings.ASSEMBLYAI_API_KEY),
        "supported_formats": ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    }

@router.post("/validate-video")
async def validate_video_file(video: UploadFile = File(...)):
    """
    Validate video file without processing
    
    Args:
        video: Video file to validate
        
    Returns:
        Validation result with file information
    """
    try:
        # Basic file validation
        allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        
        if not any(video.filename.lower().endswith(ext) for ext in allowed_extensions):
            raise HTTPException(
                status_code=400, 
                detail=f"File must be one of: {', '.join(allowed_extensions)}"
            )
        
        # Get file size
        file_size = 0
        video.file.seek(0, 2)  # Seek to end
        file_size = video.file.tell()
        video.file.seek(0)  # Reset to beginning
        
        # Check file size (100MB limit)
        max_size = 100 * 1024 * 1024  # 100MB
        if file_size > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {max_size // (1024*1024)}MB"
            )
        
        return {
            "valid": True,
            "filename": video.filename,
            "size_bytes": file_size,
            "size_mb": round(file_size / (1024*1024), 2),
            "content_type": video.content_type
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@router.get("/supported-formats")
async def get_supported_formats():
    """Get list of supported video formats"""
    return {
        "video_formats": ['.mp4', '.avi', '.mov', '.mkv', '.webm'],
        "audio_formats": ['.mp3', '.wav', '.m4a'],
        "subtitle_formats": ['.ass', '.srt']
    }

@router.get("/processing-info")
async def get_processing_info():
    """Get information about the processing pipeline"""
    return {
        "steps": [
            {
                "step": 1,
                "name": "Frame Extraction",
                "description": "Extract frames from reference video for style analysis"
            },
            {
                "step": 2,
                "name": "Frame Analysis",
                "description": "Analyze frames using OpenAI Vision API to extract text styles"
            },
            {
                "step": 3,
                "name": "Duplicate Filtering",
                "description": "Remove duplicate frames to optimize processing"
            },
            {
                "step": 4,
                "name": "Style Clustering",
                "description": "Group similar styles and rank by visual importance"
            },
            {
                "step": 5,
                "name": "Audio Processing",
                "description": "Transcribe input video audio using AssemblyAI"
            },
            {
                "step": 6,
                "name": "Sentence Splitting",
                "description": "Split transcription into sentences with timing"
            },
            {
                "step": 7,
                "name": "Word Styling",
                "description": "Apply style priorities to words using AI"
            },
            {
                "step": 8,
                "name": "Template Application",
                "description": "Apply extracted styles to words based on priorities"
            },
            {
                "step": 9,
                "name": "Word Chunking",
                "description": "Group words into subtitle chunks for optimal display"
            },
            {
                "step": 10,
                "name": "ASS Generation",
                "description": "Generate final ASS subtitle file with styled captions"
            }
        ],
        "estimated_processing_time": "2-5 minutes depending on video length",
        "max_video_length": "10 minutes recommended"
    }
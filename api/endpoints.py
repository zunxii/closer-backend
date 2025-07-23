from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import Response
from core.processor import VideoSubtitleProcessor
from config import Config, get_settings
from utils.file_utils import is_video_file

router = APIRouter()
settings = get_settings()


@router.post("/process-videos")
async def process_videos_instagram(
    reference_url: str = Form(...),
    input_video: UploadFile = File(...)
):
    """
    Download reference video from Instagram, process both videos, and return ASS subtitle content.
    """
    # Validate configuration
    Config.validate()

    if not is_video_file(input_video.filename):
        raise HTTPException(
            status_code=400,
            detail="Input video must be a supported video format"
        )

    try:
        processor = VideoSubtitleProcessor()
        ass_content = processor.process(reference_url, input_video)

        return Response(
            content=ass_content,
            media_type="application/octet-stream",
            headers={"Content-Disposition": "attachment; filename=styled_subtitles.ass"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Video Subtitle Processing API",
        "version": "1.0.0"
    }


@router.get("/config")
async def get_config_status():
    return {
        "openai_configured": bool(Config.OPENAI_API_KEY),
        "assemblyai_configured": bool(Config.ASSEMBLYAI_API_KEY),
        "supported_formats": ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    }


@router.post("/validate-video")
async def validate_video_file(video: UploadFile = File(...)):
    try:
        if not is_video_file(video.filename):
            raise HTTPException(
                status_code=400,
                detail="File must be one of: .mp4, .avi, .mov, .mkv, .webm"
            )

        video.file.seek(0, 2)
        file_size = video.file.tell()
        video.file.seek(0)

        max_size = 100 * 1024 * 1024
        if file_size > max_size:
            raise HTTPException(
                status_code=400,
                detail="File too large. Maximum allowed size is 100MB"
            )

        return {
            "valid": True,
            "filename": video.filename,
            "size_bytes": file_size,
            "size_mb": round(file_size / (1024 * 1024), 2),
            "content_type": video.content_type
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@router.get("/supported-formats")
async def get_supported_formats():
    return {
        "video_formats": ['.mp4', '.avi', '.mov', '.mkv', '.webm'],
        "audio_formats": ['.mp3', '.wav', '.m4a'],
        "subtitle_formats": ['.ass', '.srt']
    }


@router.get("/processing-info")
async def get_processing_info():
    return {
        "steps": [
            {"step": 0, "name": "Download Instagram Video", "description": "Download reference video using API"},
            {"step": 1, "name": "Frame Extraction", "description": "Extract frames from reference video"},
            {"step": 2, "name": "Frame Analysis", "description": "Analyze frames using Vision AI"},
            {"step": 3, "name": "Duplicate Filtering", "description": "Filter duplicate frames"},
            {"step": 4, "name": "Style Clustering", "description": "Group and rank styles"},
            {"step": 5, "name": "Audio Processing", "description": "Transcribe audio using AssemblyAI"},
            {"step": 6, "name": "Sentence Splitting", "description": "Split transcription into sentences"},
            {"step": 7, "name": "Word Styling", "description": "Style words using AI ranking"},
            {"step": 8, "name": "Template Application", "description": "Apply style templates"},
            {"step": 9, "name": "Word Chunking", "description": "Chunk text into subtitles"},
            {"step": 10, "name": "ASS Generation", "description": "Generate .ASS subtitle file"}
        ],
        "estimated_processing_time": "2-5 minutes",
        "max_video_length": "10 minutes recommended"
    }

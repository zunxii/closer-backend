import os
import tempfile
import shutil
from pathlib import Path
from typing import Generator

def create_temp_directory() -> str:
    """Create a temporary directory for processing"""
    return tempfile.mkdtemp()

def cleanup_temp_directory(temp_dir: str) -> None:
    """Clean up temporary directory"""
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"⚠️ Warning: Could not clean up temp directory: {e}")

def save_uploaded_file(file, file_path: str) -> None:
    """Save uploaded file to specified path"""
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

def ensure_directory_exists(directory: str) -> None:
    """Ensure directory exists, create if not"""
    os.makedirs(directory, exist_ok=True)

def read_in_chunks(data: bytes, chunk_size: int = 5242880) -> Generator[bytes, None, None]:
    """Read data in chunks for streaming"""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

def get_file_extension(filename: str) -> str:
    """Get file extension from filename"""
    return Path(filename).suffix.lower()

def is_image_file(filename: str) -> bool:
    """Check if file is an image"""
    return get_file_extension(filename) in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

def is_video_file(filename: str) -> bool:
    """Check if file is a video"""
    return get_file_extension(filename) in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
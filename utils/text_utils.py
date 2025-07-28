import os
import tempfile
import json
import shutil
from pathlib import Path
from typing import Generator, List
import re
from datetime import timedelta

def clean_json_text(raw: str) -> str:
    """Clean JSON text from markdown formatting"""
    return re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE)

def hex_to_ass_color(hex_color: str) -> str:
    """Convert hex color to ASS color format"""
    hex_color = hex_color.replace("#", "")
    if len(hex_color) != 6:
        return "&HFFFFFF"
    r, g, b = hex_color[0:2], hex_color[2:4], hex_color[4:6]
    return f"&H{b}{g}{r}".upper()

def hex_to_rgb(ass_color: str):
    """
    Converts ASS color format (&HBBGGRR) to (R, G, B).
    Input: '&HFF00FF'
    Output: (255, 0, 255)
    """
    hex_str = ass_color.strip().upper().replace("&H", "").zfill(6)
    bb = int(hex_str[0:2], 16)
    gg = int(hex_str[2:4], 16)
    rr = int(hex_str[4:6], 16)
    return (rr, gg, bb)


def ms_to_ass_time(ms: int) -> str:
    """Convert milliseconds to ASS time format"""
    t = timedelta(milliseconds=ms)
    return f"{int(t.total_seconds() // 3600)}:{int((t.total_seconds() % 3600) // 60):02}:{int(t.total_seconds() % 60):02}.{int(t.microseconds / 10000):02}"

def normalize_word(word_obj: dict) -> dict:
    """Normalize word object with default values"""
    try:
        return {
            "text": word_obj.get("text", ""),
            "fontname": word_obj.get("font", "Poppins"),
            "fontsize": int(word_obj.get("font_size", 48)),
            "primary_colour": hex_to_ass_color(word_obj.get("color", "#FFFFFF")),
            "bold": -1 if str(word_obj.get("bold", False)).lower() in ["true", "-1", "1"] else 0,
            "italic": -1 if str(word_obj.get("italic", False)).lower() in ["true", "-1", "1"] else 0,
            "outline": int(word_obj.get("outline", 1)),
            "shadow": int(word_obj.get("shadow", 0)),
            "relative_position": word_obj.get("relative_position", [1, 1]),
        }
    except Exception as e:
        print(f"⚠️ Failed to normalize word: {e}")
        return None

def get_frame_text(frame: dict) -> str:
    """Extract text from frame data"""
    return " ".join(word["text"] for word in frame["words"]).strip()

def clean_json_response(content: str) -> str:
    """Clean JSON response from OpenAI"""
    content = re.sub(r"^```(?:json)?", "", content, flags=re.MULTILINE)
    content = re.sub(r"```$", "", content, flags=re.MULTILINE)
    content = re.sub(r",\s*([}\]])", r"\1", content)
    return content.strip()

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

def parse_multiple_json_objects(text: str) -> List[dict]:
        """
        Safely parse multiple JSON objects from a single string response.
        """
        json_objects = []
        # Fallback pattern if your Python doesn't support recursive regex
        pattern = r"\{[^{}]*\}"

        try:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    obj = json.loads(match)
                    json_objects.append(obj)
                except json.JSONDecodeError as e:
                    print(f"⚠️ Skipping malformed JSON: {e}")
        except Exception as e:
            print(f"❌ Error parsing JSON: {e}")
        
        return json_objects
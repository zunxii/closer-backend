import os
import json
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
    INSTAGRAM_VIDEO_DOWNLOAD_API_KEY = os.getenv("INSTAGRAM_VIDEO_DOWNLOAD_API_KEY")

    HOST="0.0.0.0"
    PORT=8080
    
    TARGET_FPS = 1
    MAX_FRAMES = 85
    MAX_CHUNK_SIZE = 5
    PAUSE_THRESHOLD = 120
    PRIORITY_JUMP = 0.35
    STYLE_THRESHOLD = 0.15

    OPENAI_MODEL = "gpt-4o"
    OPENAI_TEMPERATURE = 0.2
    OPENAI_MAX_TOKENS = 2500

    ASSEMBLYAI_UPLOAD_URL = "https://api.assemblyai.com/v2/upload"
    ASSEMBLYAI_TRANSCRIPT_URL = "https://api.assemblyai.com/v2/transcript"
    INSTAGRAM_VIDEO_DOWNLOAD_URL = "https://apihut.in/api/download/videos"
    CHUNK_SIZE = 5242880

    DEFAULT_STYLE_NAME = "style3"
    MIN_FREQUENCY = 1
    ALPHA = 0.7
    BETA = 0.3

    FONT_DESCRIPTION_PATH = os.path.join("assets", "font_description.json")
    FONT_DESCRIPTION = None  # Will be loaded below

    @classmethod
    def load_font_description(cls):
        try:
            with open(cls.FONT_DESCRIPTION_PATH, "r", encoding="utf-8") as f:
                cls.FONT_DESCRIPTION = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Font description file not found at {cls.FONT_DESCRIPTION_PATH}")
        except json.JSONDecodeError:
            raise ValueError("Font description file is not a valid JSON")

    @classmethod
    def validate(cls):
        """Validate required config keys and load font descriptions"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required")
        if not cls.ASSEMBLYAI_API_KEY:
            raise ValueError("ASSEMBLYAI_API_KEY is required")
        
        cls.load_font_description()
        return True


def get_settings():
    """Factory function to get config instance"""
    Config.validate()
    return Config()

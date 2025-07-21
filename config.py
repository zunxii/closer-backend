import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
    
    # Processing Settings
    TARGET_FPS = 2
    MAX_FRAMES = 85
    MAX_CHUNK_SIZE = 5
    PAUSE_THRESHOLD = 120
    PRIORITY_JUMP = 0.35
    STYLE_THRESHOLD = 0.15
    
    # OpenAI Settings
    OPENAI_MODEL = "gpt-4o"
    OPENAI_TEMPERATURE = 0.2
    OPENAI_MAX_TOKENS = 1200
    
    # AssemblyAI Settings
    ASSEMBLYAI_UPLOAD_URL = "https://api.assemblyai.com/v2/upload"
    ASSEMBLYAI_TRANSCRIPT_URL = "https://api.assemblyai.com/v2/transcript"
    CHUNK_SIZE = 5242880  # 5MB
    
    # Video Processing
    DEFAULT_STYLE_NAME = "style3"
    MIN_FREQUENCY = 3
    ALPHA = 0.7
    BETA = 0.3
    
    @classmethod
    def validate(cls):
        """Validate that required configuration is present"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required")
        if not cls.ASSEMBLYAI_API_KEY:
            raise ValueError("ASSEMBLYAI_API_KEY is required")
        return True
import json
import os
import openai
from typing import List, Dict
from tqdm import tqdm
from config import Config
from utils.image_utils import image_to_base64, get_sorted_image_files
from utils.text_utils import clean_json_text, normalize_word

class FrameAnalyzer:
    def __init__(self):
        self.config = Config()
        self.client = openai.Client(api_key=Config.OPENAI_API_KEY)
        self.model = Config.OPENAI_MODEL
        self.temperature = Config.OPENAI_TEMPERATURE
        self.max_tokens = Config.OPENAI_MAX_TOKENS

    def generate_prompt(self) -> str:
        """Generate prompt for frame analysis"""
        return """
        Analyze this image and extract any visible text with styling information.
        Return JSON format with the following structure:
        {
            "words": [
                {
                    "text": "word",
                    "font": "font_name",
                    "font_size": 48,
                    "color": "#FFFFFF",
                    "bold": false,
                    "italic": false,
                    "outline": 1,
                    "shadow": 0,
                    "relative_position": [1, 1]
                }
            ]
        }
        """

    def build_reference_styles(self, style_memory: List[Dict], max_styles: int = 10) -> str:
        """Build reference styles from previous frames"""
        recent = style_memory[-max_styles:]
        seen = set()
        references = []
        
        for frame in recent:
            for word in frame["words"]:
                key = (word["fontname"], word["primary_colour"])
                if key not in seen:
                    seen.add(key)
                    references.append(f"- Font: {word['fontname']}, Color: {word['primary_colour']}")
        
        if references:
            return (
                "REFERENCE STYLES from previous frames (font + color only):\n"
                + "\n".join(references)
                + "\nUse these styles if the new text matches visually."
            )
        return ""

    def analyze_single_frame(self, image_path: str, reference_styles: str) -> Dict:
        """Analyze a single frame for text and styling"""
        image_b64 = image_to_base64(image_path)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a subtitle caption visual style extractor. Your job is to return JSON of the visible styled words in the image. Only return valid JSON with keys. Do not return any explanations or markdown.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": reference_styles},
                            {"type": "text", "text": self.generate_prompt()},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                        ],
                    },
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            raw = response.choices[0].message.content
            cleaned = clean_json_text(raw)

            try:
                parsed = json.loads(cleaned)
                words_raw = parsed["words"] if isinstance(parsed, dict) else parsed
            except Exception as e:
                print(f"❌ JSON parse error: {e}")
                return None

            normalized = [normalize_word(word) for word in words_raw]
            normalized = [w for w in normalized if w]

            return {"words": normalized}

        except Exception as e:
            print(f"❌ API error: {e}")
            return None

    def analyze_frames(self, folder: str, max_frames: int = Config.MAX_FRAMES) -> List[Dict]:
        """Analyze all frames in folder"""
        files = get_sorted_image_files(folder, max_frames)
        
        print(f"🔍 Processing {len(files)} frames from '{folder}'...")
        all_data = []
        style_memory = []

        for filename in tqdm(files, desc="Analyzing Frames"):
            image_path = os.path.join(folder, filename)
            reference_styles = self.build_reference_styles(style_memory, max_styles=10)
            
            frame_data = self.analyze_single_frame(image_path, reference_styles)
            if frame_data:
                frame_result = {"frame": filename, "words": frame_data["words"]}
                all_data.append(frame_result)
                style_memory.append(frame_result)

        return all_data
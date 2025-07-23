import os
import json
import torch
import open_clip
from PIL import Image
from torchvision import transforms
from typing import List, Dict, Any
from tqdm import tqdm
from config import get_settings


config = get_settings()
raw_fonts = config.FONT_DESCRIPTION.get("fonts", [])

# Convert list of fonts into {name: prompt} dictionary
font_descriptions = {font["name"]: font["prompt"] for font in raw_fonts}

class LocalFrameAnalyzer:
    def __init__(self, font_descriptions: Dict[str, str], device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.model = self.model.to(self.device).eval()

        self.font_names = list(font_descriptions.keys())
        self.font_prompts = list(font_descriptions.values())
        self.font_embeddings = self.encode_texts_in_chunks(self.font_prompts)

    def encode_texts_in_chunks(self, text_list, chunk_size=256):
        all_embeddings = []
        for i in range(0, len(text_list), chunk_size):
            chunk = text_list[i:i + chunk_size]
            tokens = self.tokenizer(chunk).to(self.device)
            with torch.no_grad():
                chunk_features = self.model.encode_text(tokens)
                chunk_features /= chunk_features.norm(dim=-1, keepdim=True)
            all_embeddings.append(chunk_features)
        return torch.cat(all_embeddings, dim=0)

    def analyze_frame(self, image_path: str) -> Dict[str, Any]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = self.preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_embedding = self.model.encode_image(image)
            image_embedding /= image_embedding.norm(dim=-1, keepdim=True)

        similarities = torch.nn.functional.cosine_similarity(image_embedding, self.font_embeddings)
        top_index = similarities.argmax().item()
        top_font = self.font_names[top_index]
        confidence = similarities[top_index].item()

        return {
            "words": [
                {
                    "text": "<unknown>",
                    "fontname": top_font,
                    "fontsize": 48,
                    "primary_colour": "#FFFFFF",
                    "bold": -1 if "Bold" in top_font else 0,
                    "italic": -1 if "Italic" in top_font else 0,
                    "outline": 1,
                    "shadow": 0,
                    "relative_position": [1, 1],
                    "confidence": confidence
                }
            ]
        }

    def analyze_frames(self, folder: str, max_frames: int = None) -> List[Dict[str, Any]]:
        valid_extensions = {'.jpg', '.jpeg', '.png'}
        files = sorted([f for f in os.listdir(folder) if any(f.lower().endswith(ext) for ext in valid_extensions)])
        if max_frames:
            files = files[:max_frames]

        print(f"🔍 Processing {len(files)} frames from '{folder}'...")
        results = []

        for filename in tqdm(files, desc="Analyzing Frames"):
            image_path = os.path.join(folder, filename)
            result = self.analyze_frame(image_path)
            result["frame"] = filename
            results.append(result)

        debug_json_path = os.path.join(folder, "all_frames_analysis.json")
        with open(debug_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return results

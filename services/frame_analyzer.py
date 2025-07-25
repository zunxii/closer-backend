import os
import openai
from PIL import Image
import base64

class FontAnalyzer:
    def __init__(self, api_key):
        self.client = openai.Client(api_key=api_key)

    def image_to_base64(self, image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def analyze_image(self, image_path):
        base64_image = self.image_to_base64(image_path)

        prompt = (
            "Analyze the following image and extract text styling information. "
            "For each visible text block, return a JSON object in this structure:\n\n"
        "{\n"
        '  "text": "<unknown>",\n'
        '  "fontname": top_font,\n'
        '  "fontsize": 48,\n'
        '  "primary_colour": "#FFFFFF",\n'
        '  "bold": -1 if "Bold" in top_font else 0,\n'
        '  "italic": -1 if "Italic" in top_font else 0,\n'
        '  "outline": 1,\n'
        '  "shadow": 0,\n'
        '  "relative_position": [1, 1],\n'
        '  "confidence": confidence\n'
        "}"
        )

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a font analysis assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                },
            ],
            max_tokens=2500
        )

        return response.choices[0].message.content

    def analyze_folder(self, folder_path):
        results = {}
        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(folder_path, filename)
                try:
                    print(f"Analyzing {filename}...")
                    result = self.analyze_image(image_path)
                    results[filename] = result
                except Exception as e:
                    print(f"Error analyzing {filename}: {e}")
        return results


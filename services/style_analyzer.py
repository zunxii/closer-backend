import os
import json
import base64
from PIL import Image
from openai import OpenAI
from config import get_settings  # Ensure this imports your Config class correctly


class StyleAnalyzer:
    def __init__(self):
        self.config = get_settings()
        self.client = OpenAI(api_key=self.config.OPENAI_API_KEY)

    def encode_image_to_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def send_image_to_openai(self, image_path: str) -> str:
        base64_image = self.encode_image_to_base64(image_path)

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
            model=self.config.OPENAI_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ],
            max_tokens=self.config.OPENAI_MAX_TOKENS,
            temperature=self.config.OPENAI_TEMPERATURE,
        )

        return response.choices[0].message.content

    def process_all_images(self, folder_path: str = "style_representatives_flat.json") -> dict:
        output = {}
        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                image_path = os.path.join(folder_path, filename)
                try:
                    print(f"Processing: {filename}")
                    result = self.send_image_to_openai(image_path)
                    output[filename] = result
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        return output

    def save_output(self, output: dict, output_path: str = "style_output.json"):
        with open(output_path, "w") as f:
            json.dump(output, f, indent=4)
        print(f"Saved style output to {output_path}")


# Example usage
if __name__ == "__main__":
    analyzer = StyleAnalyzer()
    results = analyzer.process_all_images("representatives")
    analyzer.save_output(results)

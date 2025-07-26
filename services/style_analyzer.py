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
            "You are an expert visual design assistant for a professional video editing tool. "
            "Your task is to perform precise visual style detection on the provided image. "
            "Extract the actual text style visible in the image — as close as possible to what a designer or typographer would identify. "
            "The data is used to auto-style subtitles in videos, so the match must be visually indistinguishable to the human eye, even if the internal font name differs slightly. "
            "Do NOT default to basic fonts like Arial, unless it is unmistakably correct. Use designer-level judgment. "
            "\n\n"
            "Guidance:\n"
            "- Look closely at curves, edges, thickness, and letterforms to identify nuanced fonts.\n"
            "- Infer size based on pixel height and visual density.\n"
            "- If the font has styles like 'Bold Italic' or 'Condensed Light', include the full designation.\n"
            "- If multiple styles exist in one image (e.g., different font sizes or types), return them as separate JSON chunks. Otherwise, return one.\n"
            "- Guess based on appearance: even if unsure of the exact name, choose the closest font visually, e.g., 'Open Sans Bold' or 'Roboto Condensed Italic'.\n"
            "- Output must be in clean JSON with no markdown, code fencing, or explanation.\n\n"
            "Output Format:\n"
            "{\n"
            '  "name": "style1",\n'
            '  "fontname": "<Exact font name visually identified>",\n'
            '  "fontsize": <estimated size in pt>,\n'
            '  "primary_colour": "&HBBGGRR" (hex with alpha channel),\n'
            '  "bold": 1 or 0,\n'
            '  "italic": 1 or 0,\n'
            '  "outline": 1 or 0,\n'
            '  "shadow": 1 or 0,\n'
            '  "frame_id": "<image filename>"\n'
            "}\n"
            "Return nothing except the valid JSON output."
            "If there are multiple styles in the image, return them as separate JSON objects, treat them as different images and make seperate for each."
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
        print(f"OpenAI response: {response.choices[0].message.content}")
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

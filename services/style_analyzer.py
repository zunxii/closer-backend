import os
import json
import base64
from typing import List, Dict, Any
from openai import OpenAI
from config import get_settings


class StyleAnalyzer:
    def __init__(self):
        self.config = get_settings()
        self.client = OpenAI(api_key=self.config.OPENAI_API_KEY)

    def encode_image_to_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def send_batch_images_to_openai(self, image_paths: List[str]) -> tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
        base64_images = []
        for path in image_paths:
            base64_images.append({
                "filename": os.path.basename(path),
                "base64": self.encode_image_to_base64(path)
            })

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
            "- Make sure to return the color in BBGGRR not RRGGBB as its going to be used directly in ass-parsing.\n"
            "- If multiple styles exist in one image (e.g., different font sizes or types), return them as separate JSON chunks. Otherwise, return one.\n"
            "- Guess based on appearance: even if unsure of the exact name, choose the closest font visually, e.g., 'Open Sans Bold' or 'Roboto Condensed Italic'.\n"
            "- Output must be in clean JSON with no markdown, code fencing, or explanation.\n\n"

            "👀 Tips:\n"
            "- Carefully observe font endings (serif vs sans), stroke contrast, and letter width.\n"
            "- Fonts like 'Poppins', 'Montserrat', 'Roboto', 'Lato', 'Oswald', 'Open Sans', etc. should be considered.\n"
            "- Avoid generic guesses like Arial or Times unless visually confirmed.\n\n"

            "🔢 Output Format:\n"
            "[\n"
            "  {\n"
            '    "frame_id": "filename.jpg",\n'
            '    "styles": [\n'
            "      {\n"
            '        "name": "<style_no> take a counter for all images",\n'
            '        "fontname": "<Exact font name from google fonts only visually identified not some generic font family like Arial>",\n'
            '        "fontsize": <number in px>,\n'
            '        "primary_colour": "<&HBBGGRR> strictly ass format color",\n'
            '        "bold": 1(for yes) or 0(for no),\n'
            '        "italic": 1(for yes) or 0(for no),\n'
            '        "outline": 1(for yes) or 0(for no),\n'
            '        "shadow": 1(for yes) or 0(for no)\n'
            "      }, ...\n"
            "    ]\n"
            "  }, ...\n"
            "]\n\n"
            "Respond ONLY with the raw JSON array. No markdown. No explanations."
        )

        message_content = [{"type": "text", "text": prompt}]
        for img in base64_images:
            message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img['base64']}"
                }
            })

        response = self.client.chat.completions.create(
            model=self.config.OPENAI_MODEL,
            messages=[{"role": "user", "content": message_content}],
            max_tokens=self.config.OPENAI_MAX_TOKENS,
            temperature=self.config.OPENAI_TEMPERATURE,
        )

        raw_response = response.choices[0].message.content.strip()
        print(f"📩 OpenAI Response:\n{raw_response}")

        parsed: List[Dict[str, Any]] = json.loads(raw_response)

        all_styles: List[Dict[str, Any]] = []
        frame_style_map: Dict[str, List[str]] = {}

        for entry in parsed:
            frame_id = entry["frame_id"]
            styles = entry.get("styles", [])
            frame_style_map[frame_id] = []
            # print(f"DEBUG styles for frame {frame_id}: {styles} (type={type(styles)})")
            for style in styles:
                style["frame_id"] = frame_id
                all_styles.append(style)
                # print(f"DEBUG frame_style_map[{frame_id}] = {frame_style_map[frame_id]} (type={type(frame_style_map[frame_id])})")
                frame_style_map[frame_id].append(style["name"])
        # print("open ai frame extraction response")
        # print(all_styles)
        # print('openai frame mapping response')
        # print(frame_style_map)

        return all_styles, frame_style_map

    def process_all_images(self, folder_path: str) -> List[Dict[str, Any]]:
        image_paths = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        ]
        if not image_paths:
            print("⚠️ No valid image files found in the folder.")
            return []

        try:
            print(f"📂 Sending {len(image_paths)} images to OpenAI...")
            all_styles, _ = self.send_batch_images_to_openai(image_paths)
            return all_styles
        except Exception as e:
            print(f"❌ Error during OpenAI processing: {e}")
            return []

    def save_output(self, output: List[Dict[str, Any]], output_path: str = "style_output.json"):
        with open(output_path, "w") as f:
            json.dump(output, f, indent=4)
        print(f"✅ Saved {len(output)} styles to {output_path}")


# ✅ Example usage
if __name__ == "__main__":
    analyzer = StyleAnalyzer()
    results = analyzer.process_all_images("representatives")
    analyzer.save_output(results)

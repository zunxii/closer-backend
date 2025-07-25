import os
import cv2
import json
import torch
import easyocr
import open_clip
import numpy as np
from PIL import Image
from collections import defaultdict, Counter


class TextStyleAnalyzer:
    def __init__(self, frame_dir="frames", output_dir="cropped_words"):
        self.frame_dir = frame_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        self.color_groups = defaultdict(list)

        self._init_clip_model()

    def _init_clip_model(self):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.model.to(self.device)
        self.model.eval()

        style_prompt_mapping = {
            "thin": "very light weight, thin font, minimal contrast",
            "regular": "standard text, regular font weight, not bold or italic",
            "bold italic": "very heavy bolded and slanted text more than 10 degrees",
            "bold": "heavy bold font, thick and strong characters",
            "italic": "slanted text with serif curves, italic font",
            "thin italic": "very light weight and slanted, thin italic font more than 10 degrees"
        }
        self.style_labels = list(style_prompt_mapping.keys())
        style_prompts = list(style_prompt_mapping.values())

        tokens = self.tokenizer(style_prompts).to(self.device)
        with torch.no_grad():
            self.text_embeds = self.model.encode_text(tokens)
            self.text_embeds /= self.text_embeds.norm(dim=-1, keepdim=True)

    def extract_words(self):
        frame_files = sorted([f for f in os.listdir(self.frame_dir) if f.lower().endswith(('.png', '.jpg'))])
        for frame_file in frame_files:
            frame_path = os.path.join(self.frame_dir, frame_file)
            image = cv2.imread(frame_path)
            if image is None:
                continue

            results = self.reader.readtext(image)
            if len(results) > 6:
                continue

            for i, (bbox, text, conf) in enumerate(results):
                if conf < 0.4 or len(text.strip()) < 2:
                    continue

                (tl, tr, br, bl) = bbox
                x_min, y_min = int(min(tl[0], bl[0])), int(min(tl[1], tr[1]))
                x_max, y_max = int(max(tr[0], br[0])), int(max(bl[1], br[1]))

                if x_max - x_min < 5 or y_max - y_min < 5:
                    continue

                cropped = image[y_min:y_max, x_min:x_max]
                word_filename = f"{os.path.splitext(frame_file)[0]}_word{i}.png"
                cv2.imwrite(os.path.join(self.output_dir, word_filename), cropped)

    def rgb_to_color_family(self, rgb):
        r, g, b = rgb
        if r > 200 and g > 200 and b < 100:
            return 'yellow'
        elif r > 200 and g < 100 and b < 100:
            return 'red'
        elif r < 100 and g > 200 and b < 100:
            return 'green'
        elif r < 100 and g < 100 and b > 200:
            return 'blue'
        elif r > 200 and g > 200 and b > 200:
            return 'white'
        elif r < 50 and g < 50 and b < 50:
            return 'black'
        elif r > 100 and b > 100 and g < 100:
            return 'purple'
        elif r < 150 and g > 150 and b > 150:
            return 'cyan'
        elif r > 160 and g > 100 and b < 50:
            return 'orange'
        else:
            return 'others'

    def get_dominant_colors(self):
        for filename in os.listdir(self.output_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(self.output_dir, filename)
                img = Image.open(path).convert("RGB")
                pixels = img.getdata()
                color_counts = Counter(pixels)
                dominant_rgb = color_counts.most_common(1)[0][0]
                color_family = self.rgb_to_color_family(dominant_rgb)
                self.color_groups[color_family].append(filename)

    def get_stroke_thickness(self, img_gray):
        edges = cv2.Canny(img_gray, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        return np.mean(dilated)

    def get_slant_angle(self, img_gray):
        edges = cv2.Canny(img_gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
        if lines is None:
            return 0.0
        angles = [(theta - np.pi/2) * 180 / np.pi for [[_, theta]] in lines]
        valid_angles = [a for a in angles if abs(a) > 5]
        return np.median(valid_angles) if valid_angles else 0.0

    def classify_by_logic(self, thickness, slant):
        is_bold = thickness > 55
        is_italic = abs(slant) > 20
        if is_bold and is_italic:
            return "bold italic"
        elif is_bold:
            return "bold"
        elif is_italic and thickness < 10:
            return "thin italic"
        elif is_italic:
            return "italic"
        elif thickness < 8:
            return "thin"
        else:
            return "regular"

    def predict_style(self, img_path):
        img = Image.open(img_path).convert("RGB")
        im_clip = self.preprocess(img).unsqueeze(0).to(self.device)

        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        thickness = self.get_stroke_thickness(img_cv)
        slant = self.get_slant_angle(img_cv)

        logic_label = self.classify_by_logic(thickness, slant)
        with torch.no_grad():
            img_feat = self.model.encode_image(im_clip)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            sims = (img_feat @ self.text_embeds.T).squeeze(0)
        best_clip = sims.argmax().item()

        return logic_label, float(sims[best_clip].item()), img_feat.squeeze()

    def analyze_styles(self, output_detailed="final_style_groups.json", output_flat="style_representatives_flat.json"):
        self.get_dominant_colors()
        final_result = {}
        flat_result = {}

        top_colors = [c for c in self.color_groups if c != "others"]
        for color in top_colors[:3]:
            style_groups = defaultdict(list)
            for filename in self.color_groups[color]:
                img_path = os.path.join(self.output_dir, filename)
                if not os.path.exists(img_path):
                    continue
                style, score, emb = self.predict_style(img_path)
                style_groups[style].append({
                    "filename": filename,
                    "score": score,
                    "embedding": emb.cpu().tolist()
                })

            color_result = {}
            for style, items in style_groups.items():
                best_item = max(items, key=lambda x: x["score"])
                color_result[style] = {
                    "representative": best_item["filename"],
                    "count": len(items),
                    "files": [i["filename"] for i in items]
                }
                flat_result[f"{color}_{style}"] = best_item["filename"]

            final_result[color] = color_result

        with open(output_detailed, "w") as f:
            json.dump(final_result, f, indent=2)
        with open(output_flat, "w") as f:
            json.dump(flat_result, f, indent=2)

        print(f"✅ Output saved to '{output_detailed}' and '{output_flat}'.")


if __name__ == "__main__":
    analyzer = TextStyleAnalyzer(frame_dir="frames", output_dir="cropped_words")
    analyzer.extract_words()
    analyzer.analyze_styles()
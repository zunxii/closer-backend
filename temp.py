from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import cv2
import os
import base64
import json
from pathlib import Path
import re
from tqdm import tqdm
from dotenv import load_dotenv
import openai
import numpy as np
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
import subprocess
import requests
import time
import io
import soundfile as sf
from typing import Generator, List
from datetime import timedelta
import tempfile
import shutil

load_dotenv()

app = FastAPI(title="Video Subtitle Processing API", version="1.0.0")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
openai_client = openai.Client(api_key=OPENAI_API_KEY)

# ===== SCRIPT 1: Frame Extraction =====
def extract_frames(video_path: str, output_folder: str, target_fps: int = 2):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        raise ValueError("Unable to determine video FPS.")

    frame_interval = int(fps / target_fps)
    frame_num = 0
    saved_count = 0

    print(f"📸 Extracting frames from: {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % frame_interval == 0:
            frame_path = os.path.join(output_folder, f'frame_{saved_count:05d}.jpg')
            cv2.imwrite(frame_path, frame)
            saved_count += 1

        frame_num += 1

    cap.release()
    print(f"Saved {saved_count} frames to: {output_folder}")
    return saved_count

# ===== SCRIPT 2: Frame Analysis =====
def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def extract_frame_number(filename):
    match = re.search(r"(\d+)", filename)
    return int(match.group(1)) if match else 0

def clean_json_text(raw):
    return re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE)

def hex_to_ass_color(hex_color):
    hex_color = hex_color.replace("#", "")
    if len(hex_color) != 6:
        return "&HFFFFFF"
    r, g, b = hex_color[0:2], hex_color[2:4], hex_color[4:6]
    return f"&H{b}{g}{r}".upper()

def normalize_word(word_obj):
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

def generate_prompt():
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

def build_reference_styles(style_memory, max_styles=10):
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

def analyze_frames(folder: str, max_frames: int = 85):
    files = sorted(
        [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png"))],
        key=extract_frame_number,
    )[:max_frames]

    print(f"🔍 Processing {len(files)} frames from '{folder}'...")
    all_data = []
    style_memory = []

    for filename in tqdm(files, desc="Analyzing Frames"):
        image_path = os.path.join(folder, filename)
        image_b64 = image_to_base64(image_path)
        reference_styles = build_reference_styles(style_memory, max_styles=10)

        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a subtitle caption visual style extractor. Your job is to return JSON of the visible styled words in the image. Only return valid JSON with keys. Do not return any explanations or markdown.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": reference_styles},
                            {"type": "text", "text": generate_prompt()},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                        ],
                    },
                ],
                temperature=0.2,
                max_tokens=1200,
            )

            raw = response.choices[0].message.content
            cleaned = clean_json_text(raw)

            try:
                parsed = json.loads(cleaned)
                words_raw = parsed["words"] if isinstance(parsed, dict) else parsed
            except Exception as e:
                print(f"❌ JSON parse error on {filename}: {e}")
                continue

            normalized = [normalize_word(word) for word in words_raw]
            normalized = [w for w in normalized if w]

            frame_data = {"frame": filename, "words": normalized}
            all_data.append(frame_data)
            style_memory.append(frame_data)

        except Exception as e:
            print(f"❌ API error on {filename}: {e}")

    return all_data

# ===== SCRIPT 3: Filter Duplicates =====
def get_frame_text(frame):
    return " ".join(word["text"] for word in frame["words"]).strip()

def filter_duplicate_frames(all_frames):
    filtered_frames = []
    i = 0

    while i < len(all_frames):
        current = all_frames[i]
        current_text = get_frame_text(current)

        if i + 1 < len(all_frames):
            next_text = get_frame_text(all_frames[i + 1])
            if current_text and current_text in next_text:
                i += 1
                continue

        filtered_frames.append(current)
        i += 1

    print(f"Original: {len(all_frames)} → Filtered: {len(filtered_frames)}")
    return filtered_frames

# ===== SCRIPT 4: Style Clustering =====
def hex_to_rgb(ass_color):
    ass_color = ass_color.replace("&H", "").zfill(6)
    b = int(ass_color[0:2], 16)
    g = int(ass_color[2:4], 16)
    r = int(ass_color[4:6], 16)
    return [r, g, b]

def compute_visual_weight(style):
    weight = style["fontsize"]
    if style["bold"] == -1:
        weight += 20
    if style["italic"] == -1:
        weight += 5
    return weight

def cluster_styles(frames):
    style_entries = []
    style_features = []
    seen_keys = set()
    style_to_word_refs = defaultdict(list)

    for frame in frames:
        for word in frame.get("words", []):
            key = (word["fontname"], word["fontsize"], word["primary_colour"], word["bold"], word["italic"])
            style_to_word_refs[key].append(word)

            if key in seen_keys:
                continue
            seen_keys.add(key)

            rgb = hex_to_rgb(word["primary_colour"])
            vec = [
                word["fontsize"],
                int(word["bold"] != 0),
                int(word["italic"] != 0),
                *rgb
            ]

            style_entries.append(word)
            style_features.append(vec)

    if not style_features:
        return []

    features = np.array(style_features)
    scaled = MinMaxScaler().fit_transform(features)

    dbscan = DBSCAN(eps=0.3, min_samples=1)
    labels = dbscan.fit_predict(scaled)

    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[label].append((idx, scaled[idx]))

    representatives = []
    cluster_keys = []

    for label, group in clusters.items():
        center = np.mean([g[1] for g in group], axis=0)
        closest_idx = min(group, key=lambda g: np.linalg.norm(g[1] - center))[0]
        rep_word = style_entries[closest_idx]

        rep_key = (
            rep_word["fontname"],
            rep_word["fontsize"],
            rep_word["primary_colour"],
            rep_word["bold"],
            rep_word["italic"]
        )

        representatives.append(rep_word)
        cluster_keys.append([(
            style_entries[idx]["fontname"],
            style_entries[idx]["fontsize"],
            style_entries[idx]["primary_colour"],
            style_entries[idx]["bold"],
            style_entries[idx]["italic"]
        ) for idx, _ in group])

    style_counts = []
    for keys in cluster_keys:
        count = sum(len(style_to_word_refs[k]) for k in keys)
        style_counts.append(count)

    alpha = 0.7
    beta = 0.3
    min_frequency = 3

    max_count = max(style_counts) if style_counts else 1
    scored_styles = []

    for word, count in zip(representatives, style_counts):
        if count < min_frequency:
            continue

        visual_weight = compute_visual_weight(word)
        frequency_weight = count / max_count
        total_score = alpha * visual_weight + beta * frequency_weight * 100

        word = word.copy()
        word["score"] = round(total_score, 2)
        scored_styles.append(word)

    scored_styles.sort(key=lambda w: w["score"])

    ranked_styles = []
    for i, style in enumerate(scored_styles):
        ranked_styles.append({
            "name": f"style{i+1}",
            "fontname": style["fontname"],
            "fontsize": style["fontsize"],
            "primary_colour": style["primary_colour"],
            "bold": style["bold"],
            "italic": style["italic"],
            "outline": style["outline"],
            "shadow": style["shadow"]
        })

    return ranked_styles

# ===== SCRIPT 5: Audio Processing =====
def mp4_to_mp3_bytes(input_path: str) -> bytes:
    command = [
        'ffmpeg',
        '-i', input_path,
        '-f', 'mp3',
        '-acodec', 'libmp3lame',
        '-vn',
        'pipe:1'
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise Exception(f"FFmpeg error: {result.stderr.decode()}")
    return result.stdout

def read_in_chunks(data: bytes, chunk_size: int = 5242880) -> Generator[bytes, None, None]:
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

def upload_to_assemblyai(audio_bytes: bytes) -> str:
    headers = {
        "authorization": ASSEMBLYAI_API_KEY,
        "transfer-encoding": "chunked"
    }
    response = requests.post(
        "https://api.assemblyai.com/v2/upload",
        headers=headers,
        data=read_in_chunks(audio_bytes),
        stream=True
    )
    response.raise_for_status()
    return response.json()['upload_url']

def transcribe_audio_url(audio_url: str) -> List[dict]:
    headers = {
        "authorization": ASSEMBLYAI_API_KEY,
        "content-type": "application/json"
    }
    response = requests.post(
        "https://api.assemblyai.com/v2/transcript",
        headers=headers,
        json={"audio_url": audio_url, "auto_chapters": False, "iab_categories": False}
    )
    transcript_id = response.json()['id']

    while True:
        polling = requests.get(f"https://api.assemblyai.com/v2/transcript/{transcript_id}", headers=headers)
        status = polling.json()['status']
        if status == 'completed':
            return polling.json()['words']
        elif status == 'error':
            raise Exception(f"Transcription failed: {polling.json()['error']}")
        time.sleep(2)

def energy_data(audio: bytes, transcription: List[dict]) -> List[dict]:
    audio_data, sample_rate = sf.read(io.BytesIO(audio))
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    for word in transcription:
        start_sec = word["start"] / 1000.0
        end_sec = word["end"] / 1000.0
        start_sample = int(start_sec * sample_rate)
        end_sample = int(end_sec * sample_rate)
        word_audio = audio_data[start_sample:end_sample]
        energy = float(np.sqrt(np.mean(word_audio**2))) if len(word_audio) > 0 else 0.0
        word["energy"] = energy

    return transcription

def process_video_audio(video_path: str):
    audio_bytes = mp4_to_mp3_bytes(video_path)
    audio_url = upload_to_assemblyai(audio_bytes)
    words = transcribe_audio_url(audio_url)
    enhanced = energy_data(audio_bytes, words)
    return enhanced

# ===== SCRIPT 6: Sentence Splitting =====
def split_into_sentences(transcription):
    sentences = []
    current_sentence = []
    start_time = None

    for word in transcription:
        word_text = word["text"]
        if not current_sentence:
            start_time = word["start"]

        current_sentence.append({
            "text": word_text,
            "start": word["start"],
            "end": word["end"],
            "energy": word["energy"]
        })

        if word_text.endswith(('.', '!', '?')):
            sentence_text = " ".join(w["text"] for w in current_sentence)
            sentence_words = [
                {"text": w["text"], "start": w["start"], "end": w["end"], "energy": w["energy"]}
                for w in current_sentence
            ]

            sentences.append({
                "text": sentence_text,
                "start_time": start_time,
                "end_time": current_sentence[-1]["end"],
                "words": sentence_words
            })
            current_sentence = []

    if current_sentence:
        sentence_text = " ".join(w["text"] for w in current_sentence)
        sentence_words = [
            {"text": w["text"], "start": w["start"], "end": w["end"], "energy": w["energy"]}
            for w in current_sentence
        ]

        sentences.append({
            "text": sentence_text,
            "start_time": start_time,
            "end_time": current_sentence[-1]["end"],
            "words": sentence_words
        })

    return sentences

# ===== SCRIPT 7: Style Words =====
def style_words(sentences, max_style_order):
    system_prompt = f"""
You are a smart assistant that styles captions for vertical short-form videos (Reels, TikToks, Shorts).

You'll be given a sentence and a list of words. Each word has:
- text
- start time (ms)
- end time (ms)
- energy (float between 0.0 and 1.0)

Your task is to return the word list with added styling:

For each word, assign:
- "priority_value" (float between 0.000 and 1.000 three decimal places*)
- "style_order" (int between 1 and {max_style_order}, where 1 is highest visual emphasis and {max_style_order} is the lowest that represent a fallback defualt style for normal words)

Follow these strict rules:
- Use the original word order, do NOT skip or reorder any words.
- Use energy and meaning to drive emphasis.
- Filler words ("a", "the", "and", "in", etc.) get low priority (0.0–0.3) and style_order 3+.
- Impactful, emotional, or energetic words get higher values.
- Make it suitable for eye-catching video captions.
Return ONLY a valid JSON list like this:

[
  {{  
    "text": "<word>",
    "start": <start>,
    "end": <end>,
    "energy": <energy>,
    "priority_value": <float>,
    "style_order": <int>
  }}
]
DO NOT include explanations, markdown, or any formatting. Just pure JSON list of styled words.
"""

    styled_sentences = []

    for i, sentence in enumerate(sentences):
        user_prompt = f"""
Sentence: "{sentence['text']}"

Words:
{json.dumps(sentence['words'], indent=2)}
"""

        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
            )

            content = response.choices[0].message.content.strip()
            content = re.sub(r"^```(?:json)?", "", content, flags=re.MULTILINE)
            content = re.sub(r"```$", "", content, flags=re.MULTILINE)
            content = re.sub(r",\s*([}\]])", r"\1", content)

            styled_words = json.loads(content)

            styled_sentences.append({
                "text": sentence["text"],
                "start_time": sentence["start_time"],
                "end_time": sentence["end_time"],
                "words": styled_words
            })

            print(f"✅ Styled sentence {i+1}/{len(sentences)}")
            time.sleep(0.8)

        except Exception as e:
            print(f"❌ Error on sentence {i+1}: {e}")
            continue

    return styled_sentences

# ===== Apply Template Styles =====
def apply_template_styles(data, total_styles=3, threshold=0.15, default_style="style3"):
    for chunk in data:
        words = chunk["words"]
        priority_values = [w["priority_value"] for w in words]

        max_p = max(priority_values)
        min_p = min(priority_values)
        delta = max_p - min_p

        if delta > threshold:
            for word in words:
                if 1 <= word["style_order"] <= total_styles:
                    word["style"] = f"style{word['style_order']}"
                else:
                    word["style"] = default_style
        else:
            for word in words:
                word["style"] = default_style

    return data

# ===== Word Chunking =====
def chunk_words_advanced(words, max_chunk_size=5, pause_threshold=120, priority_jump=0.35):
    chunks = []
    current = []

    for i, word in enumerate(words):
        current.append(word)

        if i == len(words) - 1:
            break

        next_word = words[i + 1]
        pause = next_word["start"] - word["end"]
        prio_diff = abs(next_word["priority_value"] - word["priority_value"])

        try:
            style_current = int(word["style"].replace("style", ""))
            style_next = int(next_word["style"].replace("style", ""))
            style_diff = abs(style_next - style_current)
        except:
            style_diff = 0

        punctuation_break = word["text"][-1] in ".!?,"

        if (
            pause > pause_threshold or
            len(current) >= max_chunk_size or
            prio_diff > priority_jump or
            punctuation_break
        ):
            chunks.append(current)
            current = []

    if current:
        chunks.append(current)

    return chunks

def process_chunking(caption_sentences):
    chunked_output = []

    for sentence in caption_sentences:
        chunks = chunk_words_advanced(sentence["words"])
        for chunk in chunks:
            if not chunk:
                continue
            chunk_entry = {
                "start": chunk[0]["start"],
                "end": chunk[-1]["end"],
                "words": chunk,
                "text": " ".join(word["text"] for word in chunk)
            }
            chunked_output.append(chunk_entry)

    return chunked_output

# ===== ASS Export =====
def ms_to_ass_time(ms):
    t = timedelta(milliseconds=ms)
    return f"{int(t.total_seconds() // 3600)}:{int((t.total_seconds() % 3600) // 60):02}:{int(t.total_seconds() % 60):02}.{int(t.microseconds / 10000):02}"

def generate_ass_file(caption_sentences, raw_styles):
    styles = {style["name"].lower(): style for style in raw_styles}
    style_names = sorted(styles.keys(), key=lambda s: int(''.join(filter(str.isdigit, s))))
    lowest_style = style_names[-1] if style_names else "style1"

    ass_header = """[Script Info]
Title: Styled Subtitles
ScriptType: v4.00+
Collisions: Normal
PlayResY: 720
PlayResX: 1280
Timer: 100.0000

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, Bold, Italic, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
"""

    style_lines = [
        f"Style: {s['name']},{s['fontname']},{s['fontsize']},{s['primary_colour']},{s['bold']},{s['italic']},{s['outline']},{s['shadow']},2,10,10,10,1"
        for s in raw_styles
    ]

    event_header = """
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events = []

    for sentence in caption_sentences:
        words = sentence["words"]
        if not words:
            continue

        start_time = ms_to_ass_time(words[0]["start"])
        end_time = ms_to_ass_time(words[-1]["end"])

        styled_text = []
        for word in words:
            style_key = word["style"].lower()
            s = styles.get(style_key, styles[lowest_style])
            styled_word = (
                "{\\fn" + s["fontname"] +
                f"\\fs{s['fontsize']}\\c{s['primary_colour']}" +
                ("\\b1" if s["bold"] == -1 else "") +
                f"\\shad{s['shadow']}}}" + word["text"]
            )
            styled_text.append(styled_word)

        dialogue = f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{' '.join(styled_text)}"
        events.append(dialogue)

    return "\n".join([ass_header, *style_lines, event_header, *events])

# ===== Main API Endpoint =====
@app.post("/process-videos")
async def process_videos_direct(
    reference_video: UploadFile = File(...),
    input_video: UploadFile = File(...)
):
    """
    Process two videos and return ASS content directly without creating a file.
    """
    if not OPENAI_API_KEY or not ASSEMBLYAI_API_KEY:
        raise HTTPException(status_code=500, detail="Missing required API keys")

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Save uploaded files
        ref_video_path = os.path.join(temp_dir, "reference.mp4")
        input_video_path = os.path.join(temp_dir, "input.mp4")
        frames_dir = os.path.join(temp_dir, "frames")
        
        with open(ref_video_path, "wb") as f:
            shutil.copyfileobj(reference_video.file, f)
        
        with open(input_video_path, "wb") as f:
            shutil.copyfileobj(input_video.file, f)

        # [All the processing steps remain the same...]
        
        # Step 1: Extract frames from reference video
        print("🔄 Step 1: Extracting frames from reference video...")
        frame_count = extract_frames(ref_video_path, frames_dir, target_fps=2)
        
        # Step 2: Analyze frames for styles
        print("🔄 Step 2: Analyzing frames for styles...")
        all_frames = analyze_frames(frames_dir, max_frames=85)
        
        # Step 3: Filter duplicate frames
        print("🔄 Step 3: Filtering duplicate frames...")
        filtered_frames = filter_duplicate_frames(all_frames)
        
        # Step 4: Cluster styles
        print("🔄 Step 4: Clustering styles...")
        ranked_styles = cluster_styles(filtered_frames)
        
        if not ranked_styles:
            # Create default style if no styles found
            ranked_styles = [{
                "name": "style1",
                "fontname": "Arial",
                "fontsize": 48,
                "primary_colour": "&HFFFFFF",
                "bold": 0,
                "italic": 0,
                "outline": 1,
                "shadow": 0
            }]
        
        # Step 5: Process audio from input video
        print("🔄 Step 5: Processing audio from input video...")
        transcription = process_video_audio(input_video_path)
        
        # Step 6: Split into sentences
        print("🔄 Step 6: Splitting into sentences...")
        sentences = split_into_sentences(transcription)
        
        # Step 7: Style words
        print("🔄 Step 7: Styling words...")
        styled_sentences = style_words(sentences, len(ranked_styles))
        
        # Step 8: Apply template styles
        print("🔄 Step 8: Applying template styles...")
        styled_with_templates = apply_template_styles(
            styled_sentences,
            total_styles=len(ranked_styles),
            threshold=0.15,
            default_style=f"style{len(ranked_styles)}"
        )
        
        # Step 9: Chunk words
        print("🔄 Step 9: Chunking words...")
        chunked_output = process_chunking(styled_with_templates)
        
        # Step 10: Generate ASS file
        print("🔄 Step 10: Generating ASS file...")
        ass_content = generate_ass_file(chunked_output, ranked_styles)
        
        print("✅ Processing complete!")
        
        # Return content directly without creating a file
        print("🔄 Step 10: Generating ASS file...")
        ass_content = generate_ass_file(chunked_output, ranked_styles)

        print("✅ Processing complete!")

        from fastapi.responses import Response
        return Response(
            content=ass_content,
            media_type="application/octet-stream",
            headers={"Content-Disposition": "attachment; filename=output_subtitles.ass"}
        )
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
        except Exception as cleanup_error:
            print(f"⚠️ Warning: Could not clean up temp directory: {cleanup_error}")

@app.get("/")
async def root():
    return {"message": "Video Subtitle Processing API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



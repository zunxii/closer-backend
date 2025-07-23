from typing import List, Dict, Any
from datetime import timedelta
from config import Config
from utils.text_utils import hex_to_ass_color  # ✅ Import utility function


class SubtitleGenerator:
    def __init__(self):
        self.config = Config()

    def chunk_words_advanced(
        self,
        words: List[Dict[str, Any]],
        max_chunk_size: int = None,
        pause_threshold: int = None,
        priority_jump: float = None
    ) -> List[List[Dict[str, Any]]]:
        max_chunk_size = max_chunk_size or self.config.MAX_CHUNK_SIZE
        pause_threshold = pause_threshold or self.config.PAUSE_THRESHOLD
        priority_jump = priority_jump or self.config.PRIORITY_JUMP

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

            punctuation_break = word["text"][-1] in ".!?," if word["text"] else False

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

    def process_chunking(self, caption_sentences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunked_output = []

        for sentence in caption_sentences:
            chunks = self.chunk_words_advanced(sentence["words"])
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

    def ms_to_ass_time(self, ms: int) -> str:
        t = timedelta(milliseconds=ms)
        return f"{int(t.total_seconds() // 3600)}:{int((t.total_seconds() % 3600) // 60):02}:{int(t.total_seconds() % 60):02}.{int(t.microseconds / 10000):02}"

    def generate_ass_file(
        self,
        caption_sentences: List[Dict[str, Any]],
        raw_styles: List[Dict[str, Any]]
    ) -> str:
        styles = {style["name"].lower(): style for style in raw_styles}
        style_names = sorted(styles.keys(), key=lambda s: int(''.join(filter(str.isdigit, s))))
        lowest_style = style_names[-1] if style_names else self.config.DEFAULT_STYLE_NAME

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
            f"Style: {s['name']},{s['fontname']},{s['fontsize']},{hex_to_ass_color(s['primary_colour'])},{s['bold']},{s['italic']},{s['outline']},{s['shadow']},2,10,10,10,1"
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

            start_time = self.ms_to_ass_time(words[0]["start"])
            end_time = self.ms_to_ass_time(words[-1]["end"])

            styled_text = []
            for word in words:
                style_key = word["style"].lower()
                s = styles.get(style_key, styles[lowest_style])

                styled_word = (
                    "{\\fn" + s["fontname"] +
                    f"\\fs{s['fontsize']}\\c{hex_to_ass_color(s['primary_colour'])}" +
                    ("\\b1" if s["bold"] == -1 else "") +
                    f"\\shad{s['shadow']}}}" + word["text"]
                )
                styled_text.append(styled_word)

            dialogue = f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{' '.join(styled_text)}"
            events.append(dialogue)

        return "\n".join([ass_header, *style_lines, event_header, *events])


# Global functions
def process_chunking(caption_sentences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    generator = SubtitleGenerator()
    return generator.process_chunking(caption_sentences)

def generate_ass_file(
    caption_sentences: List[Dict[str, Any]],
    raw_styles: List[Dict[str, Any]]
) -> str:
    generator = SubtitleGenerator()
    return generator.generate_ass_file(caption_sentences, raw_styles)

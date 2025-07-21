"""
ASS subtitle generation service.
Handles word chunking, template styling, and ASS file generation.
"""

from typing import List, Dict, Any
from datetime import timedelta


def chunk_words_advanced(
    words: List[Dict[str, Any]], 
    max_chunk_size: int = 5, 
    pause_threshold: int = 120, 
    priority_jump: float = 0.35
) -> List[List[Dict[str, Any]]]:
    """
    Chunk words into subtitle segments based on timing, priority, and style changes.
    
    Args:
        words: List of word objects with styling information
        max_chunk_size: Maximum number of words per chunk
        pause_threshold: Minimum pause in ms to trigger chunk break
        priority_jump: Priority difference threshold to trigger chunk break
        
    Returns:
        List[List[Dict]]: List of word chunks
    """
    chunks = []
    current = []

    for i, word in enumerate(words):
        current.append(word)

        # Check if this is the last word
        if i == len(words) - 1:
            break

        next_word = words[i + 1]
        pause = next_word["start"] - word["end"]
        prio_diff = abs(next_word["priority_value"] - word["priority_value"])

        # Calculate style difference
        try:
            style_current = int(word["style"].replace("style", ""))
            style_next = int(next_word["style"].replace("style", ""))
            style_diff = abs(style_next - style_current)
        except:
            style_diff = 0

        # Check for punctuation break
        punctuation_break = word["text"][-1] in ".!?,"

        # Determine if we should break the chunk
        if (
            pause > pause_threshold or
            len(current) >= max_chunk_size or
            prio_diff > priority_jump or
            punctuation_break
        ):
            chunks.append(current)
            current = []

    # Add any remaining words
    if current:
        chunks.append(current)

    return chunks


def process_chunking(caption_sentences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process all sentences and chunk their words into subtitle segments.
    
    Args:
        caption_sentences: List of sentence objects with styled words
        
    Returns:
        List[Dict]: List of chunked subtitle segments
    """
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


def ms_to_ass_time(ms: int) -> str:
    """
    Convert milliseconds to ASS timestamp format.
    
    Args:
        ms: Time in milliseconds
        
    Returns:
        str: ASS formatted timestamp (H:MM:SS.CS)
    """
    t = timedelta(milliseconds=ms)
    return f"{int(t.total_seconds() // 3600)}:{int((t.total_seconds() % 3600) // 60):02}:{int(t.total_seconds() % 60):02}.{int(t.microseconds / 10000):02}"


def generate_ass_file(
    caption_sentences: List[Dict[str, Any]], 
    raw_styles: List[Dict[str, Any]]
) -> str:
    """
    Generate complete ASS subtitle file content.
    
    Args:
        caption_sentences: List of chunked subtitle segments
        raw_styles: List of style definitions
        
    Returns:
        str: Complete ASS file content
    """
    # Create style lookup dictionary
    styles = {style["name"].lower(): style for style in raw_styles}
    style_names = sorted(styles.keys(), key=lambda s: int(''.join(filter(str.isdigit, s))))
    lowest_style = style_names[-1] if style_names else "style1"

    # ASS file header
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

    # Generate style definitions
    style_lines = [
        f"Style: {s['name']},{s['fontname']},{s['fontsize']},{s['primary_colour']},{s['bold']},{s['italic']},{s['outline']},{s['shadow']},2,10,10,10,1"
        for s in raw_styles
    ]

    # Events section header
    event_header = """
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    # Generate dialogue events
    events = []

    for sentence in caption_sentences:
        words = sentence["words"]
        if not words:
            continue

        start_time = ms_to_ass_time(words[0]["start"])
        end_time = ms_to_ass_time(words[-1]["end"])

        # Build styled text for each word
        styled_text = []
        for word in words:
            style_key = word["style"].lower()
            s = styles.get(style_key, styles[lowest_style])
            
            # Create inline style override for each word
            styled_word = (
                "{\\fn" + s["fontname"] +
                f"\\fs{s['fontsize']}\\c{s['primary_colour']}" +
                ("\\b1" if s["bold"] == -1 else "") +
                f"\\shad{s['shadow']}}}" + word["text"]
            )
            styled_text.append(styled_word)

        # Create dialogue line
        dialogue = f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{' '.join(styled_text)}"
        events.append(dialogue)

    # Combine all parts
    return "\n".join([ass_header, *style_lines, event_header, *events])
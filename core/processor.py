import json
import os
from typing import List, Dict, Any
from config import Config
from utils.file_utils import (
    create_temp_directory,
    cleanup_temp_directory,
    ensure_directory_exists,
    save_uploaded_file
)

from services.frame_extractor import FrameExtractor
from services.duplicate_filter import DuplicateFilter
from services.style_clusterer import StyleClusterer
from services.audio_processor import AudioProcessor
from services.sentence_splitter import SentenceSplitter
from services.word_styler import WordStyler
from services.ass_generator import SubtitleGenerator
from services.instagram_downloader import InstagramVideoDownloader
from services.text_style_analyzer import TextStyleAnalyzer
from services.style_analyzer import StyleAnalyzer
from services.clip_selector import StyleWindowFinder


class VideoSubtitleProcessor:
    def __init__(self):
        self.temp_dir = create_temp_directory()
        self.frames_dir = os.path.join(self.temp_dir, "frames")
        self.cropped_words_dir = os.path.join(self.temp_dir, "cropped_words")
        self.representative_dir = os.path.join(self.temp_dir, "representatives")
        ensure_directory_exists(self.frames_dir)
        ensure_directory_exists(self.cropped_words_dir)
        ensure_directory_exists(self.representative_dir)

        # Load font descriptions
        raw_fonts = Config.FONT_DESCRIPTION.get("fonts", [])
        self.font_descriptions = {font["name"]: font["prompt"] for font in raw_fonts}

        # Services
        self.frame_extractor = FrameExtractor()
        self.duplicate_filter = DuplicateFilter()
        self.style_clusterer = StyleClusterer()
        self.audio_processor = AudioProcessor()
        self.sentence_splitter = SentenceSplitter()
        self.word_styler = WordStyler()
        self.subtitle_generator = SubtitleGenerator()
        self.instagram_downloader = InstagramVideoDownloader()
        self.text_style_analyzer = TextStyleAnalyzer(
            frame_dir=self.frames_dir,
            output_dir=self.cropped_words_dir
        )
        self.openai_analyzer = StyleAnalyzer()
        self.window_finder = StyleWindowFinder()

    def _download_reference_video(self, reference_url: str) -> str:
        path = os.path.join(self.temp_dir, "reference.mp4")
        print(" Step 0: Downloading reference video...")
        self.instagram_downloader.download(reference_url, path)
        return path

    def _save_input_video(self, input_video) -> str:
        path = os.path.join(self.temp_dir, "input.mp4")
        save_uploaded_file(input_video, path)
        return path

    def _extract_and_analyze_text_styles(self) -> List[Dict[str, Any]]:
        print(" Step: Analyzing styles from extracted frames...")

        frame_files = sorted([
            f for f in os.listdir(self.frames_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])[:5]  # Only process up to 5 frames

        image_paths = [os.path.join(self.frames_dir, fname) for fname in frame_files]

        try:
            result = self.openai_analyzer.send_batch_images_to_openai(image_paths)

            # Defensive unpacking and validation
            if not result or not isinstance(result, tuple) or len(result) != 2:
                raise ValueError("Style analysis returned invalid result. Expected (styles, frame_style_map)")

            styles, frame_style_map = result

            if not styles or not isinstance(styles, list):
                raise ValueError("Invalid or empty styles returned.")

            if not frame_style_map or not isinstance(frame_style_map, dict):
                raise ValueError("Invalid or empty frame_style_map returned.")

            print(f" Extracted {len(styles)} styles from {len(frame_style_map)} frames.")

            # Optionally save the response
            with open("style_clusters.json", "w") as f:
                json.dump(frame_style_map, f, indent=4)

            return styles, frame_style_map

        except Exception as e:
            print(f"❌ Failed to analyze styles in batch: {e}")
            return [], {}



    def _extract_styles_from_reference(self, ref_video_path: str) -> List[Dict[str, Any]]:
        print("🎬 Step 1: Extracting styles from reference video...")

        # Audio and sentence splitting
        ref_sentences = self._process_audio(ref_video_path)

        # Style assignment and chunking
        ref_chunks = self._style_and_chunk_words(ref_sentences, 4)

        # Debug: Print the structure of ref_chunks
        print(f" Debug: ref_chunks length: {len(ref_chunks)}")
        if ref_chunks:
            print(f" Debug: First chunk structure: {ref_chunks[0]}")
            print(f" Debug: First chunk keys: {list(ref_chunks[0].keys())}")
        
        #  Finding a window that has all 4 styles
        # Define the target styles you're looking for (1, 2, 3, 4 based on style_order)
        target_styles = list(range(1, 5))  # [1, 2, 3, 4]
        
        style_window = self.window_finder.find_window(ref_chunks, target_styles)

        if not style_window:
            raise Exception("❌ Could not find a window with all style orders present in reference video.")

        start_time_ms, end_time_ms = style_window
        # Convert from milliseconds to seconds
        start_time = start_time_ms / 1000.0
        end_time = end_time_ms / 1000.0
        print(f" Style window found from {start_time}s to {end_time}s")

        # Frame extraction
        self.frame_extractor.extract_frames(
            ref_video_path,
            self.frames_dir,
            start_time=start_time,
            end_time=end_time
        )

        # Style analysis
        font_analysis_results,  frame_words_map  = self._extract_and_analyze_text_styles()

        ranked_styles, updated_frame_words_map = self.style_clusterer.cluster_styles(
            font_analysis_results,
            frame_words_map
        )

        return ranked_styles, updated_frame_words_map

    def _create_default_styles(self) -> List[Dict[str, Any]]:
        return [{
            "name": "style1",
            "fontname": "Arial",
            "fontsize": 48,
            "primary_colour": "&HFFFFFF",
            "bold": 0,
            "italic": 0,
            "outline": 1,
            "shadow": 0
        }]

    def _process_audio(self, input_video_path: str) -> List[Dict[str, Any]]:
        print(" Step 6: Transcribing audio...")
        transcription = self.audio_processor.process_video_audio(input_video_path)
        print(" Step 7: Splitting transcription into sentences...")
        return self.sentence_splitter.split_into_sentences(transcription)

    def _style_and_chunk_words(self, sentences, num_styles) -> List[Dict[str, Any]]:
        print(" Step 8: Assigning word styles...")
        styled_sentences = self.word_styler.style_words(sentences, num_styles)

        styled_with_templates = self.word_styler.apply_template_styles(
            styled_sentences,
            total_styles=num_styles,
            threshold=Config.STYLE_THRESHOLD,
            default_style=Config.DEFAULT_STYLE_NAME
        )

        print(" Step 9: Chunking styled sentences...")
        return self.subtitle_generator.process_chunking(styled_with_templates)
    
    def _style_and_chunk_words_input(self, sentences, frame_word_map) -> List[Dict[str, Any]]:
        print(" Step 8: Assigning word styles...")
        styled_sentences = self.word_styler.style_words_using_map(sentences, frame_word_map)

        styled_with_templates = self.word_styler.apply_template_styles(
            styled_sentences,
            total_styles=4,
            threshold=Config.STYLE_THRESHOLD,
            default_style=Config.DEFAULT_STYLE_NAME
        )

        print(" Step 9: Chunking styled sentences...")
        return self.subtitle_generator.process_chunking(styled_with_templates)

    def process(self, reference_url: str, input_video) -> str:
        try:
            ref_video_path = self._download_reference_video(reference_url)
            input_video_path = self._save_input_video(input_video)

            ranked_styles, updated_frame_words_map = self._extract_styles_from_reference(ref_video_path)


            if not ranked_styles:
                print("⚠️ No styles found — using fallback.")
                ranked_styles = self._create_default_styles()
            print(ranked_styles)


            sentences = self._process_audio(input_video_path)
            chunked_output = self._style_and_chunk_words_input(sentences, updated_frame_words_map)

            print(" Step 10: Generating ASS subtitle file...")
            ass_content = self.subtitle_generator.generate_ass_file(chunked_output, ranked_styles)

            print("✅ Subtitle generation complete.")
            return ass_content

        except Exception as e:
            print(f"❌ Error during processing: {e}")
            raise

        finally:
            print(" Cleaning up temp directory...")
            cleanup_temp_directory(self.temp_dir)
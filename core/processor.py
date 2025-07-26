import os
from typing import List, Dict, Any
import json
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
        font_descriptions = {font["name"]: font["prompt"] for font in raw_fonts}

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
        """Run text extraction, style classification, and OpenAI analysis."""
        print(" Step 2: Extracting text from frames...")
        self.text_style_analyzer.extract_words()

        print(" Step 3: Analyzing styles and grouping by color + logic...")
        detailed_path = os.path.join(self.temp_dir, "final_style_groups.json")
        flat_path = os.path.join(self.temp_dir, "style_representatives_flat.json")
        self.text_style_analyzer.analyze_styles(output_detailed=detailed_path, output_flat=flat_path)

        print(" Step 4: Sending representative styles to OpenAI for analysis...")
        with open(flat_path, "r") as f:
            rep_dict = json.load(f)

        openai_results = {}
        for name, relative_path in rep_dict.items():
            image_path = os.path.join(self.cropped_words_dir, relative_path)
            try:
                result = self.openai_analyzer.send_image_to_openai(image_path)
                openai_results[name] = result
            except Exception as e:
                print(f"Error analyzing image {name}: {e}")

        return list(openai_results.values())

    def _extract_styles_from_reference(self, ref_video_path: str) -> List[Dict[str, Any]]:
        print(" Step 1: Extracting frames from reference video...")
        self.frame_extractor.extract_frames(ref_video_path, self.frames_dir)

        font_analysis_results = self._extract_and_analyze_text_styles()

        print(" Step 5: Clustering styles...")
        return self.style_clusterer.cluster_styles(font_analysis_results)

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
        print(" Step 6: Extracting audio transcription...")
        transcription = self.audio_processor.process_video_audio(input_video_path)

        print(" Step 7: Splitting into sentences...")
        return self.sentence_splitter.split_into_sentences(transcription)

    def _style_and_chunk_words(self, sentences, num_styles) -> List[Dict[str, Any]]:
        print(" Step 8: Styling words with reference templates...")
        styled_sentences = self.word_styler.style_words(sentences, num_styles)

        styled_with_templates = self.word_styler.apply_template_styles(
            styled_sentences,
            total_styles=num_styles,
            threshold=Config.STYLE_THRESHOLD,
            default_style=Config.DEFAULT_STYLE_NAME
        )

        print(" Step 9: Chunking styled words...")
        return self.subtitle_generator.process_chunking(styled_with_templates)

    def process(self, reference_url: str, input_video) -> str:
        try:
            ref_video_path = self._download_reference_video(reference_url)
            input_video_path = self._save_input_video(input_video)

            ranked_styles = self._extract_styles_from_reference(ref_video_path)
            if not ranked_styles:
                print("⚠️ No styles found — falling back to default.")
                ranked_styles = self._create_default_styles()

            sentences = self._process_audio(input_video_path)
            chunked_output = self._style_and_chunk_words(sentences, len(ranked_styles))

            print(" Step 10: Generating subtitle file...")
            ass_content = self.subtitle_generator.generate_ass_file(chunked_output, ranked_styles)

            print("✅ Subtitle generation complete.")
            return ass_content

        except Exception as e:
            print(f"❌ Error during processing: {e}")
            raise Exception(f"Processing failed: {str(e)}")

        finally:
            cleanup_temp_directory(self.temp_dir)

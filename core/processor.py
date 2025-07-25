import os
from typing import List, Dict, Any

from config import Config
from services.local_frame_analyzer import LocalFrameAnalyzer
from utils.file_utils import (
    create_temp_directory,
    cleanup_temp_directory,
    ensure_directory_exists,
    save_uploaded_file
)

from services.frame_extractor import FrameExtractor
# from services.frame_analyzer import FrameAnalyzer
from services.duplicate_filter import DuplicateFilter
from services.style_clusterer import StyleClusterer
from services.audio_processor import AudioProcessor
from services.sentence_splitter import SentenceSplitter
from services.word_styler import WordStyler
from services.ass_generator import SubtitleGenerator
from services.instagram_downloader import InstagramVideoDownloader
from services.text_style_analyzer import TextStyleAnalyzer
from services.font_analyzer import FontAnalyzer


class VideoSubtitleProcessor:
    def __init__(self):
        self.temp_dir = create_temp_directory()
        self.frames_dir = os.path.join(self.temp_dir, "frames")
        self.cropped_words_dir = os.path.join(self.temp_dir, "cropped_words")
        ensure_directory_exists(self.frames_dir)
        ensure_directory_exists(self.cropped_words_dir)

        # Load font descriptions from config
        raw_fonts = Config.FONT_DESCRIPTION.get("fonts", [])
        font_descriptions = {font["name"]: font["prompt"] for font in raw_fonts}

        # Initialize service classes
        self.frame_extractor = FrameExtractor()
        self.frame_analyzer = LocalFrameAnalyzer(font_descriptions) 
        self.duplicate_filter = DuplicateFilter()
        self.style_clusterer = StyleClusterer()
        self.audio_processor = AudioProcessor()
        self.sentence_splitter = SentenceSplitter()
        self.word_styler = WordStyler()
        self.subtitle_generator = SubtitleGenerator()
        self.instagram_downloader = InstagramVideoDownloader()
        
        # Initialize new analyzers
        self.text_style_analyzer = TextStyleAnalyzer(
            frame_dir=self.frames_dir, 
            output_dir=self.cropped_words_dir
        )
        # Initialize FontAnalyzer with API key from config
        openai_api_key = getattr(Config, 'OPENAI_API_KEY', None)
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in config")
        self.font_analyzer = FontAnalyzer(openai_api_key)


    def _download_reference_video(self, reference_url: str) -> str:
        ref_path = os.path.join(self.temp_dir, "reference.mp4")
        print(" Step 0: Downloading reference video from Instagram...")
        self.instagram_downloader.download(reference_url, ref_path)
        return ref_path

    def _save_input_video(self, input_video) -> str:
        input_path = os.path.join(self.temp_dir, "input.mp4")
        save_uploaded_file(input_video, input_path)
        return input_path

    def _extract_and_analyze_text_styles(self) -> Dict[str, Any]:
        """Extract words from frames and analyze their styles"""
        print(" Step 2: Extracting text from frames...")
        self.text_style_analyzer.extract_words()
        
        print(" Step 3: Analyzing text styles with OpenAI...")
        font_analysis_results = self.font_analyzer.analyze_folder(self.cropped_words_dir)
        
        return font_analysis_results

    def _extract_styles_from_reference(self, ref_video_path: str) -> List[Dict[str, Any]]:
        print(" Step 1: Extracting frames from reference video...")
        self.frame_extractor.extract_frames(ref_video_path, self.frames_dir)

        # New integrated steps 2 and 3
        font_analysis_results = self._extract_and_analyze_text_styles()

        print(" Step 4: Clustering styles...")
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
        print(" Step 5: Processing audio from input video...")
        transcription = self.audio_processor.process_video_audio(input_video_path)

        print(" Step 6: Splitting into sentences...")
        return self.sentence_splitter.split_into_sentences(transcription)

    def _style_and_chunk_words(self, sentences, num_styles) -> List[Dict[str, Any]]:
        print(" Step 7: Styling words...")
        styled_sentences = self.word_styler.style_words(sentences, num_styles)

        print(" Step 8: Applying template styles...")
        styled_with_templates = self.word_styler.apply_template_styles(
            styled_sentences,
            total_styles=num_styles,
            threshold=Config.STYLE_THRESHOLD,
            default_style=Config.DEFAULT_STYLE_NAME
        )

        print(" Step 9: Chunking words...")
        return self.subtitle_generator.process_chunking(styled_with_templates)

    def process(self, reference_url: str, input_video) -> str:
        try:
            ref_video_path = self._download_reference_video(reference_url)
            input_video_path = self._save_input_video(input_video)

            ranked_styles = self._extract_styles_from_reference(ref_video_path)
            if not ranked_styles:
                ranked_styles = self._create_default_styles()

            sentences = self._process_audio(input_video_path)
            chunked_output = self._style_and_chunk_words(sentences, len(ranked_styles))

            print(" Step 10: Generating ASS file...")
            ass_content = self.subtitle_generator.generate_ass_file(chunked_output, ranked_styles)

            print("✅ Processing complete!")
            return ass_content

        except Exception as e:
            print(f"❌ Error during processing: {e}")
            raise Exception(f"Processing failed: {str(e)}")

        finally:
            cleanup_temp_directory(self.temp_dir)
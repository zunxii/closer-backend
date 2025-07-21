"""
Main orchestrator class for video subtitle processing.
Coordinates all processing steps and manages the complete pipeline.
"""

import os
import tempfile
import shutil
from typing import List, Dict, Any

from services.frame_extractor import extract_frames
from services.frame_analyzer import analyze_frames
from services.duplicate_filter import filter_duplicate_frames
from services.style_clusterer import cluster_styles
from services.audio_processor import process_video_audio
from services.sentence_splitter import split_into_sentences
from services.word_styler import style_words, apply_template_styles
from services.ass_generator import process_chunking, generate_ass_file


class VideoSubtitleProcessor:
    """
    Main processor class that orchestrates the complete video subtitle processing pipeline.
    """
    
    def __init__(self):
        """Initialize the processor."""
        self.temp_dir = None
        self.frames_dir = None
        
    def _setup_temp_directory(self) -> str:
        """
        Create temporary directory for processing.
        
        Returns:
            str: Path to temporary directory
        """
        self.temp_dir = tempfile.mkdtemp()
        self.frames_dir = os.path.join(self.temp_dir, "frames")
        return self.temp_dir
    
    def _cleanup_temp_directory(self):
        """Clean up temporary directory."""
        if self.temp_dir:
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as cleanup_error:
                print(f"⚠️ Warning: Could not clean up temp directory: {cleanup_error}")
    
    def _save_uploaded_files(self, reference_video, input_video) -> tuple:
        """
        Save uploaded files to temporary directory.
        
        Args:
            reference_video: Reference video file object
            input_video: Input video file object
            
        Returns:
            tuple: (reference_video_path, input_video_path)
        """
        ref_video_path = os.path.join(self.temp_dir, "reference.mp4")
        input_video_path = os.path.join(self.temp_dir, "input.mp4")
        
        with open(ref_video_path, "wb") as f:
            shutil.copyfileobj(reference_video.file, f)
        
        with open(input_video_path, "wb") as f:
            shutil.copyfileobj(input_video.file, f)
            
        return ref_video_path, input_video_path
    
    def _extract_styles_from_reference(self, ref_video_path: str) -> List[Dict[str, Any]]:
        """
        Extract styles from reference video.
        
        Args:
            ref_video_path: Path to reference video
            
        Returns:
            List[Dict]: List of extracted styles
        """
        print("🔄 Step 1: Extracting frames from reference video...")
        frame_count = extract_frames(ref_video_path, self.frames_dir, target_fps=2)
        
        print("🔄 Step 2: Analyzing frames for styles...")
        all_frames = analyze_frames(self.frames_dir, max_frames=85)
        
        print("🔄 Step 3: Filtering duplicate frames...")
        filtered_frames = filter_duplicate_frames(all_frames)
        
        print("🔄 Step 4: Clustering styles...")
        ranked_styles = cluster_styles(filtered_frames)
        
        return ranked_styles
    
    def _create_default_styles(self) -> List[Dict[str, Any]]:
        """
        Create default styles if no styles are extracted.
        
        Returns:
            List[Dict]: List of default styles
        """
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
    
    def _process_input_video_audio(self, input_video_path: str) -> List[Dict[str, Any]]:
        """
        Process audio from input video.
        
        Args:
            input_video_path: Path to input video
            
        Returns:
            List[Dict]: Enhanced transcription with energy values
        """
        print("🔄 Step 5: Processing audio from input video...")
        transcription = process_video_audio(input_video_path)
        
        print("🔄 Step 6: Splitting into sentences...")
        sentences = split_into_sentences(transcription)
        
        return sentences
    
    def _style_and_chunk_words(
        self, 
        sentences: List[Dict[str, Any]], 
        num_styles: int
    ) -> List[Dict[str, Any]]:
        """
        Style words and chunk them for subtitles.
        
        Args:
            sentences: List of sentence objects
            num_styles: Number of available styles
            
        Returns:
            List[Dict]: Chunked subtitle segments
        """
        print("🔄 Step 7: Styling words...")
        styled_sentences = style_words(sentences, num_styles)
        
        print("🔄 Step 8: Applying template styles...")
        styled_with_templates = apply_template_styles(
            styled_sentences,
            total_styles=num_styles,
            threshold=0.15,
            default_style=f"style{num_styles}"
        )
        
        print("🔄 Step 9: Chunking words...")
        chunked_output = process_chunking(styled_with_templates)
        
        return chunked_output
    
    def process_videos(self, reference_video, input_video) -> str:
        """
        Process two videos and return ASS subtitle content.
        
        Args:
            reference_video: Reference video file object for style extraction
            input_video: Input video file object for transcription
            
        Returns:
            str: ASS subtitle file content
            
        Raises:
            Exception: If processing fails
        """
        try:
            # Setup temporary directory
            self._setup_temp_directory()
            
            # Save uploaded files
            ref_video_path, input_video_path = self._save_uploaded_files(
                reference_video, input_video
            )
            
            # Extract styles from reference video
            ranked_styles = self._extract_styles_from_reference(ref_video_path)
            
            # Create default styles if none found
            if not ranked_styles:
                ranked_styles = self._create_default_styles()
            
            # Process input video audio
            sentences = self._process_input_video_audio(input_video_path)
            
            # Style and chunk words
            chunked_output = self._style_and_chunk_words(sentences, len(ranked_styles))
            
            # Generate ASS file
            print("🔄 Step 10: Generating ASS file...")
            ass_content = generate_ass_file(chunked_output, ranked_styles)
            
            print("✅ Processing complete!")
            return ass_content
            
        except Exception as e:
            print(f"❌ Error during processing: {e}")
            raise Exception(f"Processing failed: {str(e)}")
        
        finally:
            # Clean up temporary directory
            self._cleanup_temp_directory()
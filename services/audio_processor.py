"""
Audio processing service for video subtitle generation.
Handles audio extraction, transcription, and energy analysis.
"""

import os
import subprocess
import requests
import time
import io
import soundfile as sf
import numpy as np
from typing import Generator, List, Dict, Any
from config import Config


class AudioProcessor:
    """Handles audio processing operations for video files."""
    
    def __init__(self):
        self.assemblyai_api_key = Config.ASSEMBLYAI_API_KEY
        self.upload_endpoint = "https://api.assemblyai.com/v2/upload"
        self.transcript_endpoint = "https://api.assemblyai.com/v2/transcript"
        self.chunk_size = 5242880  # 5MB chunks
    
    def extract_audio_from_video(self, video_path: str) -> bytes:
        """
        Extract audio from video file and return as MP3 bytes.
        
        Args:
            video_path: Path to the input video file
            
        Returns:
            bytes: Audio data as MP3 bytes
            
        Raises:
            Exception: If FFmpeg extraction fails
        """
        command = [
            'ffmpeg',
            '-i', video_path,
            '-f', 'mp3',
            '-acodec', 'libmp3lame',
            '-vn',  # No video
            'pipe:1'  # Output to stdout
        ]
        
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            raise Exception(f"FFmpeg audio extraction failed: {e.stderr.decode()}")
    
    def _read_in_chunks(self, data: bytes) -> Generator[bytes, None, None]:
        """
        Read audio data in chunks for streaming upload.
        
        Args:
            data: Audio data bytes
            
        Yields:
            bytes: Chunks of audio data
        """
        for i in range(0, len(data), self.chunk_size):
            yield data[i:i + self.chunk_size]
    
    def upload_audio_to_assemblyai(self, audio_bytes: bytes) -> str:
        """
        Upload audio bytes to AssemblyAI and get upload URL.
        
        Args:
            audio_bytes: Audio data as bytes
            
        Returns:
            str: Upload URL for transcription
            
        Raises:
            Exception: If upload fails
        """
        headers = {
            "authorization": self.assemblyai_api_key,
            "transfer-encoding": "chunked"
        }
        
        try:
            response = requests.post(
                self.upload_endpoint,
                headers=headers,
                data=self._read_in_chunks(audio_bytes),
                stream=True
            )
            response.raise_for_status()
            return response.json()['upload_url']
        except requests.RequestException as e:
            raise Exception(f"Failed to upload audio to AssemblyAI: {str(e)}")
    
    def transcribe_audio(self, audio_url: str) -> List[Dict[str, Any]]:
        """
        Transcribe audio using AssemblyAI API.
        
        Args:
            audio_url: URL of uploaded audio file
            
        Returns:
            List[Dict]: List of transcribed words with timestamps
            
        Raises:
            Exception: If transcription fails
        """
        headers = {
            "authorization": self.assemblyai_api_key,
            "content-type": "application/json"
        }
        
        # Start transcription
        transcript_request = {
            "audio_url": audio_url,
            "auto_chapters": False,
            "iab_categories": False
        }
        
        try:
            response = requests.post(
                self.transcript_endpoint,
                headers=headers,
                json=transcript_request
            )
            response.raise_for_status()
            transcript_id = response.json()['id']
            
            # Poll for completion
            while True:
                polling_response = requests.get(
                    f"{self.transcript_endpoint}/{transcript_id}",
                    headers=headers
                )
                polling_response.raise_for_status()
                
                result = polling_response.json()
                status = result['status']
                
                if status == 'completed':
                    return result['words']
                elif status == 'error':
                    raise Exception(f"Transcription failed: {result.get('error', 'Unknown error')}")
                
                time.sleep(2)  # Wait before next poll
                
        except requests.RequestException as e:
            raise Exception(f"Transcription request failed: {str(e)}")
    
    def calculate_energy_levels(self, audio_bytes: bytes, transcription: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Calculate energy levels for each word in the transcription.
        
        Args:
            audio_bytes: Original audio data
            transcription: List of transcribed words with timestamps
            
        Returns:
            List[Dict]: Enhanced transcription with energy levels
        """
        try:
            # Load audio data
            audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
            
            # Convert to mono if stereo
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Calculate energy for each word
            enhanced_transcription = []
            
            for word in transcription:
                # Convert milliseconds to seconds
                start_sec = word["start"] / 1000.0
                end_sec = word["end"] / 1000.0
                
                # Convert to sample indices
                start_sample = int(start_sec * sample_rate)
                end_sample = int(end_sec * sample_rate)
                
                # Extract word audio segment
                word_audio = audio_data[start_sample:end_sample]
                
                # Calculate RMS energy
                if len(word_audio) > 0:
                    energy = float(np.sqrt(np.mean(word_audio**2)))
                else:
                    energy = 0.0
                
                # Add energy to word data
                enhanced_word = word.copy()
                enhanced_word["energy"] = energy
                enhanced_transcription.append(enhanced_word)
            
            return enhanced_transcription
            
        except Exception as e:
            # If energy calculation fails, return original transcription with zero energy
            print(f"Warning: Energy calculation failed: {e}")
            return [
                {**word, "energy": 0.0} for word in transcription
            ]
    
    def process_video_audio(self, video_path: str) -> List[Dict[str, Any]]:
        """
        Complete audio processing pipeline for a video file.
        
        Args:
            video_path: Path to the input video file
            
        Returns:
            List[Dict]: Enhanced transcription with energy levels
            
        Raises:
            Exception: If any step in the pipeline fails
        """
        print(f"🔄 Extracting audio from video: {video_path}")
        audio_bytes = self.extract_audio_from_video(video_path)
        
        print("🔄 Uploading audio to AssemblyAI...")
        audio_url = self.upload_audio_to_assemblyai(audio_bytes)
        
        print("🔄 Transcribing audio...")
        transcription = self.transcribe_audio(audio_url)
        
        print("🔄 Calculating energy levels...")
        enhanced_transcription = self.calculate_energy_levels(audio_bytes, transcription)
        
        print(f"✅ Audio processing complete. Found {len(enhanced_transcription)} words.")
        return enhanced_transcription
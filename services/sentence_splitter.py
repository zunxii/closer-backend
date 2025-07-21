"""
Sentence splitting service for video subtitle generation.
Splits transcribed words into logical sentences for better subtitle formatting.
"""

from typing import List, Dict, Any, Optional


class SentenceSplitter:
    """Handles splitting transcribed words into logical sentences."""
    
    def __init__(self):
        self.sentence_endings = {'.', '!', '?'}
        self.pause_indicators = {',', ';', ':'}
    
    def split_into_sentences(self, transcription: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split transcribed words into logical sentences.
        
        Args:
            transcription: List of transcribed words with timestamps and energy
            
        Returns:
            List[Dict]: List of sentences with their constituent words
        """
        if not transcription:
            return []
        
        sentences = []
        current_sentence = []
        start_time = None
        
        for word in transcription:
            word_text = word["text"]
            
            # Initialize start time for new sentence
            if not current_sentence:
                start_time = word["start"]
            
            # Add word to current sentence
            current_sentence.append({
                "text": word_text,
                "start": word["start"],
                "end": word["end"],
                "energy": word.get("energy", 0.0)
            })
            
            # Check if word ends a sentence
            if self._is_sentence_ending(word_text):
                sentence = self._create_sentence_object(current_sentence, start_time)
                sentences.append(sentence)
                current_sentence = []
                start_time = None
        
        # Handle remaining words that don't end with punctuation
        if current_sentence:
            sentence = self._create_sentence_object(current_sentence, start_time)
            sentences.append(sentence)
        
        return sentences
    
    def _is_sentence_ending(self, word_text: str) -> bool:
        """
        Check if a word ends a sentence based on punctuation.
        
        Args:
            word_text: The word text to check
            
        Returns:
            bool: True if word ends a sentence
        """
        return any(word_text.endswith(ending) for ending in self.sentence_endings)
    
    def _create_sentence_object(self, words: List[Dict[str, Any]], start_time: int) -> Dict[str, Any]:
        """
        Create a sentence object from a list of words.
        
        Args:
            words: List of words in the sentence
            start_time: Start time of the sentence
            
        Returns:
            Dict: Sentence object with metadata
        """
        if not words:
            return {
                "text": "",
                "start_time": start_time or 0,
                "end_time": start_time or 0,
                "words": []
            }
        
        sentence_text = " ".join(word["text"] for word in words)
        end_time = words[-1]["end"]
        
        return {
            "text": sentence_text,
            "start_time": start_time,
            "end_time": end_time,
            "words": words
        }
    
    def get_sentence_statistics(self, sentences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about the sentences.
        
        Args:
            sentences: List of sentence objects
            
        Returns:
            Dict: Statistics about the sentences
        """
        if not sentences:
            return {
                "total_sentences": 0,
                "total_words": 0,
                "average_words_per_sentence": 0,
                "total_duration_ms": 0,
                "average_sentence_duration_ms": 0
            }
        
        total_sentences = len(sentences)
        total_words = sum(len(sentence["words"]) for sentence in sentences)
        total_duration = sum(
            sentence["end_time"] - sentence["start_time"] 
            for sentence in sentences
        )
        
        return {
            "total_sentences": total_sentences,
            "total_words": total_words,
            "average_words_per_sentence": total_words / total_sentences,
            "total_duration_ms": total_duration,
            "average_sentence_duration_ms": total_duration / total_sentences
        }
    
    def merge_short_sentences(self, sentences: List[Dict[str, Any]], min_duration_ms: int = 1000) -> List[Dict[str, Any]]:
        """
        Merge sentences that are too short to be displayed effectively.
        
        Args:
            sentences: List of sentence objects
            min_duration_ms: Minimum duration for a sentence in milliseconds
            
        Returns:
            List[Dict]: Merged sentences
        """
        if not sentences:
            return []
        
        merged_sentences = []
        current_merge = None
        
        for sentence in sentences:
            duration = sentence["end_time"] - sentence["start_time"]
            
            if duration < min_duration_ms and current_merge is None:
                # Start a new merge
                current_merge = sentence.copy()
            elif duration < min_duration_ms and current_merge is not None:
                # Continue merging
                current_merge = self._merge_two_sentences(current_merge, sentence)
            else:
                # Sentence is long enough or we need to finalize merge
                if current_merge is not None:
                    merged_sentences.append(current_merge)
                    current_merge = None
                merged_sentences.append(sentence)
        
        # Handle any remaining merge
        if current_merge is not None:
            merged_sentences.append(current_merge)
        
        return merged_sentences
    
    def _merge_two_sentences(self, sentence1: Dict[str, Any], sentence2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two sentences into one.
        
        Args:
            sentence1: First sentence
            sentence2: Second sentence
            
        Returns:
            Dict: Merged sentence
        """
        merged_text = f"{sentence1['text']} {sentence2['text']}"
        merged_words = sentence1["words"] + sentence2["words"]
        
        return {
            "text": merged_text,
            "start_time": sentence1["start_time"],
            "end_time": sentence2["end_time"],
            "words": merged_words
        }
    
    def split_long_sentences(self, sentences: List[Dict[str, Any]], max_duration_ms: int = 5000) -> List[Dict[str, Any]]:
        """
        Split sentences that are too long to be displayed effectively.
        
        Args:
            sentences: List of sentence objects
            max_duration_ms: Maximum duration for a sentence in milliseconds
            
        Returns:
            List[Dict]: Split sentences
        """
        split_sentences = []
        
        for sentence in sentences:
            duration = sentence["end_time"] - sentence["start_time"]
            
            if duration <= max_duration_ms:
                split_sentences.append(sentence)
            else:
                # Split the sentence
                split_parts = self._split_sentence_by_duration(sentence, max_duration_ms)
                split_sentences.extend(split_parts)
        
        return split_sentences
    
    def _split_sentence_by_duration(self, sentence: Dict[str, Any], max_duration_ms: int) -> List[Dict[str, Any]]:
        """
        Split a single sentence into multiple parts based on duration.
        
        Args:
            sentence: Sentence to split
            max_duration_ms: Maximum duration per part
            
        Returns:
            List[Dict]: Split sentence parts
        """
        words = sentence["words"]
        if not words:
            return [sentence]
        
        parts = []
        current_part = []
        part_start_time = words[0]["start"]
        
        for word in words:
            current_part.append(word)
            
            # Check if adding this word exceeds the duration limit
            current_duration = word["end"] - part_start_time
            
            if current_duration >= max_duration_ms and len(current_part) > 1:
                # Create a part without the current word
                part_words = current_part[:-1]
                part = self._create_sentence_object(part_words, part_start_time)
                parts.append(part)
                
                # Start new part with current word
                current_part = [word]
                part_start_time = word["start"]
        
        # Add remaining words as final part
        if current_part:
            part = self._create_sentence_object(current_part, part_start_time)
            parts.append(part)
        
        return parts
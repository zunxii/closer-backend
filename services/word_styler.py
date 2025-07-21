"""
Word styling service for video subtitle generation.
Uses OpenAI to assign priority values and style orders to words based on energy and context.
"""

import json
import re
import time
from typing import List, Dict, Any
import openai
from config import OPENAI_API_KEY


class WordStyler:
    """Handles word styling using OpenAI for dynamic subtitle emphasis."""
    
    def __init__(self):
        self.openai_client = openai.Client(api_key=OPENAI_API_KEY)
        self.model = "gpt-4o"
        self.temperature = 0.3
        self.request_delay = 0.8  # Delay between requests to respect rate limits
    
    def style_sentences(self, sentences: List[Dict[str, Any]], max_style_order: int = 3) -> List[Dict[str, Any]]:
        """
        Apply styling to all sentences in the list.
        
        Args:
            sentences: List of sentence objects with words
            max_style_order: Maximum style order (1 = highest emphasis, max = lowest)
            
        Returns:
            List[Dict]: Sentences with styled words
        """
        if not sentences:
            return []
        
        styled_sentences = []
        
        for i, sentence in enumerate(sentences):
            try:
                print(f"🎨 Styling sentence {i+1}/{len(sentences)}: '{sentence['text'][:50]}...'")
                styled_sentence = self._style_single_sentence(sentence, max_style_order)
                styled_sentences.append(styled_sentence)
                
                # Add delay between requests
                if i < len(sentences) - 1:
                    time.sleep(self.request_delay)
                    
            except Exception as e:
                print(f"❌ Error styling sentence {i+1}: {e}")
                # Add original sentence with default styling
                fallback_sentence = self._apply_fallback_styling(sentence, max_style_order)
                styled_sentences.append(fallback_sentence)
        
        return styled_sentences
    
    def _style_single_sentence(self, sentence: Dict[str, Any], max_style_order: int) -> Dict[str, Any]:
        """
        Style a single sentence using OpenAI.
        
        Args:
            sentence: Sentence object with words
            max_style_order: Maximum style order
            
        Returns:
            Dict: Styled sentence
        """
        system_prompt = self._create_system_prompt(max_style_order)
        user_prompt = self._create_user_prompt(sentence)
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
            )
            
            content = response.choices[0].message.content.strip()
            styled_words = self._parse_openai_response(content)
            
            return {
                "text": sentence["text"],
                "start_time": sentence["start_time"],
                "end_time": sentence["end_time"],
                "words": styled_words
            }
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            raise
    
    def _create_system_prompt(self, max_style_order: int) -> str:
        """
        Create the system prompt for OpenAI.
        
        Args:
            max_style_order: Maximum style order
            
        Returns:
            str: System prompt
        """
        return f"""You are a smart assistant that styles captions for vertical short-form videos (Reels, TikToks, Shorts).

You'll be given a sentence and a list of words. Each word has:
- text
- start time (ms)
- end time (ms)
- energy (float between 0.0 and 1.0)

Your task is to return the word list with added styling:

For each word, assign:
- "priority_value" (float between 0.000 and 1.000 three decimal places)
- "style_order" (int between 1 and {max_style_order}, where 1 is highest visual emphasis and {max_style_order} is the lowest that represent a fallback default style for normal words)

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

DO NOT include explanations, markdown, or any formatting. Just pure JSON list of styled words."""
    
    def _create_user_prompt(self, sentence: Dict[str, Any]) -> str:
        """
        Create the user prompt for OpenAI.
        
        Args:
            sentence: Sentence object
            
        Returns:
            str: User prompt
        """
        return f"""Sentence: "{sentence['text']}"

Words:
{json.dumps(sentence['words'], indent=2)}"""
    
    def _parse_openai_response(self, content: str) -> List[Dict[str, Any]]:
        """
        Parse OpenAI response and extract styled words.
        
        Args:
            content: Raw response content
            
        Returns:
            List[Dict]: Parsed styled words
        """
        # Clean up the response
        content = re.sub(r"^```(?:json)?", "", content, flags=re.MULTILINE)
        content = re.sub(r"```$", "", content, flags=re.MULTILINE)
        content = re.sub(r",\s*([}\]])", r"\1", content)  # Remove trailing commas
        
        try:
            styled_words = json.loads(content)
            
            # Validate the response structure
            if not isinstance(styled_words, list):
                raise ValueError("Response is not a list")
            
            for word in styled_words:
                required_fields = ["text", "start", "end", "energy", "priority_value", "style_order"]
                if not all(field in word for field in required_fields):
                    raise ValueError(f"Missing required fields in word: {word}")
            
            return styled_words
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Failed to parse OpenAI response: {e}")
            print(f"Raw content: {content}")
            raise
    
    def _apply_fallback_styling(self, sentence: Dict[str, Any], max_style_order: int) -> Dict[str, Any]:
        """
        Apply fallback styling when OpenAI fails.
        
        Args:
            sentence: Original sentence
            max_style_order: Maximum style order
            
        Returns:
            Dict: Sentence with fallback styling
        """
        filler_words = {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", 
            "of", "with", "by", "from", "up", "about", "into", "through", "during",
            "before", "after", "above", "below", "is", "are", "was", "were", "be",
            "been", "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall", "can"
        }
        
        styled_words = []
        
        for word in sentence["words"]:
            word_text_lower = word["text"].lower().strip(".,!?;:")
            
            # Determine priority based on energy and word type
            if word_text_lower in filler_words:
                priority_value = 0.1
                style_order = max_style_order
            else:
                energy = word.get("energy", 0.0)
                priority_value = min(0.8, max(0.3, energy))
                style_order = min(max_style_order, max(1, int(3 - (energy * 2))))
            
            styled_word = {
                "text": word["text"],
                "start": word["start"],
                "end": word["end"],
                "energy": word["energy"],
                "priority_value": round(priority_value, 3),
                "style_order": style_order
            }
            
            styled_words.append(styled_word)
        
        return {
            "text": sentence["text"],
            "start_time": sentence["start_time"],
            "end_time": sentence["end_time"],
            "words": styled_words
        }
    
    def validate_styled_sentences(self, styled_sentences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate and clean up styled sentences.
        
        Args:
            styled_sentences: List of styled sentences
            
        Returns:
            List[Dict]: Validated styled sentences
        """
        validated_sentences = []
        
        for sentence in styled_sentences:
            try:
                # Validate sentence structure
                if not all(key in sentence for key in ["text", "start_time", "end_time", "words"]):
                    print(f"Warning: Invalid sentence structure, skipping: {sentence}")
                    continue
                
                # Validate words
                validated_words = []
                for word in sentence["words"]:
                    if self._validate_word(word):
                        validated_words.append(word)
                    else:
                        print(f"Warning: Invalid word, skipping: {word}")
                
                if validated_words:
                    validated_sentence = sentence.copy()
                    validated_sentence["words"] = validated_words
                    validated_sentences.append(validated_sentence)
                
            except Exception as e:
                print(f"Error validating sentence: {e}")
                continue
        
        return validated_sentences
    
    def _validate_word(self, word: Dict[str, Any]) -> bool:
        """
        Validate a single word object.
        
        Args:
            word: Word object to validate
            
        Returns:
            bool: True if valid
        """
        required_fields = ["text", "start", "end", "energy", "priority_value", "style_order"]
        
        # Check required fields
        if not all(field in word for field in required_fields):
            return False
        
        # Check data types and ranges
        try:
            if not isinstance(word["text"], str) or not word["text"].strip():
                return False
            
            if not isinstance(word["start"], (int, float)) or word["start"] < 0:
                return False
            
            if not isinstance(word["end"], (int, float)) or word["end"] < word["start"]:
                return False
            
            if not isinstance(word["energy"], (int, float)) or word["energy"] < 0:
                return False
            
            if not isinstance(word["priority_value"], (int, float)) or not (0 <= word["priority_value"] <= 1):
                return False
            
            if not isinstance(word["style_order"], int) or word["style_order"] < 1:
                return False
            
            return True
            
        except (TypeError, ValueError):
            return False
    
    def get_styling_statistics(self, styled_sentences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about the styling results.
        
        Args:
            styled_sentences: List of styled sentences
            
        Returns:
            Dict: Styling statistics
        """
        if not styled_sentences:
            return {}
        
        all_words = []
        for sentence in styled_sentences:
            all_words.extend(sentence["words"])
        
        if not all_words:
            return {}
        
        priority_values = [word["priority_value"] for word in all_words]
        style_orders = [word["style_order"] for word in all_words]
        
        # Count words by style order
        style_distribution = {}
        for style_order in style_orders:
            style_distribution[f"style{style_order}"] = style_distribution.get(f"style{style_order}", 0) + 1
        
        return {
            "total_words": len(all_words),
            "average_priority": sum(priority_values) / len(priority_values),
            "min_priority": min(priority_values),
            "max_priority": max(priority_values),
            "style_distribution": style_distribution,
            "unique_styles_used": len(set(style_orders))
        }
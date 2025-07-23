"""
Word styling service using OpenAI to assign visual emphasis to words.
Analyzes energy and semantic importance to determine styling priorities.
"""

import json
import re
import time
import openai
from typing import List, Dict, Any
from config import Config

class WordStyler:
    def __init__(self):
        self.config = Config()
        self.client = openai.Client(api_key=self.config.OPENAI_API_KEY)

    def style_words(self, sentences: List[Dict[str, Any]], max_style_order: int) -> List[Dict[str, Any]]:
        """
        Style words in sentences based on energy and semantic importance.
        
        Args:
            sentences: List of sentence objects with words
            max_style_order: Maximum style order number available
            
        Returns:
            List[Dict]: Sentences with styled words (priority_value and style_order added)
        """
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
                response = self.client.chat.completions.create(
                    model=self.config.OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.config.OPENAI_TEMPERATURE,
                    max_tokens=self.config.OPENAI_MAX_TOKENS,
                )

                content = response.choices[0].message.content.strip()
                # Clean up potential markdown formatting
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
                time.sleep(0.8)  # Rate limiting

            except Exception as e:
                print(f"❌ Error on sentence {i+1}: {e}")
                continue

        return styled_sentences

    def apply_template_styles(
        self,
        data: List[Dict[str, Any]], 
        total_styles: int = 3, 
        threshold: float = None, 
        default_style: str = None
    ) -> List[Dict[str, Any]]:
        """
        Apply template styles to styled words based on priority values.
        
        Args:
            data: List of sentence objects with styled words
            total_styles: Total number of available styles
            threshold: Minimum priority difference threshold
            default_style: Default style to use for low-priority words
            
        Returns:
            List[Dict]: Data with template styles applied
        """
        threshold = threshold or self.config.STYLE_THRESHOLD
        default_style = default_style or self.config.DEFAULT_STYLE_NAME
        
        for chunk in data:
            words = chunk["words"]
            priority_values = [w["priority_value"] for w in words]

            max_p = max(priority_values)
            min_p = min(priority_values)
            delta = max_p - min_p

            if delta > threshold:
                # Apply styles based on style_order if priority variance is significant
                for word in words:
                    if 1 <= word["style_order"] <= total_styles:
                        word["style"] = f"style{word['style_order']}"
                    else:
                        word["style"] = default_style
            else:
                # Use default style if priority variance is low
                for word in words:
                    word["style"] = default_style

        return data

# Global functions for backward compatibility
def style_words(sentences: List[Dict[str, Any]], max_style_order: int) -> List[Dict[str, Any]]:
    """Style words using WordStyler"""
    styler = WordStyler()
    return styler.style_words(sentences, max_style_order)

def apply_template_styles(
    data: List[Dict[str, Any]], 
    total_styles: int = 3, 
    threshold: float = None, 
    default_style: str = None
) -> List[Dict[str, Any]]:
    """Apply template styles using WordStyler"""
    styler = WordStyler()
    return styler.apply_template_styles(data, total_styles, threshold, default_style)
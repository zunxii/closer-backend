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
    
    def style_words_using_map(
    self,
    sentences: List[Dict[str, Any]],
    frame_word_map: Dict[str, List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
        """
        Style words like `style_words`, but enforce style compatibility
        using precomputed visual clusters (frame_word_map).
        """

        # Step 1: Use OpenAI to style semantically and energetically
        styled_sentences = self.style_words(sentences, max_style_order=4)

        # Step 2: use openAI again to map the constrains
        
        system_prompt = f"""
        You are assisting in building a smart and adaptive video editing tool, specifically focusing on improving the visual structure and styling of text overlays used in animated or subtitle-based videos. The goal is to enhance **typographic clarity**, **reduce randomness**, and introduce **visual hierarchy** that helps guide the viewer's attention effectively.

        ---

        ### Context:
        In these video frames, each sentence consists of several text fragments, and each fragment is styled using predefined "styles" (like font, weight, size, color, boldness, etc.). We have extracted these styled sentences and now want to improve their internal structure and consistency.

        ---

        ### Styled Sentences (Input):
        Each sentence is currently styled using a particular **order of styles**, which is inconsistent or arbitrary. These styles do not follow a consistent logic or hierarchy, which results in chaotic visuals. Your job is to **refine** these style orders to make them **visually appealing**, **hierarchically structured**, and **coherent across sentences**.

        Styled Sentences:
        {styled_sentences}

        ---

        ### Style Co-occurrence Mapping:
        This mapping tells you which styles have been observed together in the same frame across multiple video scenes. It's essentially a set of soft rules or constraints that describe **which styles are compatible with each other**.

        If style A appears in a frame with style B, that means A and B are visually cohesive and can be grouped or sequenced in a sentence. If two styles **do not co-occur**, they should ideally not appear together in the same sentence or right next to each other.

        Mapping:
        {frame_word_map}

        ---

        ### Objective:

        Your job is to reassign or reorder the styles used in each sentence using the above mapping as a constraint. But this is not just a mechanical task—this is about **applying design reasoning**.

        You are acting as a **typographic system designer** or **motion designer**, thinking about:
        - **How humans perceive visual structure**
        - **How styling creates rhythm and tempo**
        - **How to guide the viewer’s eye using hierarchy**

        The reordering of styles should be intelligent and reflect deep understanding of visual design.

        ---

        ### Guidelines & Reasoning:

        1. **Respect the Co-occurrence Rules:**
        - If styles A and B do not appear together in the mapping, they should not be placed in the same sentence.
        - Use the mapping as a filter to determine which style combinations are valid.

        2. **Apply Visual Hierarchy:**
        - Think of styles as ranked by importance.
        - Higher-contrast styles (like bold, large fonts, high-saturation colors) carry more emphasis and should come first or be used sparingly.
        - Subtler styles (smaller text, muted color, italic) should be used for less important fragments.
        - Create a natural flow — start with strong, high-emphasis styles and descend into softer ones.

        3. **Reduce Randomness:**
        - Across different sentences, ensure the style orders follow a consistent structure.
        - Avoid shuffling styles just for variety — consistency is key for branding and user experience.

        4. **Design for the Human Eye:**
        - Consider how a person reads a line of styled text.
        - The goal is to guide their attention, not overwhelm or confuse them.
        - Use design principles like **emphasis, contrast, alignment, repetition**, and **balance**.

        5. **Create Uniform but Flexible Patterns:**
        - Establish a typographic system or “style grammar” that feels predictable and elegant.
        - Don't rigidly enforce sameness, but ensure each sentence belongs to a consistent visual family.

        6. **Don’t Change the Sentence or Styles Themselves:**
        - Your job is not to rewrite text or invent new styles.
        - Only **reorder** existing styles within the sentence according to constraints and design logic.

        ---

        ### Output:
        return only the json without comments and markdown just cold blooded json output in a structured format.
        No explaianation needed just return the json as output only 
        - Do NOT wrap the output in triple backticks.
        - Do NOT include any explanation, markdown, commentary, or headers.
        - Ensure all keys are present for every word.
        - Maintain the original sentence word order.
        Your response will be parsed directly using `json.loads()` — any deviation will cause a failure. Be strict and precise.
        ⚠️ Critical Formatting Instructions:
        - Every word must include ALL of the following keys: `text`, `start`, `end`, `energy`, `priority_value`, `style_order`.
        - Even if you reuse or reorder existing styles, make sure every word entry contains both `"priority_value"` and `"style_order"` — without exception.
        - Do not skip any of these fields. If you're unsure, copy from the original input and just update `priority_value` and `style_order`.

        ---
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
                print('ye dekh idhr niche')
                print(content)
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
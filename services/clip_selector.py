from typing import List, Tuple, Dict, Optional
from collections import Counter, defaultdict

class StyleWindowFinder:
    def __init__(self, window_duration: float = 5.0):
        """
        Initialize with configurable window duration (default 5 seconds)
        """
        self.window_duration = window_duration * 1000  # Convert to milliseconds

    def find_window(self, segments: List[Dict], target_styles: List[int] = None) -> Optional[Tuple[float, float]]:
        """
        Finds a 5-second window that contains the most distinct style orders.
        
        Args:
            segments: List of segment dictionaries with words
            target_styles: Optional list of target styles (not used in new logic, kept for compatibility)
        
        Returns:
            Tuple of (start_time_ms, end_time_ms) or None if no suitable window found
        """
        if not segments:
            print("❌ No segments provided to StyleWindowFinder")
            return None
            
        print(f"🔍 Analyzing {len(segments)} segments for {self.window_duration/1000}s window")
        
        # Convert segments to word-level items for analysis
        word_items = []
        for segment in segments:
            if 'words' in segment:
                for word in segment['words']:
                    # Handle both 'style_order' and 'style' fields
                    style_value = None
                    if 'style_order' in word:
                        style_value = word['style_order']
                    elif 'style' in word:
                        # Extract number from style name (e.g., "style1" -> 1)
                        style_str = str(word['style'])
                        import re
                        match = re.search(r'\d+', style_str)
                        if match:
                            style_value = int(match.group())
                        else:
                            style_value = 1  # Default fallback
                    
                    if style_value is not None:
                        word_items.append({
                            'start': word.get('start', segment.get('start', 0)),
                            'end': word.get('end', segment.get('end', 0)),
                            'style_order': style_value,
                            'text': word.get('text', '')
                        })
        
        if not word_items:
            print("❌ No words with style information found in segments")
            return None
            
        print(f"🔍 Found {len(word_items)} words with styles")
        
        # Sort word items by start time
        word_items.sort(key=lambda x: x['start'])
        
        # Find all unique styles present
        all_styles = set(item['style_order'] for item in word_items)
        print(f"🔍 All styles present: {sorted(all_styles)}")
        
        if not word_items:
            print("❌ No valid word items to analyze")
            return None
        
        # Get the time range of all content
        total_start = word_items[0]['start']
        total_end = word_items[-1]['end']
        total_duration = total_end - total_start
        
        print(f"🔍 Total content duration: {total_duration/1000:.2f}s ({total_start/1000:.2f}s to {total_end/1000:.2f}s)")
        
        # If total duration is less than window duration, return the entire range
        if total_duration <= self.window_duration:
            print(f"✅ Content shorter than window duration, using entire range")
            return (total_start, total_end)
        
        # Sliding window approach to find 5-second window with most distinct styles
        best_window = None
        max_distinct_styles = 0
        best_style_distribution = {}
        
        # Calculate step size (every 0.5 seconds for good coverage)
        step_size = 500  # 0.5 seconds in milliseconds
        
        # Slide window from start to end
        current_start = total_start
        
        while current_start + self.window_duration <= total_end:
            current_end = current_start + self.window_duration
            
            # Find words within this window
            window_words = [
                word for word in word_items 
                if word['start'] >= current_start and word['end'] <= current_end
            ]
            
            if window_words:
                # Count distinct styles in this window
                window_styles = set(word['style_order'] for word in window_words)
                style_counts = Counter(word['style_order'] for word in window_words)
                
                distinct_count = len(window_styles)
                
                # Calculate a quality score based on:
                # 1. Number of distinct styles (primary factor)
                # 2. Even distribution of styles (secondary factor)
                # 3. Total word count (tertiary factor)
                
                distribution_score = 0
                if distinct_count > 1:
                    # Measure how evenly distributed the styles are
                    counts = list(style_counts.values())
                    max_count = max(counts)
                    min_count = min(counts)
                    # Lower ratio means more even distribution
                    distribution_score = min_count / max_count if max_count > 0 else 0
                
                word_count = len(window_words)
                
                # Combined quality score
                quality_score = (
                    distinct_count * 100 +  # Primary: distinct styles
                    distribution_score * 20 +  # Secondary: distribution evenness
                    min(word_count, 20) * 1  # Tertiary: word count (capped at 20)
                )
                
                # Update best window if this one is better
                if (distinct_count > max_distinct_styles or 
                    (distinct_count == max_distinct_styles and quality_score > 
                     (max_distinct_styles * 100 + best_style_distribution.get('quality', 0)))):
                    
                    best_window = (current_start, current_end)
                    max_distinct_styles = distinct_count
                    best_style_distribution = {
                        'styles': window_styles,
                        'counts': dict(style_counts),
                        'quality': quality_score,
                        'word_count': word_count
                    }
                    
                    print(f"🎯 New best window: {current_start/1000:.2f}s-{current_end/1000:.2f}s, "
                          f"{distinct_count} styles: {sorted(window_styles)}")
            
            current_start += step_size
        
        if best_window:
            start_s, end_s = best_window[0]/1000, best_window[1]/1000
            print(f"✅ Best {self.window_duration/1000}s window: {start_s:.2f}s to {end_s:.2f}s")
            print(f"🎨 Contains {max_distinct_styles} distinct styles: {sorted(best_style_distribution['styles'])}")
            print(f"📊 Style distribution: {best_style_distribution['counts']}")
            print(f"📝 Word count: {best_style_distribution['word_count']}")
            return best_window
        else:
            print("❌ No suitable window found")
            # Fallback: return first 5 seconds if available
            if total_duration > 0:
                fallback_end = min(total_start + self.window_duration, total_end)
                print(f"🔄 Fallback: using first {(fallback_end-total_start)/1000:.2f}s")
                return (total_start, fallback_end)
            return None

    def find_best_style_coverage_window(self, segments: List[Dict], min_styles: int = 2) -> Optional[Tuple[float, float]]:
        """
        Alternative method: Find window that covers at least min_styles different styles
        """
        print(f"🔍 Looking for window with at least {min_styles} different styles")
        
        word_items = []
        for segment in segments:
            if 'words' in segment:
                for word in segment['words']:
                    style_value = None
                    if 'style_order' in word:
                        style_value = word['style_order']
                    elif 'style' in word:
                        import re
                        match = re.search(r'\d+', str(word['style']))
                        if match:
                            style_value = int(match.group())
                    
                    if style_value is not None:
                        word_items.append({
                            'start': word.get('start', segment.get('start', 0)),
                            'end': word.get('end', segment.get('end', 0)),
                            'style_order': style_value,
                            'text': word.get('text', '')
                        })
        
        if not word_items:
            return None
        
        word_items.sort(key=lambda x: x['start'])
        
        # Find first window that meets minimum style requirement
        step_size = 250  # 0.25 seconds
        current_start = word_items[0]['start']
        total_end = word_items[-1]['end']
        
        while current_start + self.window_duration <= total_end:
            current_end = current_start + self.window_duration
            
            window_words = [
                word for word in word_items 
                if word['start'] >= current_start and word['end'] <= current_end
            ]
            
            if window_words:
                window_styles = set(word['style_order'] for word in window_words)
                if len(window_styles) >= min_styles:
                    print(f"✅ Found window with {len(window_styles)} styles: {sorted(window_styles)}")
                    return (current_start, current_end)
            
            current_start += step_size
        
        print(f"⚠️ Could not find window with {min_styles}+ styles, using best available")
        return self.find_window(segments)

    def _find_style_key(self, segment: Dict) -> Optional[str]:
        """Find the key that contains style information"""
        possible_keys = ['style', 'style_order', 'style_id', 'style_index', 'font_style']
        for key in possible_keys:
            if key in segment:
                return key
        return None

    def _find_start_key(self, segment: Dict) -> Optional[str]:
        """Find the key that contains start time information"""
        possible_keys = ['start', 'start_time', 'start_timestamp', 'begin']
        for key in possible_keys:
            if key in segment:
                return key
        return None

    def _find_end_key(self, segment: Dict) -> Optional[str]:
        """Find the key that contains end time information"""
        possible_keys = ['end', 'end_time', 'end_timestamp', 'finish']
        for key in possible_keys:
            if key in segment:
                return key
        return None
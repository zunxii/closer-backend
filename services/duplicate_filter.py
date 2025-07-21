from typing import List, Dict
from utils.text_utils import get_frame_text

class DuplicateFilter:
    @staticmethod
    def filter_duplicate_frames(all_frames: List[Dict]) -> List[Dict]:
        """Filter out duplicate frames based on text content"""
        filtered_frames = []
        i = 0

        while i < len(all_frames):
            current = all_frames[i]
            current_text = get_frame_text(current)

            if i + 1 < len(all_frames):
                next_text = get_frame_text(all_frames[i + 1])
                if current_text and current_text in next_text:
                    i += 1
                    continue

            filtered_frames.append(current)
            i += 1

        print(f"Original: {len(all_frames)} → Filtered: {len(filtered_frames)}")
        return filtered_frames
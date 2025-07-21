import cv2
import os
from config import Config
from utils.file_utils import ensure_directory_exists

class FrameExtractor:
    def __init__(self, target_fps: int = Config.TARGET_FPS):
        self.target_fps = target_fps

    def extract_frames(self, video_path: str, output_folder: str) -> int:
        """Extract frames from video at specified FPS"""
        ensure_directory_exists(output_folder)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if fps == 0:
            cap.release()
            raise ValueError("Unable to determine video FPS.")

        frame_interval = int(fps / self.target_fps)
        frame_num = 0
        saved_count = 0

        print(f"📸 Extracting frames from: {video_path}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num % frame_interval == 0:
                frame_path = os.path.join(output_folder, f'frame_{saved_count:05d}.jpg')
                cv2.imwrite(frame_path, frame)
                saved_count += 1

            frame_num += 1

        cap.release()
        print(f"Saved {saved_count} frames to: {output_folder}")
        return saved_count
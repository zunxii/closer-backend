import cv2
import os
from config import Config
from utils.file_utils import ensure_directory_exists

class FrameExtractor:
    def __init__(self, target_fps: int = Config.TARGET_FPS):
        self.target_fps = target_fps

    def extract_frames(self, video_path: str, output_folder: str, start_time: float = 0.0, end_time: float = None) -> int:
        """Extract frames from a selected time range in the video at the specified FPS."""
        ensure_directory_exists(output_folder)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps == 0:
            cap.release()
            raise ValueError("Unable to determine video FPS.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        # If end_time is not provided, use the full video
        if end_time is None or end_time > duration:
            end_time = duration

        if start_time < 0 or start_time >= end_time:
            cap.release()
            raise ValueError("Invalid start or end time.")

        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        frame_interval = int(fps / self.target_fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_num = start_frame
        saved_count = 0

        print(f"📸 Extracting frames from {start_time}s to {end_time}s of {video_path}")

        while cap.isOpened() and frame_num <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            if (frame_num - start_frame) % frame_interval == 0:
                frame_path = os.path.join(output_folder, f'frame_{saved_count:05d}.jpg')
                cv2.imwrite(frame_path, frame)
                saved_count += 1

            frame_num += 1

        cap.release()
        print(f"✅ Saved {saved_count} frames from {start_time}s to {end_time}s to: {output_folder}")
        return saved_count

    def clip_video(self, video_path: str, start_time: float, end_time: float, output_path: str) -> tuple:
        """
        Clip a portion of the video from start_time to end_time and save it to output_path.
        Returns (start_time, end_time) as confirmation.
        """
        ensure_directory_exists(os.path.dirname(output_path))

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            cap.release()
            raise ValueError("Unable to determine video FPS.")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID', 'avc1'

        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame

        print(f"🎬 Clipping video from {start_time}s to {end_time}s")

        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            current_frame += 1

        cap.release()
        out.release()

        print(f"✅ Clipped video saved to: {output_path}")
        return (start_time, end_time)
import os
import requests
from dotenv import load_dotenv

from config import Config

load_dotenv()

class InstagramVideoDownloader:
    def __init__(self):
        self.username = "test_user"
        self.config = Config()

        print("DEBUG - API KEY:", self.config.INSTAGRAM_VIDEO_DOWNLOAD_API_KEY)

        if not self.config.INSTAGRAM_VIDEO_DOWNLOAD_API_KEY:
            raise ValueError("Missing INSTAGRAM_VIDEO_DOWNLOAD_API_KEY in environment variables")

    def download(self, video_url: str, output_path: str) -> str:
        """Download Instagram video using API Hut and save locally."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        headers = {
            "X-Avatar-Key": self.config.INSTAGRAM_VIDEO_DOWNLOAD_API_KEY,  
            "Content-Type": "application/json"
        }

        payload = {
            "video_url": video_url,
            "type": "instagram",
            "user_id": self.username
        }

        response = requests.post(self.config.INSTAGRAM_VIDEO_DOWNLOAD_URL, json=payload, headers=headers)

        if response.status_code != 200:
            raise Exception(f"API request failed: {response.status_code} - {response.text}")

        try:
            data = response.json().get("data", [])
            video_download_url = data[0].get("url") if data else None

            if not video_download_url:
                raise Exception("No video URL found in response")

            video_response = requests.get(video_download_url, stream=True)
            if video_response.status_code != 200:
                raise Exception("Failed to download video stream")

            with open(output_path, "wb") as f:
                for chunk in video_response.iter_content(1024):
                    if chunk:
                        f.write(chunk)

            print(f" Instagram video downloaded: {output_path}")
            return output_path

        except Exception as e:
            raise Exception(f"Error processing Instagram API response: {str(e)}")

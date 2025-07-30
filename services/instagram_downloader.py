import os
import yt_dlp

class InstagramVideoDownloader:
    def __init__(self):
        self.username = "test_user"  # You can remove this if unused

    def download(self, video_url: str, output_path: str) -> str:
        """Download Instagram video using yt_dlp and save it to output_path."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # yt_dlp options
        ydl_opts = {
            'outtmpl': output_path,
            'format': 'mp4',
            'quiet': True,
            'noplaylist': True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            print(f"Instagram video downloaded: {output_path}")
            return output_path

        except Exception as e:
            raise Exception(f"Failed to download Instagram video using yt_dlp: {str(e)}")

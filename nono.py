import subprocess

def add_ass_subtitles(input_video, subtitle_file, output_video):
    command = [
        "ffmpeg",
        "-i", input_video,
        "-vf", f"ass={subtitle_file}",
        "-c:a", "copy",  # copy audio without re-encoding
        output_video
    ]

    try:
        subprocess.run(command, check=True)
        print(f"✅ Subtitles added successfully: {output_video}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error applying subtitles:\n{e}")

# Example usage
add_ass_subtitles("video.mp4", "sub.ass", "output_with_subs.mp4")

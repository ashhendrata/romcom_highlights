import os
import subprocess
import sys

clips_dir = sys.argv[1]
audio_dir = sys.argv[2]

os.makedirs(audio_dir, exist_ok=True)

for clip in os.listdir(clips_dir):
    if clip.endswith(".mp4"):
        clip_path = os.path.join(clips_dir, clip)
        audio_path = os.path.join(audio_dir, clip.replace(".mp4", ".wav"))
        subprocess.run([
            "ffmpeg", "-i", clip_path, "-q:a", "0", "-map", "a", audio_path
        ])
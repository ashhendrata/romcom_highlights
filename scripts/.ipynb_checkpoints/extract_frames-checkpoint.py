import os
import subprocess
import sys

clips_dir = sys.argv[1]
frames_dir = sys.argv[2]

os.makedirs(frames_dir, exist_ok=True)

for clip in os.listdir(clips_dir):
    if clip.endswith(".mp4"):
        clip_name = clip.replace(".mp4", "")
        output_path = os.path.join(frames_dir, clip_name)
        os.makedirs(output_path, exist_ok=True)
        clip_path = os.path.join(clips_dir, clip)
        subprocess.run([
            "ffmpeg", "-i", clip_path, os.path.join(output_path, "frame_%03d.jpg")
        ])
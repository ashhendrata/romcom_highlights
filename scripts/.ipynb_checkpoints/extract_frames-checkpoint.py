import os
import subprocess
import sys

clips_folder = sys.argv[1]
frames_folder = sys.argv[2] # where they're all going to be saved
os.makedirs(frames_folder, exist_ok=True)

for clip in os.listdir(clips_folder):
    if clip.endswith(".mp4"):
        clip_name = clip.replace(".mp4", "") # get clip id
        output_path = os.path.join(frames_folder, clip_name) #clip_001/, ...
        os.makedirs(output_path, exist_ok=True)
        clip_path = os.path.join(clips_folder, clip)
        subprocess.run([
            "ffmpeg", "-i", clip_path, os.path.join(output_path, "frame_%03d.jpg")
        ])
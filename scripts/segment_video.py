import os
import subprocess
import sys

video_path = sys.argv[1]
output_dir = sys.argv[2]

os.makedirs(output_dir, exist_ok=True)

result = subprocess.run(
    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
     "-of", "default=noprint_wrappers=1:nokey=1", video_path],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT
)
duration = float(result.stdout)

for i in range(0, int(duration), 10):
    output_file = os.path.join(output_dir, f"clip_{i//10 + 1:03d}.mp4")
    subprocess.run([
        "ffmpeg", "-i", video_path, "-ss", str(i), "-t", "10",
        "-c", "copy", output_file
    ])
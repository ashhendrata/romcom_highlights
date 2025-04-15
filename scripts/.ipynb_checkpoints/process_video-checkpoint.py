import os
import subprocess
import sys

def run(cmd):
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)

def main(video_id):
    base_path = f"dataset/{video_id}"
    raw_video = f"{base_path}/raw_video.mp4"
    clips = f"{base_path}/clips"
    audio = f"{base_path}/audio"
    frames = f"{base_path}/frames"
    transcripts = f"{base_path}/transcripts"

    os.makedirs(clips, exist_ok=True)
    os.makedirs(audio, exist_ok=True)
    os.makedirs(frames, exist_ok=True)
    os.makedirs(transcripts, exist_ok=True)

    run(f"python scripts/segment_video.py {raw_video} {clips}")

    run(f"python scripts/extract_audio.py {clips} {audio}")

    run(f"python scripts/extract_frames.py {clips} {frames}")

    run(f"whisper {raw_video} --model medium --output_format srt --output_dir {transcripts}")
    
    srt_path = f"{transcripts}/raw_video.srt"
    final_srt = f"{transcripts}/{video_id}.srt"
    if os.path.exists(srt_path):
        os.rename(srt_path, final_srt)
    run(f"python scripts/split_srt_by_clip.py {final_srt} {transcripts}")

if __name__ == "__main__":
    main(sys.argv[1])
import os
import sys
from datetime import timedelta
import srt

def get_clip_index(start_time, clip_duration=10):
    return int(start_time.total_seconds() // clip_duration)

def split_srt(input_path, output_dir, clip_duration=10):
    with open(input_path, 'r', encoding='utf-8') as f:
        subs = list(srt.parse(f.read()))
    
    clips = {}

    for sub in subs:
        index = get_clip_index(sub.start, clip_duration)
        if index not in clips:
            clips[index] = []
        clips[index].append(sub)

    os.makedirs(output_dir, exist_ok=True)

    for index, subs in clips.items():
        clip_filename = os.path.join(output_dir, f'clip{index:03d}.srt')
        with open(clip_filename, 'w', encoding='utf-8') as f:
            f.write(srt.compose(subs))

    print(f"Created {len(clips)} clip transcripts in {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/split_srt_by_clip.py dataset/video1/transcripts/video1.srt dataset/video1/transcripts")
        sys.exit(1)

    input_srt = sys.argv[1]
    output_dir = sys.argv[2]
    split_srt(input_srt, output_dir)

import os
import sys
from datetime import timedelta
import srt

def get_clip_index(start_time, clip_duration=10):
    return int(start_time.total_seconds() // clip_duration) # convert to total secs

def split_srt(input_path, output_dir, clip_duration=10): # 10s clips 
    with open(input_path, 'r', encoding='utf-8') as f:
        text = list(srt.parse(f.read()))
    
    clips = {} # group text entries by clip index
    for txt in text:
        index = get_clip_index(txt.start, clip_duration)
        if index not in clips:
            clips[index] = []
        clips[index].append(txt)

    os.makedirs(output_dir, exist_ok=True)
    for index, text in clips.items(): # making mini .srts
        clip_filename = os.path.join(output_dir, f'clip{index:03d}.srt')
        with open(clip_filename, 'w', encoding='utf-8') as f:
            f.write(srt.compose(text))

    print(f"clip transcripts created :)")

if __name__ == "__main__":
    input_srt = sys.argv[1]
    output_dir = sys.argv[2]
    split_srt(input_srt, output_dir)

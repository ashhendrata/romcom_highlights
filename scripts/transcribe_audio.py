import os
import sys
import whisper

audio_dir = sys.argv[1]
transcripts_dir = sys.argv[2]

os.makedirs(transcripts_dir, exist_ok=True)

model = whisper.load_model
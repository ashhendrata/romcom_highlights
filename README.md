# 🌹 Rom‑Com Highlight Extraction ❤️

## File Directory Structure

```text
highlights-final-project/
├── dataset/
│   ├── video1/
│   │   ├── raw_video.mp4
│   │   ├── clips/ #10s mp4 videos
│   │   ├── audio/ #clip_*.wav
│   │   ├── frames/ #clip_*/frame_*.jpg
│   │   └── transcripts/ #clip_*.srt and full transcript
│   │   └── labels.csv
│   ├── video2/ and so on
├── scripts/
│   ├── process_video.py
│   ├── prepare_labels.py
│   ├── extract_text_features.py
│   ├── extract_audio_features.py
│   ├── extract_visual_features.py
│   ├── merge_features.py
│   ├── train_model_improved.py
│   ├── train_mlp.py
│   └── predict_highlights.py
├── processed
│   ├── all_labels.csv
│   ├── final_features.pkl
│   ├── 
│   ├── 
│   ├─
├── models/
└── README.md
```

## Downloading and Preprocessing

e.g. for video5:
yt-dlp "https://youtube.com/watch?v=VIDEO_ID" -o "dataset/video5/raw_video.mp4"
python scripts/process_video.py video5

outputs:
dataset/video5/clips/*.mp4
dataset/video5/audio/*.wav
dataset/video5/frames/clip_*/frame_*.jpg
dataset/video5/transcripts/clip_*.srt


## Labelling Clips

In each labels.csv file:
clip_fname
highlight_label
emotion_present
dialogue_impactful
romantic_gesture
music_shift
plot_turning_point
credits

As long as a clip does not fall into the credits category AND ticks the box of at least one of the categories from emotion_present to plot_turning_point, it is labell

## Aggregate labels

python scripts/aggregate_data.py

output: all_labels.csv, which excludes clips where credits=1

## Extracting features

Text: python scripts/extract_text_features.py
Audio: python scripts/extract_audio_features.py
Visual: python scripts/extract_visual_features.py

merging all: python scripts/merge_features.py

## Training models: logistic regression

python scripts/train_model.py

## Generating highlight reel

python scripts/prepare_labels.py video5

python scripts/extract_text_features.py
python scripts/extract_audio_features.py
python scripts/extract_visual_features.py
python scripts/merge_features.py

python scripts/predict_highlights.py video5
output: concat.txt

to stitch it all together:
ffmpeg -f concat -safe 0 -i concat.txt -c copy highlight_reel_video5.mp4


## Evaulation






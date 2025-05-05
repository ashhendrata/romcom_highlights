# 🌹 Multimodal Highlight Detection in Rom-Coms ❤️

## Overview
This project is an end-to-end pipeline that automatically finds and stitches together the most emotionally charged “climax” moments from short romantic comedies. We split each film into 10-second clips, extract three modalities of features (text from subtitles, audio via Mel-Frequency Cepstral Coefficients (MFCCs), and visual via ResNet-18), and train a logistic‐regression classifier, a random forest classifier, and a gradient-boosted classifier to decide which clips belong in the highlight reel. 

## Replication Instructions

### An Overview: File Directory Structure

By following the instructions below, you should be able to create the following file directory structure. 

```text
highlights-final-project/
├── dataset/
│   ├── video1/
│   │   ├── raw_video.mp4 # downloaded from YouTube
│   │   ├── clips/        #10s mp4 files that make up the rom-coms (clip_000.mp4, ...)
│   │   ├── audio/        #clip_*.wav which are 10s audio clips extracted from the videos from clips/ folder (clip_000.wav, ...)
│   │   ├── frames/       #clip_*/frame_*.jpg catagorized into folders according to the 10s clips (clip_000/frame_001.jpg, ...)
│   │   └── transcripts/  #clip*.srt (clip000.srt, ...) and the full transcript named videoX.srt
│   │   └── labels.csv    # contains labels for each clip, including highlight (1) or non-highlight (0)
│   ├── video2/ and so on...
├── scripts/
│   ├── process_video.py
│   ├── count_clips.py                # prints % climactic vs non-climactic clips
│   ├── aggregate_data.py
│   ├── extract_text_features.py
│   ├── extract_audio_features.py
│   ├── extract_visual_features.py
│   ├── merge_features.py
│   ├── train_logistic.py
│   ├── train_forest.py
│   ├── train_boosting.py
│   └── predict_highlights.py
├── processed
│   ├── all_labels.csv      # aggregated and cleaned labels from all non-credits clips
│   ├── final_features.pkl  # merged text,audio, and visual features
│   ├── audio_features.csv  # extracted audio features from all non-credits clips
│   ├── text_features.csv   # extracted text features from transcripts from all non-credits clips
│   ├── visual_features.pkl # extracted visual features from all non-credits clips
├── models/
│   ├── 
└── README.md
```

### Downloading Rom-Coms

1. Find a short romantic comedy film on YouTube
2. To download it, run the following script, replacing "video6" and the YouTube URL accordingly:
```
yt-dlp -f bestvideo+bestaudio --merge-output-format mp4 "https://www.youtube.com/watch?v=k_d2Vxc6No8" -o "dataset/video6/raw_video.%(ext)s"
```
3. The video should be saved as raw_video.mp4 in an empty folder named videoX, X being the video number or id


### Labelling Clips

In each labels.csv file, there should be the following columns:
1. clip_fname         # clip_000, clip_001, ...
2. highlight_label    # marked 1 if any of the catagories below except for credits is marked 1
3. emotion_present    # Is someone expressing a strong emotion (love, heartbreak, joy, jealousy, etc.)?
4. dialogue_impactful # Does the dialogue include a confession, reunion, breakup, or emotional realization?
5. romantic_gesture   # Is there a significant romantic gesture (kiss, hug, proposal, gift, dance, etc.)?
6. music_shift        # Is there a noticeable music swell or soundtrack shift underscoring emotion?
7. plot_turning_point # Is this a plot twist or a key decision moment (e.g. choosing love over career)?
8. credits            # credits = 1 for end‐credits clips (these will be dropped)

To aggregate the labels across all the videos for training, run the following script which would generate all_labels.csv:
```
python scripts/aggregate_data.py
```

### Preprocessing Videos

1. Segment rom-coms, extract modalities, and generate transcripts by running the following script:
```
python scripts/process_video.py videoX
```
process_video.py runs the following:
- python scripts/segment_video.py {raw_video} {clips}
- python scripts/extract_audio.py {clips} {audio}                                   # MFCC mean and variance
- python scripts/extract_frames.py {clips} {frames}                                 # ResNet-18 pooled frames
- whisper {raw_video} --model medium --output_format srt --output_dir {transcripts} # generates transcripts of rom-coms
- python scripts/split_srt_by_clip.py {final_srt} {transcripts}                     # splits transcripts into 10s segments


### Extracting Features

1. To extract textual, auditory, and visual features, run the following:
```
python scripts/extract_text_features.py    # generates text_features.csv
```
```
python scripts/extract_audio_features.py   # generates audio_features.csv
```
```
python scripts/extract_visual_features.py  # generates visual_features.pkl
```
*Note: Loading a .pkl is usually much quicker than parsing a CSV, which is important here due to the massive number of frames to process.*
2. Merging all of the above with: 
```
python scripts/merge_features.py           # generates final_features.pkl
```

### Training and Evaluating a Classifier

To train the model with logistic regression, run:
```
python scripts/train_logistic.py
```

**Why use logistic regression?**
1. Logistic regression is one of the most interpretable models, making it easier to understand which features (textual, auditory, visual) contribute most to the prediction of highlight scenes.
2. Logistic regression serves as a solid baseline.
3. It trains very fast which was helpful when experimenting with labeling thresholds.

To train the model with random forest classifier, run:
```
python scripts/train_forest.py
```
**Why use random forest?**
1. Robust to irrelevant features and outliers
2. Provides feature-importance scores

To train the model with gradient boosting, run:
```
python scripts/train_boosting.py
```
**Why use gradient boosting?**
1. Builds trees sequentially to correct earlier errors—good at capturing subtle patterns
2. Handles class imbalance without needing manual resampling

### Generating a Highlight Reel

1. If the rom-com is unlabelled (i.e. doesn't have a filled labels.csv file), run the following to generate predictions.
```
python scripts/predict_highlights.py videoX
```
2. Then run this to stitch the clips together:
```
ffmpeg -f concat -safe 0 -i dataset/videoX/concat_videoX.txt -c copy dataset/videoX/highlight_reel_videoX.mp4
```

However, if the rom-com is labelled, run the following:
```
python scripts/stitch_highlights.py videoX
```


### Evaluation
We report per-class precision, recall, F1, and support on the held-out film.

#### Clip-count baseline

To see what fraction of clips are non-climactic (your majority-class baseline), run:
```
python scripts/count_clips.py
```

## Future Directions

### For preliminary stages:

While the proof‐of‐concept project demonstrates that simple multimodal features can surface emotionally charged climaxes, there are several next steps of varying feasibility to explore. First, expanding beyond 6 short films to a larger and more diverse set of rom-coms (possibly including full-length features) would improve model generalization and allow us to fine-tune deep architectures like multimodal transformers. Second, incorporating facial expression classifiers, speech sentiment analysis, or scene-graph understanding could help the model better distinguish between subtle versus overt climactic moments. For instance, a romantic gesture like a kiss would not necessarily mean the clip is a highlight. Third, a study to evaluate perceived highlight quality and adjust our definition of “highlight” based on human judgment would mitigate the limitations of human (my own) subjectivity. Finally, moving toward semi- or un-supervised methods (e.g., autoencoders trained on user-edited highlight reels) could reduce the need for manual labels and adapt dynamically to different genres.

### For the final presentation:




## Contributions

Thank you for being here! This was a solo project with the following stages:
1. **Data Collection and Annotation (about 20 hours)**: selected and downloaded 10 short rom-coms, determined labeling guidelines, and manually annotated clips.
2. **Preprocessing and Feature Handling (about 20 h)**: implemented and refined scripts for segmentation, transcript splitting, feature extraction (textual, audtory, visual), and feature merging.
3. **Modeling & Evaluation (about 15 h)**: created MFCC, BERT, and ResNet-18 feature pipelines, trained and finetuned logistic regression, random forest, and gradient-boosted classifiers.
4. **Poster Creation and Documentation (about 10 h)**: designed graphics for poster, drafted README (especially the replication instructions), and ensured steps were reproducable and detailed.






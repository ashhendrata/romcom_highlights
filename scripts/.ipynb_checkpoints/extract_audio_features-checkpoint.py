import os
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
from pathlib import Path

def extract_mfcc(wav, n=13):
    y,sr = librosa.load(wav, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n)
    return np.hstack((mfccs.mean(axis=1), mfccs.var(axis=1)))

def main():
    labels = pd.read_csv("all_labels.csv")
    labels = labels.loc[:, ~labels.columns.str.contains("^Unnamed")] # just to make sure it doesn't add more unnamed cols

    rows = []
    for _, r in tqdm(labels.iterrows(), total=len(labels), desc="Audio"):
        vid = r.video_id
        clip  = r.clip_fname
        wav = f"dataset/{vid}/audio/{clip}.wav"
        if not os.path.exists(wav): # if no audio
            continue

        feat = extract_mfcc(wav)
        tracking = {"video_id":vid, "clip_id":clip, "highlight":r.highlight_label} # flatten it and place into audio_feat_0, ...
        tracking.update({f"audio_feat_{i}":v for i,v in enumerate(feat)})
        rows.append(tracking)

    pd.DataFrame(rows).to_csv("audio_features.csv", index=False)
    print(f"audio_features.csv created :)")
    
if __name__=="__main__":
    main()
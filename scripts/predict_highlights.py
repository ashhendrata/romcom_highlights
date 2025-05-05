# this was created to simplify the process of predicting highlights of unlabelled rom_coms that do not have a labels.csv file, which many of the scripts rely on so most of this is repeated code, so for better comments go to previous files
# it produces dataset/videoX/videoX_features.pkl, dataset/videoX/predictions_videoX.csv, and dataset/videoX/concat_videoX.txt

import os
import sys
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import librosa
import joblib
import srt
from tqdm import tqdm
from PIL import Image
from sentence_transformers import SentenceTransformer
from torchvision import transforms
from torchvision.models import resnet18

def run(cmd): # to make sure everything's going well!
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)

def segment(raw_path, clips_dir): # into 10s videos with full audio
    os.makedirs(clips_dir, exist_ok=True)
    run(f"python scripts/segment_video.py {raw_path} {clips_dir}")

def extract_audio(clips_dir, audio_dir): 
    os.makedirs(audio_dir, exist_ok=True)
    run(f"python scripts/extract_audio.py {clips_dir} {audio_dir}")

def extract_frames(clips_dir, frames_dir):
    os.makedirs(frames_dir, exist_ok=True)
    run(f"python scripts/extract_frames.py {clips_dir} {frames_dir}")

def make_transcripts(raw_path, transcripts_dir, video_id):
    os.makedirs(transcripts_dir, exist_ok=True)
    run(f"whisper {raw_path} --model medium --language en --output_format srt --output_dir {transcripts_dir}")
    raw_srt = Path(transcripts_dir) / "raw_video.srt"
    final_srt = Path(transcripts_dir) / f"{video_id}.srt"
    if raw_srt.exists():
        raw_srt.rename(final_srt)
    run(f"python scripts/split_srt_by_clip.py {final_srt} {transcripts_dir}") # based on 10s clips

def load_text_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_emb(srt_path, model, dim):
    with open(srt_path, "r", encoding="utf-8") as f:
        transcript = list(srt.parse(f.read())) # parse .srt file
    text = " ".join(sub.content.strip() for sub in transcript) # merge
    return model.encode(text) if text else np.zeros(dim)

def load_visual_model(device):
    model = resnet18(pretrained=True)
    model.fc = torch.nn.Identity()
    return model.to(device).eval()

def extract_visual_feat(frames_list, model, device):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    feats = []
    for img_path in frames_list:
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x).cpu().numpy().squeeze()
        feats.append(out)
    return np.mean(feats, axis=0) if feats else np.zeros(model.fc.in_features)

def extract_mfcc_feat(wav_path, n_mfcc=13):
    y, sr = librosa.load(wav_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.hstack([mfccs.mean(axis=1), mfccs.var(axis=1)])

def main(video_id):
    base = Path("dataset")/video_id
    raw = base/"raw_video.mp4"
    
    clips_dir = base/"clips"
    audio_dir = base/"audio"
    frames_dir = base/"frames"
    transcripts_dir = base/"transcripts"
    out_dir = base

    # preprocessing scripts
    segment(raw, clips_dir)
    extract_audio(clips_dir, audio_dir)
    extract_frames(clips_dir, frames_dir)
    make_transcripts(raw, transcripts_dir, video_id)

    # feature extraction
    text_model = load_text_model()
    text_dim = text_model.get_sentence_embedding_dimension()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vis_model = load_visual_model(device)

    # loops extracting from each clip
    tracking = []
    for clip_mp4 in sorted(clips_dir.glob("*.mp4")):
        cid = clip_mp4.stem  # clip_000, ...
        rec = {"video_id":video_id, "clip_id":cid}

        # textual
        srt_p = transcripts_dir/f"{cid}.srt"
        emb   = extract_text_emb(srt_p, text_model, text_dim) if srt_p.exists() else np.zeros(text_dim)
        for i,v in enumerate(emb): rec[f"text_feat_{i}"] = float(v)

        # auditory
        wav_p = audio_dir/f"{cid}.wav"
        mfcc = extract_mfcc_feat(wav_p) if wav_p.exists() else np.zeros(26)
        for i,v in enumerate(mfcc): rec[f"audio_feat_{i}"] = float(v)

        # visual
        frames_folder = frames_dir/cid
        jpgs = sorted(frames_folder.glob("*.jpg")) if frames_folder.exists() else []
        vis_feat = extract_visual_feat(jpgs, vis_model, device)
        for i,v in enumerate(vis_feat): rec[f"visual_feat_{i}"] = float(v)

        tracking.append(rec)

    feat_df = pd.DataFrame(tracking)
    feat_pkl = out_dir/f"{video_id}_features.pkl"
    feat_df.to_pickle(feat_pkl)
    print(f"saved")

    # load model
    model_path = "highlight_clf_lr_weighted.joblib"
    if not os.path.exists(model_path):
        model_path = "highlight_clf_lr.joblib"
    clf_obj = joblib.load(model_path)
    if isinstance(clf_obj, (tuple,list)) and len(clf_obj)==2:
        pipe, thresh = clf_obj
    else:
        pipe, thresh = clf_obj, 0.5

    # predicting
    X  = feat_df.drop(columns=["video_id","clip_id"])
    probs = pipe.predict_proba(X)[:,1]
    preds = (probs >= thresh).astype(int)

    # predictions.csv creation
    out_csv = out_dir/f"predictions_{video_id}.csv"
    feat_df["prob_highlight"] = probs
    feat_df["pred_highlight"] = preds
    feat_df.to_csv(out_csv, index=False)
    print(f"predictions.csv created :)")

    # ffmpeg concat list
    concat_txt = out_dir/f"concat_{video_id}.txt"
    with open(concat_txt, "w") as f:
        for cid, p in zip(feat_df.clip_id, preds):
            if p==1:
                f.write(f"file 'dataset/{video_id}/clips/{cid}.mp4'\n")
    print(f"concat list created :)")

if __name__=="__main__":
    main(sys.argv[1])
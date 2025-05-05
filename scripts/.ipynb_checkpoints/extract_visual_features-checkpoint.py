import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
from tqdm import tqdm
from pathlib import Path

torch.backends.cudnn.benchmark = True # cuDNN autotuner

def load_model(device):
    model = resnet18(pretrained=True) # load ResNet-18
    model.fc = torch.nn.Identity() # replace final with identity so it returns input unchanged
    return model.to(device).eval()

def extract(model, device, frames):
    max_frames = 16 # for efficiency
    if len(frames) > max_frames:
        step = len(frames) // max_frames
        frames = frames[::step][:max_frames] # even
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std =[0.229,0.224,0.225])
    ])
    imgs = [transform(Image.open(f).convert("RGB")) for f in frames] # apply transform to each frame
    batch = torch.stack(imgs, 0).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast(): # forwarding the batch in mixed precision due to speed issues
        feats = model(batch)
    return feats.cpu().numpy().mean(axis=0)

def main():
    labs = pd.read_csv("all_labels.csv") # aggregated labels
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(dev)

    rows = []
    for _, r in tqdm(labs.iterrows(), total=len(labs), desc="Visual"):
        vid, clip = r.video_id, r.clip_fname
        frame_path = Path(f"dataset/{vid}/frames/{clip}")
        if not frame_path.exists(): 
            continue
        all = sorted(frame_path.glob("*.jpg")) # gather all .jpg frame paths for this clip
        if not all: 
            continue # no frames
        feat = extract(model, dev, all) # averaged CNN feature vector for this clip
        rec = {"video_id":vid, "clip_id":clip, "highlight":r.highlight_label}
        rec.update({f"visual_feat_{i}":val for i,val in enumerate(feat)}) # flatten into cols
        rows.append(rec)

    pd.DataFrame(rows).to_pickle("visual_features.pkl")
    print(f"visual_features.pkl with {len(rows)} clips created :)")

if __name__=="__main__":
    main()
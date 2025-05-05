import pandas as pd
from pathlib import Path
import srt
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def load_text(path: Path) -> str:
    txt = path.read_text(encoding="utf-8")
    subs = list(srt.parse(txt))
    return " ".join(s.content.strip() for s in subs) # full transcript

def main():
    labels = pd.read_csv("all_labels.csv")
    labels = labels.loc[:, ~labels.columns.str.contains("^Unnamed")] # just in case more unnamed cols are added again
    model = SentenceTransformer("all-MiniLM-L6-v2")
    dim   = model.get_sentence_embedding_dimension()

    tracking = []
    for _, row in tqdm(labels.iterrows(), total=len(labels), desc="Textual"):
        vid = row["video_id"]
        clip = row["clip_fname"] # clip_000, ...
        possibles = [clip, clip.replace("_","")] # due to inconsistencies in my naming (sorry!)
        srt_path = None
        for base in possibles:
            p = Path(f"dataset/{vid}/transcripts/{base}.srt")
            if p.exists():
                srt_path = p
                break
        if not srt_path:
            continue # no matching .srt
        text = load_text(srt_path)
        embbedding  = model.encode(text) if text else [0.0]*dim
        track = {"video_id": vid, "clip_id": clip, "highlight": row["highlight_label"]}
        track.update({f"text_feat_{i}": val for i, val in enumerate(embbedding)}) # flatten into keys
        tracking.append(track)

    out = pd.DataFrame(tracking)
    out.to_csv("text_features.csv", index=False)
    print(f"text_features.csv created :)")

if __name__=="__main__":
    main()
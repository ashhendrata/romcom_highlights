import pandas as pd

txt = pd.read_csv("text_features.csv")
aud = pd.read_csv("audio_features.csv")
vis = pd.read_pickle("visual_features.pkl")

df = txt.merge(aud, on=["video_id","clip_id","highlight"]).merge(vis, on=["video_id","clip_id","highlight"]) # merge on video_id + clip_id
df = df.loc[:, ~df.columns.str.contains("^Unnamed")] # in case more unnamed cols added
df.to_pickle("final_features.pkl")
print(f"final_features.pkl with {len(df)} clips created :)")
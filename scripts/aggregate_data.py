import glob
import pandas as pd

all_rows = []
for path in glob.glob("dataset/video*/labels.csv"):
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]  # added as stray cols were added during preprocessing
    vid = path.split("/")[1]
    df["video_id"] = vid
    df = df[df["credits"] != 1]  # not included if its part of the end-credits
    all_rows.append(df)

all_df = pd.concat(all_rows, ignore_index=True)
all_df.to_csv("all_labels.csv", index=False)
print(f"all_labels.csv created :)")
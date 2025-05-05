# this should stitch highlights for rom-coms that are labelled and have a labels.csv (no extractions are needed)

import sys, csv, subprocess
from pathlib import Path

def main(video_id):
    base = Path("dataset")/video_id
    labels = base/"labels.csv"
    clips = base/"clips"
    concat = base/f"concat_labeled_{video_id}.txt"
    with open(labels, newline='') as f_in, open(concat, "w") as f_out:
        reader = csv.DictReader(f_in)
        for row in reader:
            if row.get("credits","0") == "1": # skip end credits
                continue
            if row.get("highlight_label","0") == "1": # only keep clips marked as highlight
                clip_file = clips/f"{row['clip_fname']}.mp4"
                f_out.write(f"file '{clip_file}'\n")
    print(f"concat list made :)")

    out_video = base/f"highlight_reel_labelled_{video_id}.mp4"
    cmd = [
      "ffmpeg", "-y",
      "-f", "concat", "-safe", "0",
      "-i", str(concat),
      "-c", "copy", str(out_video)
    ]
    print(cmd)
    subprocess.run(cmd, check=True)
    print(f"stitched highlight reel :)}")

if __name__=="__main__":
    main(sys.argv[1])
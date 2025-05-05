import pandas as pd

def main():
    df = pd.read_csv("all_labels.csv")
    
    total = len(df)
    non_highlights = (df["highlight_label"] == 0).sum()
    highlights = (df["highlight_label"] == 1).sum()
    
    print(f"Total clips: {total}")
    print(f"Non‐highlights: {non_highlights} ({non_highlights/total:.1%})")
    print(f"Highlights: {highlights} ({highlights/total:.1%})")

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
import joblib

def main():
    df = pd.read_pickle("final_features.pkl")
    train = ["video1","video2","video3","video4"]
    test  = ["video5"]

    # split into train and test
    df_tr = df[df.video_id.isin(train)]
    df_te = df[df.video_id.isin(test)]

    # seperate features fro labels
    X_tr = df_tr.drop(columns=["video_id","clip_id","highlight"])
    y_tr = df_tr.highlight
    X_te = df_te.drop(columns=["video_id","clip_id","highlight"])
    y_te = df_te.highlight

    # scikit-learn pipeline
    pipe = Pipeline([
      ("scaler", StandardScaler()),
      ("clf", LogisticRegression(
                  max_iter=1000,
                  class_weight="balanced" # to handle imbalance
      ))
    ])

    pipe.fit(X_tr, y_tr)
    probs = pipe.predict_proba(X_te)[:,1]
    final_preds = (probs >= 0.1).astype(int)

    print("Logistic:")
    print(classification_report(y_te, final_preds, digits=3))

    joblib.dump((pipe, 0.1), "logistic.joblib")

if __name__=="__main__":
    main()
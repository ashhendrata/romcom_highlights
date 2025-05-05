import pandas as pd
import numpy as np 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
import joblib

def main():
    df = pd.read_pickle("final_features.pkl")
    train = ["video1","video2","video3","video4"]
    test  = ["video5"]

    # splitting into train and test
    df_train = df[df.video_id.isin(train)]
    df_test = df[df.video_id.isin(test)]

    # separate features from labels
    X_train = df_train.drop(columns=["video_id","clip_id","highlight"])
    y_train = df_train.highlight
    X_test = df_test.drop(columns=["video_id","clip_id","highlight"])
    y_test = df_test.highlight

    # scikit-learn pipeline
    pipe = Pipeline([
      ("scaler", StandardScaler()),
      ("clf", RandomForestClassifier(
         n_estimators=300,
         max_depth=None,
         min_samples_leaf=5,
         class_weight="balanced",
         n_jobs=-1
      ))
    ])
    pipe.fit(X_train, y_train) # training
    probs = pipe.predict_proba(X_test)[:, 1]

    final_preds = (probs >= 0.35).astype(int)

    print("Random Forest:")
    print(classification_report(y_test, final_preds, digits=3))
    joblib.dump((pipe, 0.35), "forest.joblib")
    print("forest.joblib saved :)")

if __name__=="__main__":
    main()
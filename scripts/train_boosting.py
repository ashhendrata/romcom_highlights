import pandas as pd, xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib

def main():
    df = pd.read_pickle("final_features.pkl")
    train = ["video1","video2","video3","video4"]
    test  = ["video5"]

    # splitting into training and testing
    df_train = df[df.video_id.isin(train)]
    df_test = df[df.video_id.isin(test)]

    # split features from labels
    X_train = df_train.drop(columns=["video_id","clip_id","highlight"])
    y_train = df_train.highlight
    X_test = df_test.drop(columns=["video_id","clip_id","highlight"])
    y_test = df_test.highlight

    # to deal with class imbalance
    scale = y_train.value_counts()[0] / y_train.value_counts()[1] 
    pipe = Pipeline([
      ("scaler", StandardScaler()),
      ("clf", xgb.XGBClassifier(
         n_estimators=300,
         max_depth=4,
         learning_rate=0.05,
         scale_pos_weight=scale,
         use_label_encoder=False,
         eval_metric="logloss",
         n_jobs=-1
      ))
    ])
    pipe.fit(X_train, y_train) # train 
    probs = pipe.predict_proba(X_test)[:, 1]
    final_preds = (probs >= 0.15).astype(int)

    print("XGBoost:")
    print(classification_report(y_test, final_preds, digits=3))
    joblib.dump((pipe, 0.15 ), "boosted.joblib")
    print("boosted.joblib created :)")

if __name__=="__main__":
    main()
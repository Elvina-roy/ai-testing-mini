from pathlib import Path
import json

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_data

ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)

def main():
    X_train, X_test, y_train, y_test = load_data()

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=500)),
        ]
    )

    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)

    metrics = {"roc_auc": float(auc)}
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("Saved metrics:", metrics)

if __name__ == "__main__":
    main()
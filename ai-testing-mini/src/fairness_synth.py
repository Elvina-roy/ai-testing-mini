import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_data
from src.synthetic import make_sensitive_group

ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)

def group_rates(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "tpr": float(tpr), "fpr": float(fpr)}

def main():
    X_train, X_test, y_train, y_test = load_data()

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=500)),
        ]
    )
    pipe.fit(X_train, y_train)

    p = pipe.predict_proba(X_test)[:, 1]
    group = make_sensitive_group(X_test, seed=42)

    report = {"by_group": {}}
    for g in ["A", "B"]:
        idx = (group == g).to_numpy()
        rates = group_rates(y_test[idx].to_numpy(), p[idx], threshold=0.5)
        report["by_group"][g] = rates

    tprs = [report["by_group"][g]["tpr"] for g in ["A", "B"]]
    fprs = [report["by_group"][g]["fpr"] for g in ["A", "B"]]
    report["tpr_gap"] = float(abs(tprs[0] - tprs[1]))
    report["fpr_gap"] = float(abs(fprs[0] - fprs[1]))

    (ARTIFACTS / "fairness_synth.json").write_text(json.dumps(report, indent=2))
    print("Saved fairness report:", report)

if __name__ == "__main__":
    main()
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_data

ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)

def mean_shift_score(a, b):
    denom = np.std(a) + 1e-9
    return float(abs(np.mean(a) - np.mean(b)) / denom)

def ks_statistic(a, b):
    """Simple 2-sample KS statistic without scipy."""
    a = np.sort(np.asarray(a))
    b = np.sort(np.asarray(b))
    data_all = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, data_all, side="right") / a.size
    cdf_b = np.searchsorted(b, data_all, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))

def main():
    X_train, X_test, y_train, y_test = load_data()

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=500)),
        ]
    )
    pipe.fit(X_train, y_train)

    # baseline predictions
    p_base = pipe.predict_proba(X_test)[:, 1]

    # drifted features
    X_drift = X_test.copy()
    for col in ["mean radius", "mean texture"]:
        if col in X_drift.columns:
            X_drift[col] = X_drift[col] * 1.6 + 4.0

    p_drift = pipe.predict_proba(X_drift)[:, 1]

    drift_scores = {}
    for col in ["mean radius", "mean texture"]:
        if col in X_test.columns:
            drift_scores[col] = mean_shift_score(X_test[col].to_numpy(), X_drift[col].to_numpy())

    report = {
        "mean_shift_scores": drift_scores,
        "pred_mean_base": float(np.mean(p_base)),
        "pred_mean_drift": float(np.mean(p_drift)),
        "pred_mean_shift": float(abs(np.mean(p_base) - np.mean(p_drift))),
        "pred_ks": ks_statistic(p_base, p_drift),
        "pct_pos_base": float(np.mean(p_base >= 0.5)),
        "pct_pos_drift": float(np.mean(p_drift >= 0.5)),
        "pct_pos_shift": float(abs(np.mean(p_base >= 0.5) - np.mean(p_drift >= 0.5))),
    }

    (ARTIFACTS / "drift_report.json").write_text(json.dumps(report, indent=2))
    print("Saved drift report:", report)

if __name__ == "__main__":
    main()
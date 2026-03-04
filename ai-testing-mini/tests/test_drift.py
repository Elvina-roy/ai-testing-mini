import json
from pathlib import Path

def test_feature_drift_detected():
    r = json.loads(Path("artifacts/drift_report.json").read_text())
    assert max(r["mean_shift_scores"].values()) > 0.5

def test_prediction_distribution_shift_detected():
    r = json.loads(Path("artifacts/drift_report.json").read_text())
    # KS > 0.10 обычно уже заметный сдвиг распределений
    assert r["pred_ks"] > 0.10 or r["pred_mean_shift"] > 0.02 or r["pct_pos_shift"] > 0.05
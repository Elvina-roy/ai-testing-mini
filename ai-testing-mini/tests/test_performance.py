import json
from pathlib import Path

def test_auc_gate():
    metrics = json.loads(Path("artifacts/metrics.json").read_text())
    assert metrics["roc_auc"] >= 0.95
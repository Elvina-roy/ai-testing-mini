import json
from pathlib import Path

def test_fairness_report_exists():
    p = Path("artifacts/fairness_synth.json")
    assert p.exists()

def test_fairness_gaps_within_threshold():
    r = json.loads(Path("artifacts/fairness_synth.json").read_text())

    # Synthetic: мы не ожидаем огромных перекосов.
    # Порог можно подстроить, но начнём с разумного.
    assert r["tpr_gap"] <= 0.10
    assert r["fpr_gap"] <= 0.10
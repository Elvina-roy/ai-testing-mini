# AI Testing Mini Project (Python)

Mini project demonstrating practical AI/ML testing:
- Data quality gates
- Performance gate (ROC AUC)
- Reproducibility
- Drift detection (feature + prediction distribution)
- Metamorphic robustness tests
- Synthetic fairness checks (TPR/FPR gap)

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Run tests
pytest -q
Artifacts are generated automatically before tests (see tests/conftest.py)
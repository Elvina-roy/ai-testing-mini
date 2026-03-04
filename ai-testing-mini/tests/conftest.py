import subprocess
from pathlib import Path

def _run(module: str):
    # запускаем как "python -m src.xxx" в корне проекта
    subprocess.run(["python", "-m", module], check=True)

def pytest_sessionstart(session):
    """
    Автоматически генерируем артефакты один раз перед всеми тестами.
    """
    # На всякий случай создадим папку
    Path("artifacts").mkdir(exist_ok=True)

    # 1) метрики (performance gate)
    _run("src.train")

    # 2) drift report
    _run("src.drift")

    # 3) fairness report (synthetic)
    _run("src.fairness_synth")
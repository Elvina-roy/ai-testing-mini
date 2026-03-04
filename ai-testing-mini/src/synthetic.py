import numpy as np
import pandas as pd

def make_sensitive_group(X: pd.DataFrame, seed: int = 42) -> pd.Series:
    """
    Synthetic sensitive attribute (stable):
    - почти независим от признаков, чтобы fairness-гейты были стабильны
    - фиксированный seed -> без флапа
    """
    rng = np.random.default_rng(seed)
    group = rng.choice(["A", "B"], size=len(X), p=[0.5, 0.5])
    return pd.Series(group, index=X.index, name="group")
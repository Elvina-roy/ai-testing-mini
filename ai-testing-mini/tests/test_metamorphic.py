import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_data

def build_model():
    X_train, X_test, y_train, y_test = load_data()
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=500)),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe, X_test

def test_small_input_noise_should_not_flip_too_many_predictions():
    pipe, X_test = build_model()

    p1 = pipe.predict_proba(X_test)[:, 1]

    X_noisy = X_test.copy()
    # добавим небольшой шум к паре признаков
    for col in ["mean radius", "mean texture"]:
        if col in X_noisy.columns:
            std = X_noisy[col].std()
            X_noisy[col] = X_noisy[col] + np.random.default_rng(42).normal(0, 0.02 * std, size=len(X_noisy))

    p2 = pipe.predict_proba(X_noisy)[:, 1]

    # доля примеров, где класс (>=0.5) изменился, должна быть небольшой
    flips = np.mean((p1 >= 0.5) != (p2 >= 0.5))
    assert flips < 0.05
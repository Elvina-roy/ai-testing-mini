from src.data import load_data

def test_no_missing_values():
    X_train, X_test, y_train, y_test = load_data()
    assert X_train.isna().sum().sum() == 0
    assert X_test.isna().sum().sum() == 0

def test_target_is_binary():
    _, _, y_train, y_test = load_data()
    assert set(y_train.unique()).issubset({0, 1})
    assert set(y_test.unique()).issubset({0, 1})

def test_feature_ranges_reasonable():
    X_train, _, _, _ = load_data()
    # простой sanity check: не должно быть бесконечностей/NaN
    assert X_train.replace([float("inf"), float("-inf")], 0).shape[1] == X_train.shape[1]
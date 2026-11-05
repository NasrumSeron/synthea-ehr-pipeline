import json

import numpy as np
import pandas as pd

from pipeline.model import train_and_evaluate


def synthetic_xy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n)})
    X.loc[::17, "b"] = np.nan  # imputer must cope with missing values
    y = pd.Series((X["a"] + rng.normal(scale=0.5, size=n) > 1.2).astype(int))
    return X, y


def test_metrics_fields_and_split():
    X, y = synthetic_xy()
    m = train_and_evaluate(X, y)
    assert m["n_train"] + m["n_test"] == m["n_patients"] == 400
    assert m["n_test"] == 80
    cm = m["confusion_matrix_test"]
    assert sum(cm.values()) == m["n_test"]
    assert cm["tp"] + cm["fn"] == int(round(m["prevalence_test"] * m["n_test"]))
    assert set(m["standardized_coefficients"]) == {"a", "b", "c"}
    assert len(m["auroc_cv5_folds"]) == 5


def test_split_is_stratified():
    X, y = synthetic_xy()
    m = train_and_evaluate(X, y)
    assert abs(m["prevalence_train"] - m["prevalence_test"]) < 0.01


def test_learns_signal():
    X, y = synthetic_xy()
    m = train_and_evaluate(X, y)
    assert m["auroc_test"] > 0.85 and m["auroc_cv5_mean"] > 0.85
    assert m["standardized_coefficients"]["a"] > 0


def test_deterministic():
    X, y = synthetic_xy()
    assert json.dumps(train_and_evaluate(X, y), sort_keys=True) == \
        json.dumps(train_and_evaluate(X, y), sort_keys=True)

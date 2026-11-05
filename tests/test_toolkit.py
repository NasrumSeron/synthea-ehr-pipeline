"""Toolkit helpers: recalibration maths, the JSON model, local recalibration.
Skipped if the toolkit extras are not installed (pip install -r requirements-toolkit.txt)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "toolkit"))

import multi_seed  # noqa: E402
import predict  # noqa: E402
import train_final  # noqa: E402
from pipeline import model as M  # noqa: E402


def logistic(x):
    return 1 / (1 + np.exp(-x))


def test_fit_recalibration_recovers_known_a_and_b():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.98, size=40000)
    a0, b0 = -0.4, 0.8
    y = (rng.uniform(size=p.size) < logistic(a0 + b0 * np.log(p / (1 - p)))).astype(int)
    a, b = multi_seed.fit_recalibration(y, p)
    assert a == pytest.approx(a0, abs=0.05) and b == pytest.approx(b0, abs=0.05)


def test_apply_recalibration_identity_and_order():
    p = np.array([0.05, 0.2, 0.5, 0.9])
    assert np.allclose(multi_seed.apply_recalibration(p, 0.0, 1.0), p)
    q = multi_seed.apply_recalibration(p, -0.2, 0.78)
    assert list(np.argsort(q)) == list(np.argsort(p))  # ranking unchanged for b > 0
    assert q[-1] < p[-1] and q[0] > p[0]  # b < 1 pulls extremes inwards


def synthetic_xy(n=600, seed=3):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    X.loc[::13, "b"] = np.nan
    y = pd.Series((rng.uniform(size=n) < logistic(1.5 * X["a"] - 1.5)).astype(int))
    return X, y


def test_json_model_reproduces_sklearn_exactly():
    X, y = synthetic_xy()
    pipe = M.make_model().fit(X, y)
    model = train_final.export_model(pipe, X.columns, -0.1, 0.9, 0.3, {"raw": "test"})
    ours = predict.predict_proba_from_json(model, X)
    assert np.max(np.abs(ours - pipe.predict_proba(X)[:, 1])) < 1e-12
    assert model["features"] == ["a", "b"] and len(model["coef"]) == 2


def test_stored_recalibration_is_applied_from_the_json():
    model = {"recalibration": {"a": 0.0, "b": 1.0}}
    p = np.array([0.1, 0.4, 0.8])
    assert np.allclose(predict.recalibrate(model, p), p)


def test_local_recalibration_holds_slices_apart_and_keeps_ranking():
    X, y = synthetic_xy(n=2000)
    pipe = M.make_model().fit(X, y)
    model = train_final.export_model(pipe, X.columns, 0.0, 1.0, 0.3, {"raw": "test"})
    p = predict.predict_proba_from_json(model, X) ** 0.6  # deliberately mis-calibrated
    res = predict.local_recalibration(model, p, y.to_numpy(), 0.5, 42, M)
    assert res["n_fit"] + res["n_check"] == 2000
    assert set(res["check"]) == {"raw", "stored_recal", "local_recal"}
    raw_err = abs(res["check"]["raw"]["calibration_in_the_large"])
    local_err = abs(res["check"]["local_recal"]["calibration_in_the_large"])
    assert local_err < raw_err  # the local fix moves the average risk towards the observed rate

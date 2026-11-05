import json

import numpy as np
import pandas as pd
import pytest

from pipeline import PipelineError
from pipeline.model import (calibration, choose_threshold, classification_at, net_benefit,
                            net_benefit_treat_all, split_dev_test, train_and_evaluate)


def synthetic_xy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n)})
    X.loc[::17, "b"] = np.nan  # imputer must cope with missing values
    y = pd.Series((X["a"] + rng.normal(scale=0.5, size=n) > 1.2).astype(int))
    return X, y


def test_metrics_fields_and_split():
    X, y = synthetic_xy()
    m = train_and_evaluate(X, y)
    assert m["n_dev"] + m["n_test"] == m["n_patients"] == 400
    assert m["n_test"] == 80
    for key in ("at_chosen_threshold", "at_threshold_0_5"):
        cm = m["test"][key]
        assert cm["tp"] + cm["fp"] + cm["fn"] + cm["tn"] == m["n_test"]
    assert len(m["dev_cv"]["auroc_folds"]) == 5
    assert m["features"] == ["a", "b", "c"]
    assert set(m["odds_ratio_per_sd"]) == {"a", "b", "c"}
    assert [r["threshold"] for r in m["test"]["decision_curve"]] == [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]


def test_split_is_stratified():
    m = train_and_evaluate(*synthetic_xy())
    assert abs(m["prevalence_dev"] - m["prevalence_test"]) < 0.01


def test_learns_signal():
    m = train_and_evaluate(*synthetic_xy())
    assert m["test"]["auroc"] > 0.85 and m["dev_cv"]["auroc_mean"] > 0.85
    assert m["test"]["auprc"] > m["test"]["auprc_chance_level"]
    assert m["standardized_coefficients"]["a"] > 0 and m["odds_ratio_per_sd"]["a"] > 1


def test_deterministic():
    X, y = synthetic_xy()
    assert json.dumps(train_and_evaluate(X, y), sort_keys=True) == \
        json.dumps(train_and_evaluate(X, y), sort_keys=True)


def test_dev_choices_do_not_depend_on_the_test_set():
    """Changing test-set features must not change CV results or the chosen threshold."""
    X, y = synthetic_xy()
    _, X_test, _, _ = split_dev_test(X, y)  # split depends on y only
    X2 = X.copy()
    X2.loc[X_test.index, "a"] = 0.0
    m1, m2 = train_and_evaluate(X, y), train_and_evaluate(X2, y)
    assert m1["dev_cv"] == m2["dev_cv"] and m1["threshold"] == m2["threshold"]
    assert m1["test"]["auroc"] != m2["test"]["auroc"]


def test_coefficient_names_stay_aligned_when_a_column_is_all_missing():
    X, y = synthetic_xy()
    X["b"] = np.nan
    m = train_and_evaluate(X, y)
    assert list(m["standardized_coefficients"]) == ["a", "b", "c"]
    assert m["standardized_coefficients"]["b"] == 0.0
    assert m["standardized_coefficients"]["a"] > 0


def test_too_few_positives_in_dev_set_is_a_clear_error():
    X, y = synthetic_xy(n=40)
    y[:] = 0
    y.iloc[:5] = 1  # 5 positives overall -> 4 in the development set
    with pytest.raises(PipelineError, match="development set"):
        train_and_evaluate(X, y)


# --- helpers, checked by hand ---

def test_choose_threshold_youden():
    y = [0, 0, 0, 0, 1, 1, 1]
    p = [0.1, 0.2, 0.3, 0.8, 0.5, 0.6, 0.9]
    # cut >= 0.9: sens 1/3, spec 4/4 -> J 0.333; >= 0.6: sens 2/3, spec 3/4 -> J 0.417;
    # >= 0.5: sens 3/3, spec 3/4 -> J 0.75 (unique maximum); >= 0.3: sens 1, spec 2/4 -> J 0.5
    assert choose_threshold(y, p) == 0.5
    assert choose_threshold([0, 0, 1, 1], [0.1, 0.2, 0.6, 0.9]) == 0.6  # perfect separation


def test_classification_at():
    r = classification_at([0, 0, 1, 1], [0.2, 0.6, 0.4, 0.8], 0.5)
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (1, 1, 1, 1)
    assert r["sensitivity"] == r["specificity"] == r["ppv"] == r["npv"] == 0.5


def test_net_benefit_by_hand():
    y = [1, 1, 0, 0, 0]
    p = [0.9, 0.3, 0.6, 0.1, 0.1]
    # threshold 0.5: TP 1, FP 1, n 5 -> 1/5 - 1/5 * (0.5/0.5) = 0
    assert net_benefit(y, p, 0.5) == pytest.approx(0.0)
    # threshold 0.2: TP 2, FP 1 -> 2/5 - 1/5 * 0.25 = 0.35
    assert net_benefit(y, p, 0.2) == pytest.approx(0.35)
    # treat all at 0.2: prevalence 0.4 - 0.6 * 0.25 = 0.25
    assert net_benefit_treat_all(y, 0.2) == pytest.approx(0.25)


def test_calibration_of_a_well_specified_model():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.02, 0.98, size=5000)
    y = (rng.uniform(size=5000) < p).astype(int)  # outcomes drawn from the predicted risks
    c = calibration(y, p)
    assert c["calibration_slope"] == pytest.approx(1.0, abs=0.15)
    assert abs(c["calibration_in_the_large"]) < 0.03
    assert len(c["bins_by_predicted_risk"]) == 5
    for b in c["bins_by_predicted_risk"]:
        assert b["observed"] == pytest.approx(b["mean_predicted"], abs=0.05)

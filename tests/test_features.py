"""Each feature against values computed by hand from tests/fixtures (as_of 2025-06-14)."""

import math

import pandas as pd
import pytest

from pipeline.features import FEATURES, build_features, default_as_of
from tests.conftest import AS_OF


@pytest.fixture
def feats(tables):
    return build_features(tables, AS_OF)


def col(feats, name):
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in feats[name].items()}


def test_shape_and_order(feats):
    assert list(feats.columns) == FEATURES
    assert list(feats.index) == ["p1", "p2", "p3", "p4", "p5"]


def test_default_as_of_is_latest_encounter(tables):
    assert default_as_of(tables) == AS_OF


def test_age(feats):
    # p1 birthday 15 Jun not yet reached on 14 Jun; p3 died on their 70th birthday
    assert col(feats, "age") == {"p1": 64, "p2": 45, "p3": 70, "p4": 24, "p5": 34}


def test_sex_male(feats):
    assert col(feats, "sex_male") == {"p1": 1, "p2": 0, "p3": 0, "p4": 1, "p5": 0}


def test_enc_ambulatory(feats):
    # p1: e1, e2 (duplicate row dropped), e3; wellness e5 not counted. p4: " Ambulatory " normalised
    assert col(feats, "enc_ambulatory") == {"p1": 3, "p2": 1, "p3": 0, "p4": 1, "p5": 0}


def test_enc_emergency(feats):
    assert col(feats, "enc_emergency") == {"p1": 1, "p2": 0, "p3": 2, "p4": 0, "p5": 0}


def test_enc_inpatient(feats):
    assert col(feats, "enc_inpatient") == {"p1": 0, "p2": 2, "p3": 0, "p4": 0, "p5": 0}


def test_hypertension(feats):
    assert col(feats, "hypertension") == {"p1": 1, "p2": 0, "p3": 1, "p4": 0, "p5": 0}


def test_hyperlipidemia(feats):
    # p2 via 13644009 hypercholesterolemia; p4 hypertriglyceridemia is not allowlisted
    assert col(feats, "hyperlipidemia") == {"p1": 1, "p2": 1, "p3": 0, "p4": 0, "p5": 0}


def test_bmi_latest(feats):
    # p1 latest of 30.0/32.5; p2 mean of two same-day values; p3 later text value ignored
    assert col(feats, "bmi_latest") == {"p1": 32.5, "p2": 26.0, "p3": 22.0, "p4": None, "p5": None}


def test_sbp_latest(feats):
    assert col(feats, "sbp_latest") == {"p1": 140.0, "p2": 120.0, "p3": 130.0, "p4": 110.0, "p5": None}


def test_dbp_latest(feats):
    assert col(feats, "dbp_latest") == {"p1": 90.0, "p2": 80.0, "p3": 85.0, "p4": 70.0, "p5": None}


def test_med_count(feats):
    # distinct codes, glucose-lowering drugs excluded: p1 lisinopril + atorvastatin (metformin out),
    # p2 acetaminophen, p3 aspirin (insulin out), p5 amoxicillin (x2) + ibuprofen
    assert col(feats, "med_count") == {"p1": 2, "p2": 1, "p3": 1, "p4": 0, "p5": 2}


def test_patient_with_no_events_gets_zeros_and_nans(tables):
    extra = pd.DataFrame({"Id": ["p6"], "BIRTHDATE": [pd.Timestamp("2000-01-01")],
                          "DEATHDATE": [pd.NaT], "GENDER": ["F"]})
    tables["patients"] = pd.concat([tables["patients"], extra], ignore_index=True)
    row = build_features(tables, AS_OF).loc["p6"]
    assert row["enc_ambulatory"] == 0 and row["med_count"] == 0 and row["hypertension"] == 0
    assert math.isnan(row["bmi_latest"])


def test_model_features_exclude_post_diagnosis_care():
    from pipeline.features import MODEL_FEATURES, NOT_MODEL_INPUTS
    assert MODEL_FEATURES == ["age", "sex_male", "hypertension", "hyperlipidemia",
                              "bmi_latest", "sbp_latest", "dbp_latest"]
    assert not set(MODEL_FEATURES) & set(NOT_MODEL_INPUTS)
    assert set(MODEL_FEATURES) | set(NOT_MODEL_INPUTS) == set(FEATURES)

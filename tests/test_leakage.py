"""Leakage guard: features may only read allowlisted condition/observation codes."""

import pandas as pd
import pytest

from pipeline import features as F
from pipeline.features import LeakageError, build_features
from pipeline.label import diabetes_codes
from tests.conftest import AS_OF

GLUCOSE, HBA1C = "2339-0", "4548-4"


def test_allowlists_are_exact():
    assert F.ALLOWED_CONDITION_CODES == {"59621000", "55822004", "13644009"}
    assert F.ALLOWED_OBSERVATION_CODES == {"39156-5", "8480-6", "8462-4"}


def test_allowlist_has_no_label_codes_or_prediabetes():
    assert not F.ALLOWED_CONDITION_CODES & diabetes_codes()
    assert not F.ALLOWED_CONDITION_CODES & {"15777000", "714628002"}
    assert not F.ALLOWED_OBSERVATION_CODES & {GLUCOSE, HBA1C}


@pytest.mark.parametrize("code", ["44054006", "714628002", "302870006", "80394007",
                                  "127013003", "90781000119102", "157141000119108",
                                  "368581000119106", "1551000119108", "97331000119101"])
def test_condition_accessor_rejects_non_allowlisted(tables, code):
    with pytest.raises(LeakageError):
        F.conditions_with(tables["conditions"], {code})


@pytest.mark.parametrize("code", [GLUCOSE, HBA1C])
def test_observation_accessor_rejects_non_allowlisted(tables, code):
    with pytest.raises(LeakageError):
        F.observations_with(tables["observations"], code)


def test_features_depend_only_on_allowlisted_codes(tables):
    """Deleting every non-allowlisted condition/observation row must not change any feature."""
    before = build_features(tables, AS_OF)
    stripped = dict(tables)
    stripped["conditions"] = tables["conditions"][tables["conditions"]["CODE"].isin(F.ALLOWED_CONDITION_CODES)]
    stripped["observations"] = tables["observations"][
        tables["observations"]["CODE"].isin(F.ALLOWED_OBSERVATION_CODES)]
    stripped["medications"] = tables["medications"][
        ~F.is_diabetes_medication(tables["medications"]["DESCRIPTION"])]
    pd.testing.assert_frame_equal(before, build_features(stripped, AS_OF))


def test_adding_diabetes_evidence_changes_no_feature(tables):
    """Give p5 diabetes, a glucose, an HbA1c and three diabetes drugs: no feature changes."""
    before = build_features(tables, AS_OF)
    day = pd.Timestamp("2025-01-01")
    t = dict(tables)
    t["conditions"] = pd.concat([tables["conditions"], pd.DataFrame(
        {"START": [day], "PATIENT": ["p5"], "CODE": ["44054006"],
         "DESCRIPTION": ["Diabetes mellitus type 2 (disorder)"]})], ignore_index=True)
    t["observations"] = pd.concat([tables["observations"], pd.DataFrame(
        {"DATE": [day, day], "PATIENT": ["p5", "p5"], "CODE": [GLUCOSE, HBA1C],
         "DESCRIPTION": ["Glucose", "HbA1c"], "VALUE": [250.0, 9.0]})], ignore_index=True)
    t["medications"] = pd.concat([tables["medications"], pd.DataFrame(
        {"START": [day] * 3, "PATIENT": ["p5"] * 3, "CODE": ["860975", "865098", "897122"],
         "DESCRIPTION": ["Metformin hydrochloride 500 MG", "Insulin Lispro 100 UNT/ML",
                         "3 ML liraglutide 6 MG/ML Pen Injector"]})], ignore_index=True)
    pd.testing.assert_frame_equal(before, build_features(t, AS_OF))


@pytest.mark.parametrize("desc", [
    "24 HR Metformin hydrochloride 500 MG", "insulin isophane human 70 UNT/ML",
    "3 ML liraglutide 6 MG/ML Pen Injector", "glipizide 5 MG Oral Tablet",
    "empagliflozin 10 MG Oral Tablet", "sitagliptin 100 MG Oral Tablet",
])
def test_diabetes_medications_matched(desc):
    assert F.is_diabetes_medication(pd.Series([desc])).item()


@pytest.mark.parametrize("desc", ["lisinopril 10 MG Oral Tablet", "atorvastatin 80 MG Oral Tablet"])
def test_other_medications_not_matched(desc):
    assert not F.is_diabetes_medication(pd.Series([desc])).item()

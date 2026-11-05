"""Eleven patient-level feature columns.

Leakage guard is an ALLOWLIST: condition and observation features may only read the codes
listed below, enforced by the two accessor functions. tests/test_leakage.py checks it.
"""

import re

import pandas as pd

# SNOMED CT, all active in International Edition 2026-11-01.
HYPERTENSION_CODES = frozenset({"59621000"})  # Essential hypertension
HYPERLIPIDEMIA_CODES = frozenset({"55822004", "13644009"})  # Hyperlipidemia, Hypercholesterolemia
ALLOWED_CONDITION_CODES = HYPERTENSION_CODES | HYPERLIPIDEMIA_CODES

# LOINC
BMI = "39156-5"
SYSTOLIC_BP = "8480-6"
DIASTOLIC_BP = "8462-4"
ALLOWED_OBSERVATION_CODES = frozenset({BMI, SYSTOLIC_BP, DIASTOLIC_BP})

ENCOUNTER_CLASSES = ("ambulatory", "emergency", "inpatient")

# Glucose-lowering drugs, matched on medication DESCRIPTION (case-insensitive) and excluded
# from the medication count. Covers every diabetes drug Synthea v4.0.0 emits plus common others.
DIABETES_MEDICATIONS = (
    "metformin", "insulin", "glipizide", "glyburide", "glimepiride", "gliclazide",
    "liraglutide", "semaglutide", "dulaglutide", "exenatide", "tirzepatide",
    "sitagliptin", "saxagliptin", "linagliptin", "alogliptin",
    "empagliflozin", "dapagliflozin", "canagliflozin", "ertugliflozin",
    "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "repaglinide", "nateglinide",
)
_DIABETES_MED_RE = re.compile("|".join(DIABETES_MEDICATIONS), re.IGNORECASE)

FEATURES = [
    "age", "sex_male",
    "enc_ambulatory", "enc_emergency", "enc_inpatient",
    "hypertension", "hyperlipidemia",
    "bmi_latest", "sbp_latest", "dbp_latest",
    "med_count",
]


class LeakageError(Exception):
    pass


def conditions_with(conditions, codes):
    """The only way features read conditions. Refuses codes outside the allowlist."""
    codes = frozenset(codes)
    if not codes <= ALLOWED_CONDITION_CODES:
        raise LeakageError(f"condition codes not allowlisted: {sorted(codes - ALLOWED_CONDITION_CODES)}")
    return conditions[conditions["CODE"].isin(codes)]


def observations_with(observations, code):
    """The only way features read observations. Refuses codes outside the allowlist."""
    if code not in ALLOWED_OBSERVATION_CODES:
        raise LeakageError(f"observation code not allowlisted: {code}")
    return observations[observations["CODE"] == code]


def is_diabetes_medication(description):
    return description.str.contains(_DIABETES_MED_RE, na=False)


def age_years(patients, as_of):
    """Whole years at death, or at as_of if alive."""
    end = patients["DEATHDATE"].fillna(pd.Timestamp(as_of))
    birth = patients["BIRTHDATE"]
    had_birthday = (end.dt.month > birth.dt.month) | (
        (end.dt.month == birth.dt.month) & (end.dt.day >= birth.dt.day)
    )
    return end.dt.year - birth.dt.year - (~had_birthday).astype(int)


def sex_male(patients):
    return (patients["GENDER"] == "M").astype(int)


def encounter_counts(ids, encounters):
    counts = (
        encounters[encounters["ENCOUNTERCLASS"].isin(ENCOUNTER_CLASSES)]
        .groupby(["PATIENT", "ENCOUNTERCLASS"]).size().unstack(fill_value=0)
        .reindex(index=ids, columns=list(ENCOUNTER_CLASSES), fill_value=0)
    )
    return counts.add_prefix("enc_").astype(int)


def condition_flag(ids, conditions, codes):
    hit = set(conditions_with(conditions, codes)["PATIENT"])
    return pd.Series(ids.isin(hit).astype(int).values, index=ids.values)


def latest_value(ids, observations, code):
    """Most recent numeric value; mean if several on the same latest date. NaN if none."""
    obs = observations_with(observations, code).dropna(subset=["VALUE", "DATE"])
    latest_date = obs.groupby("PATIENT")["DATE"].transform("max")
    value = obs[obs["DATE"] == latest_date].groupby("PATIENT")["VALUE"].mean()
    return value.reindex(ids.values)


def medication_count(ids, medications):
    """Distinct medication codes ever prescribed, excluding glucose-lowering drugs."""
    meds = medications[~is_diabetes_medication(medications["DESCRIPTION"])]
    return meds.groupby("PATIENT")["CODE"].nunique().reindex(ids.values, fill_value=0).astype(int)


def default_as_of(tables):
    """Reference date for age of living patients: latest encounter date in the data."""
    return tables["encounters"]["START"].max()


def build_features(tables, as_of=None):
    """One row per patient (index = patient Id, same order as patients.csv), columns = FEATURES."""
    patients = tables["patients"]
    ids = patients["Id"]
    as_of = default_as_of(tables) if as_of is None else pd.Timestamp(as_of)
    out = pd.DataFrame(index=pd.Index(ids.values, name="patient_id"))
    out["age"] = age_years(patients, as_of).values
    out["sex_male"] = sex_male(patients).values
    out = out.join(encounter_counts(ids, tables["encounters"]))
    out["hypertension"] = condition_flag(ids, tables["conditions"], HYPERTENSION_CODES)
    out["hyperlipidemia"] = condition_flag(ids, tables["conditions"], HYPERLIPIDEMIA_CODES)
    out["bmi_latest"] = latest_value(ids, tables["observations"], BMI)
    out["sbp_latest"] = latest_value(ids, tables["observations"], SYSTOLIC_BP)
    out["dbp_latest"] = latest_value(ids, tables["observations"], DIASTOLIC_BP)
    out["med_count"] = medication_count(ids, tables["medications"])
    return out[FEATURES]

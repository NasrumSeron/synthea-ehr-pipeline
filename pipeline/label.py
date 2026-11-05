"""Diabetes label (v0.1).

diabetes = 1 if the patient has any condition whose CODE is either
  - a SNOMED CT descendant-or-self of 73211009 |Diabetes mellitus| (118 codes), or
  - one of the 6 complication codes under 74627003 |Complication due to diabetes mellitus|
    that occur in the cohort (e.g. 127013003 Disorder of kidney due to diabetes mellitus).
Complications count because each one is recorded as "due to" diabetes. Prediabetes and
hyperglycaemia are not diabetes and are not in either list.

Cross-sectional (ever recorded), not a prediction over time.
"""

from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent / "data"
MELLITUS_LIST = DATA / "snomed_diabetes_mellitus_descendants.csv"
COMPLICATION_LIST = DATA / "snomed_diabetes_complications_observed.csv"


def _read_codes(path):
    codes = pd.read_csv(path, dtype=str, comment="#")
    return frozenset(codes["code"].str.strip())


def mellitus_codes():
    """<<73211009, 118 codes."""
    return _read_codes(MELLITUS_LIST)


def complication_codes():
    """The 6 observed codes under <<74627003."""
    return _read_codes(COMPLICATION_LIST)


def diabetes_codes():
    """All label codes (mellitus + observed complications)."""
    return mellitus_codes() | complication_codes()


def diabetes_label(patients, conditions, codes=None):
    """Series indexed by patient Id: 1 if any label code ever recorded, else 0."""
    codes = diabetes_codes() if codes is None else codes
    positive = set(conditions.loc[conditions["CODE"].isin(codes), "PATIENT"])
    ids = patients["Id"]
    return pd.Series(ids.isin(positive).astype(int).values, index=ids.values, name="diabetes")

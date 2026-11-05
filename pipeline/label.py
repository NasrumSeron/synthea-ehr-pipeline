"""Diabetes label: patient has any condition whose CODE is a SNOMED CT descendant-or-self of 73211009.

Cross-sectional (ever recorded), not a prediction over time.
"""

from pathlib import Path

import pandas as pd

CODE_LIST = Path(__file__).parent / "data" / "snomed_diabetes_mellitus_descendants.csv"


def diabetes_codes():
    codes = pd.read_csv(CODE_LIST, dtype=str, comment="#")
    return frozenset(codes["code"].str.strip())


def diabetes_label(patients, conditions, codes=None):
    """Series indexed by patient Id: 1 if any label code ever recorded, else 0."""
    codes = diabetes_codes() if codes is None else codes
    positive = set(conditions.loc[conditions["CODE"].isin(codes), "PATIENT"])
    ids = patients["Id"]
    return pd.Series(ids.isin(positive).astype(int).values, index=ids.values, name="diabetes")

"""Load the Synthea CSV tables the pipeline needs, reading only the columns it uses."""

from pathlib import Path

import pandas as pd

from pipeline import PipelineError

# table name -> columns read from <raw_dir>/<table>.csv
TABLES = {
    "patients": ["Id", "BIRTHDATE", "DEATHDATE", "GENDER"],
    "encounters": ["Id", "START", "PATIENT", "ENCOUNTERCLASS"],
    "conditions": ["START", "PATIENT", "CODE", "DESCRIPTION"],
    "medications": ["START", "PATIENT", "CODE", "DESCRIPTION"],
    "observations": ["DATE", "PATIENT", "CODE", "DESCRIPTION", "VALUE"],
}


def load_table(raw_dir, name):
    path = Path(raw_dir) / f"{name}.csv"
    if not path.is_file():
        raise PipelineError(f"missing input file: {path}")
    columns = TABLES[name]
    header = pd.read_csv(path, nrows=0).columns
    missing = [c for c in columns if c not in header]
    if missing:
        raise PipelineError(f"{path.name} is missing column(s): {', '.join(missing)}")
    # Everything as string; clean.py does the typing.
    return pd.read_csv(path, usecols=columns, dtype=str, keep_default_na=False, na_values=[""])


def load_tables(raw_dir):
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise PipelineError(f"raw directory not found: {raw_dir}")
    return {name: load_table(raw_dir, name) for name in TABLES}

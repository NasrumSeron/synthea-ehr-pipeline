"""Type and clean the raw tables. Returns cleaned tables plus a log of what was dropped."""

import pandas as pd

DATE_COLUMNS = {
    "patients": ["BIRTHDATE", "DEATHDATE"],
    "encounters": ["START"],
    "conditions": ["START"],
    "medications": ["START"],
    "observations": ["DATE"],
}


def to_date(series):
    """Parse ISO dates or timestamps to tz-naive calendar dates (time of day dropped)."""
    parsed = pd.to_datetime(series, errors="coerce", utc=True, format="ISO8601")
    return parsed.dt.tz_localize(None).dt.normalize()


def clean_tables(tables):
    """Return (cleaned tables, log). log[table] = {"rows_in", "rows_out", "dropped_*": n}."""
    cleaned, log = {}, {}
    for name, df in tables.items():
        df = df.copy()
        entry = {"rows_in": len(df)}
        for col in df.columns:
            if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
                df[col] = df[col].str.strip().replace("", pd.NA)
        for col in DATE_COLUMNS[name]:
            df[col] = to_date(df[col])

        n = len(df)
        df = df.drop_duplicates()
        entry["dropped_exact_duplicates"] = n - len(df)

        key = "Id" if name == "patients" else "PATIENT"
        n = len(df)
        df = df[df[key].notna()]
        entry["dropped_missing_patient_id"] = n - len(df)

        if name == "patients":
            df["GENDER"] = df["GENDER"].str.upper()
            # Synthea writes rows in thread order, which varies run to run. Sorting by Id makes
            # the patient table (and so the train/test split) independent of row order.
            df = df.sort_values("Id", kind="stable")
        if name == "encounters":
            df["ENCOUNTERCLASS"] = df["ENCOUNTERCLASS"].str.lower()
        if name == "observations":
            # Non-numeric values (e.g. survey text) become NaN; features only use numeric vitals.
            df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")

        entry["rows_out"] = len(df)
        cleaned[name] = df.reset_index(drop=True)
        log[name] = entry
    return cleaned, log

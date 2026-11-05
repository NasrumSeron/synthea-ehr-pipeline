import pandas as pd

from pipeline.clean import clean_tables


def test_exact_duplicates_dropped_and_logged(raw):
    tables, log = clean_tables(raw)
    assert log["encounters"]["dropped_exact_duplicates"] == 1
    assert log["conditions"]["dropped_exact_duplicates"] == 1
    assert len(tables["encounters"]) == 11
    assert log["patients"]["rows_out"] == 5


def test_dates_parsed_to_calendar_dates(tables):
    enc = tables["encounters"].set_index("Id")
    assert enc.loc["e4", "START"] == pd.Timestamp("2024-03-03")
    p = tables["patients"].set_index("Id")
    assert p.loc["p3", "DEATHDATE"] == pd.Timestamp("2020-03-10")
    assert pd.isna(p.loc["p1", "DEATHDATE"])


def test_strings_normalised(tables):
    assert tables["encounters"].set_index("Id").loc["e11", "ENCOUNTERCLASS"] == "ambulatory"
    assert tables["patients"].set_index("Id").loc["p4", "GENDER"] == "M"


def test_non_numeric_observation_value_becomes_nan(tables):
    obs = tables["observations"]
    assert obs["VALUE"].isna().sum() == 1


def test_rows_without_patient_id_dropped(raw):
    raw["conditions"].loc[0, "PATIENT"] = None
    _, log = clean_tables(raw)
    assert log["conditions"]["dropped_missing_patient_id"] == 1

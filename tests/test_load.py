import pytest

from pipeline import PipelineError
from pipeline.load import TABLES, load_tables
from tests.conftest import FIXTURES


def test_loads_all_tables_with_only_needed_columns(raw):
    assert set(raw) == set(TABLES)
    for name, df in raw.items():
        assert list(df.columns) == TABLES[name]


def test_raw_row_counts(raw):
    counts = {name: len(df) for name, df in raw.items()}
    assert counts == {"patients": 5, "encounters": 12, "conditions": 10,
                      "medications": 11, "observations": 17}


def test_missing_directory_raises(tmp_path):
    with pytest.raises(PipelineError, match="raw directory not found"):
        load_tables(tmp_path / "nope")


def test_missing_file_raises(tmp_path):
    with pytest.raises(PipelineError, match="missing input file"):
        load_tables(tmp_path)


def test_missing_column_raises(tmp_path):
    for name in TABLES:
        (tmp_path / f"{name}.csv").write_text((FIXTURES / f"{name}.csv").read_text())
    (tmp_path / "patients.csv").write_text("Id,BIRTHDATE\np1,1960-06-15\n")
    with pytest.raises(PipelineError, match="patients.csv is missing column"):
        load_tables(tmp_path)

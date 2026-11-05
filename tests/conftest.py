from pathlib import Path

import pandas as pd
import pytest

from pipeline.clean import clean_tables
from pipeline.load import load_tables

FIXTURES = Path(__file__).parent / "fixtures"
# Latest encounter START in the fixtures (e5) -> reference date for ages of living patients.
AS_OF = pd.Timestamp("2025-06-14")


@pytest.fixture
def raw():
    return load_tables(FIXTURES)


@pytest.fixture
def tables(raw):
    return clean_tables(raw)[0]

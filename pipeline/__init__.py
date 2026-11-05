"""ETL, feature engineering and a baseline model on Synthea synthetic EHR CSVs.

Synthetic data only. Nothing here is clinically validated.
"""


class PipelineError(Exception):
    """An input problem the user can fix (bad path, missing file or column)."""

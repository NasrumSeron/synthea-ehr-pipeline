"""End-to-end on fixtures (padded to 30 patients so the model can train) and CLI error handling."""

import json
import shutil

import pandas as pd
import pytest

from pipeline.__main__ import main
from pipeline.quality import _table
from tests.conftest import FIXTURES


@pytest.fixture
def raw_dir(tmp_path):
    """Fixture CSVs plus 25 copies of p1 (label 1) or p5 (label 0) under new IDs."""
    d = tmp_path / "raw"
    shutil.copytree(FIXTURES, d)
    patients = pd.read_csv(d / "patients.csv", dtype=str)
    conditions = pd.read_csv(d / "conditions.csv", dtype=str)
    extra_p, extra_c = [], []
    for i in range(25):
        src = "p1" if i % 4 == 0 else "p5"
        extra_p.append(patients[patients["Id"] == src].assign(Id=f"x{i:02d}"))
        extra_c.append(conditions[conditions["PATIENT"] == src].assign(PATIENT=f"x{i:02d}"))
    pd.concat([patients, *extra_p]).to_csv(d / "patients.csv", index=False)
    pd.concat([conditions, *extra_c]).to_csv(d / "conditions.csv", index=False)
    return d


def test_end_to_end(raw_dir, tmp_path):
    out = tmp_path / "out"
    assert main(["--raw", str(raw_dir), "--out", str(out)]) == 0
    table = pd.read_csv(out / "patient_table.csv", index_col=0)
    assert len(table) == len(pd.read_csv(raw_dir / "patients.csv")) == 30
    assert table.index.is_unique
    assert json.loads((out / "metrics.json").read_text())["n_patients"] == 30
    report = (out / "dq_report.md").read_text(encoding="utf-8")
    assert "## Row counts per step" in report and "missing %" in report


def test_rerun_gives_identical_outputs(raw_dir, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    main(["--raw", str(raw_dir), "--out", str(a)])
    main(["--raw", str(raw_dir), "--out", str(b)])
    for name in ("metrics.json", "dq_report.md", "patient_table.csv"):
        assert (a / name).read_bytes() == (b / name).read_bytes()


def test_row_order_does_not_change_outputs(raw_dir, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    main(["--raw", str(raw_dir), "--out", str(a)])
    shuffled = pd.read_csv(raw_dir / "patients.csv", dtype=str).sample(frac=1, random_state=1)
    shuffled.to_csv(raw_dir / "patients.csv", index=False)
    main(["--raw", str(raw_dir), "--out", str(b)])
    assert (a / "metrics.json").read_bytes() == (b / "metrics.json").read_bytes()


def test_bad_raw_path_clear_message(tmp_path, capsys):
    assert main(["--raw", str(tmp_path / "missing"), "--out", str(tmp_path / "o")]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: raw directory not found") and "Traceback" not in err


def test_too_few_positives_clear_message(tmp_path, capsys):
    assert main(["--raw", str(FIXTURES), "--out", str(tmp_path / "o")]) == 2
    assert "at least 5 patients in each label class" in capsys.readouterr().err


def test_out_is_a_file(tmp_path, capsys):
    f = tmp_path / "file.txt"
    f.write_text("x")
    with pytest.raises(SystemExit) as e:
        main(["--raw", str(FIXTURES), "--out", str(f)])
    assert e.value.code == 2
    assert "--out is not a directory" in capsys.readouterr().err


def test_missing_args_shows_usage(capsys):
    with pytest.raises(SystemExit) as e:
        main([])
    assert e.value.code == 2
    assert "usage:" in capsys.readouterr().err


def test_encounters_without_dates_clear_message(raw_dir, tmp_path, capsys):
    (raw_dir / "encounters.csv").write_text("Id,START,STOP,PATIENT,ENCOUNTERCLASS\n")
    assert main(["--raw", str(raw_dir), "--out", str(tmp_path / "o")]) == 2
    assert "no parseable START dates" in capsys.readouterr().err


def test_report_escapes_pipes_in_descriptions():
    assert _table(["a"], [("x|y\nz",)])[-1] == r"| x\|y z |"

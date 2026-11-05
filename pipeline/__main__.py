"""python -m pipeline --raw data/raw/csv --out outputs"""

import argparse
import json
import sys
from pathlib import Path

from pipeline import PipelineError
from pipeline.clean import clean_tables
from pipeline.features import build_features, default_as_of
from pipeline.label import diabetes_codes, diabetes_label
from pipeline.load import load_tables
from pipeline.model import train_and_evaluate
from pipeline.quality import build_report


def run(raw_dir, out_dir):
    raw = load_tables(raw_dir)
    raw_counts = {name: len(df) for name, df in raw.items()}
    tables, clean_log = clean_tables(raw)

    as_of = default_as_of(tables)
    X = build_features(tables, as_of)
    codes = diabetes_codes()
    y = diabetes_label(tables["patients"], tables["conditions"], codes)
    patient_table = X.assign(diabetes=y.values)

    if patient_table.index.duplicated().any():
        raise PipelineError("duplicate patient IDs in patients.csv")
    if y.nunique() < 2 or y.value_counts().min() < 5:
        raise PipelineError("need at least 5 patients in each label class to train the model")

    metrics = train_and_evaluate(X, y)
    report = build_report(raw_counts, clean_log, tables, patient_table, y, codes, as_of)

    out_dir.mkdir(parents=True, exist_ok=True)
    patient_table.to_csv(out_dir / "patient_table.csv")
    (out_dir / "dq_report.md").write_text(report, encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m pipeline",
        description="Synthea CSVs -> patient table, data-quality report, baseline model metrics.",
    )
    parser.add_argument("--raw", required=True, type=Path, help="directory with Synthea CSV export")
    parser.add_argument("--out", required=True, type=Path, help="output directory (created if missing)")
    args = parser.parse_args(argv)

    if args.out.exists() and not args.out.is_dir():
        parser.error(f"--out is not a directory: {args.out}")
    try:
        metrics = run(args.raw, args.out)
    except PipelineError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"error: {e.strerror or e}: {e.filename or ''}".rstrip(": "), file=sys.stderr)
        return 2
    cm = metrics["confusion_matrix_test"]
    print(f"patients={metrics['n_patients']} prevalence={metrics['prevalence']} "
          f"AUROC={metrics['auroc_test']} TP={cm['tp']} FP={cm['fp']} FN={cm['fn']} TN={cm['tn']}")
    print(f"wrote {args.out / 'dq_report.md'}, {args.out / 'metrics.json'}, {args.out / 'patient_table.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

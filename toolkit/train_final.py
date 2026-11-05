"""Train ONE frozen model and save it as plain JSON (toolkit).

Run from the repo folder with its .venv:
  python toolkit/train_final.py --raw data/raw/csv --model models/seed42.json

Recipe = the published v0.2 protocol: stratified 80/20 split (--split-seed), model trained on the
80 % development set, threshold (Youden's J) and recalibration (a, b) learned from 5-fold
out-of-fold development predictions. The 20 % test set is not touched here.
Saved as JSON numbers (no pickle): feature list, imputer medians, scaler means/SDs, weights,
intercept, recalibration a/b, threshold, and where it was trained.
"""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def export_model(pipe, features, rec_a, rec_b, threshold, trained_on):
    """The fitted impute -> scale -> logistic pipeline as plain JSON-ready numbers."""
    imp, sc, lr = pipe[0], pipe[1], pipe[2]
    return {
        "format": "synthea-ehr-pipeline logistic model v1 (JSON, no pickle)",
        "features": list(features),
        "imputer_median": [float(v) for v in imp.statistics_],
        "scaler_mean": [float(v) for v in sc.mean_],
        "scaler_sd": [float(v) for v in sc.scale_],
        "coef": [float(v) for v in lr.coef_[0]],
        "intercept": float(lr.intercept_[0]),
        "recalibration": {"a": rec_a, "b": rec_b,
                          "method": "logistic (outcome ~ a + b*logit(p)) on out-of-fold development predictions"},
        "threshold": {"value": threshold, "scale": "raw (uncalibrated) risk",
                      "method": "Youden's J on out-of-fold development predictions"},
        "trained_on": trained_on,
        "note": "Synthetic data only. Not clinical.",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw/csv"))
    ap.add_argument("--model", type=Path, default=Path("models/model.json"))
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--repo", type=Path, default=Path("."))
    a = ap.parse_args()
    sys.path.insert(0, str(a.repo.resolve()))
    sys.path.insert(0, str(HERE))
    import sklearn
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    import pipeline.model as M
    from multi_seed import fit_recalibration, load_xy
    from predict import predict_proba_from_json

    M.RANDOM_STATE = a.split_seed
    X, y = load_xy(a.raw)
    X_dev, X_test, y_dev, _ = M.split_dev_test(X, y)
    cv = StratifiedKFold(n_splits=M.CV_FOLDS, shuffle=True, random_state=a.split_seed)
    oof = cross_val_predict(M.make_model(), X_dev, y_dev, cv=cv, method="predict_proba")[:, 1]
    threshold = M.choose_threshold(y_dev, oof)
    rec_a, rec_b = fit_recalibration(y_dev, oof)
    pipe = M.make_model().fit(X_dev, y_dev)
    trained_on = {"raw": a.raw.as_posix(), "split_seed": a.split_seed,
                  "n_development": int(len(y_dev)), "prevalence_development": round(float(y_dev.mean()), 4),
                  "date": dt.date.today().isoformat(), "sklearn": sklearn.__version__}
    model = export_model(pipe, X.columns, rec_a, rec_b, threshold, trained_on)
    # Self-check: the JSON numbers must reproduce scikit-learn's predictions exactly.
    diff = float(np.max(np.abs(predict_proba_from_json(model, X_test) - pipe.predict_proba(X_test)[:, 1])))
    if diff > 1e-9:
        sys.exit(f"error: JSON model does not reproduce sklearn (max diff {diff})")
    a.model.parent.mkdir(parents=True, exist_ok=True)
    a.model.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    print(f"saved {a.model.resolve()}  (n_dev={len(y_dev)}, threshold={threshold:.4f}, "
          f"recal a={rec_a:.4f} b={rec_b:.4f}, JSON vs sklearn max diff {diff:.1e})")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # one clear line, no traceback
        sys.exit(f"error: {type(e).__name__}: {e}")

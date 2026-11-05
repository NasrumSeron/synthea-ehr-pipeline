"""Score a cohort with a frozen JSON model, without retraining (toolkit).

  python toolkit/predict.py --model models/seed42.json --raw data/cohorts/p2000_s7/csv --out runs/ext_p2000
  python toolkit/predict.py ... --local-recal 0.5 --save-model models/seed42_recal.json

Writes predictions.csv (per patient: risk, recalibrated risk, flag). If the cohort has labels
(Synthea always does; skip with --no-labels) it also writes evaluation.json / evaluation.md:
an external validation - TP/FP/FN/TN, sensitivity, PPV, AUROC, AUPRC and calibration before
vs after the stored recalibration.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def predict_proba_from_json(model, X):
    """Impute -> scale -> linear -> sigmoid, using only the numbers in the JSON model."""
    X = X[model["features"]].astype(float).to_numpy()
    med = np.array(model["imputer_median"])
    X = np.where(np.isnan(X), med, X)
    z = (X - np.array(model["scaler_mean"])) / np.array(model["scaler_sd"])
    return 1 / (1 + np.exp(-(z @ np.array(model["coef"]) + model["intercept"])))


def recalibrate(model, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    r = model["recalibration"]
    return 1 / (1 + np.exp(-(r["a"] + r["b"] * np.log(p / (1 - p)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-labels", action="store_true")
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument("--local-recal", type=float, default=None, metavar="FRACTION",
                    help="learn a new a/b on this fraction of the new cohort (stratified), check on the rest")
    ap.add_argument("--recal-seed", type=int, default=42)
    ap.add_argument("--save-model", type=Path, default=None,
                    help="with --local-recal: write a copy of the model carrying the local a/b")
    a = ap.parse_args()
    if a.local_recal is not None and not 0.1 <= a.local_recal <= 0.9:
        sys.exit("error: --local-recal must be between 0.1 and 0.9")
    if a.local_recal is not None and a.no_labels:
        sys.exit("error: --local-recal needs labels (the outcome) in the new cohort")
    if a.save_model and a.local_recal is None:
        sys.exit("error: --save-model is only used with --local-recal")
    sys.path.insert(0, str(a.repo.resolve()))
    sys.path.insert(0, str(HERE))
    from sklearn.metrics import average_precision_score, roc_auc_score

    import pipeline.model as M
    from multi_seed import load_xy

    model = json.loads(a.model.read_text(encoding="utf-8"))
    X, y = load_xy(a.raw)
    missing = [f for f in model["features"] if f not in X.columns]
    if missing:
        sys.exit(f"error: cohort lacks model features {missing}")
    p = predict_proba_from_json(model, X)
    p_rec = recalibrate(model, p)
    thr = model["threshold"]["value"]
    out = pd.DataFrame({"patient_id": X.index, "risk": p.round(4), "risk_recalibrated": p_rec.round(4),
                        "flag": (p >= thr).astype(int)})
    if not a.no_labels:
        out["diabetes"] = y.values
    a.out.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out / "predictions.csv", index=False)
    print(f"scored {len(out)} patients with {a.model.name}; flagged {int(out['flag'].sum())} at threshold {thr:.4f}")
    if a.no_labels:
        return

    yv = y.to_numpy()
    ev = {"model": a.model.as_posix(), "cohort": a.raw.as_posix(), "n": int(len(yv)),
          "prevalence": round(float(yv.mean()), 4),
          "trained_on_prevalence": model["trained_on"]["prevalence_development"],
          "auroc": round(float(roc_auc_score(yv, p)), 4), "auprc": round(float(average_precision_score(yv, p)), 4),
          "at_threshold": M.classification_at(yv, p, thr),
          "calibration_before": M.calibration(yv, p),
          "calibration_after_stored_recalibration": M.calibration(yv, p_rec)}
    (a.out / "evaluation.json").write_text(json.dumps(ev, indent=2) + "\n", encoding="utf-8")
    cb, ca, c = ev["calibration_before"], ev["calibration_after_stored_recalibration"], ev["at_threshold"]
    md = [f"# External validation: {a.model.name} on {a.raw}", "", "Frozen model, no retraining. Synthetic data only.", "",
          "| | |", "|---|---|",
          f"| Patients / prevalence (trained on) | {ev['n']} / {ev['prevalence']} ({ev['trained_on_prevalence']}) |",
          f"| AUROC / AUPRC | {ev['auroc']} / {ev['auprc']} |",
          f"| At threshold {c['threshold']} | TP {c['tp']} Â· FP {c['fp']} Â· FN {c['fn']} Â· TN {c['tn']}; "
          f"sens {c['sensitivity']}, spec {c['specificity']}, PPV {c['ppv']} |",
          f"| Calibration before: slope / in-the-large / Brier | {cb['calibration_slope']} / {cb['calibration_in_the_large']} / {cb['brier']} |",
          f"| After stored recalibration | {ca['calibration_slope']} / {ca['calibration_in_the_large']} / {ca['brier']} |", "",
          "Risk groups (predicted -> observed), before: " + " | ".join(f"{b['mean_predicted']}->{b['observed']}" for b in cb["bins_by_predicted_risk"]),
          "", "After: " + " | ".join(f"{b['mean_predicted']}->{b['observed']}" for b in ca["bins_by_predicted_risk"])]
    (a.out / "evaluation.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[4:]))
    if a.local_recal is not None:
        res = local_recalibration(model, p, yv, a.local_recal, a.recal_seed, M)
        res["cohort"] = a.raw.as_posix()
        write_local_recal(res, a.out, model, a.model, a.save_model)


def local_recalibration(model, p, y, fraction, seed, M):
    """Split the NEW cohort (stratified) into a recalibration slice and a check slice.
    Learn a/b on the slice from the frozen model's raw risks; compare on the check slice:
    raw vs stored recalibration vs local recalibration. The model's weights never change."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    from multi_seed import apply_recalibration, fit_recalibration
    idx = np.arange(len(y))
    i_fit, i_chk = train_test_split(idx, train_size=fraction, stratify=y, random_state=seed)
    a_loc, b_loc = fit_recalibration(y[i_fit], p[i_fit])
    p_chk, y_chk = p[i_chk], y[i_chk]
    versions = {"raw": p_chk, "stored_recal": recalibrate(model, p_chk),
                "local_recal": apply_recalibration(p_chk, a_loc, b_loc)}
    res = {"fraction": fraction, "seed": seed, "n_fit": int(len(i_fit)), "n_check": int(len(i_chk)),
           "local_a": round(a_loc, 4), "local_b": round(b_loc, 4),
           "stored_a": round(model["recalibration"]["a"], 4), "stored_b": round(model["recalibration"]["b"], 4),
           "check_auroc": round(float(roc_auc_score(y_chk, p_chk)), 4), "check": {}}
    for name, q in versions.items():
        c = M.calibration(y_chk, q)
        res["check"][name] = {"brier": c["brier"], "calibration_slope": c["calibration_slope"],
                              "calibration_in_the_large": c["calibration_in_the_large"],
                              "bins": c["bins_by_predicted_risk"]}
    return res


def write_local_recal(res, out, model, model_path, save_model):
    (out / "local_recalibration.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    rows = ["", f"## Local recalibration ({res['n_fit']} patients to learn a/b, {res['n_check']} to check)", "",
            f"Stored a/b (from training cohort): {res['stored_a']} / {res['stored_b']}. "
            f"Local a/b (from this cohort): {res['local_a']} / {res['local_b']}. "
            f"AUROC on the check slice: {res['check_auroc']} for all three (ranking unchanged).", "",
            "| Check slice | Calibration slope | Calibration-in-the-large | Brier | Top risk group: predicted -> observed |",
            "|---|---|---|---|---|"]
    for name, c in res["check"].items():
        top = c["bins"][-1]
        rows.append(f"| {name} | {c['calibration_slope']} | {c['calibration_in_the_large']} | {c['brier']} | "
                    f"{top['mean_predicted']} -> {top['observed']} |")
    with open(out / "evaluation.md", "a", encoding="utf-8") as fh:
        fh.write("\n".join(rows) + "\n")
    print("\n".join(rows))
    if save_model:
        m2 = json.loads(json.dumps(model))
        m2["recalibration"] = {"a": res["local_a"], "b": res["local_b"],
                               "method": "local logistic recalibration on the new cohort",
                               "learned_on": {"cohort": res.get("cohort"), "n": res["n_fit"],
                                              "fraction": res["fraction"], "seed": res["seed"]},
                               "previous": model["recalibration"]}
        m2["parent_model"] = model_path.as_posix()
        save_model.parent.mkdir(parents=True, exist_ok=True)
        save_model.write_text(json.dumps(m2, indent=2) + "\n", encoding="utf-8")
        print(f"saved recalibrated copy: {save_model.resolve()} (weights unchanged; only a/b replaced)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # one clear line, no traceback
        sys.exit(f"error: {type(e).__name__}: {e}")

"""Run the synthea-ehr-pipeline model over several seeds, optionally with recalibration.
Toolkit. Run from the repo folder with its .venv (pip install -r requirements-toolkit.txt):

  # different development/test splits of one cohort (fast)
  python toolkit/multi_seed.py --mode split --n-seeds 10 --raw data/raw/csv
  # new Synthea cohorts (slow: generates each cohort with Java)
  python toolkit/multi_seed.py --mode cohort --n-seeds 5 --population 1000
  # your own seeds, with recalibration
  python toolkit/multi_seed.py --mode split --seeds 1 7 42 --recalibrate

--n-seeds draws seeds from --master-seed, so "5 random seeds" is repeatable.
Outputs (default runs/multi_<mode>): runs.csv, summary.csv, summary.md, charts, index.html.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402

SYNTHEA_FIXED = ["-r", "20261101", "-e", "20261101", "--exporter.csv.export", "true",
                 "--exporter.fhir.export", "false", "--exporter.hospital.fhir.export", "false",
                 "--exporter.practitioner.fhir.export", "false"]


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_recalibration(y, proba):
    """Logistic recalibration: outcome ~ a + b * logit(proba). Returns (a, b)."""
    fit = LogisticRegression(C=1e12, max_iter=1000).fit(logit(proba).reshape(-1, 1), y)
    return float(fit.intercept_[0]), float(fit.coef_[0][0])


def apply_recalibration(proba, a, b):
    return 1 / (1 + np.exp(-(a + b * logit(proba))))


def load_xy(raw_dir):
    from pipeline.clean import clean_tables
    from pipeline.features import MODEL_FEATURES, build_features, default_as_of
    from pipeline.label import diabetes_label
    from pipeline.load import load_tables
    tables, _ = clean_tables(load_tables(raw_dir))
    X = build_features(tables, default_as_of(tables))
    y = diabetes_label(tables["patients"], tables["conditions"])
    return X[MODEL_FEATURES], y


def evaluate(X, y, split_seed, recalibrate):
    """One run of the published protocol with RANDOM_STATE = split_seed; optional recalibration."""
    import pipeline.model as M
    M.RANDOM_STATE = split_seed  # model functions read this at call time
    m = M.train_and_evaluate(X, y)
    t = m["test"]
    row = {"n_patients": m["n_patients"], "prevalence": m["prevalence"],
           "dev_cv_auroc": m["dev_cv"]["auroc_mean"], "test_auroc": t["auroc"], "test_auprc": t["auprc"],
           "brier": t["calibration"]["brier"], "cal_slope": t["calibration"]["calibration_slope"],
           "cal_in_large": t["calibration"]["calibration_in_the_large"],
           "threshold": m["threshold"]["value"],
           "sensitivity": t["at_chosen_threshold"]["sensitivity"],
           "specificity": t["at_chosen_threshold"]["specificity"],
           "ppv": t["at_chosen_threshold"]["ppv"]}
    row.update({f"OR_{k}": v for k, v in m["odds_ratio_per_sd"].items()})
    if recalibrate:
        # Same split and folds as above. Recalibration is learned on out-of-fold DEVELOPMENT
        # predictions only; the test set is used once, to compare before vs after.
        X_dev, X_test, y_dev, y_test = M.split_dev_test(X, y)
        cv = StratifiedKFold(n_splits=M.CV_FOLDS, shuffle=True, random_state=split_seed)
        oof = cross_val_predict(M.make_model(), X_dev, y_dev, cv=cv, method="predict_proba")[:, 1]
        a, b = fit_recalibration(y_dev, oof)
        proba = M.make_model().fit(X_dev, y_dev).predict_proba(X_test)[:, 1]
        after = apply_recalibration(proba, a, b)
        cal = M.calibration(y_test, after)
        auc_after = round(float(roc_auc_score(y_test, after)), 4)
        if b > 0 and auc_after != t["auroc"]:
            print(f"  note: AUROC changed after recalibration ({t['auroc']} -> {auc_after})")
        row.update({"recal_a": round(a, 4), "recal_b": round(b, 4),
                    "brier_after": cal["brier"], "cal_slope_after": cal["calibration_slope"],
                    "cal_in_large_after": cal["calibration_in_the_large"], "test_auroc_after": auc_after})
    return row


def make_cohort(repo, data_root, population, seed):
    out = data_root / f"p{population}_s{seed}"
    csv = out / "csv"
    if (csv / "patients.csv").is_file():
        print(f"  cohort p{population}_s{seed}: using existing {csv}")
        return csv
    jar = repo / "tools" / "synthea-with-dependencies.jar"
    if not jar.is_file():
        sys.exit(f"error: {jar} not found - run scripts\\generate_data.ps1 once first")
    print(f"  cohort p{population}_s{seed}: generating with Synthea (takes minutes)...")
    cmd = ["java", "-jar", str(jar), "-p", str(population), "-s", str(seed), "-cs", str(seed),
           *SYNTHEA_FIXED, "--exporter.baseDirectory", str(out)]
    log = data_root / f"p{population}_s{seed}.log"
    data_root.mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc != 0 or not (csv / "patients.csv").is_file():
        raise RuntimeError(f"Synthea failed (exit {rc}); see {log}")
    return csv


def charts(df, out, recal):
    made = []
    fig, ax = plt.subplots(figsize=(7, 4))
    cols = ["dev_cv_auroc", "test_auroc", "test_auprc", "cal_slope"] + (["cal_slope_after"] if recal else [])
    for i, c in enumerate(cols):
        ax.scatter([i] * len(df), df[c], alpha=0.7)
        ax.hlines(df[c].mean(), i - 0.25, i + 0.25, color="black")
    ax.axhline(1, ls=":", color="grey")
    ax.set_xticks(range(len(cols)), cols, rotation=20)
    ax.set_title(f"Spread across {len(df)} seeds (black line = mean)")
    fig.tight_layout(); fig.savefig(out / "1_metrics_spread.png", dpi=120); plt.close(fig)
    made.append("1_metrics_spread.png")

    or_cols = [c for c in df.columns if c.startswith("OR_")]
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, c in enumerate(or_cols):
        ax.scatter(df[c], [i] * len(df), alpha=0.7)
    ax.axvline(1, ls="--", color="grey")
    ax.set_xscale("log")
    ax.set_xticks([0.5, 0.75, 1, 1.5, 2, 3, 4])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks(range(len(or_cols)), [c[3:] for c in or_cols])
    ax.set(xlabel="odds ratio per SD (log scale)", title="Odds ratios per seed (cross 1 = not stable)")
    fig.tight_layout(); fig.savefig(out / "2_odds_ratios_spread.png", dpi=120); plt.close(fig)
    made.append("2_odds_ratios_spread.png")

    if recal:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for k, (b, a_, ttl, ideal) in enumerate([("cal_slope", "cal_slope_after", "Calibration slope", 1),
                                                  ("brier", "brier_after", "Brier score (lower = better)", None)]):
            for _, r in df.iterrows():
                ax[k].plot([0, 1], [r[b], r[a_]], "o-", alpha=0.6)
            if ideal is not None:
                ax[k].axhline(ideal, ls=":", color="grey")
            ax[k].set_xticks([0, 1], ["before", "after recalibration"])
            ax[k].set_title(ttl + " (test set, one line per seed)")
        fig.tight_layout(); fig.savefig(out / "3_recalibration.png", dpi=120); plt.close(fig)
        made.append("3_recalibration.png")
    return made


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["split", "cohort"], required=True)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", help="explicit seeds (overrides --n-seeds)")
    ap.add_argument("--master-seed", type=int, default=2026, help="draws the --n-seeds seeds")
    ap.add_argument("--raw", type=Path, default=Path("data/raw/csv"), help="split mode: cohort to use")
    ap.add_argument("--population", type=int, default=1000, help="cohort mode: Synthea -p")
    ap.add_argument("--split-seed", type=int, default=42, help="cohort mode: split seed kept fixed")
    ap.add_argument("--recalibrate", action="store_true")
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument("--data-root", type=Path, default=Path("data/cohorts"))
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    repo = a.repo.resolve()
    sys.path.insert(0, str(repo))
    seeds = a.seeds or sorted(np.random.default_rng(a.master_seed).choice(
        np.arange(1, 100000), size=a.n_seeds, replace=False).tolist())
    out = a.out or Path(f"runs/multi_{a.mode}{'_recal' if a.recalibrate else ''}")
    out.mkdir(parents=True, exist_ok=True)
    print(f"mode={a.mode} seeds={seeds} recalibrate={a.recalibrate} -> {out.resolve()}")

    X = y = None
    if a.mode == "split":
        X, y = load_xy(a.raw)
    rows = []
    for s in seeds:
        try:
            if a.mode == "cohort":
                X, y = load_xy(make_cohort(repo, a.data_root.resolve(), a.population, s))
                row = evaluate(X, y, a.split_seed, a.recalibrate)
            else:
                row = evaluate(X, y, s, a.recalibrate)
            rows.append({"seed": s, **row})
            print(f"  seed {s}: test AUROC {row['test_auroc']} slope {row['cal_slope']}"
                  + (f" -> {row['cal_slope_after']} (a={row['recal_a']}, b={row['recal_b']})" if a.recalibrate else ""))
        except Exception as e:  # keep going; record the failure
            print(f"  seed {s}: FAILED ({type(e).__name__}: {e})")
            rows.append({"seed": s, "error": f"{type(e).__name__}: {e}"})

    df = pd.DataFrame(rows)
    df.to_csv(out / "runs.csv", index=False)
    ok = df[df["error"].isna()] if "error" in df.columns else df
    if ok.empty:
        sys.exit("all seeds failed - see runs.csv")
    num = ok.drop(columns=["seed"]).select_dtypes("number")
    summary = num.agg(["mean", "std", "min", "max"]).T.round(4)
    or_cols = [c for c in num.columns if c.startswith("OR_")]
    summary["seeds_above_1"] = [int((ok[c] > 1).sum()) if c in or_cols else "" for c in summary.index]
    summary.to_csv(out / "summary.csv")
    md = [f"# Multi-seed summary ({a.mode} mode, {len(ok)} of {len(df)} seeds OK)", "",
          f"Seeds: {seeds}. Recalibration: {a.recalibrate}. Synthetic data only.", "",
          "```", summary.to_string(), "```"]
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    made = charts(ok, out, a.recalibrate)
    html = ["<html><head><meta charset='utf-8'><title>multi-seed</title></head>",
            "<body style='font-family:sans-serif;max-width:1100px;margin:auto'>",
            f"<h1>Multi-seed: {a.mode} mode, {len(ok)} seeds</h1>",
            "<pre>" + summary.to_string() + "</pre>"]
    html += [f"<img src='{p}' style='max-width:100%'>" for p in made]
    (out / "index.html").write_text("\n".join(html + ["</body></html>"]), encoding="utf-8")
    print(f"wrote runs.csv, summary.csv, summary.md, {len(made)} charts, index.html to {out.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # one clear line, no traceback
        sys.exit(f"error: {type(e).__name__}: {e}")

"""Simple charts for a synthea-ehr-pipeline run (toolkit).

Usage (from the repo folder, with its .venv active):
    python toolkit/make_plots.py --out runs/baseline --plots runs/baseline/plots

Reads <out>/metrics.json and <out>/patient_table.csv, writes PNGs + index.html to --plots.
ROC / PR curves re-create the same development/test split and model with the pipeline's own
functions, then check the test AUROC matches metrics.json.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve  # noqa: E402


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs", type=Path, help="pipeline output folder")
    ap.add_argument("--plots", default="runs/plots", type=Path, help="where to write charts")
    ap.add_argument("--repo", default=".", type=Path, help="repo folder (for the pipeline package)")
    a = ap.parse_args()
    sys.path.insert(0, str(a.repo.resolve()))
    from pipeline.features import MODEL_FEATURES
    from pipeline.model import make_model, split_dev_test

    m = json.loads((a.out / "metrics.json").read_text(encoding="utf-8"))
    pt = pd.read_csv(a.out / "patient_table.csv", index_col=0)
    a.plots.mkdir(parents=True, exist_ok=True)
    t = m["test"]
    made = []

    # 1 ROC + PR (recomputed, checked against metrics.json)
    X, y = pt[MODEL_FEATURES], pt["diabetes"]
    X_dev, X_test, y_dev, y_test = split_dev_test(X, y)
    proba = make_model().fit(X_dev, y_dev).predict_proba(X_test)[:, 1]
    auc = round(float(roc_auc_score(y_test, proba)), 4)
    if auc != t["auroc"]:
        print(f"warning: recomputed AUROC {auc} != metrics.json {t['auroc']}")
    fpr, tpr, _ = roc_curve(y_test, proba)
    prec, rec, _ = precision_recall_curve(y_test, proba)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    ax[0].plot(fpr, tpr, label=f"model (AUROC {t['auroc']})")
    ax[0].plot([0, 1], [0, 1], "--", color="grey", label="chance")
    th = t["at_chosen_threshold"]
    ax[0].scatter([1 - th["specificity"]], [th["sensitivity"]], color="red", zorder=3,
                  label=f"chosen threshold {th['threshold']}")
    ax[0].set(xlabel="1 - specificity", ylabel="sensitivity", title="ROC (test set)")
    ax[0].legend(loc="lower right")
    ax[1].plot(rec, prec, label=f"model (AUPRC {t['auprc']})")
    ax[1].axhline(t["auprc_chance_level"], ls="--", color="grey", label=f"chance ({t['auprc_chance_level']})")
    ax[1].set(xlabel="recall (sensitivity)", ylabel="precision (PPV)", title="Precision-recall (test set)")
    ax[1].legend(loc="upper right")
    made.append(save(fig, a.plots / "1_roc_pr.png"))

    # 2 Calibration (from metrics.json bins)
    bins = t["calibration"]["bins_by_predicted_risk"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    ax.plot([b["mean_predicted"] for b in bins], [b["observed"] for b in bins], "o-",
            label=f"model (slope {t['calibration']['calibration_slope']}, Brier {t['calibration']['brier']})")
    ax.set(xlabel="mean predicted risk", ylabel="observed proportion with diabetes",
           title="Calibration (test set, 5 groups)", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left")
    made.append(save(fig, a.plots / "2_calibration.png"))

    # 3 Decision curve
    dc = t["decision_curve"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    xs = [d["threshold"] for d in dc]
    ax.plot(xs, [d["model"] for d in dc], "o-", label="model")
    ax.plot(xs, [d["treat_all"] for d in dc], "s--", label="treat all")
    ax.plot(xs, [0] * len(xs), ":", color="grey", label="treat none")
    ax.set(xlabel="threshold probability", ylabel="net benefit", title="Decision curve (test set)",
           ylim=(min(-0.05, min(d["model"] for d in dc)), max(d["treat_all"] for d in dc) + 0.05))
    ax.legend()
    made.append(save(fig, a.plots / "3_decision_curve.png"))

    # 4 Odds ratios per SD
    ors = pd.Series(m["odds_ratio_per_sd"]).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(ors.index, ors.values, color=["tab:blue" if v >= 1 else "tab:orange" for v in ors.values])
    ax.axvline(1, color="grey", ls="--")
    ax.set_xscale("log")
    ax.set_xticks([0.5, 0.75, 1, 1.5, 2, 3])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set(xlabel="odds ratio per SD (log scale)", title="Associations (model fitted on development set)")
    made.append(save(fig, a.plots / "4_odds_ratios.png"))

    # 5 Feature distributions by label
    cols = [c for c in ("age", "bmi_latest", "sbp_latest", "dbp_latest") if c in pt]
    fig, axes = plt.subplots(1, len(cols), figsize=(3.2 * len(cols), 4))
    for ax, c in zip(axes, cols):
        groups = [pt.loc[pt["diabetes"] == k, c].dropna() for k in (0, 1)]
        ax.boxplot(groups, tick_labels=["no diabetes", "diabetes"])
        ax.set_title(c)
    fig.suptitle("Feature distributions by label (all patients)")
    made.append(save(fig, a.plots / "5_features_by_label.png"))

    # 6 Prevalence by age band and by binary features
    bands = pd.cut(pt["age"], bins=[0, 18, 30, 40, 50, 60, 70, 200], right=False,
                   labels=["<18", "18-29", "30-39", "40-49", "50-59", "60-69", "70+"])
    prev = pt.groupby(bands, observed=False)["diabetes"].agg(["mean", "size"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(prev.index.astype(str), prev["mean"])
    for i, (p, n) in enumerate(zip(prev["mean"], prev["size"])):
        ax[0].text(i, p, f"n={n}", ha="center", va="bottom", fontsize=8)
    ax[0].set(xlabel="age band", ylabel="proportion with diabetes", title="Prevalence by age band")
    flags = [c for c in ("sex_male", "hypertension", "hyperlipidemia") if c in pt]
    vals = [(f"{c}={k}", pt.loc[pt[c] == k, "diabetes"].mean()) for c in flags for k in (0, 1)]
    ax[1].bar([v[0] for v in vals], [v[1] for v in vals])
    ax[1].set(ylabel="proportion with diabetes", title="Prevalence by 0/1 feature")
    ax[1].tick_params(axis="x", rotation=30)
    made.append(save(fig, a.plots / "6_prevalence.png"))

    html = ["<html><head><meta charset='utf-8'><title>synthea-ehr-pipeline charts</title></head>",
            "<body style='font-family:sans-serif;max-width:1100px;margin:auto'>",
            "<h1>synthea-ehr-pipeline charts</h1>",
            f"<p>Synthetic data only. Test AUROC {t['auroc']}, AUPRC {t['auprc']}, "
            f"n = {m['n_patients']}, prevalence {m['prevalence']}.</p>"]
    html += [f"<h2>{p[2:-4].replace('_', ' ')}</h2><img src='{p}' style='max-width:100%'>" for p in made]
    (a.plots / "index.html").write_text("\n".join(html + ["</body></html>"]), encoding="utf-8")
    print(f"wrote {len(made)} charts + index.html to {a.plots.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # one clear line, no traceback
        sys.exit(f"error: {type(e).__name__}: {e}")

"""Baseline logistic regression with a development / test protocol (v0.2).

1. Split once: 80 % development, 20 % test (stratified, seed 42).
2. Everything that is chosen is chosen on the development set only:
   5-fold stratified CV estimates, and the decision threshold (Youden's J on
   out-of-fold development predictions).
3. The final model is fitted on the whole development set and the test set is used
   once: discrimination (AUROC, AUPRC), calibration (Brier, calibration-in-the-large,
   calibration slope, binned table) and decision-curve net benefit.
"""

import math

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipeline import PipelineError

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
CALIBRATION_BINS = 5
DCA_THRESHOLDS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)


def make_model():
    # Imputer and scaler are fitted on training data only, inside the pipeline.
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    )


def split_dev_test(X, y):
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)


def choose_threshold(y, proba):
    """Threshold maximising Youden's J (sensitivity + specificity - 1). First maximum wins."""
    fpr, tpr, thresholds = roc_curve(y, proba)
    finite = np.isfinite(thresholds)
    j = (tpr - fpr)[finite]
    return float(thresholds[finite][int(np.argmax(j))])


def classification_at(y, proba, threshold):
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    def ratio(a, b):
        return round(a / b, 4) if b else None

    return {"threshold": round(float(threshold), 4), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "sensitivity": ratio(tp, tp + fn), "specificity": ratio(tn, tn + fp),
            "ppv": ratio(tp, tp + fp), "npv": ratio(tn, tn + fn)}


def net_benefit(y, proba, threshold):
    """Decision-curve net benefit: TP/n - FP/n * pt/(1-pt) (Vickers & Elkin 2006)."""
    y = np.asarray(y)
    pred = np.asarray(proba) >= threshold
    n = len(y)
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    return tp / n - fp / n * threshold / (1 - threshold)


def net_benefit_treat_all(y, threshold):
    prevalence = float(np.mean(y))
    return prevalence - (1 - prevalence) * threshold / (1 - threshold)


def calibration(y, proba):
    """Brier score, calibration-in-the-large (observed - mean predicted), calibration slope
    (logistic regression of the outcome on logit(predicted); ideal 1) and a binned table."""
    y = np.asarray(y)
    p = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    slope = None
    if len(np.unique(y)) == 2:
        # Very large C = effectively unpenalised.
        fit = LogisticRegression(C=1e12, max_iter=1000).fit(logit, y)
        slope = round(float(fit.coef_[0][0]), 4)
    order = np.argsort(p, kind="stable")
    bins = []
    for idx in np.array_split(order, min(CALIBRATION_BINS, len(p))):
        bins.append({"n": int(len(idx)), "mean_predicted": round(float(p[idx].mean()), 4),
                     "observed": round(float(y[idx].mean()), 4)})
    return {"brier": round(float(brier_score_loss(y, p)), 4),
            "calibration_in_the_large": round(float(y.mean() - p.mean()), 4),
            "calibration_slope": slope,
            "bins_by_predicted_risk": bins}


def train_and_evaluate(X, y):
    X_dev, X_test, y_dev, y_test = split_dev_test(X, y)
    if y_dev.value_counts().reindex([0, 1], fill_value=0).min() < CV_FOLDS:
        raise PipelineError(f"need at least {CV_FOLDS} patients in each label class in the development set")

    # --- development set only ---
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_auroc = cross_val_score(make_model(), X_dev, y_dev, cv=cv, scoring="roc_auc")
    cv_auprc = cross_val_score(make_model(), X_dev, y_dev, cv=cv, scoring="average_precision")
    oof = cross_val_predict(make_model(), X_dev, y_dev, cv=cv, method="predict_proba")[:, 1]
    threshold = choose_threshold(y_dev, oof)
    model = make_model().fit(X_dev, y_dev)

    # --- test set, used once ---
    proba = model.predict_proba(X_test)[:, 1]
    coefs = model[-1].coef_[0]
    return {
        "model": "LogisticRegression (median impute + standard scale)",
        "features": list(X.columns),
        "split": {"test_size": TEST_SIZE, "stratified": True, "random_state": RANDOM_STATE},
        "n_patients": int(len(y)),
        "n_dev": int(len(y_dev)),
        "n_test": int(len(y_test)),
        "prevalence": round(float(y.mean()), 4),
        "prevalence_dev": round(float(y_dev.mean()), 4),
        "prevalence_test": round(float(y_test.mean()), 4),
        "dev_cv": {
            "folds": CV_FOLDS,
            "auroc_folds": [round(float(a), 4) for a in cv_auroc],
            "auroc_mean": round(float(cv_auroc.mean()), 4),
            "auroc_std": round(float(cv_auroc.std()), 4),
            "auprc_mean": round(float(cv_auprc.mean()), 4),
        },
        "threshold": {"method": "Youden's J on out-of-fold development predictions",
                      "value": round(threshold, 4)},
        "test": {
            "auroc": round(float(roc_auc_score(y_test, proba)), 4),
            "auprc": round(float(average_precision_score(y_test, proba)), 4),
            "auprc_chance_level": round(float(y_test.mean()), 4),
            "calibration": calibration(y_test, proba),
            "at_chosen_threshold": classification_at(y_test, proba, threshold),
            "at_threshold_0_5": classification_at(y_test, proba, 0.5),
            "decision_curve": [
                {"threshold": t, "model": round(net_benefit(y_test, proba, t), 4),
                 "treat_all": round(net_benefit_treat_all(y_test, t), 4), "treat_none": 0.0}
                for t in DCA_THRESHOLDS],
        },
        "standardized_coefficients": {f: round(float(c), 4) for f, c in zip(X.columns, coefs)},
        "odds_ratio_per_sd": {f: round(math.exp(float(c)), 4) for f, c in zip(X.columns, coefs)},
    }

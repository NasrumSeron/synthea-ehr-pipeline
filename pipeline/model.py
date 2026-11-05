"""Baseline: logistic regression on scaled features, stratified 80/20 split."""

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
THRESHOLD = 0.5


def make_model():
    # Imputer and scaler are fitted on the training split only, inside the pipeline.
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    )


def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    model = make_model().fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    coefs = model[-1].coef_[0]
    # Few positives in one test split makes its AUROC noisy; CV on the same data shows the spread.
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_auc = cross_val_score(make_model(), X, y, cv=cv, scoring="roc_auc")
    return {
        "model": "LogisticRegression (median impute + standard scale)",
        "split": {"test_size": TEST_SIZE, "stratified": True, "random_state": RANDOM_STATE},
        "n_patients": int(len(y)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "prevalence": round(float(y.mean()), 4),
        "prevalence_train": round(float(y_train.mean()), 4),
        "prevalence_test": round(float(y_test.mean()), 4),
        "auroc_test": round(float(roc_auc_score(y_test, proba)), 4),
        f"auroc_cv{CV_FOLDS}_mean": round(float(cv_auc.mean()), 4),
        f"auroc_cv{CV_FOLDS}_std": round(float(cv_auc.std()), 4),
        f"auroc_cv{CV_FOLDS}_folds": [round(float(a), 4) for a in cv_auc],
        "threshold": THRESHOLD,
        "confusion_matrix_test": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "standardized_coefficients": {f: round(float(c), 4) for f, c in zip(X.columns, coefs)},
    }

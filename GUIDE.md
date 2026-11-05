# Practitioner's guide

How to run, understand, change and extend this repo — from a first run to a frozen model tested on new data.
Written for someone in their first data-science role; **Senior notes** mark the judgement calls.

> **Synthetic data only. Not clinical.** The question the model answers is which recorded patient characteristics are *associated* with having diabetes in a synthetic cohort. It is not a risk-prediction model.

A formatted version of this guide is in [`docs/index.html`](docs/index.html) (also served with GitHub Pages).

---

## In 60 seconds

| | |
|---|---|
| **What it is** | A tested pipeline: raw EHR-style CSVs → one row per patient → data-quality report → logistic regression with honest evaluation |
| **What you can do with it** | Reproduce, change features or models safely, test robustness across seeds, freeze a model, score new data, recalibrate it for a new site |
| **What it is not** | Clinical evidence, a Singapore model, or a prediction model (yet) |
| **Time to first result** | About 15 minutes |

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate                   # macOS/Linux: source .venv/bin/activate
pip install -r requirements-toolkit.txt  # pipeline + charts
python -m pytest -q                      # expect 81 passed
powershell -File scripts/generate_data.ps1   # macOS/Linux: bash scripts/generate_data.sh
python -m pipeline --raw data/raw/csv --out runs/baseline
python toolkit/make_plots.py --out runs/baseline --plots runs/baseline/plots
```

You need Python 3.12+ and, for data generation only, Java 17+. The pipeline should print `test AUROC=0.9144 AUPRC=0.6088 Brier=0.084 threshold=0.2472`. If it does not, stop and find out why before changing anything.

Everything you generate goes into `runs/`, `models/` or `data/cohorts/`. Git ignores those folders, so experiments never end up in a commit by accident.

---

## 1. The map

```
 A Generate ─► B Load ─► C Clean ─► D Features ─► E Label ─► F Quality ─► G Model + evaluate
 (Synthea)    └──────────────────── python -m pipeline ────────────────────────────┘
                                                                                   │
 H Look (charts) ◄─────────────────────────────────────────────────────────────────┘
 I Robustness (many seeds)  ─►  J Freeze, score new data, recalibrate locally
```

| Phase | What happens | Where | What you can change | What to check |
|---|---|---|---|---|
| A. Generate | Synthea simulates patients and writes CSVs | `scripts/generate_data.*` | Population, seeds, dates, place | Rows in `patients.csv` |
| B. Load | Reads 5 tables, only the columns used; clear error if anything is missing | `pipeline/load.py` | Add a column if a new feature needs it | Error names the file and column |
| C. Clean | Trims text, parses dates, drops duplicates and rows without a patient ID, sorts patients | `pipeline/clean.py` | Cleaning rules | Row counts per step in `dq_report.md` |
| D. Features | 11 columns per patient, read through an allowlist; 7 go into the model | `pipeline/features.py` | `MODEL_FEATURES`, allowlists, new features | Leakage tests still pass |
| E. Label | diabetes = 1 if any of 124 SNOMED CT codes is recorded | `pipeline/label.py`, `pipeline/data/*.csv` | Code lists | "Codes outside the label" in `dq_report.md` |
| F. Quality | Missing %, min/max, codes found and not found, excluded drugs | `pipeline/quality.py` | What is reported | Anything implausible (age 110, systolic BP 42) |
| G. Model | 80/20 split, 5-fold CV on development, threshold, final fit, test set used once | `pipeline/model.py` | Split %, folds, seed, model, threshold rule | `metrics.json` |
| H. Look | Six charts and an HTML page | `toolkit/make_plots.py` | Which charts | Calibration and ROC shape |
| I. Robustness | Same recipe over many splits or new cohorts | `toolkit/multi_seed.py` | Seeds, mode, recalibration | The spread, not one number |
| J. Freeze + deploy | Save one model as JSON; score new data; recalibrate on the new site | `toolkit/train_final.py`, `toolkit/predict.py` | Recalibration share and seed | External AUROC and calibration |

**Senior notes.** Phases B to F are deterministic: the same CSVs give identical outputs. Only phase G uses randomness, fixed by one seed.

---

## 2. Change one thing at a time

Change one thing on a branch, judge it on development data across many splits, and never use the test set to decide.

1. **Branch:** `git switch -c exp/age-spline`
2. **Change one thing:** a label list, a feature, a model setting or a split setting (section 3).
3. **Test:** `python -m pytest -q`. Leakage tests must still pass. Tests built around a locked value (such as the model-input list) fail by design; update them only for a deliberate change.
4. **Compare on development data:** `python toolkit/multi_seed.py --mode split --n-seeds 10`. Read `dev_cv_auroc` and `cal_slope` as mean ± SD.
5. **Decide:** keep it only if it beats the baseline by more than the seed-to-seed spread. Otherwise `git restore .` and `git switch main`.
6. **Log it:** branch, change, mean ± SD before and after, kept or dropped.

**Rules that keep it honest**

- The `test_*` numbers in `metrics.json` are for the very end. Choosing versions by test AUROC turns the test set into a second development set.
- Every new feature goes through the allowlist and gets a leakage test. Ask: *is this known before the outcome?*
- A new package means `requirements.txt`, `pip-audit` and a security check again.

**Senior notes.** Ten splits of the same patients share most of their data, so their spread understates the real uncertainty. Cohort mode (new patients) or bootstrap intervals (not built) give a firmer answer.

---

## 3. Settings you can change

**Data generation** (Synthea flags inside `scripts/generate_data.*`, or run `java -jar tools/synthea-with-dependencies.jar ...` yourself)

| Flag | Default | Effect |
|---|---|---|
| `-p` population | 1000 (1,146 rows incl. deceased) | Bigger = steadier estimates, longer run |
| `-s` / `-cs` seeds | 42 / 42 | A new seed = a new synthetic population |
| `-r` / `-e` dates | 20261101 | Fix the simulation clock; keep fixed |
| `-g`, `-a 40-70` | none | Restrict sex or age range |
| state, city | Massachusetts | US places only |

**Pipeline code** (change on a branch)

| File: setting | Default | Effect |
|---|---|---|
| `model.py`: `RANDOM_STATE` | 42 | Which patients are in development vs test |
| `model.py`: `TEST_SIZE` | 0.2 | Share kept for the final test (section 4) |
| `model.py`: `CV_FOLDS` | 5 | Folds inside the development set |
| `model.py`: `make_model()` | median impute → scale → logistic regression | The model (section 5) |
| `model.py`: `choose_threshold()` | Youden's J | Threshold rule (section 6) |
| `model.py`: `CALIBRATION_BINS`, `DCA_THRESHOLDS` | 5; 0.05–0.5 | Calibration and decision-curve detail |
| `features.py`: `MODEL_FEATURES` | 7 inputs | What the model sees |
| `features.py`: allowlists | 3 condition codes, 3 LOINC codes | What features may read |
| `features.py`: `DIABETES_MEDICATIONS` | 25 names | Drugs left out of `med_count` |
| `label.py` + `pipeline/data/*.csv` | 118 + 6 codes | What counts as diabetes |

**Toolkit flags**

| Script | Key flags | Defaults |
|---|---|---|
| `python -m pipeline` | `--raw`, `--out` | required |
| `toolkit/make_plots.py` | `--out`, `--plots` | `outputs`, `runs/plots` |
| `toolkit/multi_seed.py` | `--mode split\|cohort`, `--n-seeds`, `--seeds`, `--master-seed`, `--population`, `--split-seed`, `--recalibrate`, `--raw`, `--out` | 5 seeds from master seed 2026; cohorts cached in `data/cohorts/` |
| `toolkit/train_final.py` | `--raw`, `--model`, `--split-seed` | `data/raw/csv`, `models/model.json`, 42 |
| `toolkit/predict.py` | `--model`, `--raw`, `--out`, `--no-labels`, `--local-recal F`, `--recal-seed`, `--save-model` | recalibration off |

---

## 4. Splits, folds and seeds

Size the test set by the number of *outcome cases* it holds, not by a percentage. Here a 20% test set holds only about 37 diabetes cases.

| Choice | Default | Trade-off | When to change |
|---|---|---|---|
| Test share | 0.2 | Bigger test = steadier test numbers, less to train on | 0.3 gives about 55 cases here; on small data use repeated CV or bootstrap instead of one split |
| CV folds | 5 | More folds = bigger training folds, slower | 10 on larger data; stay at 5 when each fold holds few cases |
| Stratification | on | Keeps the 16% rate in every split | Keep on; it can make fold-to-fold spread look smaller than the true uncertainty |
| Split seed | 42 | Which patients are tested | Never pick the best-looking seed; vary it to measure noise |
| Cohort seed | 42 | Which synthetic population | Vary it to see whether a finding replicates |

**What the noise looks like here.** Across 10 splits of the published cohort, test AUROC ranged 0.900–0.949 and the test calibration slope 0.86–1.35.

- **Split mode** answers *did I get a lucky split?* Same patients reshuffled; seconds per run.
- **Cohort mode** answers *does it hold in a new population?* New patients; about a minute per 300. Sex was stable across 10 splits but flipped between two cohorts — only cohort mode shows that.

**Senior notes.** scikit-learn prefers 5- or 10-fold CV to leave-one-out. With several rows per patient (visits), use `StratifiedGroupKFold` with patient as the group. Riley et al. 2020 give formal sample-size criteria for model development.

---

## 5. Other models and features

Work in this order: question → label → features → model. Here the label fix moved test AUROC from 0.66 to 0.94, more than any model swap is likely to.

| Try (all in scikit-learn) | Why | Expect |
|---|---|---|
| Penalised logistic regression, penalty tuned by inner CV | Guards against over-confident weights | Small change with 7 inputs |
| Age as a curve (`SplineTransformer`) | Diabetes is near zero under 30, then rises steeply | Calibration slope closer to 1 |
| `HistGradientBoostingClassifier` | Finds non-linear effects and interactions | Little gain on 7 tabular inputs; loses odds ratios |
| Deep learning | Only for sequences, free text or images | Not suited to this table |

A systematic review found no AUROC advantage for machine learning over logistic regression in low-bias clinical studies (Christodoulou et al. 2019). Keep the simpler model unless the gain clearly beats the seed spread.

**Adding a feature**

1. Write a function in `features.py` that reads data only through `conditions_with()` or `observations_with()`.
2. Add its codes to the allowlist; add the column to `FEATURES` and, if the model should use it, `MODEL_FEATURES`.
3. Write a test with a value worked out by hand from `tests/fixtures`.
4. Check: known before the outcome? Could a programme act on it?

**Senior notes.** Choose predictors from clinical knowledge before seeing results; adding or dropping features because of one run tends to overfit.

---

## 6. Calibration, recalibration and thresholds

The model's weights decide **who ranks above whom**. Recalibration corrects **how big the percentages are**. The threshold decides **who gets flagged**. Change them separately.

**Calibration** asks: of the patients given about 20% risk, do about 20 in 100 have diabetes? It is read from a line fitted on the log-odds scale between predicted and actual outcome. Slope *b* = 1 and intercept *a* = 0 is perfect; *b* < 1 means over-confident, *a* < 0 means risks too high overall.

**Recalibration** applies that line to every prediction *p*:

```
p_new = 1 / (1 + exp(-(a + b × ln(p / (1 − p)))))
```

| Kind | a and b come from | Use when | Command |
|---|---|---|---|
| None | — | Calibration is already good | default |
| Stored | Out-of-fold predictions on the training cohort | Checking the model's own calibration | `multi_seed.py --recalibrate`; saved by `train_final.py` |
| Local | A share of the **new** site's labelled data; the rest checks it | Moving the model to a new population | `predict.py --local-recal 0.5` |

**Result on a new 2,322-patient cohort** (model trained on the published cohort, never retrained; 1,161 patients to learn a/b, 1,161 to check):

| Version | Calibration slope | Calibration-in-the-large | Brier | Top risk group: predicted vs observed |
|---|---|---|---|---|
| Raw | 0.813 | −0.030 | 0.0957 | 0.64 vs 0.50 |
| Stored a/b | 0.860 | −0.028 | 0.0945 | 0.62 vs 0.50 |
| Local a/b | 0.934 | −0.0003 | 0.0918 | 0.52 vs 0.50 |

AUROC on the check half was 0.8832 for all three: ranking never changes.

**When not to recalibrate.** With little data the correction is itself noisy: on two 300-patient cohorts it made calibration worse (slope 0.96 → 1.18 and 1.14 → 1.48).

**Thresholds.** The repo uses Youden's J on development data (0.2472), which weighs a missed case and a false alarm equally. For screening, pick the threshold from the decision curve or the real cost of each error. `predict.py` flags on raw risk at the stored threshold, so recalibration does not change who is flagged.

**Senior notes — the updating ladder.** Lightest to heaviest: intercept only → intercept and slope (this toolkit) → re-estimate some weights → full retrain. Climb only as far as the new data's size supports.

---

## 7. Freeze a model and test it on new data

External validation means: train once, freeze, score data the model has never seen — **without retraining**. Cohort mode in `multi_seed.py` retrains every time, so it is not this test.

```bash
python toolkit/train_final.py --raw data/raw/csv --model models/seed42.json
python toolkit/multi_seed.py --mode cohort --seeds 7 --population 2000      # makes data/cohorts/p2000_s7
python toolkit/predict.py --model models/seed42.json --raw data/cohorts/p2000_s7/csv --out runs/ext_p2000
python toolkit/predict.py --model models/seed42.json --raw data/cohorts/p2000_s7/csv --out runs/ext_p2000_local \
       --local-recal 0.5 --save-model models/seed42_recal.json
```

- **The JSON model** holds the 7 feature names, imputer medians, scaler means and SDs, 7 weights and an intercept, a and b, the threshold, and where it was trained. That is all a logistic regression is. JSON is used instead of pickle because a pickle file can run code when opened. `train_final.py` checks that the JSON reproduces scikit-learn exactly.
- **Outputs:** `predictions.csv` (risk, recalibrated risk, flag per patient) and, with outcomes, `evaluation.md` (TP/FP/FN/TN, sensitivity, PPV, AUROC, calibration). `--no-labels` scores data without outcomes.
- **First result:** the frozen model kept its ranking on the new cohort (AUROC 0.888 vs 0.914 internally) but over-predicted the top-risk group (64% vs 51%). Ranking usually travels better than calibration.

**Senior notes.** Keep the recalibration slice and the check slice apart, as `--local-recal` does. Learning a/b on all the new data and reporting calibration on that same data overstates the fix.

---

## 8. Adapting it: Singapore, other data, other diseases, policy

The pipeline is a scaffold: load → clean → features → label → quality → model → validate. Each adaptation changes specific files and leaves the rest alone.

| Goal | What changes | What stays | First step |
|---|---|---|---|
| **Singapore relevance** | Data source (approved access only); label lists from local coding; features: ethnicity with results by subgroup, Asian BMI bands (23, 27.5), waist, lipids | Cleaning, allowlist, protocol, toolkit | Agree data governance; map local tables to the 5 tables `load.py` expects |
| **A synthetic Singapore cohort** (idea) | Synthea demographics from Census 2020 (planning area × ethnicity × sex); disease modules recalibrated to local rates | Everything downstream | Build the demographics file; check prevalence against NPHS |
| **Other open data** (e.g. MIMIC-IV demo) | A loader that maps its tables to the 5 expected tables; ICD instead of SNOMED code lists | Features logic, protocol | Write `load_mimic.py`; rebuild label and allowlist code lists |
| **Another disease** (e.g. chronic kidney disease) | Label code list; features and allowlist; leakage list (its treatments and diagnostic tests) | Pipeline, protocol, toolkit | Pull the code list from a terminology server; list what must never be a feature |
| **True prediction** | Index date, look-back window, exclude existing cases, fixed horizon | Everything else | Add an index-date step before features |
| **Policy questions** ("which campaign?") | Different data and methods: modifiable factors (diet, sugary drinks, activity) are not in EHR data; attributable-fraction or causal methods; policy evaluation needs trials or before/after designs | The data-quality and documentation habits | Start from survey or cohort data, not this EHR table |

**Senior notes.** For policy, the evidence chain matters: a Singapore trial found the Nutri-Grade label cut sugar *purchased* from beverages by 1.51 g per serving (Shin et al. 2023). That is an intermediate outcome; linking it to fewer diabetes cases needs further steps this repo does not cover.

---

## 9. Reading the outputs

Read discrimination, calibration and usefulness as three separate questions; a model can pass one and fail another.

| Metric | Plain meaning | Good looks like | Published (v0.2) |
|---|---|---|---|
| AUROC | Chance a random diabetic patient scores above a random non-diabetic one | 0.5 = coin toss, 1 = perfect | 0.9144 |
| AUPRC | Precision averaged over all sensitivities; focuses on the 16% minority | Well above chance (= prevalence) | 0.6088 vs 0.1609 |
| Brier score | Mean squared gap between predicted risk and outcome | Lower is better | 0.084 |
| Calibration slope | Predictions too spread out (< 1) or too bunched (> 1) | Close to 1 | 0.9526 |
| Calibration-in-the-large | Observed rate minus average prediction | Close to 0 | −0.0066 |
| Sensitivity / specificity | Share of cases flagged / non-cases not flagged | Set by the use case | 0.81 / 0.89 |
| PPV | Share of flagged patients who have diabetes | Falls when prevalence falls | 0.58 |
| Net benefit | Benefit of acting on the model at a threshold, net of false alarms | Above treat-all and treat-none | above both, 0.05–0.5 |
| Odds ratio per SD | Change in odds per standard deviation, others held fixed | Stable across seeds | age 2.92, hypertension 2.91 |

**Senior notes.** For a 0/1 feature, "per SD" is not the odds ratio of having vs not having it. These are adjusted associations inside Synthea's rules: hyperlipidemia triples raw prevalence (13% → 41%) but its adjusted odds ratio is about 1.06, because age explains most of it.

---

## 10. Common mistakes and a checklist

| Mistake | Looks like | Fix |
|---|---|---|
| Narrow label | Patients with diabetes codes counted as healthy (v0 missed 94) | Read "codes outside the label" in `dq_report.md` |
| Leakage | Suspiciously high AUROC; treatments or diagnostic tests as features | Allowlist + leakage tests |
| Post-diagnosis features | Visit counts as the top predictor (v0.1) | Keep care-utilisation columns out of the model |
| Tuning on the test set | Picking the version or seed with the best test AUROC | Decide on development CV across seeds |
| Trusting one split | One calibration slope taken as truth | `multi_seed.py`; report mean ± SD |
| Recalibrating on tiny data | Calibration gets worse after the fix | Recalibrate only with enough new data |
| Calling association prediction | "Predicts diabetes" from whole-record features | Say "associated with"; time-anchor to predict |

**Before you share a result**

- [ ] `pytest` passes; leakage tests unchanged or deliberately updated
- [ ] Baseline reproduces (test AUROC 0.9144)
- [ ] Change judged on development data across 10+ split seeds
- [ ] Test set used once, after the decision
- [ ] Calibration reported, not only AUROC
- [ ] Threshold rule stated and fits the use case
- [ ] Limits written down: synthetic, US-modelled, association, no lifestyle data
- [ ] Nothing from `runs/`, `models/` or `data/` committed

---

## Sources

- Synthea — https://github.com/synthetichealth/synthea · Walonoski et al., JAMIA 2018, doi:10.1093/jamia/ocx079
- Chen et al. 2019, validity of Synthea data — doi:10.1186/s12911-019-0793-0
- scikit-learn cross-validation guide — https://scikit-learn.org/stable/modules/cross_validation.html
- Christodoulou et al. 2019, ML vs logistic regression — J Clin Epidemiol 110:12–22
- Van Calster et al. 2019, calibration — https://pmc.ncbi.nlm.nih.gov/articles/PMC6912996
- Vickers & Elkin 2006, decision curve analysis — https://pubmed.ncbi.nlm.nih.gov/17099194/
- Riley et al. 2020, sample size for prediction models — doi:10.1136/bmj.m441
- Seah et al. 2023, T2D risk by ethnicity in Singapore — doi:10.1136/bmjdrc-2023-003385
- Shin, Puri & Finkelstein 2023, Nutri-Grade randomised trial — doi:10.1186/s12966-023-01422-4
- WHO Expert Consultation 2004, BMI for Asian populations — doi:10.1016/S0140-6736(03)15268-3
- Census of Population 2020, resident population by planning area, ethnic group and sex — data.gov.sg

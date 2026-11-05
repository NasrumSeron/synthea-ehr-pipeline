# Data-quality report

Synthetic data (Synthea). Not clinical data; nothing here is clinically validated.

Age reference date for living patients (latest encounter in data): 2026-10-31

## Row counts per step

| table | raw rows | exact duplicates dropped | missing patient id dropped | rows after clean |
|---|---|---|---|---|
| patients | 1146 | 0 | 0 | 1146 |
| encounters | 71918 | 0 | 0 | 71918 |
| conditions | 41437 | 1 | 0 | 41436 |
| medications | 58033 | 317 | 0 | 57716 |
| observations | 943338 | 2671 | 0 | 940667 |

- Patient table rows: 1146 (patients.csv rows: 1146)
- Duplicate patient IDs in patient table: 0

## Label (diabetes = any condition code in <<73211009, 118 codes)

Label codes that occur in conditions.csv: 2 of 118

| code | description | patients |
|---|---|---|
| 44054006 | Diabetes mellitus type 2 (disorder) | 88 |
| 427089005 | Diabetes mellitus due to cystic fibrosis (disorder) | 1 |

- Positive patients: 88 of 1146 (prevalence 0.0768)

### Diabetes-related condition codes NOT in the label set

Not used as label or feature. Patients with these codes but no label code are labelled 0.

| code | description | patients | of which label = 0 |
|---|---|---|---|
| 714628002 | Prediabetes (finding) | 441 | 377 |
| 127013003 | Disorder of kidney due to diabetes mellitus (disorder) | 128 | 94 |
| 90781000119102 | Microalbuminuria due to type 2 diabetes mellitus (disorder) | 104 | 82 |
| 157141000119108 | Proteinuria due to type 2 diabetes mellitus (disorder) | 77 | 61 |
| 80394007 | Hyperglycemia (disorder) | 44 | 0 |
| 368581000119106 | Neuropathy due to type 2 diabetes mellitus (disorder) | 24 | 0 |
| 1551000119108 | Nonproliferative diabetic retinopathy due to type II diabetes mellitus | 16 | 0 |
| 97331000119101 | Macular edema and retinopathy due to type 2 diabetes mellitus (disorder) | 1 | 0 |

- Patients with a diabetes-complication / hyperglycaemia code (prediabetes excluded) but label = 0: 94

### Prediabetes (never a feature)

| code | description | patients |
|---|---|---|
| 15777000 | Prediabetes (inactive in SNOMED CT Intl 2026-11-01) | 0 |
| 714628002 | Prediabetes (finding) | 441 |

## Leakage guard (allowlist)

- Condition codes features may read: 13644009, 55822004, 59621000
- Observation codes features may read: 39156-5, 8462-4, 8480-6
- No glucose / HbA1c observations are read.

### Hypertension / lipid codes in data but NOT allowlisted (not used)

| code | description | patients |
|---|---|---|
| 302870006 | Hypertriglyceridemia (disorder) | 86 |

- Allowlisted code 13644009 does not occur in this data.

### Medications excluded from med_count (glucose-lowering)

| description | rows | patients |
|---|---|---|
| 24 HR Metformin hydrochloride 500 MG Extended Release Oral Tablet | 1432 | 45 |
| 3 ML liraglutide 6 MG/ML Pen Injector | 7 | 1 |
| Insulin Lispro 100 UNT/ML Injectable Solution [Humalog] | 208 | 2 |
| insulin  regular  human 100 UNT/ML Injectable Solution | 13 | 13 |
| insulin isophane  human 70 UNT/ML / insulin  regular  human 30 UNT/ML Injectable Suspension [Humulin] | 6234 | 98 |

## Features

| column | missing % | min | max | mean |
|---|---|---|---|---|
| age | 0.0 | 0 | 110 | 40.55 |
| sex_male | 0.0 | 0 | 1 | 0.49 |
| enc_ambulatory | 0.0 | 0 | 811 | 35.02 |
| enc_emergency | 0.0 | 0 | 163 | 2.3 |
| enc_inpatient | 0.0 | 0 | 50 | 1.12 |
| hypertension | 0.0 | 0 | 1 | 0.21 |
| hyperlipidemia | 0.0 | 0 | 1 | 0.11 |
| bmi_latest | 2.0 | 13.4 | 45.6 | 26.47 |
| sbp_latest | 0.0 | 56.6 | 177.7 | 117.72 |
| dbp_latest | 0.0 | 26 | 119.5 | 78.9 |
| med_count | 0.0 | 0 | 31 | 5.65 |
| diabetes | 0.0 | 0 | 1 | 0.08 |

Missing values are median-imputed inside the model pipeline (fitted on the training split).

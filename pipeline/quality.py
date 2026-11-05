"""Data-quality report (markdown). Deterministic: no timestamps or absolute paths."""

from pipeline import features as F

PREDIABETES_CODES = {"15777000": "Prediabetes (inactive in SNOMED CT Intl 2026-11-01)",
                     "714628002": "Prediabetes (finding)"}
_DIABETES_TEXT = r"diabet|prediab|hyperglyc"
_HTN_LIPID_TEXT = r"hypertens|lipid|cholesterol|triglycerid"


def _cell(value):
    """Cell text comes from the CSVs: escape pipes and drop newlines so it can't break the table."""
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return lines


def _codes_by_patients(conditions, mask):
    """[(code, first description seen, distinct patients)], most patients first."""
    grouped = (conditions[mask].groupby("CODE", dropna=False)
               .agg(DESCRIPTION=("DESCRIPTION", "first"), PATIENT=("PATIENT", "nunique"))
               .reset_index().sort_values(["PATIENT", "CODE"], ascending=[False, True]))
    return list(grouped.itertuples(index=False, name=None))


def _fmt(x):
    return "" if x != x else f"{x:.2f}".rstrip("0").rstrip(".")


def build_report(raw_counts, clean_log, tables, patient_table, label, label_codes, as_of):
    conditions, meds = tables["conditions"], tables["medications"]
    n = len(patient_table)
    out = ["# Data-quality report", "",
           "Synthetic data (Synthea). Not clinical data; nothing here is clinically validated.", "",
           f"Age reference date for living patients (latest encounter in data): {as_of:%Y-%m-%d}", ""]

    out += ["## Row counts per step", ""]
    out += _table(["table", "raw rows", "exact duplicates dropped", "missing patient id dropped", "rows after clean"],
                  [(t, raw_counts[t], e["dropped_exact_duplicates"], e["dropped_missing_patient_id"], e["rows_out"])
                   for t, e in clean_log.items()])
    dup = int(patient_table.index.duplicated().sum())
    out += ["", f"- Patient table rows: {n} (patients.csv rows: {raw_counts['patients']})",
            f"- Duplicate patient IDs in patient table: {dup}", ""]

    out += [f"## Label (diabetes = any condition code in <<73211009, {len(label_codes)} codes)", ""]
    present = _codes_by_patients(conditions, conditions["CODE"].isin(label_codes))
    out += [f"Label codes that occur in conditions.csv: {len(present)} of {len(label_codes)}", ""]
    out += _table(["code", "description", "patients"], present)
    out += ["", f"- Positive patients: {int(label.sum())} of {n} (prevalence {label.mean():.4f})", ""]

    positives = set(label[label == 1].index)
    related = conditions["DESCRIPTION"].str.contains(_DIABETES_TEXT, case=False, na=False) & \
        ~conditions["CODE"].isin(label_codes)
    rows = []
    for code, desc, pts in _codes_by_patients(conditions, related):
        ids = set(conditions.loc[related & (conditions["CODE"] == code), "PATIENT"])
        rows.append((code, desc, pts, len(ids - positives)))
    out += ["### Diabetes-related condition codes NOT in the label set", "",
            "Not used as label or feature. Patients with these codes but no label code are labelled 0.", ""]
    out += _table(["code", "description", "patients", "of which label = 0"], rows)
    any_related = set(conditions.loc[related & ~conditions["CODE"].isin(PREDIABETES_CODES), "PATIENT"])
    out += ["", f"- Patients with a diabetes-complication / hyperglycaemia code (prediabetes excluded) "
                f"but label = 0: {len(any_related - positives)}", ""]

    out += ["### Prediabetes (never a feature)", ""]
    out += _table(["code", "description", "patients"],
                  [(c, d, conditions.loc[conditions["CODE"] == c, "PATIENT"].nunique())
                   for c, d in PREDIABETES_CODES.items()])

    out += ["", "## Leakage guard (allowlist)", "",
            f"- Condition codes features may read: {', '.join(sorted(F.ALLOWED_CONDITION_CODES))}",
            f"- Observation codes features may read: {', '.join(sorted(F.ALLOWED_OBSERVATION_CODES))}",
            "- No glucose / HbA1c observations are read.", ""]
    htn = conditions["DESCRIPTION"].str.contains(_HTN_LIPID_TEXT, case=False, na=False) & \
        ~conditions["CODE"].isin(F.ALLOWED_CONDITION_CODES)
    out += ["### Hypertension / lipid codes in data but NOT allowlisted (not used)", ""]
    out += _table(["code", "description", "patients"], _codes_by_patients(conditions, htn))
    for code in sorted(F.ALLOWED_CONDITION_CODES):
        if not (conditions["CODE"] == code).any():
            out += ["", f"- Allowlisted code {code} does not occur in this data."]
    excluded = meds[F.is_diabetes_medication(meds["DESCRIPTION"])]
    excl = excluded.groupby("DESCRIPTION").agg(rows=("CODE", "size"), patients=("PATIENT", "nunique"))
    out += ["", "### Medications excluded from med_count (glucose-lowering)", ""]
    out += _table(["description", "rows", "patients"],
                  [(d, r.rows, r.patients) for d, r in excl.sort_index().iterrows()])

    out += ["", "## Features", ""]
    stats = []
    for col in patient_table.columns:
        s = patient_table[col]
        stats.append((col, f"{s.isna().mean() * 100:.1f}", _fmt(s.min()), _fmt(s.max()), _fmt(s.mean())))
    out += _table(["column", "missing %", "min", "max", "mean"], stats)
    out += ["", "Missing values are median-imputed inside the model pipeline (fitted on the training split).", ""]
    return "\n".join(out)

#!/usr/bin/env python3
"""
================================================================================
NIRNAY ML TRAINING PIPELINE
================================================================================
National Infrastructure Risk & Nodal Action Intelligence

Predicts: "Will this project become delayed within the next 6 months?"
Target:   future_delayed_6m  from  nirnay_future_risk_training.csv

Usage:    python train_nirnay_model.py
          (or:  py train_nirnay_model.py  on Windows)

Produces:
  models/   nirnay_delay_model.pkl, nirnay_preprocessor.pkl,
            nirnay_pipeline.pkl, nirnay_model_metadata.json,
            nirnay_feature_list.json, nirnay_thresholds.json
  outputs/  nirnay_test_predictions.csv, nirnay_model_comparison.csv,
            nirnay_feature_importance.csv, nirnay_training_report.json,
            nirnay_dashboard_predictions.csv, nirnay_dashboard_predictions.json

Author:   NIRNAY ML Pipeline (automated)
Version:  1.0.0
================================================================================
"""

import os, sys, json, warnings, datetime, traceback
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, precision_recall_curve, roc_curve,
)
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# ==============================================================================
# CONFIGURATION
# ==============================================================================
RANDOM_STATE = 42
MODEL_VERSION = "1.0.0"
TRAINING_DATE = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "Dataset"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

TARGET_COL = "future_delayed_6m"

FUTURE_LEAKAGE_COLS = [
    "future_delayed_6m", "future_risk_level_6m",
    "future_time_overrun_months", "future_cost_overrun_pct", "target_date_6m",
]

ID_META_COLS = [
    "project_code", "project_id", "project_name", "report_date",
    "approval_date", "original_commissioning", "revised_commissioning",
    "anticipated_commissioning", "original_completion_date",
    "revised_completion_date", "anticipated_completion_date",
    "source_format", "risk_level_derived", "risk_score_derived",
    "target_next_risk_level", "target_horizon_months",
]

THRESHOLD_CANDIDATES = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

FEATURE_DISPLAY_NAMES = {
    "time_overrun_months": "Time overrun",
    "cost_overrun_pct": "Cost overrun",
    "expenditure_ratio": "Expenditure ratio",
    "cost_variance_pct": "Cost variance",
    "physical_progress": "Physical progress",
    "original_cost": "Original cost",
    "anticipated_cost": "Anticipated cost",
    "expenditure": "Cumulative expenditure",
    "project_age_months": "Project age",
    "months_remaining": "Months remaining",
    "cost_growth_ratio": "Cost escalation ratio",
    "schedule_pressure": "Schedule pressure",
    "exp_progress_gap": "Expenditure-progress gap",
    "has_revised_cost": "Has revised cost",
    "is_currently_delayed": "Currently delayed",
    "high_overrun_flag": "High time overrun (>24m)",
    "low_progress_flag": "Low physical progress (<30%)",
    "progress_exp_ratio": "Progress-expenditure alignment",
    "remaining_vs_overrun": "Remaining time vs overrun",
    "cost_overrun_flag": "Cost overrun exists",
    "progress_3m_change": "3-month progress change",
    "exp_3m_change": "3-month expenditure change",
    "prev_physical_progress": "Previous physical progress",
    "prev_expenditure_ratio": "Previous expenditure ratio",
}

np.random.seed(RANDOM_STATE)


# ==============================================================================
# UTILITIES
# ==============================================================================
def sep(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}\n")

def safe_div(a, b, default=0.0):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(b != 0, a / b, default)
    return np.where(np.isfinite(r), r, default)

def parse_date_flex(series):
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=False)
    mask = parsed.isna() & series.notna()
    if mask.any():
        alt = series[mask].astype(str).apply(
            lambda x: pd.to_datetime(f"01/{x}", format="%d/%m/%Y", errors="coerce")
            if "/" in str(x) else pd.NaT
        )
        parsed.loc[mask] = alt
    return parsed

def fail_if(cond, msg):
    if cond:
        print(f"\n{'!'*70}\n  FATAL: {msg}\n{'!'*70}\n")
        sys.exit(1)


# ==============================================================================
# SECTION 1: DATA AUDIT
# ==============================================================================
def audit_csv(filepath, label):
    print(f"\n--- {label}: {filepath.name} ---")
    if not filepath.exists():
        print(f"  FILE NOT FOUND: {filepath}")
        return None
    df = pd.read_csv(filepath, low_memory=False)
    n, c = df.shape
    pid = next((x for x in ["project_code","project_id"] if x in df.columns), None)
    npid = df[pid].nunique() if pid else "N/A"
    drng = "N/A"
    if "report_date" in df.columns:
        d = pd.to_datetime(df["report_date"], errors="coerce")
        drng = f"{d.min()} -> {d.max()}"
    miss = df.isnull().sum()
    misspct = (miss / len(df) * 100).round(1)
    print(f"  Rows: {n:,}  Cols: {c}  Dupes: {df.duplicated().sum()}  "
          f"Projects: {npid}  Dates: {drng}")
    print(f"  Columns: {list(df.columns)}")
    for col in df.columns:
        m = f"  ({misspct[col]}% miss)" if miss[col] > 0 else ""
        print(f"    {col:45s} {str(df[col].dtype):10s}{m}")
    return df

def run_data_audit():
    sep("SECTION 1: DATA AUDIT")
    files = {
        "TARGET": DATASET_DIR / "nirnay_future_risk_training.csv",
        "LATEST (inference)": DATASET_DIR / "nirnay_latest_projects.csv",
        "ML TRAINING (hist)": DATASET_DIR / "nirnay_ml_training_dataset.csv",
        "ML FEATURES (ref)": DATASET_DIR / "nirnay_ml_features.csv",
        "PROJECT-MONTH (enrich)": DATASET_DIR / "nirnay_project_month_dataset.csv",
    }
    ds = {}
    for lbl, p in files.items():
        ds[lbl] = audit_csv(p, lbl)

    # Relationship
    t, pm = ds["TARGET"], ds["PROJECT-MONTH (enrich)"]
    if t is not None and pm is not None:
        ti = set(t["project_code"].unique())
        pi = set(pm["project_id"].unique())
        ov = ti & pi
        print(f"\n  Overlap: target={len(ti)} pm={len(pi)} matched={len(ov)} unmatched={len(ti-pi)}")
    print("\n  Dataset Roles:")
    print("    nirnay_future_risk_training  -> Training + Target (future_delayed_6m)")
    print("    nirnay_project_month_dataset -> Historical enrichment (trends)")
    print("    nirnay_ml_training_dataset   -> Historical reference")
    print("    nirnay_ml_features           -> Feature reference / validation")
    print("    nirnay_latest_projects       -> Inference (current project predictions)")
    return ds


# ==============================================================================
# SECTION 2: LOAD & STANDARDIZE
# ==============================================================================
def standardize_target_cols(df):
    return df.rename(columns={
        "project_code": "project_id",
        "original_cost": "original_cost",
        "revised_cost": "revised_cost",
        "anticipated_cost": "anticipated_cost",
        "expenditure": "expenditure",
        "time_overrun_months": "time_overrun_months",
        "cost_overrun_pct": "cost_overrun_pct",
        "current_cost_variance_pct": "cost_variance_pct",
        "expenditure_ratio_pct": "expenditure_ratio",
    })

def standardize_enrich_cols(df):
    return df.rename(columns={
        "original_cost_cr": "original_cost",
        "revised_cost_cr": "revised_cost",
        "anticipated_cost_cr": "anticipated_cost",
        "cumulative_expenditure_cr": "expenditure",
        "delay_months_calc": "time_overrun_months",
        "cost_overrun_pct_calc": "cost_overrun_pct",
        "expenditure_pct_of_anticipated": "expenditure_ratio",
        "physical_progress_pct": "physical_progress",
        "project_age_months": "project_age_months",
        "months_to_anticipated_completion": "months_remaining",
    })

def load_target():
    sep("SECTION 2: DATA LOADING & VALIDATION")
    path = DATASET_DIR / "nirnay_future_risk_training.csv"
    fail_if(not path.exists(), f"Target file not found: {path}")
    df = pd.read_csv(path, low_memory=False)

    fail_if(TARGET_COL not in df.columns, f"Target column '{TARGET_COL}' missing")
    fail_if("project_code" not in df.columns and "project_id" not in df.columns, "Project ID missing")
    fail_if("report_date" not in df.columns, "report_date missing")
    fail_if(df[TARGET_COL].isna().sum() > 0, f"Target has {df[TARGET_COL].isna().sum()} missing values")

    unique_t = set(df[TARGET_COL].dropna().unique())
    fail_if(not unique_t.issubset({0, 1, 0.0, 1.0}), f"Target invalid values: {unique_t}")

    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    fail_if(df["report_date"].isna().all(), "All report_dates unparseable")

    df = standardize_target_cols(df)

    # Project age from approval_date
    df["approval_date_parsed"] = parse_date_flex(df.get("approval_date", pd.Series(dtype=str)))
    if df["approval_date_parsed"].notna().any():
        df["project_age_months"] = ((df["report_date"] - df["approval_date_parsed"]).dt.days / 30.44).round(0)
    else:
        df["project_age_months"] = np.nan

    # Months remaining from anticipated_commissioning
    df["antic_parsed"] = parse_date_flex(df.get("anticipated_commissioning", pd.Series(dtype=str)))
    if df["antic_parsed"].notna().any():
        df["months_remaining"] = ((df["antic_parsed"] - df["report_date"]).dt.days / 30.44).round(0)
    else:
        df["months_remaining"] = np.nan

    # Dedup
    dup_n = df.duplicated(subset=["project_id", "report_date"], keep=False).sum()
    if dup_n > 0:
        print(f"  WARNING: {dup_n} duplicate (project_id, report_date). Keeping first.")
        df = df.drop_duplicates(subset=["project_id", "report_date"], keep="first")

    dist = df[TARGET_COL].value_counts()
    print(f"  Loaded: {df.shape[0]:,} rows, {df.shape[1]} cols")
    for v, c in dist.items():
        lbl = "DELAYED" if v == 1 else "NOT DELAYED"
        print(f"    {lbl} ({v}): {c:,} ({c/len(df)*100:.1f}%)")
    print(f"  Dates: {df['report_date'].min().date()} -> {df['report_date'].max().date()}")
    print(f"  Projects: {df['project_id'].nunique()}")
    return df


# ==============================================================================
# SECTION 3: STATUS LEAKAGE AUDIT
# ==============================================================================
def audit_status(df):
    sep("SECTION 3: STATUS LEAKAGE AUDIT")
    if "status" not in df.columns:
        print("  `status` not present. Skipping.")
        return df, False

    uv = df["status"].unique()
    print(f"  Unique status values ({len(uv)}):")
    for s in sorted(uv, key=str):
        print(f"    '{s}': {(df['status']==s).sum():,}")

    print(f"\n  Cross-tab: status x {TARGET_COL}")
    ct = pd.crosstab(df["status"], df[TARGET_COL], margins=True)
    ct_pct = pd.crosstab(df["status"], df[TARGET_COL], normalize="index") * 100
    print(ct.to_string())
    print(f"\n  % delayed by status:")
    for sv in ct_pct.index:
        if sv != "All":
            dp = ct_pct.loc[sv, 1] if 1 in ct_pct.columns else 0
            print(f"    '{sv}': {dp:.1f}% become delayed")

    sl = df["status"].str.lower().str.strip()
    is_del = sl.isin(["delayed","slow","stalled","behind"]) | sl.str.contains("delay", na=False)

    if is_del.any():
        corr = np.corrcoef(is_del.astype(int), df[TARGET_COL])[0, 1]
        print(f"\n  Correlation(is_currently_delayed, {TARGET_COL}): {corr:.4f}")
        if abs(corr) > 0.90:
            print("  !!! VERY HIGH correlation — EXCLUDING status entirely.")
            return df, True
        elif abs(corr) > 0.70:
            print("  !! HIGH correlation. Using is_currently_delayed with awareness.")
            df["is_currently_delayed"] = is_del.astype(int)
            return df, False
        else:
            print("  OK. Moderate correlation. Using is_currently_delayed.")
            df["is_currently_delayed"] = is_del.astype(int)
            return df, False
    else:
        print("  No delay-related status values. Safe.")
        return df, False


# ==============================================================================
# SECTION 4: SHARED FEATURE ENGINE (NirnayFeatureEngine)
# ==============================================================================
class NirnayFeatureEngine:
    """
    SHARED feature engineering for training AND inference.
    Same logic, same column order — guarantees consistency.
    """
    def __init__(self):
        self.fitted = False
        self.sector_enc = {}
        self.state_enc = {}
        self.feature_names = []
        self.feature_medians = {}
        self.exclude_status = False

    def fit_transform(self, df, exclude_status=False):
        self.exclude_status = exclude_status
        f = self._build(df)

        if "sector" in f.columns:
            uv = f["sector"].dropna().unique()
            self.sector_enc = {v: i for i, v in enumerate(sorted(uv, key=str))}
            f["sector"] = f["sector"].map(self.sector_enc).fillna(-1)
        if "state" in f.columns:
            uv = f["state"].dropna().unique()
            self.state_enc = {v: i for i, v in enumerate(sorted(uv, key=str))}
            f["state"] = f["state"].map(self.state_enc).fillna(-1)

        f = f.select_dtypes(include=[np.number])
        medians = f.median().to_dict()
        # Replace NaN medians (from all-NaN columns) with 0.0
        self.feature_medians = {k: (v if pd.notna(v) else 0.0) for k, v in medians.items()}
        self.feature_names = list(f.columns)
        self.fitted = True
        for c in f.columns:
            f[c] = f[c].fillna(self.feature_medians[c])
        return f

    def transform(self, df):
        assert self.fitted, "Must fit first!"
        f = self._build(df)
        if "sector" in f.columns:
            f["sector"] = f["sector"].map(self.sector_enc).fillna(-1)
        if "state" in f.columns:
            f["state"] = f["state"].map(self.state_enc).fillna(-1)
        f = f.select_dtypes(include=[np.number])
        for c in self.feature_names:
            if c not in f.columns:
                f[c] = self.feature_medians.get(c, 0.0)
        f = f[self.feature_names]
        for c in f.columns:
            f[c] = f[c].fillna(self.feature_medians.get(c, 0.0))
        return f

    def _build(self, df):
        f = pd.DataFrame(index=df.index)

        # Cost
        f["original_cost"] = pd.to_numeric(df.get("original_cost"), errors="coerce")
        f["anticipated_cost"] = pd.to_numeric(df.get("anticipated_cost"), errors="coerce")
        f["expenditure"] = pd.to_numeric(df.get("expenditure"), errors="coerce")
        f["expenditure_ratio"] = pd.to_numeric(df.get("expenditure_ratio"), errors="coerce")
        f["cost_overrun_pct"] = pd.to_numeric(df.get("cost_overrun_pct"), errors="coerce")
        cv = df.get("cost_variance_pct", df.get("cost_overrun_pct"))
        f["cost_variance_pct"] = pd.to_numeric(cv, errors="coerce") if cv is not None else np.nan

        f["cost_growth_ratio"] = safe_div(f["anticipated_cost"].values, f["original_cost"].values, 1.0)
        rev = pd.to_numeric(df.get("revised_cost"), errors="coerce")
        f["has_revised_cost"] = (rev.notna() & (rev != f["original_cost"])).astype(int)
        f["cost_overrun_flag"] = (f["cost_overrun_pct"] > 0).astype(int)

        # Schedule
        f["time_overrun_months"] = pd.to_numeric(df.get("time_overrun_months"), errors="coerce")
        f["months_remaining"] = pd.to_numeric(df.get("months_remaining"), errors="coerce")
        f["project_age_months"] = pd.to_numeric(df.get("project_age_months"), errors="coerce")
        f["schedule_pressure"] = safe_div(f["time_overrun_months"].values,
                                          np.maximum(f["project_age_months"].values, 1), 0.0)
        f["remaining_vs_overrun"] = safe_div(f["months_remaining"].values,
                                             np.maximum(f["time_overrun_months"].values, 1), 1.0)
        f["high_overrun_flag"] = (f["time_overrun_months"] > 24).astype(int)

        # Progress
        f["physical_progress"] = pd.to_numeric(df.get("physical_progress"), errors="coerce")
        f["exp_progress_gap"] = f["expenditure_ratio"] - f["physical_progress"]
        f["progress_exp_ratio"] = safe_div(f["physical_progress"].values,
                                           np.maximum(f["expenditure_ratio"].values, 1), 0.0)
        f["low_progress_flag"] = (f["physical_progress"] < 30).astype(int)

        # Status
        if not self.exclude_status and "is_currently_delayed" in df.columns:
            f["is_currently_delayed"] = df["is_currently_delayed"].astype(int)

        # Historical trends (if enriched)
        for tc in ["progress_3m_change", "exp_3m_change",
                    "prev_physical_progress", "prev_expenditure_ratio"]:
            if tc in df.columns:
                f[tc] = pd.to_numeric(df[tc], errors="coerce")

        # Categorical
        if "sector" in df.columns:
            f["sector"] = df["sector"].astype(str)
        if "state" in df.columns:
            f["state"] = df["state"].astype(str)

        return f

    def get_display_name(self, feat):
        return FEATURE_DISPLAY_NAMES.get(feat, feat.replace("_", " ").title())


# ==============================================================================
# SECTION 5: HISTORICAL ENRICHMENT
# ==============================================================================
def _normalize_pid(series):
    """Normalize project IDs: strip leading zeros, convert to string."""
    return series.astype(str).str.strip().str.lstrip("0")

def enrich(target_df):
    sep("SECTION 5: HISTORICAL ENRICHMENT")
    pm_path = DATASET_DIR / "nirnay_project_month_dataset.csv"
    if not pm_path.exists():
        print("  Project-month dataset not found. Skipping.")
        return target_df

    pm = pd.read_csv(pm_path, low_memory=False)
    pm = standardize_enrich_cols(pm)
    pm["report_date"] = pd.to_datetime(pm["report_date"], errors="coerce")

    # Normalize project IDs to handle type mismatch
    # Target: "020100044" (string with leading zero)
    # PM:      20100044   (int, no leading zero)
    target_df["project_id"] = _normalize_pid(target_df["project_id"])
    pm["project_id"] = _normalize_pid(pm["project_id"])

    id_overlap = set(target_df["project_id"].unique()) & set(pm["project_id"].unique())
    print(f"  ID normalization: {len(id_overlap)} projects matched after stripping leading zeros")

    pm = pm.sort_values(["project_id", "report_date"]).reset_index(drop=True)

    # Lag features
    pm["prev_physical_progress"] = pm.groupby("project_id")["physical_progress"].shift(1)
    pm["prev_expenditure_ratio"] = pm.groupby("project_id")["expenditure_ratio"].shift(1)
    pm["p3m_ago"] = pm.groupby("project_id")["physical_progress"].shift(3)
    pm["progress_3m_change"] = pm["physical_progress"] - pm["p3m_ago"]
    pm["e3m_ago"] = pm.groupby("project_id")["expenditure_ratio"].shift(3)
    pm["exp_3m_change"] = pm["expenditure_ratio"] - pm["e3m_ago"]

    enrich_cols = [
        "project_id", "report_date", "physical_progress", "project_age_months",
        "months_remaining", "sector", "state", "agency",
        "prev_physical_progress", "prev_expenditure_ratio",
        "progress_3m_change", "exp_3m_change",
    ]
    enrich_cols = [c for c in enrich_cols if c in pm.columns]
    pme = pm[enrich_cols].drop_duplicates(subset=["project_id", "report_date"], keep="last")

    existing = set(target_df.columns) - {"project_id", "report_date"}
    new_cols = [c for c in enrich_cols if c not in existing]
    merge_cols = list(dict.fromkeys(["project_id", "report_date"] +
                                     [c for c in new_cols if c not in ["project_id","report_date"]]))

    before = len(target_df)

    # Try exact match first
    enriched = target_df.merge(pme[merge_cols], on=["project_id", "report_date"],
                                how="left", suffixes=("", "_enr"))
    for col in new_cols:
        if col in ["project_id", "report_date"]:
            continue
        ec = f"{col}_enr"
        if ec in enriched.columns:
            enriched[col] = enriched[col].fillna(enriched[ec])
            enriched.drop(columns=[ec], inplace=True)

    matched = enriched["physical_progress"].notna().sum() if "physical_progress" in enriched.columns else 0
    print(f"  Exact (project_id, report_date) match: {matched:,} ({matched/len(enriched)*100:.1f}%)")

    # If exact match rate is low, try merge_asof (closest previous date)
    if matched / len(enriched) < 0.3:
        print("  Low match rate. Using per-project closest-date lookup...")
        # Drop the failed enrichment columns
        drop_cols = [c for c in new_cols if c in enriched.columns and c not in ["project_id","report_date"]]
        enriched = enriched.drop(columns=drop_cols, errors="ignore")

        value_cols = [c for c in merge_cols if c not in ["project_id", "report_date"]]

        # Sort BOTH sides by report_date (required for merge_asof)
        target_sorted = target_df.sort_values("report_date").reset_index(drop=True)
        pme_sorted = pme.sort_values("report_date").reset_index(drop=True)

        try:
            enriched = pd.merge_asof(
                target_sorted,
                pme_sorted[["project_id", "report_date"] + value_cols],
                on="report_date",
                by="project_id",
                direction="backward",
                suffixes=("", "_asof"),
            )
            for col in value_cols:
                ac = f"{col}_asof"
                if ac in enriched.columns:
                    enriched[col] = enriched[col].fillna(enriched[ac])
                    enriched.drop(columns=[ac], inplace=True)
        except Exception as e:
            print(f"  merge_asof failed: {e}")
            print("  Falling back to latest-snapshot-per-project join...")
            enriched = target_df.copy()
            # Get latest snapshot per project from PM
            latest_pm = pme.sort_values("report_date").groupby("project_id").last()[value_cols].reset_index()
            enriched = enriched.merge(latest_pm, on="project_id", how="left", suffixes=("", "_pm"))
            for col in value_cols:
                pc = f"{col}_pm"
                if pc in enriched.columns:
                    enriched[col] = enriched[col].fillna(enriched[pc])
                    enriched.drop(columns=[pc], inplace=True)

        matched = enriched["physical_progress"].notna().sum() if "physical_progress" in enriched.columns else 0
        print(f"  After fallback: {matched:,} ({matched/len(enriched)*100:.1f}%)")

    print(f"  Rows: {before:,} -> {len(enriched):,}")
    new_feat = [c for c in new_cols if c not in ["project_id","report_date"]]
    print(f"  New features: {new_feat}")
    return enriched


# ==============================================================================
# SECTION 6: TIME-AWARE SPLIT
# ==============================================================================
def time_split(df, target_col):
    sep("SECTION 6: TIME-AWARE SPLIT")
    df = df.sort_values("report_date").reset_index(drop=True)
    dates = df["report_date"].dropna().sort_values()
    n = len(dates)
    c1 = dates.iloc[int(n * 0.70)]
    c2 = dates.iloc[int(n * 0.85)]

    tr = df[df["report_date"] < c1].copy()
    va = df[(df["report_date"] >= c1) & (df["report_date"] < c2)].copy()
    te = df[df["report_date"] >= c2].copy()

    fail_if(tr["report_date"].max() > va["report_date"].min(), "Train/valid temporal violation")

    for nm, sp in [("TRAIN", tr), ("VALID", va), ("TEST", te)]:
        nd = (sp[target_col] == 1).sum()
        print(f"  {nm:6s}: {len(sp):>6,} | {sp['report_date'].min().date()} -> "
              f"{sp['report_date'].max().date()} | projects: {sp['project_id'].nunique():>4} | "
              f"delayed: {nd:,} ({nd/len(sp)*100:.1f}%)")
    return tr, va, te


# ==============================================================================
# SECTION 7: CLASS IMBALANCE
# ==============================================================================
def check_imbalance(y):
    sep("SECTION 7: CLASS IMBALANCE")
    c = pd.Series(y).value_counts()
    t = len(y)
    r = c.get(0, 0) / max(c.get(1, 1), 1)
    print(f"  Not delayed (0): {c.get(0,0):,} ({c.get(0,0)/t*100:.1f}%)")
    print(f"  Delayed     (1): {c.get(1,0):,} ({c.get(1,0)/t*100:.1f}%)")
    print(f"  Ratio: {r:.2f}:1")
    if r > 3:
        print("  -> Using class_weight='balanced'. SMOTE NOT applied.")
    else:
        print("  -> Reasonable distribution.")
    return r


# ==============================================================================
# SECTION 8: MODEL TRAINING
# ==============================================================================
def get_models(spw):
    m = OrderedDict()
    m["Logistic Regression"] = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs")
    m["Random Forest"] = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", max_depth=15,
        min_samples_leaf=10, random_state=RANDOM_STATE, n_jobs=-1)
    m["XGBoost"] = xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        scale_pos_weight=spw, random_state=RANDOM_STATE,
        use_label_encoder=False, eval_metric="logloss", verbosity=0)
    m["HistGradientBoosting"] = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=6,
        class_weight="balanced", random_state=RANDOM_STATE)
    m["LightGBM"] = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        is_unbalance=True, random_state=RANDOM_STATE, verbosity=-1, force_col_wise=True)
    return m

def metrics(yt, yp, ypr):
    m = {
        "accuracy": accuracy_score(yt, yp),
        "precision": precision_score(yt, yp, zero_division=0),
        "recall": recall_score(yt, yp, zero_division=0),
        "f1": f1_score(yt, yp, zero_division=0),
        "roc_auc": roc_auc_score(yt, ypr),
        "pr_auc": average_precision_score(yt, ypr),
        "confusion_matrix": confusion_matrix(yt, yp).tolist(),
    }
    cm = confusion_matrix(yt, yp)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        m["delayed_recall"] = tp / max(tp + fn, 1)
        m["delayed_precision"] = tp / max(tp + fp, 1)
        m["false_negative_rate"] = fn / max(tp + fn, 1)
        m["false_positive_rate"] = fp / max(tn + fp, 1)
    else:
        m.update({"delayed_recall": 0, "delayed_precision": 0,
                  "false_negative_rate": 1, "false_positive_rate": 0})
    return m

def train_all(models, Xtr, ytr, Xva, yva):
    sep("SECTION 8: MODEL TRAINING")
    res, trained = {}, {}
    for name, mdl in models.items():
        print(f"  Training: {name}...", end=" ", flush=True)
        if name == "Logistic Regression":
            sc = StandardScaler()
            Xts = sc.fit_transform(Xtr)
            Xvs = sc.transform(Xva)
            mdl.fit(Xts, ytr)
            yp = mdl.predict(Xvs)
            ypr = mdl.predict_proba(Xvs)[:, 1]
            trained[name] = {"model": mdl, "scaler": sc}
        else:
            mdl.fit(Xtr, ytr)
            yp = mdl.predict(Xva)
            ypr = mdl.predict_proba(Xva)[:, 1]
            trained[name] = {"model": mdl, "scaler": None}
        m = metrics(yva, yp, ypr)
        res[name] = m
        print(f"F1={m['f1']:.3f} Recall={m['recall']:.3f} PR-AUC={m['pr_auc']:.3f}")
    return res, trained


# ==============================================================================
# SECTION 9: MODEL COMPARISON
# ==============================================================================
def print_comparison(res):
    sep("SECTION 9: MODEL COMPARISON")
    hdr = f"{'MODEL':30s} | {'PREC':>6s} | {'RECALL':>6s} | {'F1':>6s} | {'ROC-AUC':>7s} | {'PR-AUC':>7s} | {'FNR':>5s}"
    print(f"  {hdr}")
    print(f"  {'-'*len(hdr)}")
    for n, m in res.items():
        print(f"  {n:30s} | {m['precision']:6.3f} | {m['recall']:6.3f} | "
              f"{m['f1']:6.3f} | {m['roc_auc']:7.3f} | {m['pr_auc']:7.3f} | "
              f"{m['false_negative_rate']:5.3f}")


# ==============================================================================
# SECTION 10: THRESHOLD OPTIMIZATION
# ==============================================================================
def optimize_threshold(mi, Xva, yva):
    sep("SECTION 10: THRESHOLD OPTIMIZATION")
    mdl, sc = mi["model"], mi.get("scaler")
    ypr = mdl.predict_proba(sc.transform(Xva) if sc else Xva)[:, 1]

    print(f"  {'THRESH':>8s} | {'PREC':>6s} | {'RECALL':>6s} | {'F1':>6s} | {'FPR':>5s} | {'FNR':>5s}")
    print(f"  {'-'*50}")

    best_t, best_s = 0.50, -1
    for t in THRESHOLD_CANDIDATES:
        yp = (ypr >= t).astype(int)
        p = precision_score(yva, yp, zero_division=0)
        r = recall_score(yva, yp, zero_division=0)
        f = f1_score(yva, yp, zero_division=0)
        cm = confusion_matrix(yva, yp)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / max(tn+fp, 1)
            fnr = fn / max(tp+fn, 1)
        else:
            fpr, fnr = 0, 1
        s = 0.50 * r + 0.30 * f + 0.20 * p
        mk = " <- BEST" if s > best_s else ""
        if s > best_s:
            best_s, best_t = s, t
        print(f"  {t:8.2f} | {p:6.3f} | {r:6.3f} | {f:6.3f} | {fpr:5.3f} | {fnr:5.3f}{mk}")

    print(f"\n  SELECTED THRESHOLD: {best_t}")
    return best_t


# ==============================================================================
# SECTION 11: RISK LEVEL CALIBRATION
# ==============================================================================
def calibrate_risk(mi, Xva, yva, threshold):
    sep("SECTION 11: RISK LEVEL CALIBRATION")
    mdl, sc = mi["model"], mi.get("scaler")
    ypr = mdl.predict_proba(sc.transform(Xva) if sc else Xva)[:, 1]

    print(f"  Prob distribution: min={ypr.min():.4f} 25th={np.percentile(ypr,25):.4f} "
          f"med={np.percentile(ypr,50):.4f} 75th={np.percentile(ypr,75):.4f} max={ypr.max():.4f}")

    cfgs = [
        {"name": "Quartile-based", "low": 0.25, "med": 0.50, "high": 0.75},
        {"name": "Threshold-aligned", "low": threshold*0.5, "med": threshold,
         "high": threshold + (1-threshold)*0.5},
        {"name": "Conservative", "low": 0.20, "med": 0.45, "high": 0.70},
    ]
    dm = yva == 1
    for c in cfgs:
        lvls = np.where(ypr >= c["high"], "Critical",
               np.where(ypr >= c["med"], "High",
               np.where(ypr >= c["low"], "Medium", "Low")))
        cts = pd.Series(lvls).value_counts()
        ch_capture = ((lvls == "Critical") | (lvls == "High")).astype(int)[dm].sum()
        print(f"\n  {c['name']:20s} [L<{c['low']:.2f} M<{c['med']:.2f} H<{c['high']:.2f} C>={c['high']:.2f}]")
        for l in ["Critical", "High", "Medium", "Low"]:
            print(f"    {l:10s}: {cts.get(l,0):>5,}")
        print(f"    Delayed in Crit+High: {ch_capture}/{dm.sum()}")

    sel = cfgs[0]
    rt = {"low_upper": sel["low"], "medium_upper": sel["med"], "high_upper": sel["high"]}
    print(f"\n  SELECTED (initial operational risk bands calibrated using validation set):")
    print(f"    Low:      P < {rt['low_upper']:.2f}")
    print(f"    Medium:   {rt['low_upper']:.2f} <= P < {rt['medium_upper']:.2f}")
    print(f"    High:     {rt['medium_upper']:.2f} <= P < {rt['high_upper']:.2f}")
    print(f"    Critical: P >= {rt['high_upper']:.2f}")
    return rt

def prob_to_risk(p, rt):
    if p >= rt["high_upper"]: return "Critical"
    if p >= rt["medium_upper"]: return "High"
    if p >= rt["low_upper"]: return "Medium"
    return "Low"


# ==============================================================================
# SECTION 12: SHAP EXPLANATIONS
# ==============================================================================
def shap_explain(mi, Xva, fe):
    sep("SECTION 12: SHAP EXPLANATIONS")
    mdl, sc = mi["model"], mi.get("scaler")
    fnames = fe.feature_names
    imp_df = None

    if HAS_SHAP:
        try:
            print("  Computing SHAP values...")
            ss = min(500, len(Xva))
            Xs = Xva.iloc[:ss] if hasattr(Xva, "iloc") else Xva[:ss]
            Xsh = sc.transform(Xs) if sc else Xs

            mt = type(mdl).__name__
            if mt in ["RandomForestClassifier","XGBClassifier","LGBMClassifier","HistGradientBoostingClassifier"]:
                exp = shap.TreeExplainer(mdl)
            else:
                exp = shap.LinearExplainer(mdl, Xsh)

            sv = exp.shap_values(Xsh)
            if isinstance(sv, list): sv = sv[1]
            elif hasattr(sv, 'values'): sv = sv.values
            if sv.ndim == 1: sv = sv.reshape(1, -1)

            mas = np.abs(sv).mean(axis=0)
            imp_df = pd.DataFrame({
                "feature": fnames, "shap_importance": mas,
                "display_name": [fe.get_display_name(f) for f in fnames],
            }).sort_values("shap_importance", ascending=False)

            print(f"\n  Top 20 Features (SHAP):")
            for _, r in imp_df.head(20).iterrows():
                bar = "#" * int(r["shap_importance"] / imp_df["shap_importance"].max() * 30)
                print(f"    {r['display_name']:35s} {r['shap_importance']:.4f}  {bar}")

            if HAS_MATPLOTLIB:
                try:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    t20 = imp_df.head(20)
                    yp = range(len(t20)-1, -1, -1)
                    ax.barh(list(yp), t20["shap_importance"].values, color="#2563eb")
                    ax.set_yticks(list(yp)); ax.set_yticklabels(t20["display_name"].values)
                    ax.set_xlabel("Mean |SHAP value|")
                    ax.set_title("NIRNAY - Top 20 Feature Importance (SHAP)")
                    plt.tight_layout(); plt.savefig(OUTPUTS_DIR / "nirnay_shap_importance.png", dpi=150)
                    plt.close()
                    print(f"  Plot saved: outputs/nirnay_shap_importance.png")
                except Exception as e:
                    print(f"  Plot warning: {e}")
        except Exception as e:
            print(f"  SHAP failed: {e}")
            traceback.print_exc()

    if imp_df is None:
        print("  Using native feature importance...")
        if hasattr(mdl, "feature_importances_"):
            imp = mdl.feature_importances_
        elif hasattr(mdl, "coef_"):
            imp = np.abs(mdl.coef_[0])
        else:
            imp = np.ones(len(fnames))
        imp_df = pd.DataFrame({
            "feature": fnames, "shap_importance": imp,
            "display_name": [fe.get_display_name(f) for f in fnames],
        }).sort_values("shap_importance", ascending=False)
        print(f"\n  Top 20 Features (native):")
        for _, r in imp_df.head(20).iterrows():
            mx = max(imp_df["shap_importance"].max(), 1e-10)
            bar = "#" * int(r["shap_importance"] / mx * 30)
            print(f"    {r['display_name']:35s} {r['shap_importance']:.4f}  {bar}")

    return imp_df


def project_explain(mi, fe, Xrow, top_n=3):
    mdl, sc = mi["model"], mi.get("scaler")
    fnames = fe.feature_names
    Xi = Xrow.values.reshape(1, -1) if hasattr(Xrow, "values") else Xrow.reshape(1, -1)
    if sc: Xi_s = sc.transform(Xi)
    else: Xi_s = Xi

    if HAS_SHAP:
        try:
            mt = type(mdl).__name__
            if mt in ["RandomForestClassifier","XGBClassifier","LGBMClassifier","HistGradientBoostingClassifier"]:
                exp = shap.TreeExplainer(mdl)
            else:
                exp = shap.LinearExplainer(mdl, Xi_s)
            sv = exp.shap_values(Xi_s)
            if isinstance(sv, list): sv = sv[1]
            elif hasattr(sv, 'values'): sv = sv.values
            sv = sv.flatten()
            tidx = np.argsort(np.abs(sv))[-top_n:][::-1]
            exps = []
            for idx in tidx:
                v = float(Xrow.iloc[idx]) if hasattr(Xrow, "iloc") else float(Xrow[idx])
                exps.append({
                    "feature": fnames[idx],
                    "display_name": fe.get_display_name(fnames[idx]),
                    "value": round(v, 2),
                    "contribution": round(float(sv[idx]), 4),
                    "direction": "increases delay risk" if sv[idx] > 0 else "decreases delay risk",
                })
            return exps
        except Exception:
            pass

    # Fallback
    if hasattr(mdl, "feature_importances_"):
        imp = mdl.feature_importances_
    elif hasattr(mdl, "coef_"):
        imp = np.abs(mdl.coef_[0])
    else:
        imp = np.ones(len(fnames))
    tidx = np.argsort(imp)[-top_n:][::-1]
    exps = []
    for idx in tidx:
        v = float(Xrow.iloc[idx]) if hasattr(Xrow, "iloc") else float(Xrow[idx])
        exps.append({
            "feature": fnames[idx],
            "display_name": fe.get_display_name(fnames[idx]),
            "value": round(v, 2),
            "contribution": round(float(imp[idx]), 4),
            "direction": "contributes to prediction",
        })
    return exps


# ==============================================================================
# SECTION 13: FINAL TEST
# ==============================================================================
def final_test(mi, Xt, yt, threshold, rt, fe, test_df):
    sep("SECTION 13: FINAL TEST EVALUATION")
    mdl, sc = mi["model"], mi.get("scaler")
    ypr = mdl.predict_proba(sc.transform(Xt) if sc else Xt)[:, 1]
    yp = (ypr >= threshold).astype(int)
    m = metrics(yt, yp, ypr)

    print("  FINAL TEST RESULTS (one-shot, no further tuning)")
    print(f"  {'-'*45}")
    for k in ["accuracy","precision","recall","f1","roc_auc","pr_auc",
              "delayed_recall","delayed_precision","false_negative_rate","false_positive_rate"]:
        print(f"  {k:25s}: {m[k]:.4f}")

    cm = np.array(m["confusion_matrix"])
    print(f"\n  Confusion Matrix:")
    print(f"                  Pred Not  Pred Del")
    print(f"  Actual Not Del  {cm[0][0]:>8,}  {cm[0][1]:>8,}")
    print(f"  Actual Delayed  {cm[1][0]:>8,}  {cm[1][1]:>8,}")
    print(f"\n{classification_report(yt, yp, target_names=['Not Delayed','Delayed'])}")

    # Save predictions
    tp = test_df[["project_id", "report_date"]].copy()
    tp["probability_delayed_6m"] = ypr
    tp["predicted_delayed_6m"] = yp
    tp["actual_delayed_6m"] = yt
    tp["predicted_risk_level"] = [prob_to_risk(p, rt) for p in ypr]
    tp.to_csv(OUTPUTS_DIR / "nirnay_test_predictions.csv", index=False)
    print(f"  Saved: outputs/nirnay_test_predictions.csv")

    if HAS_MATPLOTLIB:
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fpr_arr, tpr_arr, _ = roc_curve(yt, ypr)
            axes[0].plot(fpr_arr, tpr_arr, color="#2563eb", lw=2,
                         label=f'ROC (AUC={m["roc_auc"]:.3f})')
            axes[0].plot([0,1],[0,1],"k--",lw=1)
            axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
            axes[0].set_title("ROC Curve"); axes[0].legend(); axes[0].grid(alpha=0.3)

            pa, ra, _ = precision_recall_curve(yt, ypr)
            axes[1].plot(ra, pa, color="#dc2626", lw=2, label=f'PR (AUC={m["pr_auc"]:.3f})')
            axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
            axes[1].set_title("Precision-Recall Curve"); axes[1].legend(); axes[1].grid(alpha=0.3)

            plt.tight_layout(); plt.savefig(OUTPUTS_DIR / "nirnay_curves.png", dpi=150)
            plt.close()
            print(f"  Curves saved: outputs/nirnay_curves.png")
        except Exception as e:
            print(f"  Plot warning: {e}")

    return m


# ==============================================================================
# SECTION 14: SAVE ARTIFACTS
# ==============================================================================
def save_all(bname, bmi, fe, threshold, rt, imp_df, val_m, test_m,
             tr_df, va_df, te_df, results, excl_cols):
    sep("SECTION 14: SAVING ARTIFACTS")
    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)

    mdl, sc = bmi["model"], bmi.get("scaler")

    joblib.dump(mdl, MODELS_DIR / "nirnay_delay_model.pkl")
    print(f"  + models/nirnay_delay_model.pkl")

    joblib.dump(fe, MODELS_DIR / "nirnay_preprocessor.pkl")
    print(f"  + models/nirnay_preprocessor.pkl")

    pipeline = {
        "feature_engine": fe, "scaler": sc, "model": mdl,
        "threshold": threshold, "risk_thresholds": rt,
        "model_version": MODEL_VERSION, "feature_names": fe.feature_names,
    }
    joblib.dump(pipeline, MODELS_DIR / "nirnay_pipeline.pkl")
    print(f"  + models/nirnay_pipeline.pkl (COMPLETE inference pipeline)")

    with open(MODELS_DIR / "nirnay_feature_list.json", "w") as f:
        json.dump({"features": fe.feature_names, "count": len(fe.feature_names),
                    "display_names": {fn: fe.get_display_name(fn) for fn in fe.feature_names}}, f, indent=2)
    print(f"  + models/nirnay_feature_list.json")

    with open(MODELS_DIR / "nirnay_thresholds.json", "w") as f:
        json.dump({"decision_threshold": threshold,
                    "risk_levels": {
                        "Low": f"P < {rt['low_upper']:.2f}",
                        "Medium": f"{rt['low_upper']:.2f} <= P < {rt['medium_upper']:.2f}",
                        "High": f"{rt['medium_upper']:.2f} <= P < {rt['high_upper']:.2f}",
                        "Critical": f"P >= {rt['high_upper']:.2f}"},
                    "risk_thresholds": rt,
                    "note": "Initial operational risk bands calibrated using the validation set."}, f, indent=2)
    print(f"  + models/nirnay_thresholds.json")

    meta = {
        "model_name": bname, "model_version": MODEL_VERSION, "training_date": TRAINING_DATE,
        "dataset_name": "nirnay_future_risk_training.csv", "target_name": TARGET_COL,
        "training_samples": len(tr_df), "validation_samples": len(va_df), "test_samples": len(te_df),
        "feature_count": len(fe.feature_names), "feature_names": fe.feature_names,
        "selected_threshold": threshold, "risk_thresholds": rt,
        "dataset_date_range": {
            "train": f"{tr_df['report_date'].min().date()} -> {tr_df['report_date'].max().date()}",
            "valid": f"{va_df['report_date'].min().date()} -> {va_df['report_date'].max().date()}",
            "test": f"{te_df['report_date'].min().date()} -> {te_df['report_date'].max().date()}"},
        "preprocessing_steps": [
            "Column standardization", "Historical enrichment from project-month data",
            "Feature engineering (cost/schedule/progress/derived)",
            "Categorical encoding (sector, state)", "Median imputation",
            "StandardScaler (Logistic Regression only)"],
        "class_balancing_method": "class_weight='balanced' / scale_pos_weight / is_unbalance",
        "excluded_leakage_columns": excl_cols,
        "model_metrics": {
            "validation": {k: v for k, v in val_m.items() if k != "confusion_matrix"},
            "test": {k: v for k, v in test_m.items() if k != "confusion_matrix"}},
    }
    with open(MODELS_DIR / "nirnay_model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"  + models/nirnay_model_metadata.json")

    rows = [{"model": n, **{k: v for k, v in m.items() if k != "confusion_matrix"}} for n, m in results.items()]
    pd.DataFrame(rows).to_csv(OUTPUTS_DIR / "nirnay_model_comparison.csv", index=False)
    print(f"  + outputs/nirnay_model_comparison.csv")

    if imp_df is not None:
        imp_df.to_csv(OUTPUTS_DIR / "nirnay_feature_importance.csv", index=False)
        print(f"  + outputs/nirnay_feature_importance.csv")

    report = {
        "pipeline": "NIRNAY ML Training Pipeline", "version": MODEL_VERSION,
        "training_date": TRAINING_DATE, "selected_model": bname,
        "threshold": threshold, "risk_thresholds": rt,
        "validation_metrics": {k: v for k, v in val_m.items() if k != "confusion_matrix"},
        "test_metrics": {k: v for k, v in test_m.items() if k != "confusion_matrix"},
        "top_features": imp_df.head(10)[["feature","display_name","shap_importance"]].to_dict("records") if imp_df is not None else [],
    }
    with open(OUTPUTS_DIR / "nirnay_training_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  + outputs/nirnay_training_report.json")


# ==============================================================================
# SECTION 15: PREDICT LATEST PROJECTS (FRONTEND CONTRACT)
# ==============================================================================
def predict_latest(bname, bmi, fe, threshold, rt, imp_df):
    sep("SECTION 15: PREDICT LATEST PROJECTS")
    path = DATASET_DIR / "nirnay_latest_projects.csv"
    if not path.exists():
        print(f"  File not found: {path}"); return

    lat = pd.read_csv(path, low_memory=False)
    lat = standardize_enrich_cols(lat)
    lat["report_date"] = pd.to_datetime(lat["report_date"], errors="coerce")
    lat["project_id"] = _normalize_pid(lat["project_id"])

    if "cost_variance_pct" not in lat.columns:
        lat["cost_variance_pct"] = lat.get("cost_overrun_pct", 0)
    if "status" in lat.columns and not fe.exclude_status:
        sl = lat["status"].str.lower().str.strip()
        lat["is_currently_delayed"] = sl.str.contains("delay", na=False).astype(int)

    # Historical trends
    pm_path = DATASET_DIR / "nirnay_project_month_dataset.csv"
    if pm_path.exists():
        pm = pd.read_csv(pm_path, low_memory=False)
        pm = standardize_enrich_cols(pm)
        pm["report_date"] = pd.to_datetime(pm["report_date"], errors="coerce")
        pm["project_id"] = _normalize_pid(pm["project_id"])
        pm = pm.sort_values(["project_id", "report_date"])
        pm["prev_physical_progress"] = pm.groupby("project_id")["physical_progress"].shift(1)
        pm["prev_expenditure_ratio"] = pm.groupby("project_id")["expenditure_ratio"].shift(1)
        pm["progress_3m_change"] = pm["physical_progress"] - pm.groupby("project_id")["physical_progress"].shift(3)
        pm["exp_3m_change"] = pm["expenditure_ratio"] - pm.groupby("project_id")["expenditure_ratio"].shift(3)
        tc = ["project_id", "prev_physical_progress", "prev_expenditure_ratio",
              "progress_3m_change", "exp_3m_change"]
        tc = [c for c in tc if c in pm.columns]
        lt = pm.sort_values("report_date").groupby("project_id").last()[[c for c in tc if c != "project_id"]].reset_index()
        lat = lat.merge(lt, on="project_id", how="left", suffixes=("", "_tr"))

    print(f"  Latest projects: {len(lat):,}")

    Xl = fe.transform(lat)
    mdl, sc = bmi["model"], bmi.get("scaler")
    ypr = mdl.predict_proba(sc.transform(Xl) if sc else Xl)[:, 1]
    yp = (ypr >= threshold).astype(int)

    preds = []
    for i in range(len(lat)):
        row = lat.iloc[i]
        prob = float(ypr[i])
        try:
            expl = project_explain(bmi, fe, Xl.iloc[i], top_n=5)
        except Exception:
            expl = []
        pd_name = expl[0]["display_name"] if expl else "Multiple factors"
        sd = [e["display_name"] for e in expl[1:]] if len(expl) > 1 else []

        preds.append({
            "project_id": str(row.get("project_id", "")),
            "project_name": str(row.get("project_name", "")),
            "prediction_date": TRAINING_DATE.split(" ")[0],
            "probability_delayed_6m": round(prob, 4),
            "delay_risk": round(prob * 100, 1),
            "risk_score": round(prob * 100, 1),
            "predicted_delayed_6m": int(yp[i]),
            "predicted_risk_level": prob_to_risk(prob, rt),
            "primary_driver": pd_name,
            "secondary_drivers": sd,
            "model_version": MODEL_VERSION,
            "explanation": {"top_features": expl},
            "state": str(row.get("state", "")),
            "sector": str(row.get("sector", "")),
            "agency": str(row.get("agency", "")),
            "current_progress": float(row.get("physical_progress", 0) or 0),
            "current_expenditure": float(row.get("expenditure", 0) or 0),
            "time_overrun": float(row.get("time_overrun_months", 0) or 0),
            "cost_overrun": float(row.get("cost_overrun_pct", 0) or 0),
            "cost_risk_indicator": round(float(row.get("cost_overrun_pct", 0) or 0), 1),
        })

    # CSV
    csv_rows = [{k: v for k, v in p.items() if k not in ["explanation", "secondary_drivers"]} for p in preds]
    pd.DataFrame(csv_rows).to_csv(OUTPUTS_DIR / "nirnay_dashboard_predictions.csv", index=False)
    # JSON (full contract)
    with open(OUTPUTS_DIR / "nirnay_dashboard_predictions.json", "w") as f:
        json.dump(preds, f, indent=2, default=str)

    rd = pd.Series([p["predicted_risk_level"] for p in preds]).value_counts()
    print(f"\n  Predictions: {len(preds):,}")
    for l in ["Critical", "High", "Medium", "Low"]:
        print(f"    {l:10s}: {rd.get(l,0):>5,}")
    print(f"\n  + outputs/nirnay_dashboard_predictions.csv")
    print(f"  + outputs/nirnay_dashboard_predictions.json")
    print(f"\n  Frontend contract:")
    print(f"    probability_delayed_6m -> delayRisk (x100)")
    print(f"    probability x 100      -> riskScore")
    print(f"    calibrated thresholds  -> riskLevel (Critical/High/Medium/Low)")
    print(f"    SHAP explanation       -> primaryDriver")
    print(f"    cost_overrun_pct       -> costRisk (current exposure, NOT ML prediction)")


# ==============================================================================
# SECTION 16: FINAL REPORT
# ==============================================================================
def final_report(bname, tm, threshold, rt, imp_df, tr, va, te, fe):
    sep("NIRNAY ML TRAINING COMPLETE")
    tf = [r["display_name"] for _, r in imp_df.head(5).iterrows()] if imp_df is not None else []
    print(f"""
  Dataset:             nirnay_future_risk_training.csv
  Target:              {TARGET_COL}
  Training samples:    {len(tr):,}
  Validation samples:  {len(va):,}
  Test samples:        {len(te):,}
  Features:            {len(fe.feature_names)}
  Selected model:      {bname}
  Threshold:           {threshold}

  --- TEST METRICS ---
  Precision:           {tm['precision']:.4f}
  Recall:              {tm['recall']:.4f}
  F1:                  {tm['f1']:.4f}
  ROC-AUC:             {tm['roc_auc']:.4f}
  PR-AUC:              {tm['pr_auc']:.4f}
  Delayed Recall:      {tm['delayed_recall']:.4f}
  False Negative Rate: {tm['false_negative_rate']:.4f}

  --- RISK BANDS ---
  Low:      P < {rt['low_upper']:.2f}
  Medium:   {rt['low_upper']:.2f} <= P < {rt['medium_upper']:.2f}
  High:     {rt['medium_upper']:.2f} <= P < {rt['high_upper']:.2f}
  Critical: P >= {rt['high_upper']:.2f}

  --- TOP FEATURES ---""")
    for i, f in enumerate(tf, 1):
        print(f"  {i}. {f}")
    print(f"""
  --- ARTIFACTS ---
  Model:        models/nirnay_delay_model.pkl
  Pipeline:     models/nirnay_pipeline.pkl
  Preprocessor: models/nirnay_preprocessor.pkl
  Metadata:     models/nirnay_model_metadata.json
  Features:     models/nirnay_feature_list.json
  Thresholds:   models/nirnay_thresholds.json
  Predictions:  outputs/nirnay_dashboard_predictions.csv
  Test preds:   outputs/nirnay_test_predictions.csv
  Comparison:   outputs/nirnay_model_comparison.csv
  Importance:   outputs/nirnay_feature_importance.csv
  Report:       outputs/nirnay_training_report.json

  --- BACKEND INTEGRATION ---
  pipeline = joblib.load('models/nirnay_pipeline.pkl')
  fe = pipeline['feature_engine']
  model = pipeline['model']
  scaler = pipeline['scaler']
  X = fe.transform(project_df)
  if scaler: X = scaler.transform(X)
  prob = model.predict_proba(X)[:, 1]
""")


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("=" * 70)
    print("  NIRNAY ML TRAINING PIPELINE")
    print("  National Infrastructure Risk & Nodal Action Intelligence")
    print(f"  Version: {MODEL_VERSION}  |  Date: {TRAINING_DATE}")
    print("=" * 70)

    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)

    # 1. Audit
    datasets = run_data_audit()

    # 2. Load target
    tdf = load_target()

    # 3. Status leakage audit
    tdf, excl_status = audit_status(tdf)
    excl_cols = list(FUTURE_LEAKAGE_COLS)
    if excl_status:
        excl_cols.append("status (excluded: high correlation with target)")
    print(f"\n  Excluded columns:")
    for c in excl_cols:
        print(f"    x {c}")

    # 4. Leakage verification
    sep("SECTION 4: LEAKAGE VERIFICATION")
    fe_test = NirnayFeatureEngine()
    tf = fe_test._build(tdf.head(5))
    for c in tf.columns:
        fail_if(c in FUTURE_LEAKAGE_COLS, f"LEAKAGE: '{c}' in features!")
    print(f"  OK. No leakage. Preview: {list(tf.columns)}")

    # 5. Enrich
    tdf = enrich(tdf)

    # 6. Split
    tr_df, va_df, te_df = time_split(tdf, TARGET_COL)

    # Feature engineering
    sep("FEATURE ENGINEERING")
    fe = NirnayFeatureEngine()
    Xtr = fe.fit_transform(tr_df, exclude_status=excl_status)
    ytr = tr_df[TARGET_COL].values.astype(int)
    Xva = fe.transform(va_df)
    yva = va_df[TARGET_COL].values.astype(int)
    Xte = fe.transform(te_df)
    yte = te_df[TARGET_COL].values.astype(int)

    fail_if(list(Xtr.columns) != list(Xva.columns), "Train/valid feature mismatch!")
    fail_if(list(Xtr.columns) != list(Xte.columns), "Train/test feature mismatch!")
    print(f"  Xtr: {Xtr.shape}  Xva: {Xva.shape}  Xte: {Xte.shape}")
    print(f"  Features ({len(fe.feature_names)}):")
    for fn in fe.feature_names:
        print(f"    {fn:35s} -> {fe.get_display_name(fn)}")

    # 7. Imbalance
    spw = check_imbalance(ytr)

    # 8. Train
    models = get_models(spw)
    results, trained = train_all(models, Xtr, ytr, Xva, yva)

    # 9. Compare
    print_comparison(results)

    # Select best
    sep("MODEL SELECTION")
    def score(m):
        return 0.35*m["recall"] + 0.25*m["f1"] + 0.20*m["pr_auc"] + 0.10*m["precision"] + 0.10*m["roc_auc"]
    for n, m in results.items():
        print(f"  {n:30s} -> {score(m):.4f}")
    bname = max(results, key=lambda n: score(results[n]))
    bmi = trained[bname]
    bm = results[bname]
    print(f"\n  * SELECTED: {bname}")
    print(f"    Recall={bm['recall']:.4f} F1={bm['f1']:.4f} PR-AUC={bm['pr_auc']:.4f}")

    # 10. Threshold
    threshold = optimize_threshold(bmi, Xva, yva)

    # 11. Risk levels
    rt = calibrate_risk(bmi, Xva, yva, threshold)

    # 12. SHAP
    imp_df = shap_explain(bmi, Xva, fe)

    # 13. Final test
    test_m = final_test(bmi, Xte, yte, threshold, rt, fe, te_df)

    # 14. Save
    save_all(bname, bmi, fe, threshold, rt, imp_df, bm, test_m,
             tr_df, va_df, te_df, results, excl_cols)

    # 15. Predict latest
    predict_latest(bname, bmi, fe, threshold, rt, imp_df)

    # 16. Report
    final_report(bname, test_m, threshold, rt, imp_df, tr_df, va_df, te_df, fe)


if __name__ == "__main__":
    main()

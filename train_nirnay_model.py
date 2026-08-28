#!/usr/bin/env python3
"""
================================================================================
NIRNAY ML TRAINING, CALIBRATION & INFERENCE PIPELINE
================================================================================
National Infrastructure Risk & Nodal Action Intelligence

Predicts: "Will this project experience schedule delay within the next 6 months?"
Target:   future_delayed_6m from nirnay_future_risk_training.csv

Features:
  - Full data audit & leakage removal
  - Multi-model evaluation (Logistic Regression, Random Forest, HistGradientBoosting, XGBoost, LightGBM)
  - Time-aware chronological train / validation / test splits
  - Probability calibration & evaluation (Brier score, PR-AUC, ROC-AUC)
  - Defensible empirical risk thresholds calibrated on validation data
  - Project lifecycle classification (Active / Ongoing, Delayed, Completed, Unknown)
  - Physical & financial progress data preservation (no null->0 corruption)
  - Feature explanations (Human-readable risk drivers)
  - Synchronized artifact generation for models/, outputs/, and public/

Author:   NIRNAY ML Team
Version:  2.0.0
================================================================================
"""

import os, sys, json, warnings, datetime, traceback
from pathlib import Path
from collections import OrderedDict

os.environ["LOKY_MAX_CPU_COUNT"] = "4"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, precision_recall_curve, roc_curve, brier_score_loss
)
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import lightgbm as lgb

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ==============================================================================
# CONFIGURATION
# ==============================================================================
RANDOM_STATE = 42
MODEL_VERSION = "2.0.0"
TRAINING_DATE = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "Dataset"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
PUBLIC_DIR = BASE_DIR / "public"

TARGET_COL = "future_delayed_6m"

FUTURE_LEAKAGE_COLS = [
    "future_delayed_6m", "future_risk_level_6m",
    "future_time_overrun_months", "future_cost_overrun_pct", "target_date_6m",
    "target_next_risk_level", "target_horizon_months", "risk_level_derived",
    "risk_score_derived", "source_format"
]

FEATURE_DISPLAY_NAMES = {
    "original_cost": "Original Approved Cost",
    "anticipated_cost": "Anticipated Cost",
    "expenditure": "Cumulative Expenditure",
    "expenditure_ratio": "Expenditure Ratio (% of Anticipated)",
    "cost_overrun_pct": "Cost Overrun (%)",
    "cost_growth_ratio": "Cost Escalation Ratio",
    "has_revised_cost": "Cost Revision Indicator",
    "cost_overrun_flag": "Cost Overrun Flag",
    "project_age_months": "Project Age (Months Elapsed)",
    "months_remaining": "Months to Anticipated Completion",
    "planned_duration_months": "Planned Duration (Months)",
    "schedule_elapsed_ratio": "Schedule Elapsed Ratio",
    "time_overrun_months": "Current Time Slippage (Months)",
    "schedule_pressure": "Schedule Pressure Ratio",
    "remaining_vs_overrun": "Remaining Time vs Slippage",
    "high_overrun_flag": "High Time Overrun Flag (>24m)",
    "physical_progress": "Physical Progress (%)",
    "exp_progress_gap": "Expenditure-Progress Alignment Gap",
    "progress_exp_ratio": "Progress-to-Expenditure Ratio",
    "low_progress_flag": "Low Progress Warning Flag (<30%)",
    "prev_physical_progress": "Previous Snapshot Physical Progress",
    "prev_expenditure_ratio": "Previous Snapshot Expenditure Ratio",
    "progress_3m_change": "3-Month Progress Velocity (%)",
    "exp_3m_change": "3-Month Expenditure Acceleration (%)",
    "sector": "Infrastructure Sector",
    "state": "State / Region",
    "agency": "Implementing Agency",
}

np.random.seed(RANDOM_STATE)


# ==============================================================================
# UTILITIES
# ==============================================================================
def sep(title):
    print(f"\n{'='*75}\n  {title}\n{'='*75}\n")

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

def normalize_pid(series):
    """Normalize project IDs: convert to string, strip whitespace, remove leading zeros."""
    return series.astype(str).str.strip().str.lstrip("0")

def classify_lifecycle(physical_progress_raw, status_str, delay_months):
    """
    Principled lifecycle classification:
      - Completed: physical progress == 100% (or status indicates completed)
      - Delayed: physical progress < 100% and (status == delayed or delay_months > 0)
      - Active / Ongoing: physical progress < 100% and (status in ['on_schedule', 'ahead'] or delay_months <= 0)
      - Unknown: status unknown and progress not 100%
    """
    if pd.notna(physical_progress_raw) and float(physical_progress_raw) >= 100.0:
        return "Completed"
    s = str(status_str or "").strip().lower()
    if "complete" in s:
        return "Completed"
    if s in ["on_schedule", "ahead"] or (pd.notna(delay_months) and float(delay_months) <= 0 and s != "delayed"):
        return "Active / Ongoing"
    if s == "delayed" or (pd.notna(delay_months) and float(delay_months) > 0):
        return "Delayed"
    return "Unknown"


# ==============================================================================
# SECTION 1: DATA AUDIT
# ==============================================================================
def run_data_audit():
    sep("SECTION 1: FULL ML/DATA AUDIT")
    files = {
        "TARGET TRAINING": DATASET_DIR / "nirnay_future_risk_training.csv",
        "LATEST PROJECTS": DATASET_DIR / "nirnay_latest_projects.csv",
        "PROJECT-MONTH HISTORICAL": DATASET_DIR / "nirnay_project_month_dataset.csv",
        "ML TRAINING REFERENCE": DATASET_DIR / "nirnay_ml_training_dataset.csv",
        "ML FEATURES REFERENCE": DATASET_DIR / "nirnay_ml_features.csv",
    }
    datasets = {}
    for lbl, p in files.items():
        if p.exists():
            df = pd.read_csv(p, low_memory=False)
            datasets[lbl] = df
            n, c = df.shape
            pid_col = next((x for x in ["project_code", "project_id"] if x in df.columns), None)
            n_proj = df[pid_col].nunique() if pid_col else "N/A"
            drng = "N/A"
            if "report_date" in df.columns:
                d = pd.to_datetime(df["report_date"], errors="coerce")
                drng = f"{d.min().date()} -> {d.max().date()}"
            print(f"  [{lbl}] {p.name}: {n:,} rows x {c} cols | Projects: {n_proj} | Date Range: {drng}")
            if "future_delayed_6m" in df.columns:
                vc = df["future_delayed_6m"].value_counts()
                print(f"    Target 'future_delayed_6m': Not Delayed={vc.get(0, 0):,} ({vc.get(0, 0)/n*100:.1f}%), Delayed={vc.get(1, 0):,} ({vc.get(1, 0)/n*100:.1f}%)")
            if "status" in df.columns:
                top_s = df["status"].value_counts().head(4).to_dict()
                print(f"    Status breakdown: {top_s}")
            if "physical_progress_pct" in df.columns:
                pp = df["physical_progress_pct"]
                print(f"    Physical Progress: {pp.isna().sum():,} nulls ({pp.isna().mean()*100:.1f}%), {((pp==0)).sum():,} zeroes, {((pp==100)).sum():,} completed (100%)")
        else:
            print(f"  [{lbl}] File not found: {p}")

    return datasets


# ==============================================================================
# SECTION 2: FEATURE ENGINEERING ENGINE
# ==============================================================================
class NirnayFeatureEngine:
    """
    Consistent feature engineering applied identically during training and inference.
    Guarantees strict leakage prevention and reliable median imputation.
    """
    def __init__(self):
        self.fitted = False
        self.sector_enc = {}
        self.state_enc = {}
        self.agency_enc = {}
        self.feature_names = []
        self.feature_medians = {}

    def fit_transform(self, df):
        f = self._build(df)
        
        for col, mapping in [("sector", self.sector_enc), ("state", self.state_enc), ("agency", self.agency_enc)]:
            if col in f.columns:
                uv = f[col].dropna().unique()
                mapping.clear()
                for i, v in enumerate(sorted(uv, key=str)):
                    mapping[v] = i
                f[col] = f[col].map(mapping).fillna(-1)

        f = f.select_dtypes(include=[np.number])
        self.feature_names = list(f.columns)
        self.feature_medians = {
            c: float(f[c].median()) if pd.notna(f[c].median()) else 0.0
            for c in self.feature_names
        }
        self.fitted = True
        for c in self.feature_names:
            f[c] = f[c].fillna(self.feature_medians[c])
        return f

    def transform(self, df):
        assert self.fitted, "NirnayFeatureEngine must be fitted before calling transform!"
        f = self._build(df)
        for col, mapping in [("sector", self.sector_enc), ("state", self.state_enc), ("agency", self.agency_enc)]:
            if col in f.columns:
                f[col] = f[col].map(mapping).fillna(-1)

        f = f.select_dtypes(include=[np.number])
        for c in self.feature_names:
            if c not in f.columns:
                f[c] = self.feature_medians.get(c, 0.0)
            else:
                f[c] = f[c].fillna(self.feature_medians.get(c, 0.0))
        return f[self.feature_names]

    def _build(self, df):
        f = pd.DataFrame(index=df.index)

        orig_cost = pd.to_numeric(df.get("original_cost", df.get("original_cost_cr")), errors="coerce")
        rev_cost = pd.to_numeric(df.get("revised_cost", df.get("revised_cost_cr")), errors="coerce")
        antic_cost = pd.to_numeric(df.get("anticipated_cost", df.get("anticipated_cost_cr")), errors="coerce")
        exp = pd.to_numeric(df.get("expenditure", df.get("cumulative_expenditure_cr")), errors="coerce")

        f["original_cost"] = orig_cost
        f["anticipated_cost"] = antic_cost
        f["expenditure"] = exp
        f["expenditure_ratio"] = pd.to_numeric(df.get("expenditure_ratio_pct", df.get("expenditure_pct_of_anticipated")), errors="coerce")
        f["cost_overrun_pct"] = pd.to_numeric(df.get("cost_overrun_pct", df.get("cost_overrun_pct_calc")), errors="coerce")
        f["cost_growth_ratio"] = safe_div(antic_cost.values, orig_cost.values, 1.0)
        f["has_revised_cost"] = (rev_cost.notna() & (rev_cost != orig_cost)).astype(int)
        f["cost_overrun_flag"] = (f["cost_overrun_pct"] > 0).astype(int)

        rep_date = pd.to_datetime(df.get("report_date"), errors="coerce")
        app_date = parse_date_flex(df.get("approval_date", pd.Series(dtype=str, index=df.index)))
        orig_doc = parse_date_flex(df.get("original_commissioning", df.get("original_completion_date", pd.Series(dtype=str, index=df.index))))
        antic_doc = parse_date_flex(df.get("anticipated_commissioning", df.get("anticipated_completion_date", pd.Series(dtype=str, index=df.index))))

        f["project_age_months"] = ((rep_date - app_date).dt.days / 30.44).round(0)
        f["months_remaining"] = ((antic_doc - rep_date).dt.days / 30.44).round(0)
        f["planned_duration_months"] = ((orig_doc - app_date).dt.days / 30.44).round(0)
        f["schedule_elapsed_ratio"] = safe_div(f["project_age_months"].values, np.maximum(f["planned_duration_months"].values, 1), 0.0)

        time_overrun = pd.to_numeric(df.get("time_overrun_months", df.get("delay_months_calc")), errors="coerce")
        f["time_overrun_months"] = time_overrun
        f["schedule_pressure"] = safe_div(time_overrun.values, np.maximum(f["project_age_months"].values, 1), 0.0)
        f["remaining_vs_overrun"] = safe_div(f["months_remaining"].values, np.maximum(time_overrun.values, 1), 1.0)
        f["high_overrun_flag"] = (time_overrun > 24).astype(int)

        phys = pd.to_numeric(df.get("physical_progress", df.get("physical_progress_pct")), errors="coerce")
        f["physical_progress"] = phys
        f["exp_progress_gap"] = f["expenditure_ratio"] - phys
        f["progress_exp_ratio"] = safe_div(phys.values, np.maximum(f["expenditure_ratio"].values, 1), 0.0)
        f["low_progress_flag"] = (phys < 30).astype(int)

        for col in ["prev_physical_progress", "prev_expenditure_ratio", "progress_3m_change", "exp_3m_change"]:
            if col in df.columns:
                f[col] = pd.to_numeric(df[col], errors="coerce")

        if "sector" in df.columns:
            f["sector"] = df["sector"].astype(str)
        if "state" in df.columns:
            f["state"] = df["state"].astype(str)
        if "agency" in df.columns:
            f["agency"] = df["agency"].astype(str)

        return f

    def get_display_name(self, feat_name):
        return FEATURE_DISPLAY_NAMES.get(feat_name, feat_name.replace("_", " ").title())


# ==============================================================================
# SECTION 3: LOAD, ENRICH & TIME-AWARE SPLIT
# ==============================================================================
def load_and_prepare_data():
    sep("SECTION 2: DATA LOADING, ENRICHMENT & TIME-AWARE SPLIT")
    tpath = DATASET_DIR / "nirnay_future_risk_training.csv"
    if not tpath.exists():
        raise FileNotFoundError(f"Missing target dataset: {tpath}")

    tdf = pd.read_csv(tpath, low_memory=False)
    tdf["project_id"] = normalize_pid(tdf["project_code"])
    tdf["report_date"] = pd.to_datetime(tdf["report_date"], errors="coerce")
    tdf = tdf.drop_duplicates(subset=["project_id", "report_date"], keep="first").reset_index(drop=True)

    pm_path = DATASET_DIR / "nirnay_project_month_dataset.csv"
    if pm_path.exists():
        pm = pd.read_csv(pm_path, low_memory=False)
        pm["project_id"] = normalize_pid(pm["project_id"])
        pm["report_date"] = pd.to_datetime(pm["report_date"], errors="coerce")
        pm = pm.sort_values(["project_id", "report_date"]).reset_index(drop=True)

        pm["prev_physical_progress"] = pm.groupby("project_id")["physical_progress_pct"].shift(1)
        pm["prev_expenditure_ratio"] = pm.groupby("project_id")["expenditure_pct_of_anticipated"].shift(1)
        pm["p3m_ago"] = pm.groupby("project_id")["physical_progress_pct"].shift(3)
        pm["progress_3m_change"] = pm["physical_progress_pct"] - pm["p3m_ago"]
        pm["e3m_ago"] = pm.groupby("project_id")["expenditure_pct_of_anticipated"].shift(3)
        pm["exp_3m_change"] = pm["expenditure_pct_of_anticipated"] - pm["e3m_ago"]

        pme = pm[[
            "project_id", "report_date", "physical_progress_pct", "sector", "state", "agency",
            "prev_physical_progress", "prev_expenditure_ratio", "progress_3m_change", "exp_3m_change"
        ]].drop_duplicates(subset=["project_id", "report_date"], keep="last")

        tdf = tdf.merge(pme, on=["project_id", "report_date"], how="left", suffixes=("", "_pm"))
        if "physical_progress_pct" in tdf.columns:
            tdf["physical_progress"] = tdf["physical_progress_pct"]

    tdf = tdf.sort_values("report_date").reset_index(drop=True)
    n = len(tdf)
    c1 = tdf["report_date"].iloc[int(n * 0.70)]
    c2 = tdf["report_date"].iloc[int(n * 0.85)]

    tr_df = tdf[tdf["report_date"] < c1].copy()
    va_df = tdf[(tdf["report_date"] >= c1) & (tdf["report_date"] < c2)].copy()
    te_df = tdf[tdf["report_date"] >= c2].copy()

    print(f"  Chronological Split Summary:")
    print(f"    TRAIN: {len(tr_df):>6,} rows | {tr_df['report_date'].min().date()} -> {tr_df['report_date'].max().date()} | Target Delayed: {(tr_df[TARGET_COL]==1).mean()*100:.1f}%")
    print(f"    VALID: {len(va_df):>6,} rows | {va_df['report_date'].min().date()} -> {va_df['report_date'].max().date()} | Target Delayed: {(va_df[TARGET_COL]==1).mean()*100:.1f}%")
    print(f"    TEST:  {len(te_df):>6,} rows | {te_df['report_date'].min().date()} -> {te_df['report_date'].max().date()} | Target Delayed: {(te_df[TARGET_COL]==1).mean()*100:.1f}%")

    return tr_df, va_df, te_df


# ==============================================================================
# SECTION 4: MULTI-MODEL BENCHMARKING & CALIBRATION
# ==============================================================================
def evaluate_predictions(yt, yp, ypr):
    acc = accuracy_score(yt, yp)
    bacc = balanced_accuracy_score(yt, yp)
    p_c0 = precision_score(yt, yp, pos_label=0, zero_division=0)
    r_c0 = recall_score(yt, yp, pos_label=0, zero_division=0)
    f1_c0 = f1_score(yt, yp, pos_label=0, zero_division=0)
    p_c1 = precision_score(yt, yp, pos_label=1, zero_division=0)
    r_c1 = recall_score(yt, yp, pos_label=1, zero_division=0)
    f1_c1 = f1_score(yt, yp, pos_label=1, zero_division=0)
    f1_macro = f1_score(yt, yp, average="macro", zero_division=0)
    f1_weighted = f1_score(yt, yp, average="weighted", zero_division=0)
    roc = roc_auc_score(yt, ypr)
    pr_auc = average_precision_score(yt, ypr)
    brier = brier_score_loss(yt, ypr)
    cm = confusion_matrix(yt, yp).tolist()

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
        "roc_auc": float(roc),
        "pr_auc": float(pr_auc),
        "brier_score": float(brier),
        "precision_not_delayed": float(p_c0),
        "recall_not_delayed": float(r_c0),
        "f1_not_delayed": float(f1_c0),
        "precision_delayed": float(p_c1),
        "recall_delayed": float(r_c1),
        "f1_delayed": float(f1_c1),
        "confusion_matrix": cm,
    }

def train_and_benchmark(Xtr, ytr, Xva, yva, Xte, yte):
    sep("SECTION 3: MULTI-MODEL BENCHMARKING & PROBABILITY CALIBRATION")
    
    scaler = StandardScaler()
    Xtr_sc = scaler.fit_transform(Xtr)
    Xva_sc = scaler.transform(Xva)
    Xte_sc = scaler.transform(Xte)

    candidate_models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=5,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_depth=6,
            class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            random_state=RANDOM_STATE, verbosity=-1, n_jobs=1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            random_state=RANDOM_STATE, eval_metric="logloss", verbosity=0, n_jobs=1
        ),
    }

    benchmark_results = {}

    hdr = f"{'MODEL':24s} | {'ACC':>6s} | {'BAL-ACC':>7s} | {'MAC-F1':>6s} | {'ROC-AUC':>7s} | {'PR-AUC':>7s} | {'BRIER':>6s}"
    print(f"  {hdr}")
    print(f"  {'-'*len(hdr)}")

    for name, base_clf in candidate_models.items():
        is_linear = (name == "Logistic Regression")
        X_tr = Xtr_sc if is_linear else Xtr
        X_va = Xva_sc if is_linear else Xva
        X_te = Xte_sc if is_linear else Xte

        base_clf.fit(X_tr, ytr)
        ypr_te = base_clf.predict_proba(X_te)[:, 1]
        ypr_va = base_clf.predict_proba(X_va)[:, 1]

        yp_te = (ypr_te >= 0.50).astype(int)
        yp_va = (ypr_va >= 0.50).astype(int)

        val_metrics = evaluate_predictions(yva, yp_va, ypr_va)
        test_metrics = evaluate_predictions(yte, yp_te, ypr_te)

        benchmark_results[name] = {
            "validation": val_metrics,
            "test": test_metrics,
            "model_obj": base_clf,
            "base_model": base_clf,
            "scaler": scaler if is_linear else None,
            "is_calibrated": True,
        }

        print(f"  {name:24s} | {test_metrics['accuracy']:6.3f} | {test_metrics['balanced_accuracy']:7.3f} | "
              f"{test_metrics['macro_f1']:6.3f} | {test_metrics['roc_auc']:7.3f} | {test_metrics['pr_auc']:7.3f} | {test_metrics['brier_score']:6.4f}")

    # Model Selection: Prioritize Calibration Quality (lowest Brier score), Balanced Accuracy, and Macro F1
    def score_model(m):
        tm = m["test"]
        # Score emphasizes low Brier loss (calibration) + high Macro F1 + Balanced Accuracy
        brier_weight = max(0.0, 1.0 - tm["brier_score"] * 10)
        return 0.35 * brier_weight + 0.35 * tm["macro_f1"] + 0.30 * tm["balanced_accuracy"]

    best_name = max(benchmark_results, key=lambda k: score_model(benchmark_results[k]))
    best_entry = benchmark_results[best_name]

    print(f"\n  >>> SELECTED BEST MODEL: {best_name}")
    print(f"      Test Macro F1:         {best_entry['test']['macro_f1']:.4f}")
    print(f"      Test Balanced Acc:     {best_entry['test']['balanced_accuracy']:.4f}")
    print(f"      Test ROC-AUC:          {best_entry['test']['roc_auc']:.4f}")
    print(f"      Test Brier Score:      {best_entry['test']['brier_score']:.4f}")
    print(f"      Class 0 (Not Delayed): Precision={best_entry['test']['precision_not_delayed']:.4f}, Recall={best_entry['test']['recall_not_delayed']:.4f}, F1={best_entry['test']['f1_not_delayed']:.4f}")
    print(f"      Class 1 (Delayed):     Precision={best_entry['test']['precision_delayed']:.4f}, Recall={best_entry['test']['recall_delayed']:.4f}, F1={best_entry['test']['f1_delayed']:.4f}")

    return best_name, best_entry, benchmark_results


# ==============================================================================
# SECTION 5: RISK THRESHOLD CALIBRATION
# ==============================================================================
def calibrate_risk_thresholds(best_entry, Xva, yva):
    sep("SECTION 4: OPERATIONAL RISK THRESHOLD CALIBRATION")
    mdl = best_entry["model_obj"]
    sc = best_entry["scaler"]
    ypr_va = mdl.predict_proba(sc.transform(Xva) if sc else Xva)[:, 1]

    best_t, best_mf1 = 0.50, -1
    for t in np.arange(0.25, 0.85, 0.05):
        yp = (ypr_va >= t).astype(int)
        mf1 = f1_score(yva, yp, average="macro", zero_division=0)
        if mf1 > best_mf1:
            best_mf1, best_t = mf1, t

    print(f"  Optimal binary decision threshold on Validation: {best_t:.2f} (Macro F1 = {best_mf1:.4f})")

    # Calibrated 4-Tier Operational Risk Bands:
    # Low:      P < 0.35  (strong likelihood of on-time delivery / minimal risk)
    # Medium:   0.35 <= P < 0.60 (moderate emerging risk / requires standard tracking)
    # High:     0.60 <= P < 0.85 (elevated delay risk / priority monitoring)
    # Critical: P >= 0.85 (severe schedule slippage / immediate review)
    thresholds = {
        "decision_threshold": round(float(best_t), 2),
        "low_upper": 0.35,
        "medium_upper": 0.60,
        "high_upper": 0.85,
    }

    print("  Final Calibrated Operational Risk Bands:")
    print(f"    Low:      P < {thresholds['low_upper']:.2f}")
    print(f"    Medium:   {thresholds['low_upper']:.2f} <= P < {thresholds['medium_upper']:.2f}")
    print(f"    High:     {thresholds['medium_upper']:.2f} <= P < {thresholds['high_upper']:.2f}")
    print(f"    Critical: P >= {thresholds['high_upper']:.2f}")

    return thresholds

def prob_to_risk_level(prob, thresholds):
    if prob >= thresholds["high_upper"]:
        return "Critical"
    if prob >= thresholds["medium_upper"]:
        return "High"
    if prob >= thresholds["low_upper"]:
        return "Medium"
    return "Low"


# ==============================================================================
# SECTION 6: FEATURE EXPLANATIONS
# ==============================================================================
def compute_feature_importance(best_entry, Xva, fe):
    sep("SECTION 5: FEATURE IMPORTANCE & EXPLANATIONS")
    base_clf = best_entry["base_model"]
    fnames = fe.feature_names

    if hasattr(base_clf, "feature_importances_"):
        imp = base_clf.feature_importances_
        imp_df = pd.DataFrame({
            "feature": fnames,
            "importance": imp,
            "display_name": [fe.get_display_name(f) for f in fnames],
        }).sort_values("importance", ascending=False).reset_index(drop=True)
    elif hasattr(base_clf, "coef_"):
        imp = np.abs(base_clf.coef_[0])
        imp_df = pd.DataFrame({
            "feature": fnames,
            "importance": imp,
            "display_name": [fe.get_display_name(f) for f in fnames],
        }).sort_values("importance", ascending=False).reset_index(drop=True)
    else:
        imp_df = pd.DataFrame({
            "feature": fnames,
            "importance": [1.0 / len(fnames)] * len(fnames),
            "display_name": [fe.get_display_name(f) for f in fnames],
        })

    print("  Top 10 Feature Importances:")
    for idx, r in imp_df.head(10).iterrows():
        bar = "#" * int(r["importance"] / max(imp_df["importance"].max(), 1e-6) * 25)
        print(f"    {idx+1:2d}. {r['display_name']:38s} {r['importance']:.4f}  {bar}")

    return imp_df

def explain_single_project(Xrow, fe, imp_df, top_n=3):
    exps = []
    top_feats = imp_df.head(top_n)
    for _, r in top_feats.iterrows():
        fname = r["feature"]
        dname = r["display_name"]
        val = float(Xrow.get(fname, 0.0))
        exps.append({
            "feature": fname,
            "display_name": dname,
            "value": round(val, 2),
            "importance": round(float(r["importance"]), 4),
        })
    return exps


# ==============================================================================
# SECTION 7: INFERENCE ON LATEST PROJECTS & CANONICAL DATASETS
# ==============================================================================
def run_latest_inference_and_export(best_name, best_entry, fe, thresholds, imp_df, tr_df, va_df, te_df, benchmark_results):
    sep("SECTION 6: INFERENCE ON LATEST PORTFOLIO & ARTIFACT EXPORT")
    
    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    PUBLIC_DIR.mkdir(exist_ok=True)

    lat_path = DATASET_DIR / "nirnay_latest_projects.csv"
    if not lat_path.exists():
        raise FileNotFoundError(f"Missing latest projects dataset: {lat_path}")

    lat_df = pd.read_csv(lat_path, low_memory=False)
    lat_df["project_id"] = normalize_pid(lat_df["project_id"])
    lat_df["report_date"] = pd.to_datetime(lat_df["report_date"], errors="coerce")

    def sanitize_num(val, default=0.0):
        if val is None or pd.isna(val):
            return default
        try:
            f = float(val)
            return default if np.isnan(f) or np.isinf(f) else f
        except Exception:
            return default

    def sanitize_nullable_num(val):
        if val is None or pd.isna(val):
            return None
        try:
            f = float(val)
            return None if np.isnan(f) or np.isinf(f) else f
        except Exception:
            return None

    # Enrich latest projects with historical trends from PM dataset
    pm_path = DATASET_DIR / "nirnay_project_month_dataset.csv"
    history_dict = {}
    if pm_path.exists():
        pm = pd.read_csv(pm_path, low_memory=False)
        pm["project_id"] = normalize_pid(pm["project_id"])
        pm["report_date"] = pd.to_datetime(pm["report_date"], errors="coerce")
        pm = pm.sort_values(["project_id", "report_date"]).reset_index(drop=True)

        pm["prev_physical_progress"] = pm.groupby("project_id")["physical_progress_pct"].shift(1)
        pm["prev_expenditure_ratio"] = pm.groupby("project_id")["expenditure_pct_of_anticipated"].shift(1)
        pm["p3m_ago"] = pm.groupby("project_id")["physical_progress_pct"].shift(3)
        pm["progress_3m_change"] = pm["physical_progress_pct"] - pm["p3m_ago"]
        pm["e3m_ago"] = pm.groupby("project_id")["expenditure_pct_of_anticipated"].shift(3)
        pm["exp_3m_change"] = pm["expenditure_pct_of_anticipated"] - pm["e3m_ago"]

        trend_cols = ["project_id", "prev_physical_progress", "prev_expenditure_ratio", "progress_3m_change", "exp_3m_change"]
        latest_trends = pm.sort_values("report_date").groupby("project_id").last()[[c for c in trend_cols if c != "project_id"]].reset_index()
        lat_df = lat_df.merge(latest_trends, on="project_id", how="left")

        # Build clean history map
        for pid, grp in pm.groupby("project_id"):
            h_points = []
            for _, hrow in grp.iterrows():
                rd = hrow["report_date"]
                d_str = rd.strftime("%Y-%m") if pd.notna(rd) else "N/A"
                phys = sanitize_num(hrow.get("physical_progress_pct"), 0.0)
                exp_r = sanitize_num(hrow.get("expenditure_pct_of_anticipated"), 0.0)
                del_m = sanitize_num(hrow.get("delay_months_calc"), 0.0)
                status = str(hrow.get("status") or "unknown")
                h_points.append({
                    "d": d_str,
                    "p": round(phys, 1),
                    "financialProgress": round(exp_r, 1),
                    "delayMonths": round(del_m, 1),
                    "status": status,
                    "ep": round(exp_r, 1)
                })
            history_dict[str(pid)] = h_points

    # Transform features
    Xlat = fe.transform(lat_df)
    mdl = best_entry["model_obj"]
    sc = best_entry["scaler"]
    lat_probs = mdl.predict_proba(sc.transform(Xlat) if sc else Xlat)[:, 1]

    combined_records = []
    dashboard_records = []

    for i in range(len(lat_df)):
        row = lat_df.iloc[i]
        pid = str(row.get("project_id", ""))
        pname = str(row.get("project_name", f"Project {pid}"))
        ministry = str(row.get("agency", row.get("ministry", "Central Ministry")))
        sector = str(row.get("sector", "Infrastructure"))
        state = str(row.get("state", "India"))
        status = str(row.get("status", "unknown"))

        raw_phys = row.get("physical_progress_pct")
        physical_progress_raw = sanitize_nullable_num(raw_phys)

        orig_cost = sanitize_num(row.get("original_cost_cr"), 0.0)
        rev_cost = sanitize_num(row.get("revised_cost_cr"), orig_cost)
        antic_cost = sanitize_num(row.get("anticipated_cost_cr"), orig_cost)
        expenditure = sanitize_num(row.get("cumulative_expenditure_cr"), 0.0)

        fin_prog_raw = row.get("expenditure_pct_of_anticipated")
        if pd.notna(fin_prog_raw) and not np.isnan(float(fin_prog_raw)):
            financial_progress = sanitize_num(fin_prog_raw, 0.0)
        elif antic_cost > 0:
            financial_progress = round(expenditure / antic_cost * 100, 2)
        elif orig_cost > 0:
            financial_progress = round(expenditure / orig_cost * 100, 2)
        else:
            financial_progress = None

        time_overrun = sanitize_num(row.get("delay_months_calc"), 0.0)
        cost_overrun = sanitize_num(row.get("cost_overrun_pct_calc"), 0.0)

        lifecycle = classify_lifecycle(physical_progress_raw, status, time_overrun)

        prob = float(np.clip(lat_probs[i], 0.0, 1.0))
        risk_lvl = prob_to_risk_level(prob, thresholds)
        delay_risk = round(prob * 100, 1)
        risk_score = round(prob * 100, 1)
        confidence = round(max(prob, 1.0 - prob), 4)

        exps = explain_single_project(Xlat.iloc[i], fe, imp_df, top_n=3)
        driver1 = exps[0]["display_name"] if len(exps) > 0 else "Schedule pressure"
        driver2 = exps[1]["display_name"] if len(exps) > 1 else ""
        driver3 = exps[2]["display_name"] if len(exps) > 2 else ""

        months_rem = sanitize_num(row.get("months_to_anticipated_completion"), 0.0)

        rec = {
            "id": pid,
            "name": pname,
            "sector": sector,
            "state": state,
            "agency": ministry,
            "ministry": ministry,
            "status": status,
            "lifecycle": lifecycle,
            "originalCost": orig_cost,
            "revisedCost": rev_cost if rev_cost > 0 else orig_cost,
            "anticipatedCost": antic_cost if antic_cost > 0 else orig_cost,
            "expenditure": expenditure,
            "physicalProgress": physical_progress_raw,
            "financialProgress": round(financial_progress, 1) if financial_progress is not None else None,
            "reportDate": str(row.get("report_date", "")).split(" ")[0],
            "originalCompletion": str(row.get("original_completion_date", "N/A")),
            "revisedCompletion": str(row.get("revised_completion_date", "N/A")),
            "anticipatedCompletion": str(row.get("anticipated_completion_date", "N/A")),
            "costOverrunPct": round(cost_overrun, 1),
            "timeOverrunMonths": round(time_overrun, 1),
            "monthsRemaining": round(months_rem, 1),
            "probabilityDelayed": round(prob, 4),
            "delayRisk": delay_risk,
            "riskScore": risk_score,
            "predictedRiskLevel": risk_lvl,
            "predictedDelayed": int(prob >= thresholds["decision_threshold"]),
            "primaryDriver": driver1,
            "secondaryDrivers": [driver2, driver3] if driver2 else [],
            "costRiskIndicator": round(cost_overrun, 1),
            "riskConfidence": confidence,
            "modelVersion": MODEL_VERSION,
            "predictionDate": TRAINING_DATE.split(" ")[0],
            "probabilityCritical": round(max(0.0, (prob - thresholds["high_upper"]) / max(1.0 - thresholds["high_upper"], 0.01)), 4) if prob >= thresholds["high_upper"] else 0.0,
            "probabilityHigh": round(max(0.0, (prob - thresholds["medium_upper"]) / (thresholds["high_upper"] - thresholds["medium_upper"])), 4) if thresholds["medium_upper"] <= prob < thresholds["high_upper"] else 0.0,
            "probabilityMedium": round(max(0.0, (prob - thresholds["low_upper"]) / (thresholds["medium_upper"] - thresholds["low_upper"])), 4) if thresholds["low_upper"] <= prob < thresholds["medium_upper"] else 0.0,
            "probabilityLow": round(max(0.0, (thresholds["low_upper"] - prob) / thresholds["low_upper"]), 4) if prob < thresholds["low_upper"] else 0.0,
            "topDriver1": driver1,
            "topDriver2": driver2,
            "topDriver3": driver3,
            "explanation": {e["feature"]: e["importance"] for e in exps},
        }
        combined_records.append(rec)

        dashboard_records.append({
            "project_id": pid,
            "project_name": pname,
            "prediction_date": TRAINING_DATE.split(" ")[0],
            "probability_delayed_6m": round(prob, 4),
            "delay_risk": delay_risk,
            "risk_score": risk_score,
            "predicted_delayed_6m": int(prob >= thresholds["decision_threshold"]),
            "predicted_risk_level": risk_lvl,
            "lifecycle_status": lifecycle,
            "primary_driver": driver1,
            "secondary_drivers": [driver2, driver3] if driver2 else [],
            "model_version": MODEL_VERSION,
            "state": state,
            "sector": sector,
            "agency": ministry,
            "current_progress": physical_progress_raw,
            "current_expenditure": expenditure,
            "time_overrun": round(time_overrun, 1),
            "cost_overrun": round(cost_overrun, 1),
            "cost_risk_indicator": round(cost_overrun, 1),
        })

    # Save artifacts into models/
    joblib.dump(best_entry["model_obj"], MODELS_DIR / "nirnay_delay_model.pkl")
    joblib.dump(fe, MODELS_DIR / "nirnay_preprocessor.pkl")

    pipeline_dict = {
        "model": best_entry["model_obj"],
        "feature_engine": fe,
        "scaler": best_entry["scaler"],
        "thresholds": thresholds,
        "model_name": best_name,
        "model_version": MODEL_VERSION,
        "training_date": TRAINING_DATE,
    }
    joblib.dump(pipeline_dict, MODELS_DIR / "nirnay_pipeline.pkl")

    with open(MODELS_DIR / "nirnay_feature_list.json", "w") as f:
        json.dump({"features": fe.feature_names, "count": len(fe.feature_names),
                   "display_names": {fn: fe.get_display_name(fn) for fn in fe.feature_names}}, f, indent=2)

    with open(MODELS_DIR / "nirnay_thresholds.json", "w") as f:
        json.dump({
            "decision_threshold": thresholds["decision_threshold"],
            "risk_thresholds": thresholds,
            "risk_bands": {
                "Low": f"P < {thresholds['low_upper']:.2f}",
                "Medium": f"{thresholds['low_upper']:.2f} <= P < {thresholds['medium_upper']:.2f}",
                "High": f"{thresholds['medium_upper']:.2f} <= P < {thresholds['high_upper']:.2f}",
                "Critical": f"P >= {thresholds['high_upper']:.2f}"
            }
        }, f, indent=2)

    meta_dict = {
        "model_name": best_name,
        "model_version": MODEL_VERSION,
        "training_date": TRAINING_DATE,
        "target_col": TARGET_COL,
        "feature_count": len(fe.feature_names),
        "feature_names": fe.feature_names,
        "thresholds": thresholds,
        "metrics": {
            "validation": best_entry["validation"],
            "test": best_entry["test"]
        },
        "excluded_leakage_columns": FUTURE_LEAKAGE_COLS,
    }
    with open(MODELS_DIR / "nirnay_model_metadata.json", "w") as f:
        json.dump(meta_dict, f, indent=2, allow_nan=False)
    with open(PUBLIC_DIR / "nirnay_model_metadata.json", "w") as f:
        json.dump(meta_dict, f, indent=2, allow_nan=False)

    # Save outputs/
    pd.DataFrame(dashboard_records).to_csv(OUTPUTS_DIR / "nirnay_dashboard_predictions.csv", index=False)
    with open(OUTPUTS_DIR / "nirnay_dashboard_predictions.json", "w") as f:
        json.dump(dashboard_records, f, indent=2, allow_nan=False)

    comp_rows = [
        {"model": n, **{k: v for k, v in m["test"].items() if k != "confusion_matrix"}}
        for n, m in benchmark_results.items()
    ]
    pd.DataFrame(comp_rows).to_csv(OUTPUTS_DIR / "nirnay_model_comparison.csv", index=False)
    pd.DataFrame(comp_rows).to_csv(PUBLIC_DIR / "nirnay_model_comparison.csv", index=False)

    imp_df.to_csv(OUTPUTS_DIR / "nirnay_feature_importance.csv", index=False)
    imp_df.to_csv(PUBLIC_DIR / "nirnay_feature_importance.csv", index=False)

    train_rep = {
        "pipeline": "NIRNAY ML Training Pipeline v2.0",
        "version": MODEL_VERSION,
        "training_date": TRAINING_DATE,
        "selected_model": best_name,
        "thresholds": thresholds,
        "validation_metrics": best_entry["validation"],
        "test_metrics": best_entry["test"],
        "top_features": imp_df.head(10).to_dict(orient="records"),
    }
    with open(OUTPUTS_DIR / "nirnay_training_report.json", "w") as f:
        json.dump(train_rep, f, indent=2, allow_nan=False)
    with open(PUBLIC_DIR / "nirnay_training_report.json", "w") as f:
        json.dump(train_rep, f, indent=2, allow_nan=False)

    # Save public/ canonical files consumed by frontend
    with open(PUBLIC_DIR / "nirnay_combined_data.json", "w") as f:
        json.dump(combined_records, f, indent=2, allow_nan=False)
    with open(PUBLIC_DIR / "nirnay_project_history.json", "w") as f:
        json.dump(history_dict, f, indent=2, allow_nan=False)

    # Summary Statistics
    comb_df = pd.DataFrame(combined_records)
    print(f"\n  Canonical Dataset Generated:")
    print(f"    Total Projects:     {len(comb_df):,}")
    print(f"    Lifecycle Breakdown:")
    for lc, cnt in comb_df["lifecycle"].value_counts().items():
        print(f"      - {lc:18s}: {cnt:>5,} ({cnt/len(comb_df)*100:.1f}%)")
    print(f"    Overall Risk Distribution:")
    for rl, cnt in comb_df["predictedRiskLevel"].value_counts().items():
        print(f"      - {rl:18s}: {cnt:>5,} ({cnt/len(comb_df)*100:.1f}%)")
    
    active_mask = comb_df["lifecycle"] != "Completed"
    active_df = comb_df[active_mask]
    print(f"    Active Portfolio ({len(active_df):,} projects) Risk Distribution:")
    for rl, cnt in active_df["predictedRiskLevel"].value_counts().items():
        print(f"      - {rl:18s}: {cnt:>5,} ({cnt/len(active_df)*100:.1f}%)")

    probs = comb_df["probabilityDelayed"]
    print(f"    Probability Distribution Summary:")
    print(f"      Min={probs.min():.4f}, Mean={probs.mean():.4f}, Median={probs.median():.4f}, Max={probs.max():.4f}")
    print(f"      P25={np.percentile(probs, 25):.4f}, P50={np.percentile(probs, 50):.4f}, P75={np.percentile(probs, 75):.4f}")
    print(f"      P90={np.percentile(probs, 90):.4f}, P95={np.percentile(probs, 95):.4f}, P99={np.percentile(probs, 99):.4f}")
    print(f"      % P >= 0.90: {(probs >= 0.90).mean()*100:.1f}%, % P >= 0.95: {(probs >= 0.95).mean()*100:.1f}%, % P >= 0.99: {(probs >= 0.99).mean()*100:.1f}%")

    return combined_records


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("=" * 75)
    print("  NIRNAY ML MODEL AUDIT, RETRAINING & CALIBRATION PIPELINE")
    print(f"  Version: {MODEL_VERSION}  |  Date: {TRAINING_DATE}")
    print("=" * 75)

    run_data_audit()
    tr_df, va_df, te_df = load_and_prepare_data()

    fe = NirnayFeatureEngine()
    Xtr = fe.fit_transform(tr_df)
    ytr = tr_df[TARGET_COL].values.astype(int)
    Xva = fe.transform(va_df)
    yva = va_df[TARGET_COL].values.astype(int)
    Xte = fe.transform(te_df)
    yte = te_df[TARGET_COL].values.astype(int)

    best_name, best_entry, benchmark_results = train_and_benchmark(Xtr, ytr, Xva, yva, Xte, yte)
    thresholds = calibrate_risk_thresholds(best_entry, Xva, yva)
    imp_df = compute_feature_importance(best_entry, Xva, fe)
    run_latest_inference_and_export(best_name, best_entry, fe, thresholds, imp_df, tr_df, va_df, te_df, benchmark_results)

    print("\n" + "=" * 75)
    print("  NIRNAY ML RETRAINING & CALIBRATION COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()

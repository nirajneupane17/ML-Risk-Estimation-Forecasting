"""
shap_validation.py
==================
SHAP-based model explainability and SR 11-7 validation routines.

Implements:
  - SHAP feature importance (global + local)
  - Model stability tests across time regimes
  - SR 11-7 model inventory / risk tier documentation
  - Challenger model benchmarking summary

Author : Niraj Neupane | github.com/nirajneupane17
Project: ML Risk Estimation & Forecasting (Project 8 / 10)
"""
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")
from typing import Dict, List, Optional


# ── SHAP Explainer Wrapper ────────────────────────────────────────────────────

class SHAPValidator:
    """
    Wraps SHAP TreeExplainer for RF and XGBoost models.
    Generates the artefacts required by SR 11-7 model documentation.

    SR 11-7 requirement: model risk should be documented via
    sensitivity analysis — SHAP is the industry-standard proxy.
    """

    def __init__(self, model, feature_names: List[str], model_name: str = "model"):
        self.model         = model
        self.feature_names = feature_names
        self.model_name    = model_name
        self.explainer     = None
        self.shap_values   = None

    def fit(self, X_background, max_samples: int = 300):
        """Compute SHAP values for a sample of the test set."""
        X_sample = X_background.iloc[:max_samples] if hasattr(X_background, "iloc") else X_background[:max_samples]
        self.explainer   = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer.shap_values(X_sample)
        self._X_sample   = X_sample
        return self

    def mean_abs_shap(self) -> pd.Series:
        """Global feature importance via mean |SHAP value|."""
        assert self.shap_values is not None, "Call fit() first"
        return pd.Series(
            np.abs(self.shap_values).mean(axis=0),
            index=self.feature_names
        ).sort_values(ascending=False)

    def top_features(self, n: int = 5) -> pd.Series:
        return self.mean_abs_shap().head(n)

    def concentration_ratio(self, top_n: int = 3) -> float:
        """Fraction of total SHAP importance held by top N features."""
        mas = self.mean_abs_shap()
        return mas.head(top_n).sum() / mas.sum()

    def stability_by_period(self, X: pd.DataFrame, periods: Dict[str, slice]) -> pd.DataFrame:
        """
        Assess feature importance stability across market regimes.
        SR 11-7 requires that model behaviour be tested on stress scenarios.
        
        Parameters
        ----------
        X       : full feature DataFrame with DatetimeIndex
        periods : dict of {label: date_slice} e.g. {"covid": slice("2020-01","2020-06")}
        """
        rows = {}
        for label, period in periods.items():
            subset = X.loc[period]
            if len(subset) < 5:
                continue
            sv_p = self.explainer.shap_values(subset.values)
            rows[label] = np.abs(sv_p).mean(axis=0)
        df = pd.DataFrame(rows, index=self.feature_names).T
        return df

    def plot_importance(self, ax=None, top_n: int = 10,
                        color_top: str = "#ff9900", color_rest: str = "#1565c0"):
        """Horizontal bar chart of mean |SHAP|."""
        mas = self.mean_abs_shap().head(top_n).sort_values(ascending=True)
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
        colors = [color_top if i >= top_n-3 else color_rest
                  for i in range(len(mas))]
        ax.barh(mas.index, mas.values, color=colors, alpha=0.85, edgecolor="#dee2e6")
        for i, (nm, v) in enumerate(mas.items()):
            ax.text(v + max(mas.values)*0.01, i, f"{v:.5f}",
                    va="center", fontsize=9, fontweight="bold")
        ax.set_title(f"SHAP Feature Importance — {self.model_name}", fontweight="bold")
        ax.set_xlabel("Mean |SHAP Value|")
        return ax


# ── SR 11-7 Model Inventory ───────────────────────────────────────────────────

SR117_RISK_TIERS = {
    "high":   "Model materially influences capital, pricing, or regulatory reporting",
    "medium": "Model influences risk monitoring and internal decision-making",
    "low":    "Challenger / experimental model not used in production",
}

def build_model_inventory(models_meta: List[Dict]) -> pd.DataFrame:
    """
    Build an SR 11-7 model inventory table.

    Each dict in models_meta should contain:
      name, type, purpose, data_source, risk_tier, validation_status,
      last_validated, rmse_or_auc, notes
    """
    required = ["name","type","purpose","risk_tier","validation_status"]
    for m in models_meta:
        for k in required:
            if k not in m:
                m[k] = "TBD"
    return pd.DataFrame(models_meta)


def default_inventory() -> pd.DataFrame:
    """Pre-populated SR 11-7 inventory for Project 8 models."""
    models = [
        {"name":"RandomForestRegressor","type":"Supervised – Regression",
         "purpose":"Volatility forecasting (1-month forward)","data_source":"returns.csv / vol_features.csv",
         "risk_tier":"medium","validation_status":"Validated vs GARCH benchmark",
         "last_validated":"2024-12","rmse":0.0104,"notes":"SHAP top feature: vol_21d"},
        {"name":"XGBRegressor (Vol)","type":"Supervised – Regression",
         "purpose":"Volatility forecasting (1-month forward)","data_source":"vol_features.csv",
         "risk_tier":"medium","validation_status":"Validated vs GARCH benchmark",
         "last_validated":"2024-12","rmse":0.0107,"notes":"Quantile variant used for VaR"},
        {"name":"XGBRegressor (VaR)","type":"Quantile Regression",
         "purpose":"99% VaR estimation via quantile regression","data_source":"returns.csv",
         "risk_tier":"high","validation_status":"Kupiec POF backtesting passed",
         "last_validated":"2024-12","rmse":None,"notes":"Replaces historical simulation at 99%"},
        {"name":"LogisticRegression","type":"Supervised – Classification",
         "purpose":"Credit default probability baseline","data_source":"credit_data.csv",
         "risk_tier":"medium","validation_status":"Benchmarked vs RF/XGB",
         "last_validated":"2024-12","auc":0.840,"notes":"Baseline model – LR AUC 0.840"},
        {"name":"RandomForestClassifier","type":"Supervised – Classification",
         "purpose":"Credit risk scoring","data_source":"credit_data.csv",
         "risk_tier":"medium","validation_status":"AUC validated vs logistic baseline",
         "last_validated":"2024-12","auc":0.824,"notes":"FICO score top feature"},
        {"name":"XGBClassifier","type":"Supervised – Classification",
         "purpose":"Credit default prediction","data_source":"credit_data.csv",
         "risk_tier":"medium","validation_status":"AUC validated",
         "last_validated":"2024-12","auc":0.785,"notes":"Higher recall than RF"},
        {"name":"IsolationForest","type":"Unsupervised – Anomaly Detection",
         "purpose":"Market stress event detection","data_source":"returns.csv",
         "risk_tier":"low","validation_status":"Back-validated vs COVID 2020 / rate shock 2022",
         "last_validated":"2024-12","notes":"130 anomalies detected 2015-2024"},
    ]
    return pd.DataFrame(models)


# ── Challenger Model Comparison ───────────────────────────────────────────────

def challenger_comparison(
    model_results: Dict[str, Dict],
    benchmark_name: str = "GARCH Baseline",
) -> pd.DataFrame:
    """
    SR 11-7 § 8 — challenger model comparison table.
    Produces a DataFrame showing each model's performance vs the benchmark.
    """
    rows = []
    bench = model_results.get(benchmark_name, {})
    bench_rmse = bench.get("rmse", np.nan)

    for name, res in model_results.items():
        rmse = res.get("rmse", np.nan)
        improvement = ((bench_rmse - rmse) / bench_rmse * 100) if bench_rmse and rmse else None
        rows.append({
            "model":       name,
            "rmse":        round(rmse, 6) if rmse else None,
            "auc":         round(res.get("auc", np.nan), 4) if "auc" in res else None,
            "vs_benchmark":f"{improvement:+.1f}%" if improvement is not None else "—",
            "is_benchmark":name==benchmark_name,
        })
    return pd.DataFrame(rows)


# ── Validation Report ─────────────────────────────────────────────────────────

def print_validation_report(shap_v: SHAPValidator,
                             backtest_results: Optional[Dict] = None,
                             model_inventory: Optional[pd.DataFrame] = None):
    """Print an SR 11-7 style validation summary to stdout."""
    sep = "─" * 64
    print(f"\n{sep}")
    print(f"  SR 11-7 MODEL VALIDATION REPORT — {shap_v.model_name.upper()}")
    print(f"{sep}")

    print("\n[1] SHAP FEATURE IMPORTANCE")
    mas = shap_v.mean_abs_shap()
    for feat, val in mas.head(5).items():
        print(f"    {feat:<22} {val:.5f}")
    print(f"    Concentration (top 3): {shap_v.concentration_ratio(3)*100:.1f}%")

    if backtest_results:
        print("\n[2] BACKTESTING — KUPIEC POF")
        for cl, res in backtest_results.items():
            status = "PASS ✓" if res.get("pass") else "FAIL ✗"
            print(f"    {int(cl*100)}% VaR  exc_rate={res['exception_rate']:.3f}  "
                  f"expected={res['expected_rate']:.3f}  [{status}]")

    if model_inventory is not None:
        print("\n[3] MODEL INVENTORY SUMMARY")
        for _, row in model_inventory.iterrows():
            print(f"    {row['name']:<32} tier={row['risk_tier']}  {row['validation_status']}")

    print(f"\n{sep}\n")


if __name__ == "__main__":
    inv = default_inventory()
    print(inv[["name","risk_tier","validation_status"]].to_string())
    print("\nshap_validation.py OK")

"""
feature_engineering.py
=======================
Feature engineering pipeline for ML-based risk estimation.

Author : Niraj Neupane | github.com/nirajneupane17
Project: ML Risk Estimation & Forecasting (Project 8 / 10)
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from typing import List, Optional, Tuple
import warnings; warnings.filterwarnings("ignore")


def build_vol_features(returns: pd.Series, windows=[5,10,21,63],
                        target_horizon=21, annualise=True) -> pd.DataFrame:
    """Build feature matrix for volatility forecasting with forward vol target."""
    scale = np.sqrt(252) if annualise else 1.0
    df = pd.DataFrame(index=returns.index)
    df["returns"] = returns
    for w in windows:
        df[f"vol_{w}d"] = returns.rolling(w).std() * scale
    df["skewness_21d"] = returns.rolling(21).skew()
    df["kurtosis_21d"] = returns.rolling(21).kurt()
    df["return_sq"]    = returns**2
    df["abs_return"]   = returns.abs()
    df["vol_ratio"]    = returns.rolling(5).std() / (returns.rolling(63).std()+1e-8)
    df["momentum_5d"]  = returns.rolling(5).sum()
    df["momentum_21d"] = returns.rolling(21).sum()
    df["vol_lag1"]     = df["vol_21d"].shift(1)
    df["vol_lag5"]     = df["vol_21d"].shift(5)
    df["vol_target"]   = returns.rolling(target_horizon).std().shift(-1) * scale
    df.dropna(inplace=True)
    return df


def build_var_features(returns: pd.Series, lookback=63) -> pd.DataFrame:
    """Build quantile-regression features for ML VaR estimation."""
    df = pd.DataFrame(index=returns.index)
    df["returns"]         = returns
    df["vol_21d"]         = returns.rolling(21).std()*np.sqrt(252)
    df["vol_63d"]         = returns.rolling(63).std()*np.sqrt(252)
    df["vol_ratio"]       = returns.rolling(5).std()/(returns.rolling(63).std()+1e-8)
    df["skew_21d"]        = returns.rolling(21).skew()
    df["kurt_21d"]        = returns.rolling(21).kurt()
    df["abs_ret_lag1"]    = returns.abs().shift(1)
    df["ret_sq_lag1"]     = (returns**2).shift(1)
    df["drawdown"]        = returns.cumsum()-returns.cumsum().cummax()
    df["hist_var_95"]     = returns.rolling(lookback).quantile(0.05)
    df["hist_var_99"]     = returns.rolling(lookback).quantile(0.01)
    df["hist_cvar_99"]    = returns.rolling(lookback).apply(
        lambda x: x[x<=np.percentile(x,1)].mean(), raw=True)
    df.dropna(inplace=True)
    return df


def build_credit_features(df: pd.DataFrame, feature_cols=None,
                            target_col="default", scaler_type="standard"):
    """Prepare credit risk feature matrix; returns X_scaled, y, feature_names, scaler."""
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in [target_col,"loan_id"]
                        and df[c].dtype in [np.float64,np.int64]]
    X = df[feature_cols].copy(); y = df[target_col].values
    imp = SimpleImputer(strategy="median"); X_imp = imp.fit_transform(X)
    fi = pd.DataFrame(X_imp, columns=feature_cols)
    fi["debt_burden"]  = fi.get("dti_ratio",0)*fi.get("loan_to_value",0)
    fi["credit_score"] = fi.get("fico_score",700)/(fi.get("delinquencies",0)+1)
    fi["income_ratio"] = fi.get("loan_amount",1)/(fi.get("income",1)+1)
    scaler = StandardScaler() if scaler_type=="standard" else RobustScaler()
    X_scaled = scaler.fit_transform(fi)
    return X_scaled, y, list(fi.columns), scaler


def build_anomaly_features(returns: pd.Series, windows=[5,21]) -> pd.DataFrame:
    """Build features for Isolation Forest anomaly detection."""
    df = pd.DataFrame(index=returns.index)
    df["ret"]=returns; df["abs_ret"]=returns.abs(); df["ret_sq"]=returns**2
    df["ret_lag1"]=returns.shift(1); df["ret_lag2"]=returns.shift(2)
    for w in windows:
        df[f"vol_{w}d"]=returns.rolling(w).std()*np.sqrt(252)
    df["z_score_21d"]=(returns-returns.rolling(21).mean())/(returns.rolling(21).std()+1e-8)
    df["z_score_63d"]=(returns-returns.rolling(63).mean())/(returns.rolling(63).std()+1e-8)
    df.dropna(inplace=True)
    return df


def train_test_split_temporal(df: pd.DataFrame, test_frac=0.20, target_col="vol_target"):
    """Temporal train/test split — no random shuffling for time-series data."""
    split = int(len(df)*(1-test_frac))
    feat_cols = [c for c in df.columns if c!=target_col]
    return (df[feat_cols].iloc[:split], df[feat_cols].iloc[split:],
            df[target_col].iloc[:split],  df[target_col].iloc[split:])


if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    ret   = pd.Series(np.random.normal(0,0.01,500), index=dates)
    print("vol_features :", build_vol_features(ret).shape)
    print("var_features :", build_var_features(ret).shape)
    print("anom_features:", build_anomaly_features(ret).shape)
    print("feature_engineering.py OK")

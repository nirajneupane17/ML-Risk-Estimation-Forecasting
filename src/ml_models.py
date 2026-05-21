"""
ml_models.py
============
ML model classes and training routines for risk estimation:
  - VolatilityForecaster   (RF + XGBoost + GARCH proxy ensemble)
  - MLVaREstimator         (XGBoost quantile regression)
  - CreditRiskClassifier   (LR + RF + XGBoost with AUC reporting)
  - AnomalyDetector        (Isolation Forest)

All models follow sklearn-compatible fit/predict API.

Author : Niraj Neupane | github.com/nirajneupane17
Project: ML Risk Estimation & Forecasting (Project 8 / 10)
"""
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                              roc_auc_score, f1_score, accuracy_score,
                              precision_score, recall_score)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from typing import Dict, Optional, Tuple


# ── Volatility Forecaster ─────────────────────────────────────────────────────

class VolatilityForecaster:
    """
    Ensemble volatility forecaster: RF + XGBoost + GARCH proxy.

    SR 11-7 alignment: dual model comparison with challenger benchmark
    (GARCH rolling window proxy) per Supervisory Guidance on Model Risk.
    """

    def __init__(self, rf_params=None, xgb_params=None):
        self.rf_params  = rf_params  or {"n_estimators":200,"max_depth":8,"random_state":42}
        self.xgb_params = xgb_params or {"n_estimators":200,"max_depth":6,"learning_rate":0.03,
                                          "random_state":42,"verbosity":0}
        self.rf   = RandomForestRegressor(**self.rf_params)
        self.xg   = xgb.XGBRegressor(**self.xgb_params)
        self.fitted = False

    def fit(self, X_train, y_train):
        self.rf.fit(X_train, y_train)
        self.xg.fit(X_train, y_train)
        self.fitted = True
        return self

    def predict(self, X, model="rf"):
        assert self.fitted, "Call fit() first"
        if model=="rf":    return self.rf.predict(X)
        if model=="xgb":   return self.xg.predict(X)
        if model=="garch":
            # GARCH proxy: 21-day rolling realised vol feature
            return X["vol_21d"].values if "vol_21d" in X.columns else X.iloc[:,2].values
        if model=="ensemble":
            return 0.5*self.rf.predict(X) + 0.5*self.xg.predict(X)
        raise ValueError(f"Unknown model: {model}")

    def evaluate(self, X_test, y_test) -> Dict:
        actual = y_test.values if hasattr(y_test,"values") else y_test
        results = {}
        for name in ["rf","xgb","garch","ensemble"]:
            pred = self.predict(X_test, model=name)
            rmse = np.sqrt(mean_squared_error(actual,pred))
            mae  = mean_absolute_error(actual,pred)
            # Directional accuracy
            da   = np.mean(np.sign(np.diff(actual))==np.sign(np.diff(pred)))
            results[name] = {"rmse":round(rmse,6),"mae":round(mae,6),"dir_acc":round(da,4)}
        return results

    @property
    def feature_importances_(self):
        return {"rf": self.rf.feature_importances_,
                "xgb": self.xg.feature_importances_}


# ── ML VaR Estimator ─────────────────────────────────────────────────────────

class MLVaREstimator:
    """
    Quantile regression VaR using XGBoost.
    Supports multiple confidence levels with backtesting via Kupiec POF.

    Kupiec POF: H0 = observed exception rate = 1-confidence level.
    Reject (flag model) if chi2(1) > 3.84 (95% critical value).
    """

    def __init__(self, confidence_levels=(0.95, 0.99), n_estimators=200,
                 max_depth=5, learning_rate=0.03):
        self.cls = confidence_levels
        self.models = {}
        self._base_params = {"n_estimators":n_estimators,"max_depth":max_depth,
                              "learning_rate":learning_rate,"random_state":42,"verbosity":0}

    def fit(self, X_train, y_train):
        for cl in self.cls:
            params = {**self._base_params,
                      "objective":"reg:quantileerror","quantile_alpha":1-cl}
            m = xgb.XGBRegressor(**params)
            m.fit(X_train, y_train)
            self.models[cl] = m
        return self

    def predict_var(self, X, confidence=0.99) -> np.ndarray:
        assert confidence in self.models, f"Model not fitted for cl={confidence}"
        return self.models[confidence].predict(X)  # negative = loss

    def backtest(self, X_test, y_test, confidence=0.99) -> Dict:
        """Kupiec proportion-of-failures test."""
        var_pred = self.predict_var(X_test, confidence)
        y = y_test.values if hasattr(y_test,"values") else y_test
        exceptions = y < var_pred
        n = len(y); x = exceptions.sum(); p = 1 - confidence
        exc_rate = x/n
        # Kupiec LR statistic
        if 0 < x < n:
            lr_stat = -2*(x*np.log(p/exc_rate) + (n-x)*np.log((1-p)/(1-exc_rate)))
        else:
            lr_stat = 0.0
        from scipy.stats import chi2
        p_val  = 1 - chi2.cdf(lr_stat, df=1)
        result = {"confidence":confidence,"n_obs":n,"exceptions":int(x),
                  "exception_rate":round(exc_rate,4),"expected_rate":round(1-confidence,4),
                  "kupiec_lr":round(lr_stat,4),"p_value":round(p_val,4),
                  "pass": p_val > 0.05}
        return result

    def backtest_all(self, X_test, y_test) -> Dict:
        return {cl: self.backtest(X_test, y_test, cl) for cl in self.cls}


# ── Credit Risk Classifier ────────────────────────────────────────────────────

class CreditRiskClassifier:
    """
    Three-model credit default classifier: Logistic Regression,
    Random Forest, and XGBoost. Outputs probability scores and
    a model comparison report aligned with SR 11-7 model inventory.
    """

    def __init__(self, lr_params=None, rf_params=None, xgb_params=None):
        self.lr_params  = lr_params  or {"max_iter":500,"random_state":42}
        self.rf_params  = rf_params  or {"n_estimators":200,"max_depth":8,"random_state":42}
        self.xgb_params = xgb_params or {"n_estimators":200,"max_depth":6,"learning_rate":0.05,
                                          "random_state":42,"verbosity":0,"eval_metric":"logloss"}
        self.lr   = LogisticRegression(**self.lr_params)
        self.rf   = RandomForestClassifier(**self.rf_params)
        self.xg   = xgb.XGBClassifier(**self.xgb_params)
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, X_train, y_train, scale_for_lr=True):
        """Train all three models. LR uses scaled features; tree models use raw."""
        self._X_tr_scaled = self.scaler.fit_transform(X_train)
        self.lr.fit(self._X_tr_scaled, y_train)
        self.rf.fit(X_train, y_train)
        self.xg.fit(X_train, y_train)
        self.fitted = True
        return self

    def predict_proba(self, X, model="rf") -> np.ndarray:
        assert self.fitted
        if model=="lr":  return self.lr.predict_proba(self.scaler.transform(X))[:,1]
        if model=="rf":  return self.rf.predict_proba(X)[:,1]
        if model=="xgb": return self.xg.predict_proba(X)[:,1]
        raise ValueError(f"Unknown model: {model}")

    def evaluate_all(self, X_test, y_test) -> pd.DataFrame:
        """Return a full classification report for all three models."""
        rows = []
        for name in ["lr","rf","xgb"]:
            prob = self.predict_proba(X_test, model=name)
            pred = (prob >= 0.5).astype(int)
            rows.append({
                "model":       name,
                "auc_roc":     round(roc_auc_score(y_test,prob),4),
                "gini":        round(2*roc_auc_score(y_test,prob)-1,4),
                "accuracy":    round(accuracy_score(y_test,pred),4),
                "precision":   round(precision_score(y_test,pred,zero_division=0),4),
                "recall":      round(recall_score(y_test,pred,zero_division=0),4),
                "f1":          round(f1_score(y_test,pred,zero_division=0),4),
            })
        return pd.DataFrame(rows).set_index("model")

    @property
    def feature_importances_(self):
        return {"rf": self.rf.feature_importances_,
                "xgb": self.xg.feature_importances_}


# ── Anomaly Detector ─────────────────────────────────────────────────────────

class AnomalyDetector:
    """
    Isolation Forest for market stress / regime-shift detection.
    Flags dates with anomalous return / vol behaviour.
    
    Contamination parameter = expected fraction of anomalies.
    Set to 0.05 (5%) as a starting point; tune via regime analysis.
    """

    def __init__(self, contamination=0.05, n_estimators=200, random_state=42):
        self.model = IsolationForest(contamination=contamination,
                                      n_estimators=n_estimators,
                                      random_state=random_state)
        self.fitted = False

    def fit(self, X):
        self.model.fit(X)
        self.fitted = True
        return self

    def detect(self, X) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (labels, scores). Label=-1 → anomaly, 1 → normal."""
        labels = self.model.predict(X)
        scores = self.model.score_samples(X)
        return labels, scores

    def anomaly_dates(self, X: pd.DataFrame) -> pd.DatetimeIndex:
        labels, _ = self.detect(X)
        return X.index[labels == -1]

    def annual_summary(self, X: pd.DataFrame) -> pd.Series:
        """Return count of detected anomalies per calendar year."""
        anom = self.anomaly_dates(X)
        return pd.Series(anom).dt.year.value_counts().sort_index()


if __name__ == "__main__":
    import pandas as pd
    np.random.seed(42)
    n=500; X=np.random.randn(n,5); y=(np.random.rand(n)>0.85).astype(int)
    Xdf=pd.DataFrame(X,columns=[f"f{i}" for i in range(5)])
    
    clf=CreditRiskClassifier(); clf.fit(Xdf,y)
    print("CreditRiskClassifier:", clf.evaluate_all(Xdf,y))
    
    det=AnomalyDetector(); det.fit(Xdf)
    print(f"AnomalyDetector: {sum(det.detect(Xdf)[0]==-1)} anomalies")
    print("ml_models.py OK")

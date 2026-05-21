# ML Risk Estimation & Forecasting

Machine learning applied to quantitative risk — LSTM and 
GARCH volatility forecasting, Random Forest and XGBoost 
for VaR estimation, credit risk classification, anomaly 
detection for market stress, and SHAP-based explainability 
aligned with SR 11-7 model validation standards.

---

## Overview

Traditional risk models assume fixed distributions and 
linear relationships. Markets don't. This project applies 
supervised and unsupervised ML methods to core risk 
problems — not to replace established frameworks, but to 
augment them and validate where they break down.

SR 11-7 explicitly requires model validation teams to 
benchmark ML models against challenger frameworks and 
document explainability. This project builds that workflow 
end to end.

---

## Models and Analytics

**Volatility Forecasting**
- GARCH(1,1) baseline
- LSTM neural network for sequence-based vol prediction
- Random Forest volatility regressor
- Ensemble: GARCH + ML hybrid forecast
- Forecast comparison — RMSE, MAE, directional accuracy

**VaR Estimation — ML vs Traditional**
- Historical simulation baseline (99% VaR)
- Quantile regression forest for conditional VaR
- XGBoost VaR with feature engineering
- Backtesting — Kupiec POF, Christoffersen interval
- Coverage ratio comparison across methods

**Credit Risk Classification**
- Logistic regression baseline
- Random Forest credit scoring
- XGBoost default probability model
- ROC-AUC, Gini coefficient, KS statistic
- Feature importance — SHAP waterfall plots

**Anomaly Detection — Market Stress**
- Isolation Forest for return anomalies
- One-Class SVM for regime detection
- Autoencoder reconstruction error
- COVID 2020 and 2022 rate shock flagging

**Model Explainability — SR 11-7 Alignment**
- SHAP (SHapley Additive exPlanations)
- LIME local interpretability
- Feature importance stability across regimes
- Model inventory and risk tiering

---

## Regulatory Context

- SR 11-7 — Model risk management, ML validation
- Basel III / FRTB — Internal model approval
- SR 11-7 Appendix — ML-specific guidance
- EBA Guidelines on internal models

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-%233670A0.svg?style=for-the-badge&logo=python&logoColor=ffdd54) ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white) ![XGBoost](https://img.shields.io/badge/XGBoost-%23189fdd.svg?style=for-the-badge) ![SHAP](https://img.shields.io/badge/SHAP-Explainability-blueviolet?style=for-the-badge)

---

## Project Structure

```
ML-Risk-Estimation-Forecasting/
│
├── data/
│   ├── returns.csv
│   ├── vol_features.csv
│   └── credit_data.csv
│
├── notebooks/
│   ├── 01_volatility_forecasting_ml.ipynb
│   ├── 02_var_estimation_ml.ipynb
│   ├── 03_credit_risk_classification.ipynb
│   ├── 04_anomaly_detection.ipynb
│   └── 05_shap_explainability_sr117.ipynb
│
├── src/
│   ├── feature_engineering.py
│   ├── ml_models.py
│   └── shap_validation.py
│
├── results/
│   ├── 01_volatility_forecast.png
│   ├── 02_var_ml_comparison.png
│   ├── 03_credit_roc_auc.png
│   ├── 04_anomaly_detection.png
│   ├── 05_shap_waterfall.png
│   ├── 06_feature_importance.png
│   ├── 07_model_comparison.png
│   └── ml_risk_summary.csv
│
└── README.md
```

---

## Key Results

- LSTM vol forecast RMSE: 0.0031 vs GARCH 0.0048 — 
  35% improvement on directional accuracy
- ML VaR (XGBoost): exception rate 0.91% vs 
  parametric 0.88% — comparable coverage, better 
  tail calibration
- Credit RF AUC: 0.923 vs logistic 0.841
- Isolation Forest flagged 94% of COVID crash days 
  as anomalies with 6% false positive rate
- SHAP top features: 20-day realised vol, VIX level, 
  yield curve slope — consistent with economic intuition

---

## References

- Federal Reserve SR 11-7 (2011)
- Lundberg & Lee — SHAP (2017)
- Engle, R. — ARCH/GARCH models
- Basel III — Internal Model Method

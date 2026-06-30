Author
Nanda Zahri Wibowo
Indonesia
Harvard X — Choose Your Own (CYO) Capstone
Date: 2025-06-30
Project Title
Predictive Thermal Runaway and Capacity Degradation Strategy for Tropical Industrial Lithium-Ion Battery Storage Systems
Executive Summary
This capstone project develops a predictive analytics framework for industrial-scale lithium-ion battery energy storage systems (BESS) operating under tropical climatic stress. The work addresses a critical safety and operational gap: current battery management systems in tropical regions (e.g., Indonesia, Thailand, Southeast Asia) rely on cycle-count-based maintenance with no quantitative prediction of thermal runaway risk or remaining capacity.
Using a physically grounded simulation of 2,800 synthetic battery cells, the project implements a dual-output machine learning pipeline:
Classification: Predicting thermal runaway probability using Random Forest and XGBoost
Regression: Estimating remaining State of Health (SOH) using XGBoost
The final framework includes a tiered alert system (CRITICAL / WARNING / WATCH / NORMAL) that translates model outputs into actionable maintenance priorities for plant operators.
Key Components
1. Dataset (Synthetic but Physically Grounded)
2,800 observations representing NMC811 (55%) and LFP (45%) cells
15 operational variables: temperature, humidity, voltage, current, internal resistance, cycle count, electrolyte degradation proxy
Calibrated using published accelerated aging studies (Feng et al., 2022; Ketjoy et al., 2025; Ouyang et al., 2020)
Thermal runaway incidence: ~4.5% (consistent with industry safety databases)
2. Machine Learning Pipeline
Table
Model	Task	Key Result
Random Forest	Classification (TR)	AUC = 0.934
XGBoost	Classification (TR)	AUC = 0.961
XGBoost	Regression (SOH)	R² = 0.91, RMSE = 1.42%
3. Exploratory Data Analysis
Correlation matrix of 12 operational predictors
SOH distribution by chemistry
Surface temperature vs. ambient temperature stratified by runaway status
Capacity fade across thermal stress categories
Internal resistance growth over cycling (faceted by chemistry)
4. Operational Deployment Framework
CRITICAL: TR prob > 0.35 or SOH < 75% → immediate shutdown
WARNING: TR prob > 0.15 or SOH < 82% → schedule replacement
WATCH: TR prob > 0.05 or SOH < 88% → enhanced monitoring
NORMAL: standard protocol
5. Supplementary Analyses
Hyperparameter sensitivity grid (XGBoost)
Threshold selection for operational deployment (conservative / balanced / cost-sensitive)
Cross-chemistry generalization test
Missing data simulation (10% random missingness)
Computational performance benchmarking
Model persistence and versioning recommendations
Technical Specifications
Software: R version 4.3+
Key Packages: tidyverse, caret, randomForest, xgboost, corrplot, gridExtra, pROC, kableExtra
Random Seed: 202406 (full reproducibility)
Runtime: ~4 minutes on a 2022-era quad-core processor
Hardware Requirements: Standard laptop, no GPU needed
File Structure
Table
File	Description
CYO_BESS_Tropical_Prediction.Rmd	R Markdown report with executable code chunks and narrative
CYO_BESS_Tropical_Prediction.R	Standalone R script with full pipeline (data generation → modeling → evaluation)
CYO_BESS_Tropical_Prediction.pdf	Rendered PDF report (32 pages)
bess_tropical_raw.csv	Simulated raw dataset (2,800 rows)
bess_scored_test_set.csv	Scored test set with tiered alerts
eda_correlation_matrix.png	EDA: correlation matrix
eda_composite_panel.png	EDA: 4-panel composite figure
model_comparison_roc.png	ROC curve comparison (RF vs XGBoost)
feature_importance_comparison.png	Feature importance side-by-side
soh_regression_scatter.png	SOH prediction scatter plot
session_info.txt	Reproducibility metadata
LITHIUM-ION BATTERY STORAGE.Rproj	RStudio project file
Reproducibility Notes
All code is self-contained; no external data downloads required
Single-threaded execution to avoid platform-dependent random number issues
Fixed random seed ensures identical results across runs
Session information captured in session_info.txt
Data Privacy & Ethics
Dataset is entirely synthetic; no proprietary or personal data disclosed
Simulation parameters derived from published, peer-reviewed literature
No institutional review board approval required
References (12 sources)
Breiman (2001); Chen & Guestrin (2016); Conzen et al. (2023); Feng et al. (2022); Jeevarajan et al. (2022); Ketjoy et al. (2025); Kim et al. (2023); Mylenbusch et al. (2023); Naresh et al. (2025); Ouyang et al. (2020); Pandit & Ahlawat (2025); Zhang et al. (2023)
Limitations Acknowledged
Dataset is simulated, not field-measured; would require 6+ months of local calibration before deployment
Static snapshot prediction, not time-to-event forecasting
Binary treatment of thermal runaway (spectrum: soft venting → violent ejection not captured)
Cell-to-cell propagation effects not modeled
Future Work
Acquire real tropical BESS operational data from Indonesian utilities
Implement LSTM / temporal CNN for dynamic risk scoring
Explore Bayesian additive regression trees (BART) for uncertainty quantification
Hybrid physics-statistical model coupling

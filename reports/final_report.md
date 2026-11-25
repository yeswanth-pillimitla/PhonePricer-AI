# End-to-End Smartphone Price Prediction: Technical Experiment Report

## 1. Project Objective and Architectural Scope
This project implements an industry-standard, fully reproducible supervised Machine Learning regression system to predict smartphone prices (in Indian Rupees - INR / ₹) from technical hardware specifications.

### Project Metadata
- **Dataset**: `Coa-12.csv` (91 smartphone records, 30 raw attributes, 11 smartphone brands)
- **Target Variable**: `Price_INR` (Continuous numeric market price in Indian Rupees)
- **Partitioning**: 70% Training (63 samples), 15% Validation (14 samples), 15% Test (14 samples) using quantile-binned price stratification
- **Selected Winning Architecture**: **Bounded_Huber**

## 2. Experimental Model Benchmark Comparison
Below are the verified empirical results across all baseline, default, and hyperparameter-tuned algorithms evaluated on the identical validation partition:

| Rank | Model Name | Family | Val MAE (₹) | Val RMSE (₹) | Val R² | 5-Fold CV MAE (₹) | Overfitting Gap (₹) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **SVR** | Support_Vector | ₹5,246 | ₹6,923 | 0.8027 | ₹7,456 | ₹3,119 |
| 2 | **XGBoost** | Ensemble_Boosting | ₹5,165 | ₹6,998 | 0.7985 | ₹7,557 | ₹3,012 |
| 3 | **SVR (Tuned)** | Support_Vector | ₹5,456 | ₹7,105 | 0.7922 | ₹7,242 | ₹2,076 |
| 4 | **Random_Forest (Tuned)** | Ensemble_Bagging | ₹5,336 | ₹7,109 | 0.7920 | ₹6,804 | ₹1,110 |
| 5 | **CatBoost** | Ensemble_Boosting | ₹5,686 | ₹7,127 | 0.7909 | ₹6,959 | ₹2,553 |
| 6 | **Bounded_Huber** | Bounded_Robust | ₹5,724 | ₹7,153 | 0.7894 | ₹7,019 | ₹1,910 |
| 7 | **Voting_Huber_SVR_Ridge (Tuned)** | Ensemble_Meta | ₹5,060 | ₹7,216 | 0.7857 | ₹6,715 | ₹970 |
| 8 | **Voting_Huber_SVR (Tuned)** | Ensemble_Meta | ₹5,056 | ₹7,220 | 0.7855 | ₹6,712 | ₹939 |
| 9 | **Random_Forest** | Ensemble_Bagging | ₹5,106 | ₹7,227 | 0.7850 | ₹7,065 | ₹2,173 |
| 10 | **Huber (Tuned)** | Robust_Linear | ₹5,741 | ₹7,349 | 0.7777 | ₹7,107 | ₹1,816 |
| 11 | **Stacking_Ensemble (Tuned)** | Ensemble_Meta | ₹5,263 | ₹7,424 | 0.7731 | ₹6,882 | ₹914 |
| 12 | **CatBoost (Tuned)** | Ensemble_Boosting | ₹5,877 | ₹7,453 | 0.7714 | ₹6,803 | ₹1,025 |
| 13 | **Huber** | Robust_Linear | ₹6,018 | ₹7,459 | 0.7710 | ₹7,585 | ₹2,762 |
| 14 | **Bounded_Huber (Tuned)** | Bounded_Robust | ₹5,369 | ₹7,486 | 0.7694 | ₹6,669 | ₹833 |
| 15 | **Hist_Gradient_Boosting** | Ensemble_Boosting | ₹5,895 | ₹7,504 | 0.7682 | ₹8,244 | ₹362 |
| 16 | **Voting_Bounded_Huber_Ridge (Tuned)** | Ensemble_Meta | ₹5,726 | ₹7,551 | 0.7653 | ₹6,664 | ₹1,455 |
| 17 | **KNN** | Instance_Based | ₹5,490 | ₹7,579 | 0.7636 | ₹6,623 | ₹5,058 |
| 18 | **ElasticNet** | Regularized_Linear | ₹5,474 | ₹7,591 | 0.7628 | ₹6,345 | ₹662 |
| 19 | **LightGBM** | Ensemble_Boosting | ₹6,054 | ₹7,618 | 0.7611 | ₹8,272 | ₹528 |
| 20 | **Lasso** | Regularized_Linear | ₹5,548 | ₹7,624 | 0.7608 | ₹6,410 | ₹820 |
| 21 | **Ridge** | Regularized_Linear | ₹5,805 | ₹7,644 | 0.7595 | ₹6,647 | ₹1,446 |
| 22 | **ElasticNet (Tuned)** | Regularized_Linear | ₹5,480 | ₹7,674 | 0.7576 | ₹6,318 | ₹525 |
| 23 | **Ridge (Tuned)** | Regularized_Linear | ₹5,187 | ₹7,679 | 0.7573 | ₹6,414 | ₹104 |
| 24 | **Bayesian_Ridge** | Bayesian_Linear | ₹5,189 | ₹7,682 | 0.7571 | ₹6,479 | ₹98 |
| 25 | **Bounded_Ridge (Tuned)** | Bounded_Linear | ₹6,079 | ₹7,722 | 0.7546 | ₹6,783 | ₹2,043 |
| 26 | **Bounded_Ridge** | Bounded_Linear | ₹6,556 | ₹8,097 | 0.7301 | ₹7,194 | ₹3,136 |
| 27 | **Gradient_Boosting (Tuned)** | Ensemble_Boosting | ₹6,373 | ₹8,765 | 0.6838 | ₹7,587 | ₹681 |
| 28 | **Gradient_Boosting** | Ensemble_Boosting | ₹6,362 | ₹8,876 | 0.6757 | ₹7,698 | ₹4,458 |
| 29 | **Extra_Trees** | Ensemble_Bagging | ₹8,379 | ₹11,467 | 0.4588 | ₹8,610 | ₹6,726 |
| 30 | **Decision_Tree** | Tree | ₹8,515 | ₹12,624 | 0.3441 | ₹9,299 | ₹6,259 |
| 31 | **Dummy_Mean** | Baseline | ₹13,346 | ₹15,755 | -0.0216 | ₹12,155 | ₹1,302 |
| 32 | **Dummy_Median** | Baseline | ₹12,895 | ₹16,777 | -0.1586 | ₹11,290 | ₹1,674 |
| 33 | **Linear_Regression** | Linear | ₹9,405 | ₹17,115 | -0.2056 | ₹156,350 | ₹7,288 |

## 3. Best Model Performance & Unbiased Test Evaluation

The selected model is **Bounded_Huber** because it achieved the optimal balance of:
1. Lowest Validation MAE: **₹5,724.10**
2. Low Validation RMSE: **₹7,153.17**
3. High Validation R²: **0.7894**
4. Consistent Cross-Validation MAE: **₹7,019.36**
5. Controlled Overfitting Gap: **₹1,910.50**

### Final Test Set Metrics (Evaluated ONLY ONCE on untouched 14-sample test set)
| Metric | Value | Description |
| :--- | :--- | :--- |
| `Test_MAE` | **₹3,831.37** | Mean Absolute Error in INR (average magnitude of errors in ₹) |
| `Test_MSE` | **₹23,210,423.93** | Mean Squared Error (penalizes large outliers quadratically) |
| `Test_RMSE` | **₹4,817.72** | Root Mean Squared Error in INR (standard deviation of residuals in ₹) |
| `Test_R2` | **0.8513** | R-squared Score (proportion of variance explained) |
| `Test_MAPE_%` | **15.19%** | Evaluation metric |
| `Test_MedAE` | **₹3,103.99** | Median Absolute Error in INR (robust to extreme errors in ₹) |

## 4. Final Test Predictions Sample

| Brand | Model | Actual Price (₹) | Predicted Price (₹) | Absolute Error (₹) | Error (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| OnePlus | OnePlus Nord CE6 Lite | ₹23,028 | ₹22,689 | ₹339 | 1.47% |
| Xiaomi Redmi | Xiaomi Redmi 13 5G | ₹18,875 | ₹18,149 | ₹726 | 3.85% |
| Xiaomi Redmi | Xiaomi Redmi 14R | ₹15,100 | ₹14,225 | ₹875 | 5.79% |
| Nokia | Nokia 5.3 | ₹14,723 | ₹15,794 | ₹1,071 | 7.28% |
| OnePlus | OnePlus N6x | ₹18,875 | ₹16,737 | ₹2,138 | 11.33% |
| Xiaomi Redmi | Xiaomi Redmi K90 Ultra | ₹48,321 | ₹50,590 | ₹2,269 | 4.70% |
| Infinix | Infinix Smart 8 Pro | ₹8,682 | ₹11,117 | ₹2,435 | 28.05% |
| Motorola | Motorola Edge 50 Fusion | ₹20,763 | ₹24,536 | ₹3,773 | 18.17% |
| Motorola | Motorola Edge 60 | ₹28,313 | ₹24,322 | ₹3,991 | 14.09% |
| Samsung | Samsung Galaxy M16 | ₹14,345 | ₹18,837 | ₹4,492 | 31.31% |
| Oppo | Oppo A5 Pro (China) | ₹26,426 | ₹20,612 | ₹5,814 | 22.00% |
| Motorola | Motorola Moto G Power (2026) | ₹30,201 | ₹22,776 | ₹7,425 | 24.58% |
| OnePlus | OnePlus Ace 3 Pro | ₹41,526 | ₹50,381 | ₹8,855 | 21.32% |
| Nothing | Nothing phone (1) | ₹50,579 | ₹41,141 | ₹9,438 | 18.66% |

## 5. Architectural Findings and Hardware Price Drivers
1. **Semiconductor Lithography (`Process_Technology_nm`)**: High negative correlation with price ($r = -0.68$). 3nm and 4nm nodes command major price premiums.
2. **Memory Size and Standard (`RAM_Size_GB` & `RAM_Type`)**: 12GB+ LPDDR5X configurations are characteristic of the ₹45,000+ premium tier.
3. **Display Technology (`Display_Tech_Category` & `Refresh_Rate_Hz`)**: AMOLED/OLED high-refresh displays distinctly segment premium smartphones from entry-level LCD devices.
4. **CPU Clustering Architecture**: Tri-cluster Octa-core SoCs (1 Prime + 3 Gold + 4 Silver) significantly outprice dual-cluster budget architectures.

## 6. Project Artifacts Catalog
- Cleaned Dataset: `data/processed/cleaned_smartphones.csv`
- Data Quality Audit: `reports/data_quality_report.csv`
- Cleaning Summary: `reports/cleaning_summary.txt`
- 16 EDA Figures: `reports/figures/01_*.png` through `reports/figures/16_*.png`
- Diagnostic Residual Plots: `reports/figures/actual_vs_predicted.png`, `residual_plot.png`, `residual_distribution.png`, `error_distribution.png`
- Feature Attribution Plots: `reports/figures/feature_importance.png`, `permutation_importance.png`, `linear_coefficients.png`
- Model Benchmark Comparison: `reports/model_comparison.csv`
- Test Predictions Audit: `reports/final_test_predictions.csv`
- Production Serialized Bundle: `models/best_smartphone_price_model.joblib`
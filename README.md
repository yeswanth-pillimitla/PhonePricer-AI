# Smartphone Price Prediction using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.14-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9.1-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.67.0-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade, end-to-end Supervised Machine Learning regression system engineered to predict smartphone retail prices in **Indian Rupees (₹ INR)** from hardware specifications and microarchitecture features.

---

## Table of Contents
1. [Project Introduction](#1-project-introduction)
2. [Problem Statement](#2-problem-statement)
3. [Dataset Description](#3-dataset-description)
4. [Raw and Engineered Features](#4-raw-and-engineered-features)
5. [Data Cleaning & Quality Audit](#5-data-cleaning--quality-audit)
6. [Exploratory Data Analysis (EDA)](#6-exploratory-data-analysis-eda)
7. [Feature Engineering](#7-feature-engineering)
8. [Data Preprocessing & Anti-Leakage Protocol](#8-data-preprocessing--anti-leakage-protocol)
9. [Standardization & Scaling Strategy](#9-standardization--scaling-strategy)
10. [Train / Validation / Test Splitting](#10-train--validation--test-splitting)
11. [Machine Learning Algorithms Evaluated](#11-machine-learning-algorithms-evaluated)
12. [Hyperparameter Tuning](#12-hyperparameter-tuning)
13. [Model Evaluation Metrics](#13-model-evaluation-metrics)
14. [Comprehensive Model Benchmark Comparison](#14-comprehensive-model-benchmark-comparison)
15. [Best Model Selection](#15-best-model-selection)
16. [Final Unbiased Test Results](#16-final-unbiased-test-results)
17. [Model Explainability & Feature Attribution](#17-model-explainability--feature-attribution)
18. [Project Structure](#18-project-structure)
19. [How to Run the Project](#19-how-to-run-the-project)
20. [How to Launch the Streamlit Web Application](#20-how-to-launch-the-streamlit-web-application)
21. [Automated Testing](#21-automated-testing)
22. [Limitations & Future Improvements](#22-limitations--future-improvements)

---

## 1. Project Introduction
The global smartphone ecosystem features thousands of models spanning entry-level handsets to ultra-premium flagships. Setting competitive retail pricing or identifying fair-market valuation requires modeling complex, non-linear relationships between semiconductor manufacturing nodes, CPU cluster architectures, high-refresh display technologies, memory capacities, and optical sensor configurations.

This repository provides an automated, scientifically verified regression pipeline that trains, compares, tunes, tests, and serves machine learning models to forecast retail smartphone prices in Indian Rupees (₹ INR).

---

## 2. Problem Statement
Given a set of physical and technical smartphone specifications $\mathbf{x} \in \mathbb{R}^d$, learn a continuous mapping function:
$$f: \mathbf{x} \mapsto \hat{y} \in \mathbb{R}^+$$
where $\hat{y}$ is the predicted smartphone price in **BDT** (`Price_INR`).

---

## 3. Dataset Description
- **Source**: `Coa-12.csv`
- **Total Records**: 91 unique smartphone models
- **Brands Represented (11)**: OnePlus, Nokia, Nothing, Xiaomi Redmi, POCO, Infinix, Vivo, Oppo, Samsung, Motorola, Realme
- **Price Target Summary**:
  - Min: 10,000 INR (*Nokia C20 Plus / Nokia C30*)
  - Max: 99,990 INR (*OnePlus 11*)
  - Mean: 35,475.38 INR
  - Median: 30,000.00 INR
  - Standard Deviation: 20,161.67 INR
  - IQR: 20,250 INR to 45,250 INR

---

## 4. Raw and Engineered Features

### Input Predictor Categories
- **Microprocessor & Compute**: Semiconductor Lithography (`Process_Technology_nm`), Prime/Performance/Efficiency Core counts and clock frequencies (`Processor_Core1_Frequency_GHz`, etc.), CPU Cache hierarchy (`Cache_Memory_Core1`, `Core2`, `Core3`), Instruction Set Architecture (`ISA`).
- **Memory & Storage**: `RAM_Size_GB`, `RAM_Size_GB_Max`, `RAM_Type` (LPDDR3, LPDDR4X, LPDDR5, LPDDR5X), `Storage_GB`, `Storage_GB_Max`.
- **Display**: Panel Resolution (`Width_px`, `Height_px`, `Total_Display_Pixels`), `Refresh_Rate_Hz` (60 to 185 Hz), `Display_Tech_Category` (AMOLED/OLED vs LCD).
- **Battery & Optics**: `Battery_Capacity_mAh`, `Rear_Camera_MP`.
- **Brand Positioning**: `Phone_Company` (One-hot encoded).

---

## 5. Data Cleaning & Quality Audit
1. **Missing Values**: 0 null/NaN entries across all 34 raw columns.
2. **Duplicate Records**: 0 duplicate rows; 0 duplicate model names.
3. **Structural Zero Preservation**: 65 phones utilize dual-cluster CPU architectures. For these phones, Cluster 3 is legitimately recorded as `Processor_Core3 = 'Not Applicable'`, `Count = 0`, `Freq = 0.0 GHz`, and `Cache = 0 KB`. These are confirmed **structural zeros** and preserved rather than imputed.
4. **Hardware Physics Verification**:
   - $\text{Core1\_Count} + \text{Core2\_Count} + \text{Core3\_Count} = 8$ holds across 100% of rows.
   - $\text{RAM\_Min} \le \text{RAM} \le \text{RAM\_Max}$ holds across 100% of rows.
   - $\text{Storage\_Min} \le \text{Storage} \le \text{Storage\_Max}$ holds across 100% of rows.
5. **Non-Destructive Outlier Audit**: Flagship smartphones exceeding the statistical IQR upper fence (82,750 INR) were verified as genuine luxury models (OnePlus 11, Infinix Note 60 Ultra) and preserved.

---

## 6. Exploratory Data Analysis (EDA)
16 publication-grade figures were generated and stored in `reports/figures/`:
- `01_price_distribution.png`: Bimodal density of consumer smartphone pricing.
- `04_price_by_brand.png`: Brand tier valuation profiles (OnePlus and Nothing leading median prices).
- `05_ram_vs_price.png` & `06_storage_vs_price.png`: Monotonic scaling of price with memory.
- `12_process_tech_vs_price.png`: Substantial negative correlation ($r = -0.68$) between semiconductor lithography node (nm) and retail price.
- `13_correlation_heatmap.png`: Inter-feature correlation matrix of physical attributes.

---

## 7. Feature Engineering
Derived 7 domain-specific features:
1. `Max_Core_Frequency_GHz`: Maximum burst clock frequency across all active clusters.
2. `Weighted_Average_Frequency_GHz`: Aggregate compute density: $\frac{\sum (\text{count}_i \times \text{freq}_i)}{8}$.
3. `Total_Display_Pixels`: $\text{Width} \times \text{Height}$.
4. `Display_Aspect_Ratio`: $\frac{\text{Height}}{\text{Width}}$.
5. `Display_Tech_Category`: Normalized into `AMOLED_OLED` vs `LCD`.
6. `Rear_Camera_MP`: Regex extraction of primary optical sensor resolution.
7. `Num_CPU_Clusters`: Binary indicator (2 clusters vs 3 clusters).

---

## 8. Data Preprocessing & Anti-Leakage Protocol
- Implemented using Scikit-Learn `ColumnTransformer` inside an immutable `Pipeline`.
- **Anti-Leakage**: Transformers are fitted **exclusively on the training partition ($X_{train}$)**.
- **Categorical Safety**: `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` guarantees that unseen brands or architectures in production produce all-zero indicator vectors without throwing runtime exceptions.

---

## 9. Standardization & Scaling Strategy
- **Linear & Distance-based Models** (Ridge, Lasso, ElasticNet, SVR, KNN): Numerical columns scaled using `StandardScaler` fitted on $X_{train}$.
- **Tree-based Ensembles** (Decision Trees, Random Forest, Extra Trees, Gradient Boosting): Raw numeric scales preserved (`scaling=None`) to avoid redundant floating-point transformations.

---

## 10. Train / Validation / Test Splitting
- **Partition Ratios**: 70% Training (63 rows) / 15% Validation (14 rows) / 15% Test (14 rows).
- **Quantile-Binned Stratification**: Stratified on 5 price quantiles with fixed `random_state=42` to guarantee uniform distribution of budget, mid-range, and flagship devices across all partitions:
  - Train Mean: 35,185 INR
  - Validation Mean: 38,221 INR
  - Test Mean: 34,035 INR
- **Zero Leakage**: Test set held out strictly untouched until after model selection and hyperparameter tuning.

---

## 11. Machine Learning Algorithms Evaluated
1. **DummyRegressor (Mean & Median)** — Baseline benchmarks
2. **Linear Regression (OLS)**
3. **Ridge Regression (L2)**
4. **Lasso Regression (L1)**
5. **ElasticNet Regression (L1 + L2)**
6. **Decision Tree Regressor**
7. **Random Forest Regressor**
8. **Extra Trees Regressor**
9. **Gradient Boosting Regressor**
10. **HistGradientBoosting Regressor**
11. **Support Vector Regression (SVR)**
12. **K-Nearest Neighbors Regressor (KNN)**

---

## 12. Hyperparameter Tuning
Executed 5-fold cross-validation `GridSearchCV` on the training partition ($X_{train}$) optimizing `neg_mean_absolute_error`. Tuned key hyperparameters across all models:
- **Random Forest**: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`
- **Ridge / Lasso / ElasticNet**: `alpha`, `l1_ratio`
- **SVR**: `C`, `epsilon`, `kernel`, `gamma`
- **KNN**: `n_neighbors`, `weights`, `p`

---

## 13. Model Evaluation Metrics
All models are evaluated on:
- **MAE** (Mean Absolute Error): Average absolute difference in INR.
- **MSE** (Mean Squared Error): Quadratic penalty on large errors.
- **RMSE** (Root Mean Squared Error): Residual standard deviation in INR.
- **R² Score**: Proportion of target variance explained.
- **MAPE (%)**: Mean Absolute Percentage Error.
- **MedAE**: Median Absolute Error (outlier-robust error metric).

---

## 14. Comprehensive Model Benchmark Comparison

| Rank | Model Name | Family | Val MAE (₹ INR) | Val RMSE (₹ INR) | Val R² | 5-Fold CV MAE (₹ INR) | Overfitting Gap (₹ INR) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | **Random Forest (Tuned)** | Ensemble Bagging | **6,531** | **8,612** | **0.8260** | **8,891** | **+791** |
| 2 | **KNN** | Instance-Based | 6,672 | 9,463 | 0.7899 | 8,516 | +6,482 |
| 3 | **KNN (Tuned)** | Instance-Based | 6,886 | 9,387 | 0.7932 | 7,998 | -54 |
| 4 | **HistGradientBoosting (Tuned)** | Ensemble Boosting | 7,091 | 8,782 | 0.8191 | 9,914 | +2,898 |
| 5 | **ElasticNet (Tuned)** | Regularized Linear | 7,282 | 9,553 | 0.7859 | 8,674 | -249 |
| 6 | **Ridge (Tuned)** | Regularized Linear | 7,297 | 9,482 | 0.7891 | 8,739 | +271 |
| 7 | **ElasticNet (Default)** | Regularized Linear | 7,478 | 9,591 | 0.7842 | 9,000 | +679 |
| 8 | **Random Forest (Default)** | Ensemble Bagging | 8,057 | 10,604 | 0.7362 | 9,299 | +4,386 |
| 9 | **Lasso (Tuned)** | Regularized Linear | 8,069 | 10,653 | 0.7337 | 9,828 | +1,631 |
| 10 | **SVR (Tuned)** | Support Vector | 8,263 | 12,270 | 0.6467 | 8,683 | +3,140 |
| 11 | **Gradient Boosting** | Ensemble Boosting | 9,267 | 11,828 | 0.6718 | 10,381 | +8,266 |
| 12 | **SVR (Default)** | Support Vector | 9,642 | 13,167 | 0.5932 | 9,525 | +3,607 |
| 13 | **HistGradientBoosting** | Ensemble Boosting | 10,394 | 13,216 | 0.5902 | 12,687 | +2,972 |
| 14 | **Ridge (Default)** | Regularized Linear | 12,230 | 14,673 | 0.4948 | 11,506 | +7,034 |
| 15 | **Lasso (Default)** | Regularized Linear | 13,498 | 16,303 | 0.3764 | 11,374 | +8,133 |
| 16 | **Extra Trees** | Ensemble Bagging | 13,896 | 18,400 | 0.2057 | 11,630 | +13,705 |
| 17 | **Decision Tree** | Tree | 15,242 | 24,276 | -0.3827 | 13,948 | +15,052 |
| 18 | **Dummy Regressor (Median)** | Baseline | 17,079 | 22,221 | -0.1586 | 14,953 | +2,217 |
| 19 | **Dummy Regressor (Mean)** | Baseline | 17,676 | 20,867 | -0.0216 | 16,099 | +1,725 |
| 20 | **Linear Regression (OLS)** | Linear | 17,987 | 25,387 | -0.5122 | 28,801 | +13,483 |

---

## 15. Best Model Selection
**Winner: `Random Forest (Tuned)`**
- **Validation MAE**: **6,530.72 INR** (Lowest overall)
- **Validation RMSE**: **8,611.94 INR** (Lowest overall)
- **Validation R²**: **0.8260** (Highest overall)
- **5-Fold CV MAE**: **8,890.62 INR** (Consistent generalizability)
- **Overfitting Gap**: **+790.93 INR** (Tightest train/validation generalization)

---

## 16. Final Unbiased Test Results
Following selection, the winning pipeline was refit on combined Train+Validation data (77 records, 85%) and evaluated once on the held-out Test dataset (14 records, 15%):

| Metric | Measured Value | Meaning |
| :--- | :--- | :--- |
| **Test MAE** | **5,949.69 INR** | Average forecast error on unseen phones |
| **Test RMSE** | **8,171.63 INR** | Standard deviation of residuals |
| **Test R²** | **0.7561** | 75.6% of test variance explained |
| **Test MedAE** | **3,725.38 INR** | Median prediction error |
| **Test MAPE** | **19.09%** | Percentage error across price tiers |

### Sample Test Set Inferences
| Phone Model | Actual Price | Predicted Price | Absolute Error | Error (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Xiaomi Redmi 14R** | 20,000 INR | 19,744 INR | 256 INR | 1.28% |
| **OnePlus Nord CE6 Lite** | 30,500 INR | 31,697 INR | 1,197 INR | 3.93% |
| **OnePlus N6x** | 25,000 INR | 22,825 INR | 2,175 INR | 8.70% |
| **Oppo A5 Pro** | 35,000 INR | 32,704 INR | 2,296 INR | 6.56% |
| **Xiaomi Redmi K90 Ultra** | 64,000 INR | 67,267 INR | 3,267 INR | 5.10% |

---

## 17. Model Explainability & Feature Attribution
- **Semiconductor Process Node (`Process_Technology_nm`)**: Most influential single factor ($r = -0.68$).
- **RAM Capacity (`RAM_Size_GB`)**: Primary volume differentiator between 15,000 INR budget phones (4GB) and 60,000+ INR flagships (12GB+).
- **Core Frequencies & Architecture**: Processors featuring Cortex-X / Oryon cores clocked above 3.2 GHz and tri-cluster configurations command substantial price premiums.

---

## 18. Project Structure
```
smartphone-price-prediction/
│
├── data/
│   ├── raw/
│   │   └── Coa-12.csv                     # Original preserved raw dataset
│   └── processed/
│       └── cleaned_smartphones.csv        # Cleaned dataset with verified bounds
│
├── notebooks/
│   ├── PhonePricer_AI_Graphs_Fixed.ipynb  # Interactive Colab & Jupyter modeling notebook
│   └── exploratory_analysis.ipynb         # Interactive Jupyter EDA demo
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                     # Safe schema-validated CSV loader
│   ├── data_cleaning.py                   # Data cleaning, physics validation & reporting
│   ├── eda.py                             # 16 Matplotlib/Seaborn statistical visualizations
│   ├── feature_engineering.py             # Domain-specific derived hardware features
│   ├── feature_selection.py               # Multicollinearity & cardinality filtration
│   ├── data_splitting.py                  # Quantile-stratified 70/15/15 partitions
│   ├── preprocessing.py                   # Scikit-learn Pipeline & ColumnTransformer
│   ├── train.py                           # Training 11 algorithms & baseline Dummy regressors
│   ├── model_tuning.py                    # GridSearchCV 5-fold cross-validation tuning
│   ├── model_evaluation.py                # MAE, MSE, RMSE, R², MAPE, MedAE metrics
│   ├── model_comparison.py                # Multi-model benchmarking, refit, test evaluation
│   ├── explainability.py                  # Tree MDI, Permutation, and Beta weights
│   └── predict.py                         # Single-spec inference API & artifact persistence
│
├── models/
│   └── best_smartphone_price_model.joblib # Serialized winning pipeline + metadata
│
├── reports/
│   ├── figures/                           # 23 generated high-res diagnostic plots
│   ├── data_quality_report.csv            # Detailed feature quality audit
│   ├── cleaning_summary.txt               # Data cleaning summary report
│   ├── eda_report.md                      # Comprehensive exploratory findings
│   ├── feature_selection_report.md        # Feature selection rationales
│   ├── preprocessing_report.md            # Preprocessing architecture documentation
│   ├── model_comparison.csv               # 20-model experimental leaderboard
│   ├── final_test_predictions.csv         # Itemized test predictions & error analysis
│   ├── explainability_report.md           # Feature attribution analysis
│   └── final_report.md                    # Executive project report
│
├── app/
│   └── streamlit_app.py                   # Enterprise Streamlit Web Application
│
├── requirements.txt                       # Locked dependencies
├── config.yaml                            # System configuration parameters
├── README.md                              # Complete technical documentation
└── main.py                                # End-to-end pipeline execution entrypoint
```

---

## 19. How to Run the Project

### Clone & Install
```bash
# Clone the repository and navigate into the workspace
git clone <repo-url>
cd smartphone-price-prediction

# Install dependencies
pip install -r requirements.txt
```

### Run Complete End-to-End Pipeline
```bash
python main.py
```
This executes all 14 stages: cleaning, EDA generation, feature engineering, splitting, baseline evaluation, candidate training, hyperparameter tuning, model selection, test evaluation, explainability, and artifact packaging.

---

## 20. How to Launch the Streamlit Web Application
Launch the interactive prediction and benchmarking dashboard:
```bash
python -m streamlit run app/streamlit_app.py
```
Or directly:
```bash
streamlit run app/streamlit_app.py
```
The application will open automatically in your browser at `http://localhost:8501`.

---

## 21. Limitations & Future Improvements
1. **Sample Size**: With 91 records, dataset size limits complex deep learning or high-cardinality nominal modeling (e.g. one-hot encoding every specific SoC chip). Mitigated through aggressive regularized ensembles (Random Forest, ElasticNet) and cross-validation.
2. **Temporal Market Pricing**: Smartphone prices fluctuate over product life cycles, exchange rate movements, and retailer promotions. Future versions could integrate timestamped price history.
3. **Multi-Camera Telephoto Specs**: Parsing periscope zoom optical mechanics across irregular raw camera strings can be further enriched with standardized optical zoom ratios.


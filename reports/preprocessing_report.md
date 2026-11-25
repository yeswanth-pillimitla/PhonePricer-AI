# Data Preprocessing and Leakage Prevention Report

## 1. Architectural Design
Data preprocessing is engineered using Scikit-Learn `Pipeline` and `ColumnTransformer` constructs.
Crucially, preprocessing is encapsulated **inside** the model pipeline so that fitting happens **exclusively on the training dataset**.

### Dual-Pipeline Strategy
1. **Linear / Distance-based Pipeline (`scaling='standard'` / `'robust'`)**:
   - Used by: Linear Regression, Ridge, Lasso, ElasticNet, SVR, and K-Nearest Neighbors.
   - Numerical steps: `SimpleImputer(strategy='median')` -> `StandardScaler()`.
   - Ensures gradients and Euclidean distances are not distorted by differences in measurement units (e.g., mAh vs GHz).

2. **Tree-based Pipeline (`scaling=None`)**:
   - Used by: Decision Trees, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting.
   - Numerical steps: `SimpleImputer(strategy='median')` (preserves raw numeric feature scales).
   - Avoids redundant floating-point scaling transformations that do not alter monotonic tree partition decisions.

## 2. Categorical Encoding with Out-of-Vocabulary Safety
- Categorical columns: `Phone_Company`, `RAM_Type`, `ISA`, `Display_Tech_Category`.
- Encoded using: `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
- **Robustness Guarantee**: Any unseen smartphone brand, RAM type, or architecture encountered in production or test sets will produce an all-zero one-hot row rather than throwing an exception.

## 3. Strict Anti-Leakage Protocol
- Neither validation data nor test data is ever passed to `preprocessor.fit()`.
- The identical fitted transformer learned from $X_{train}$ is used to transform $X_{val}$, $X_{test}$, and user inputs in the Streamlit application.
- Target variable `Price_INR` is isolated and never scaled or included in transformer inputs.
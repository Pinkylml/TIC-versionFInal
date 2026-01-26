# CRISP-DM Phase 4: Modeling Audit Report

## 1. Executive Summary
This audit covers the **entire** modeling pipeline in `Entrenamiento_XGBoost(1).ipynb`, including data loading, feature engineering, and the rigorous dual-stage training process (Initial Screening -> Hyperparameter Tuning).

**Overall Assessment:** The methodology is **statistically sound** and **advanced**. It employs Survival Analysis (AFT) and a specific 3-scenario validation strategy to ensure robustness against dataset shifts. The final tuning phase reveals that correct stratification (S2) provides the most reliable generalization, despite Random splitting (S1) showing higher (but potentially overfit) cross-validation scores.

## 2. Notebook Settings & Environment
*   **Notebook:** `04_Data_modeling/Entrenamiento_XGBoost(1).ipynb`
*   **Dependencies:** `xgboost` (GPU enabled), `lifelines`, `scikit-survival` (sksurv), `sklearn`.
*   **Hardware:** The code leverages `device='cuda'` for accelerated training.

## 3. Data Integrity & Feature Engineering
*   **Alignment:** Datasets A and B are perfectly aligned with **103 columns** after processing.
*   **Categorical Handling:** Custom `carrera_clean` logic successfully normalizes inconsistent labels.
*   **Feature Selection:** The `Cohorte` variable was excluded based on statistical testing (Cox/RSF) to prevent bias.

## 4. Modeling Strategy: RSF vs. XGBoost AFT

The pipeline compares two distinct approaches:
1.  **Random Survival Forest (RSF):** Used as a baseline for non-linear interactions.
2.  **XGBoost AFT (Winner):** The primary model, utilizing the Accelerated Failure Time objective (`survival:aft`) which is ideal for the interval-censored data structure (`T_Lower`, `T_Upper`).

### 4.1 Validation Strategy
Three specific splitting scenarios were evaluated to test robustness:
*   **S1 (Random):** Naïve random split.
*   **S2 (Stratified by Event):** Preserves censorship rates. **(Recommended)**
*   **S3 (Stratified by Career):** Balances career representation.

## 5. Hyperparameter Tuning & Results (CRITICAL)

The final phase involved a `RandomizedSearch` (30 iterations) followed by a final fit. The results present an interesting trade-off between Cross-Validation (CV) potential and actual Test generalization.

### 5.1 Final Tuning Results Table

| Dataset | Split | CV Score (Mean) | CV Std | Test Score | Best Iter |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A** | **S1 (Random)** | **0.6769** | 0.068 | 0.6115 | 338 |
| A | S2 (Stratified) | 0.6475 | 0.046 | **0.6620** | 788 |
| A | S3 (Career) | 0.6496 | 0.044 | 0.6418 | 589 |
| **B** | **S1 (Random)** | **0.6742** | 0.070 | 0.6034 | 667 |
| B | S2 (Stratified) | 0.6594 | 0.056 | 0.6417 | 206 |

### 5.2 Analysis of the "Winner"
The user questioned the selection of **S1 (Random)** which appears at the top due to the highest `CV Score` (0.6769).

**Explanation:**
*   **Why S1 ranks first:** Automated selection logic typically prioritizes the **Metrics on Training/CV data** (`cv_cindex_mean`). S1 achieved the highest peak likely due to a "lucky" random split that made the training folds easier to predict.
*   **The Discrepancy:** Note the huge gap between CV (0.677) and Test (0.612) for S1. This indicates **overfitting** to the specific random split.
*   **The Real Statistical Winner:** **S2 (Stratified by Event)** is the superior model. It has a slightly lower CV score (0.648) but generalizes much better to the Test set (0.662). Its standard deviation is also lower (0.046 vs 0.068), proving it is more stable.

## 6. Audit Conclusion
The notebook successfully identifies a high-potential model. While the automated sorting might put **S1** on top due to raw CV numbers, a data scientific review of the table confirms that **Dataset A - Scenario S2 (Stratified)** is the most reliable configuration for deployment, offering a C-index of ~0.66 with high stability.

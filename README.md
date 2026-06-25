# TIA Risk Prediction: Effect of Influenza Vaccination on First Occurrence of Transient Ischemic Attack

## Table of Contents
1. [What Is This Project?](#1-what-is-this-project)
2. [Why It Matters](#2-why-it-matters)
3. [Data Source](#3-data-source)
4. [Analysis Pipeline](#4-analysis-pipeline)
5. [Feature Engineering](#5-feature-engineering)
6. [Models and Why Each Was Chosen](#6-models-and-why-each-was-chosen)
7. [Handling Class Imbalance](#7-handling-class-imbalance)
8. [Model Evaluation Strategy](#8-model-evaluation-strategy)
9. [Interpretability: SHAP](#9-interpretability-shap)
10. [The Vaccine Hypothesis in Depth](#10-the-vaccine-hypothesis-in-depth)
11. [Challenges and Limitations](#11-challenges-and-limitations)
12. [How to Run](#12-how-to-run)
13. [Practical Applications](#13-practical-applications)
14. [How to Improve from Here](#14-how-to-improve-from-here)

---

## 1. What Is This Project?

This project builds a machine learning system to predict the **first occurrence of a Transient Ischemic Attack (TIA)** — commonly called a "mini-stroke" — using publicly available US population health data.

A TIA produces temporary stroke-like symptoms (sudden numbness, vision loss, confusion, difficulty speaking) that resolve within 24 hours. Unlike a full stroke, no permanent damage occurs. However, a TIA is a medical emergency: roughly **10–15% of TIA patients suffer a full stroke within 90 days**, with the highest risk concentrated in the first 48 hours. Early identification of high-risk individuals enables preventive intervention before that window closes.

**Primary research question:** *What are the major predictors of first TIA occurrence in the US general population, and does influenza vaccination reduce risk — particularly in adults aged 65 and older?*

The vaccine hypothesis is not arbitrary. Acute infections are known to trigger inflammatory responses that destabilize atherosclerotic plaques and promote thrombosis. Several observational studies suggest that influenza vaccination reduces the rate of cardiovascular events. This project tests whether that signal is detectable and quantifiable in cross-sectional population survey data.

---

## 2. Why It Matters

**Clinically:** TIA is massively underdiagnosed. Many patients dismiss symptoms or are not seen until after the time-sensitive risk window has passed. A risk score derived from routine clinical variables — age, blood pressure, diabetes status, vaccination history — could be embedded into an electronic health record to flag high-risk patients who have not yet had a TIA but are statistically likely to have one.

**Epidemiologically:** TIA incidence in the US is estimated at 200,000–500,000 per year, but the true number is higher because mild events go unreported. Identifying modifiable risk factors (like vaccination) at the population level informs public health policy.

**Economically:** A stroke costs an average of $140,000 in the first year of care. Preventing even a fraction of the strokes that follow a TIA has enormous economic value. A low-cost screening tool that prioritizes preventive care is cost-effective even at modest sensitivity.

**Scientifically:** The causal pathway between influenza and cerebrovascular events is biologically plausible but not yet confirmed. If the vaccine signal holds up across multiple modeling approaches and is stable in SHAP analysis, it strengthens the case for a prospective clinical trial.

---

## 3. Data Source

### NHANES — National Health and Nutrition Examination Survey

NHANES is an ongoing CDC program that interviews and physically examines a nationally representative sample of Americans. It is one of the most comprehensive and rigorously designed cross-sectional health surveys in the world. Data are released in two-year cycles and are entirely free and publicly available.

**Three cycles were pooled to maximize sample size:**

| Cycle | Years | Participants | NHANES Code |
|-------|-------|-------------|-------------|
| I | 2015–2016 | ~9,971 | Suffix `_I` |
| J | 2017–2018 | ~9,254 | Suffix `_J` |
| P | 2017–2020 pre-pandemic | ~15,560 | Prefix `P_` |

Pooling is standard practice in NHANES analyses when studying low-prevalence outcomes. TIA affects only 1–3% of the survey population, so a single two-year cycle would yield fewer than 200 positive cases — too few for robust modeling.

**Modules used from each cycle:**

| Module | Variables | Purpose |
|--------|-----------|---------|
| DEMO | Age, sex, race, income, education | Demographic controls |
| MCQ | MCQ160F (stroke/TIA), MCQ160E/B/D (cardiac), MCQ220 (cancer) | Target variable + cardiac comorbidities |
| IMQ | IMQ011 (flu shot past 12 months) | Primary predictor of interest |
| BPQ | BPQ020 (hypertension diagnosis) | Risk factor |
| BPX / BPXO | Systolic & diastolic BP measurements | Augments questionnaire hypertension |
| DIQ | DIQ010 (diabetes) | Risk factor |
| SMQ | SMQ020 (ever smoked), SMQ040 (current) | Risk factor |
| BMX | BMXBMI (BMI), BMXWAIST (waist circumference) | Obesity indicators |
| TCHOL | LBXTC (total cholesterol) | Cardiovascular risk |
| HDL | LBDHDD (HDL cholesterol) | Cardiovascular risk |
| ALQ | ALQ130 / ALQ121 (drinks per week) | Lifestyle factor |
| PAQ | PAD680 (sedentary time) | Lifestyle factor |

### Why NHANES and Not Claims Data?

CMS claims data captures what is billed, not what is measured. NHANES provides physical examination results (measured blood pressure, blood draws for cholesterol) alongside self-reported history, making the feature set richer and less subject to coding artifacts. NHANES is also free and requires no data use agreement for its public-use files.

### Data Access Note

CDC restructured its NHANES file hosting in 2022. Files now live at:
```
https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{YEAR}/DataFiles/{FILE}.XPT
```
Pre-pandemic cycle P files use a reversed naming convention (`P_DEMO.xpt` instead of `DEMO_P.XPT`) and are co-located with cycle J files under `/2017/DataFiles/`. The download cell in the notebook handles this automatically and caches files locally after the first run.

---

## 4. Analysis Pipeline

The notebook (`tia_influenza_vaccine_predictor.ipynb`) executes a complete end-to-end analysis in 15 sections. Here is how each piece fits together and why decisions were made the way they were.

### Step 1 — Data Acquisition
Each of the 36 XPT (SAS transport) files is downloaded from the CDC, validated as a genuine XPT file (by checking the binary header `HEADER RECORD*`), and cached locally. Corrupt HTML error pages that were previously saved to disk are detected and deleted before re-downloading. This guards against partial downloads from interrupted sessions.

### Step 2 — Module Integration per Cycle
Within each NHANES cycle, the 12 modules are merged on `SEQN` (the respondent ID) using left joins anchored to the demographics file. Left joins are used because not every respondent completes every questionnaire — respondents who skipped the immunization module remain in the dataset with NaN for vaccination status rather than being dropped.

A `safe_cols` helper ensures that if a particular column was renamed or dropped in a given cycle (which happens between NHANES cycles), the merge silently skips it rather than crashing.

### Step 3 — Pooling Across Cycles
The three per-cycle dataframes are concatenated with a `cycle` label column added. This produces a pooled dataset of approximately 35,000 respondents before target filtering. A `cycle` dummy is not explicitly entered as a model feature, but it implicitly captures survey year effects through shared covariates.

### Step 4 — Target Variable Definition
The outcome variable (`stroke_tia`) is derived from `MCQ160F`:
- 1 → "Yes, ever had a stroke" → positive class (1)
- 2 → "No" → negative class (0)
- 7, 9, or missing → excluded (refused / don't know)

After dropping rows with missing target, roughly 97–99% of remaining respondents are TIA-negative and 1–3% are TIA-positive.

### Step 5 — Feature Engineering
Twenty-nine candidate features are derived from raw survey variables (see Section 5 below). The engineering step also harmonizes variable names across cycles, handles NHANES coding conventions (e.g., 1=Yes/2=No rather than 1=Yes/0=No), and creates interaction terms relevant to the vaccine hypothesis.

### Step 6 — Exploratory Data Analysis
Before any modeling, six visualizations are generated:
- Age distribution by stroke status (TIA patients are older)
- TIA rate by vaccination status (primary hypothesis check)
- TIA rate by age group (non-linearity of risk with age)
- Correlation of risk factors with outcome (quick magnitude check)
- BMI distribution by stroke status
- Vaccine × senior interaction (key effect modification plot)

Chi-square tests on categorical features and descriptive statistics by outcome group provide a statistical sense-check before investing in model training.

### Steps 7–9 — Model Training and Optimization
Seven model classes are trained on the same stratified split (70% train / 15% validation / 15% test). XGBoost and LightGBM are further optimized with Optuna for 60 trials each. See Section 6 for model rationale.

### Steps 10–11 — Evaluation and Threshold Selection
The best model by validation ROC-AUC is evaluated once on the held-out test set. The classification threshold is not left at the default 0.5 — it is chosen using Youden's J statistic (maximizing sensitivity + specificity simultaneously on the validation ROC curve). In clinical screening, the cost of missing a true TIA patient is far higher than a false alarm, so this threshold is often shifted further toward sensitivity.

### Steps 12–13 — SHAP Interpretability and Vaccine Effect Analysis
SHAP values decompose each model prediction into per-feature contributions, enabling clinically credible explanations. The vaccine effect is visualized in three ways: a SHAP dependence plot, a scatter plot colored by senior status, and a bar chart of mean SHAP by vaccine × age group combination.

### Steps 14–15 — Cross-Validation and Artifact Saving
Five-fold stratified cross-validation on the combined train+validation set gives a more stable estimate of generalization performance. The final model, imputer, scaler, feature list, and optimal threshold are saved as a pickle artifact for production inference.

---

## 5. Feature Engineering

Raw NHANES variables use numeric codes that need to be translated into model-ready features. The engineering choices reflect domain knowledge about TIA pathophysiology.

### Age
- `age` (continuous): Direct from RIDAGEYR. Age is the single strongest demographic predictor of TIA.
- `age_sq`: Squared age captures the accelerating risk curve in older adults without requiring a non-linear model to discover it.
- `is_senior`: Binary flag for 65+, consistent with Medicare eligibility and a common cut-point in cardiovascular research.

### Vaccine Interaction (Primary Hypothesis)
- `flu_vaccine`: Binary, from IMQ011.
- `flu_x_senior`: Product of `flu_vaccine` × `is_senior`. This interaction term allows the model to detect a differential protective effect of vaccination in older adults — the hypothesized mechanism. Without this explicit term, tree models might find the interaction but linear models would not.

### Hypertension
Hypertension is derived from two sources and combined:
1. Self-reported diagnosis (BPQ020): captures treated/diagnosed hypertension
2. Measured blood pressure (BPX/BPXO modules): SBP ≥ 140 or DBP ≥ 90

Participants with unknown questionnaire response but elevated measured BP are classified as hypertensive. This two-source approach reduces undercounting from undiagnosed hypertension, which is common in lower-income and under-screened populations.

### Cardiac Score
CHD, CHF, and angina (MCQ160E, MCQ160B, MCQ160D) are individually coded and summed into a `cardiac_score` (0–3). This composite reduces dimensionality while preserving additive risk.

### Alcohol
NHANES changed its alcohol frequency variable between cycles. Cycle I uses `ALQ130` (drinks per week); cycles J and P add `ALQ121`. The code looks for either column and uses whichever is present, defining heavy drinking as ≥14 drinks/week (the clinical threshold for "heavy use").

### Cholesterol
NHANES provides both total cholesterol (`LBXTC`) and HDL (`LBDHDD`). Low HDL (< 40 mg/dL) is a known cardiovascular risk factor and is coded as a binary flag.

### Race/Ethnicity
Five race/ethnicity dummies are created from RIDRETH1. Race is included not as a biological predictor but as a proxy for systemic factors (access to healthcare, historical screening rates, dietary patterns) that affect both vaccination uptake and TIA risk.

---

## 6. Models and Why Each Was Chosen

### Logistic Regression (baseline)
The interpretable linear baseline. Coefficients directly express log-odds of TIA per unit change in each feature. If a more complex model does not meaningfully outperform logistic regression, the simpler model should be preferred for clinical deployment (easier to audit, faster to run, no black-box concerns).

### Decision Tree
A single tree is included as a pedagogical baseline that produces easily visualizable decision rules ("if age > 70 AND hypertension AND not vaccinated, predict TIA"). In practice, single trees overfit and are dominated by ensemble methods, but they are useful for stakeholder communication.

### Random Forest
Bagging over many trees reduces variance without introducing strong assumptions. RF handles missing data and mixed feature types well. It is resistant to irrelevant features and provides a natural feature importance measure. It tends to underperform gradient boosting on tabular data in practice but is more robust to hyperparameter choices.

### Gradient Boosting (sklearn)
Sequential residual fitting. Each tree corrects the errors of its predecessors. GBM typically outperforms RF on tabular data and is less sensitive to outliers than logistic regression. The sklearn implementation is slower than XGBoost/LightGBM but more familiar to many practitioners.

### XGBoost
A highly optimized gradient boosting implementation with built-in regularization (L1 and L2), efficient handling of missing values (learned split directions), and hardware-accelerated tree construction. `scale_pos_weight` is set to the negative/positive class ratio to handle imbalance without SMOTE. XGBoost is one of the most reliable off-the-shelf models for structured/tabular prediction tasks.

### LightGBM
Leaf-wise tree growth (vs. depth-wise in XGBoost) finds better splits with fewer trees, making it faster to train. `class_weight='balanced'` is used instead of `scale_pos_weight`. LightGBM handles large datasets well and is particularly strong when the number of features is large relative to sample size.

### PyTorch MLP (Neural Network)
A three-layer feed-forward network with batch normalization, GELU activations, dropout (0.35), and a one-cycle learning rate schedule. The network is trained with `BCEWithLogitsLoss` weighted by the class imbalance ratio. The MLP is included to test whether non-linear interactions not captured by boosted trees improve performance. On tabular data of this size (~25,000 rows, 29 features), deep learning rarely outperforms well-tuned gradient boosting, but it is a useful comparison point.

---

## 7. Handling Class Imbalance

TIA affects approximately 1–3% of the survey population. This severe imbalance means a naive model that predicts "no TIA" for everyone achieves 97–99% accuracy but is clinically worthless.

Three complementary strategies are used:

**1. SMOTE (Synthetic Minority Oversampling Technique)**
Applied to the training set only, after the train/val/test split. SMOTE generates synthetic positive-class examples by interpolating between real positive-class nearest neighbors in feature space. It is never applied to the validation or test sets, which must reflect the true population distribution. Applying SMOTE to the full dataset before splitting would cause data leakage (synthetic examples derived from test-set neighbors).

**2. Class weights**
XGBoost uses `scale_pos_weight = n_negative / n_positive`. LightGBM, Random Forest, GBM, and the MLP use `class_weight='balanced'` or an equivalent `pos_weight` tensor. This makes the loss function penalize misclassification of the minority class more heavily.

**3. Threshold calibration**
The default 0.5 decision threshold assumes equal cost for false positives and false negatives. In TIA screening, a missed high-risk patient (false negative) is far more costly than an unnecessary follow-up referral (false positive). Youden's J selects the threshold that maximizes sensitivity + specificity simultaneously on the validation ROC curve. In deployment, the threshold can be shifted further toward sensitivity based on clinical tolerance for false alarms.

---

## 8. Model Evaluation Strategy

### Why PR-AUC, not just ROC-AUC?

ROC-AUC is insensitive to class imbalance because it treats false positives relative to the full negative population, which is very large. A model can have a high ROC-AUC while still performing poorly on the minority class.

PR-AUC (Precision-Recall AUC) focuses only on the positive class: how precise is the model when it predicts TIA, and how many true TIA cases does it capture? This is the metric that matters clinically. A random classifier achieves a PR-AUC equal to the prevalence rate (~0.02), so any useful model must substantially exceed this baseline.

### Data Split Rationale

A stratified 70/15/15 split (train/validation/test) ensures both the validation and test sets contain proportional representation of TIA cases. The validation set is used for model selection and threshold tuning. The test set is touched exactly once — only for the final selected model — to produce an unbiased estimate of real-world performance. Cross-validation is run separately on the combined train+validation set to estimate variance.

### Hyperparameter Optimization

Optuna runs 60 trials for both XGBoost and LightGBM, searching over learning rate, tree depth, subsampling, regularization coefficients, and minimum child weight using Tree-structured Parzen Estimator (TPE) — a Bayesian method that learns which hyperparameter regions are promising and focuses search there, unlike random search.

---

## 9. Interpretability: SHAP

SHAP (SHapley Additive exPlanations) is grounded in cooperative game theory. Each feature's SHAP value measures how much it contributed to moving a single prediction away from the population average.

**Why SHAP is required for clinical ML:**
- It provides per-patient explanations ("this patient's TIA risk is high because of their age and uncontrolled blood pressure, partially offset by their vaccination status")
- It exposes surprising model behaviors that may indicate data artifacts rather than genuine biology
- Regulatory and institutional review bodies increasingly expect ML-aided clinical tools to be explainable

**Three SHAP visualizations are produced:**
1. **Bar chart of mean |SHAP|** — global feature importance, averaged over the test set
2. **Beeswarm plot** — shows both direction and magnitude of each feature's effect across all test samples
3. **Dependence plot for flu_vaccine** — shows how the vaccine's SHAP contribution varies with other features (especially `is_senior`)

---

## 10. The Vaccine Hypothesis in Depth

The biological rationale for influenza vaccine protecting against TIA:

1. **Acute infection triggers inflammation.** Influenza causes systemic inflammatory responses (elevated CRP, fibrinogen, IL-6) that increase blood viscosity and promote platelet aggregation.
2. **Inflammation destabilizes plaques.** In patients with pre-existing atherosclerosis, acute infection can rupture vulnerable plaques in carotid or cerebral arteries, triggering thrombus formation.
3. **Vaccination reduces infection frequency and severity.** By preventing or attenuating influenza episodes, the vaccine removes a trigger for acute cardiovascular events.
4. **Seasonal patterns support the hypothesis.** TIA and stroke incidence peak in winter months, coinciding with influenza season.

**What the model tests:**
The `flu_vaccine` feature and the `flu_x_senior` interaction term quantify whether vaccinated individuals — especially those 65 and older, who have higher cardiovascular baseline risk — have lower predicted TIA probability after controlling for all other risk factors.

A negative SHAP value for `flu_vaccine = 1` supports a protective effect. A stronger negative SHAP in seniors than in younger adults supports the age-dependent protective mechanism.

**Caveats:**
NHANES is cross-sectional, not longitudinal. The survey asks "did you receive a flu shot in the past 12 months?" and "have you ever had a stroke/TIA?" These are not temporally ordered — the stroke may have occurred before the most recent vaccine. The model can identify correlations but **cannot establish causation**. A prospective cohort or randomized trial would be needed to confirm the effect.

---

## 11. Challenges and Limitations

### Data Challenges

**Cross-sectional design**
NHANES captures a snapshot in time. The TIA outcome (`MCQ160F`) asks "have you ever had a stroke?" — it does not record when the stroke occurred or whether vaccination preceded or followed it. This conflation of timing limits causal inference. A longitudinal dataset with event dates (e.g., Medicare claims linked to vaccination records) would be more appropriate.

**Small positive class**
Even with three cycles pooled (~35,000 respondents), fewer than 700 TIA-positive cases are available after filtering. SMOTE and class weighting help, but models trained on small positive classes are vulnerable to overfitting to the specific demographic characteristics of that minority sample.

**Self-reported outcomes**
"Have you ever had a stroke or TIA?" is answered by the respondent, not derived from a medical record. Recall bias, misdiagnosis (many TIAs go unrecognized), and social desirability effects all introduce noise in the target variable. Some respondents may report a TIA they did not have; many more may fail to report one they did.

**Module non-response**
Not all NHANES participants complete all questionnaire modules. Immunization module completion rates vary by cycle. Participants who skip the IMQ module have unknown vaccination status, not "unvaccinated" status. Treating these as missing and imputing with the median may underestimate vaccine coverage.

**NHANES cycle restructuring**
The pre-pandemic cycle P renamed files (e.g., `DEMO_P.XPT` → `P_DEMO.xpt`) and changed some variable names. The ALQ (alcohol) module used different column names in cycle I (`ALQ130`) vs. later cycles (`ALQ121`). The blood pressure exam module was renamed from `BPX` to `BPXO` in cycle P with different column names. These inconsistencies require careful harmonization and mean that some features have missingness patterns driven by cycle rather than by the respondent.

### Model-Specific Challenges

**Logistic Regression**
Assumes a linear decision boundary in the feature space. TIA risk likely involves non-linear interactions (e.g., age × hypertension × diabetes) that logistic regression cannot capture without explicit polynomial features. Class imbalance, even with `class_weight='balanced'`, often results in low precision at clinically useful sensitivity levels.

*How to address:* Add polynomial interaction features or use PLR (penalized logistic regression with interaction terms). Calibrate predicted probabilities with Platt scaling.

**Decision Tree**
Single trees are high-variance — small changes in training data produce very different trees. They tend to capture only one or two dominant splits before overfitting. They underperform ensembles on virtually every clinical prediction task.

*How to address:* Use only for communication/visualization; rely on ensemble methods for prediction.

**Random Forest**
Random Forest tends to underperform boosted trees on tabular data with mixed feature importance. It also does not handle missing values natively (requires imputation beforehand). Feature importances can be biased toward high-cardinality continuous features.

*How to address:* Use SHAP importances (which correct for this bias) instead of Gini importances. Consider extra-trees variant for faster training.

**Gradient Boosting (sklearn)**
The sklearn implementation does not parallelize tree construction and is slow on large datasets. It is also more prone to overfitting without careful regularization (max_depth, min_samples_leaf, subsample). It lacks native missing value handling.

*How to address:* Use XGBoost or LightGBM for production. They are faster, better regularized, and handle missing values natively.

**XGBoost**
Can overfit when `n_estimators` is large without early stopping. The `scale_pos_weight` parameter helps with imbalance but does not interact with SMOTE-augmented training data in an obvious way — the effective class ratio changes when SMOTE has been applied.

*How to address:* Use early stopping on a validation set. When using SMOTE, set `scale_pos_weight=1` (balanced after oversampling) or skip SMOTE and rely solely on `scale_pos_weight`.

**LightGBM**
Leaf-wise growth can create very deep, imbalanced trees that overfit on small datasets. The `num_leaves` parameter (not `max_depth`) controls model complexity, which surprises practitioners familiar with other frameworks.

*How to address:* Set `num_leaves` conservatively (≤ 63 for a dataset of this size). Use `min_child_samples` to prevent splits on too few training examples.

**PyTorch MLP**
Neural networks typically require far more data than 25,000 samples to outperform well-tuned gradient boosting on tabular data. Batch normalization requires batch sizes large enough to compute stable statistics — with 256 samples per batch and ~17,500 training examples, this is borderline. The MLP is also sensitive to learning rate schedules and can fail to converge or converge to poor local minima in individual runs.

*How to address:* On this dataset size, treat the MLP as an experimental comparison rather than the production model. For genuinely larger datasets (500k+ samples), neural networks become more competitive. TabNet or FT-Transformer architectures are specifically designed for tabular data.

### Evaluation Challenges

**Threshold sensitivity**
The choice of decision threshold dramatically affects the sensitivity/specificity tradeoff. Youden's J is one approach, but it weights sensitivity and specificity equally. If clinical protocol requires ≥ 80% sensitivity (to not miss more than 1 in 5 TIA patients), the threshold should be tuned to that constraint directly.

**Calibration**
A model may rank patients correctly (high ROC-AUC) without assigning accurate absolute probabilities. If a model predicts 15% TIA risk for a patient, it should ideally be right about 15% of the time for that score. Uncalibrated models are dangerous in clinical use. Calibration is not assessed in the current notebook.

---

## 12. How to Run

### Environment Setup

The notebook is designed to run from the conda base environment on Windows:

```powershell
# Activate base conda environment (already active in Anaconda Prompt)
conda activate base

# Start Jupyter
jupyter notebook tia_influenza_vaccine_predictor.ipynb
```

### Execution Order

Run cells top to bottom. The first cell installs any missing packages. Cells are designed to be idempotent — re-running the download cell uses cached files from `data/nhanes/` and does not re-download.

**Expected runtimes (CPU):**

| Section | Approximate Time |
|---------|----------------|
| Data download (first run) | 5–10 min |
| Data download (cached) | < 5 sec |
| Feature engineering + EDA | < 1 min |
| Base model training (6 models) | 2–5 min |
| MLP training (80 epochs) | 5–15 min |
| Optuna optimization (120 trials) | 20–40 min |
| SHAP computation | 3–10 min |

Total first-run time on CPU: approximately 45–75 minutes. With a CUDA GPU, MLP training and SHAP KernelExplainer calls are significantly faster.

### Outputs

| File | Contents |
|------|---------|
| `data/nhanes/*.XPT` | Cached NHANES survey files |
| `eda_overview.png` | Exploratory data analysis plots |
| `model_comparison.png` | ROC/PR/metric comparison charts |
| `test_set_evaluation.png` | Final model confusion matrix and ROC |
| `shap_summary.png` | Feature importance and beeswarm plots |
| `flu_vaccine_effect.png` | Vaccine SHAP dependence plots |
| `tia_prediction_model.pkl` | Saved model, imputer, scaler, threshold |

---

## 13. Practical Applications

### Clinical Decision Support
The saved pickle artifact exposes a `predict_tia_risk(patient_dict)` function that takes a dictionary of patient features and returns:
- `risk_score`: probability of TIA history (0–1)
- `high_risk`: Boolean flag based on the tuned threshold
- `risk_label`: "HIGH" or "LOW"

This can be embedded in an EHR as a background scoring service that flags patients for follow-up without requiring physician input at the point of care.

### Population Health Management
Insurance companies and health systems can apply the model to their member/patient population to identify and proactively contact high-risk individuals — scheduling wellness visits, ensuring vaccination, or escalating blood pressure management.

### Vaccine Policy Justification
If the influenza vaccine signal is stable across model types (consistent negative SHAP values for `flu_vaccine = 1`) and strong in seniors, this supports expanding vaccine access and coverage in the 65+ population as a cardiovascular preventive measure — not just for respiratory protection.

### Research Hypothesis Generation
The SHAP dependence plots reveal which features interact with vaccine status. Unexpected interactions (e.g., the vaccine effect being stronger in diabetics, or absent in current smokers) generate new hypotheses for confirmatory studies.

### Baseline for Claims-Linked Studies
This model can serve as a baseline comparison for a richer study that links NHANES data with Medicare claims, which would provide actual TIA event dates, medication histories, and repeated vaccination records — enabling proper longitudinal analysis.

---

## 14. How to Improve from Here

### Data Improvements

**Use longitudinal data with event timestamps**
The fundamental limitation of this analysis is the cross-sectional design. The strongest improvement would be replacing NHANES with a dataset where TIA events are recorded with dates — such as Medicare claims or a hospital EHR cohort — enabling survival analysis and proper temporal ordering of predictors and outcomes.

**Link to vaccination registries**
Rather than relying on the single question "did you get a flu shot in the past 12 months?", link to state immunization registries to obtain full vaccination histories including year, timing, and vaccine type. This enables dose-response analysis (consistent multi-year vaccination vs. single-year).

**Add medication features**
Anticoagulants (warfarin, apixaban), statins, and antihypertensives are major confounders that NHANES partially captures but that claims data captures completely. Including medication status would substantially improve model accuracy and reduce confounding.

**Expand to additional NHANES cycles**
Cycles 2009–2010 through 2013–2014 also contain stroke outcomes and immunization data. Including them would approximately double the dataset size and the number of positive cases.

### Modeling Improvements

**Survival analysis**
For longitudinal data, replace binary classification with a Cox proportional hazards model or a discrete-time hazard model. These approaches correctly handle right-censoring (participants who have not yet had a TIA) and produce time-to-event predictions rather than binary risk labels.

**Calibration**
Add probability calibration after model training using `CalibratedClassifierCV` with isotonic regression or Platt scaling. This makes predicted probabilities interpretable as true population-level risk estimates.

**Propensity score methods**
To move closer to causal inference on the vaccine effect, use inverse probability of treatment weighting (IPTW) with a propensity score model for vaccination. This partially adjusts for the non-random nature of who gets vaccinated, reducing confounding.

**Ensemble/stacking**
Combine predictions from multiple models using a meta-learner (logistic regression on the held-out validation predictions of each base model). Stacking typically outperforms any individual model and is particularly effective when base models have complementary error patterns.

**Uncertainty quantification**
Add conformal prediction intervals or Monte Carlo dropout (in the MLP) to provide prediction intervals alongside point estimates. A high-risk prediction with wide uncertainty is clinically different from a high-risk prediction with narrow uncertainty.

**Fairness analysis**
Evaluate model performance separately by race/ethnicity, sex, and income group. Clinical ML models frequently perform worse for underrepresented subgroups, which can exacerbate existing health disparities. Fairness metrics (equalized odds, calibration by subgroup) should be part of any deployment evaluation.

### Infrastructure Improvements

**Unit tests**
Add `pytest` tests covering: correct handling of missing BPQ/DIQ/SMQ modules, consistent feature column ordering before prediction, SMOTE applied only to training data, and threshold loading from saved artifact.

**Automated retraining pipeline**
NHANES releases new two-year cycles periodically. An automated pipeline could detect new cycle releases, download and harmonize the new files, retrain the model, compare performance to the previous version, and promote the new model if it improves PR-AUC above a threshold.

**REST API wrapper**
Wrap `predict_tia_risk()` in a FastAPI endpoint with input validation, logging, and monitoring. This makes the model consumable by EHR systems without requiring Python expertise.

---

## Summary Table

| Dimension | Detail |
|-----------|--------|
| **Target** | First occurrence of stroke/TIA (MCQ160F), binary |
| **Data** | NHANES 2015–16, 2017–18, 2017–20 pooled; ~35,000 respondents |
| **Positive class rate** | ~1–3% (severe imbalance) |
| **Features** | 29 engineered features across demographics, vitals, comorbidities, lifestyle, labs |
| **Key hypothesis** | Influenza vaccination reduces TIA risk, especially in adults 65+ |
| **Models** | LR · DT · RF · GBM · XGBoost · LightGBM · PyTorch MLP |
| **Imbalance strategy** | SMOTE (train only) + class weights + threshold calibration |
| **Primary metric** | PR-AUC (Precision-Recall AUC) |
| **Threshold selection** | Youden's J on validation ROC |
| **Interpretability** | SHAP TreeExplainer / LinearExplainer / KernelExplainer |
| **Tuning** | Optuna TPE, 60 trials each for XGBoost and LightGBM |
| **Artifact** | `tia_prediction_model.pkl` — model + imputer + scaler + threshold |
| **Core limitation** | Cross-sectional design prevents causal inference on vaccine effect |
| **Next step** | Longitudinal claims data + survival analysis + propensity weighting |

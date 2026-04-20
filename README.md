# Forensic-Integrated ML Pipeline for Actuarial Technical Pricing
### VERITAS-RISK  | French Motor Third-Party Liability (MTPL) Pricing Engine

**Author:** Victor Sunarko | **Domain:** Actuarial Science and Financial Forensics

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Polars](https://img.shields.io/badge/Polars-1.39.0-orange)](https://pola.rs)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1.4-green)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0-green)](https://lightgbm.readthedocs.io)
[![CatBoost](https://img.shields.io/badge/CatBoost-Latest-yellow)](https://catboost.ai)
[![Optuna](https://img.shields.io/badge/Optuna-4.7.0-purple)](https://optuna.org)

---

## A Note on Transparency

This project was built by Victor Sunarko with (partial) assistance from Claude (by Anthropic) as a large language model (LLM) collaborator. Claude helped with code structure, debugging specific pipeline issues, and documentation formatting at various points during development. All actuarial methodology decisions, the Forensic-First philosophy, the architectural choices between modeling approaches, the interpretation of results, and the direction of the entire project were driven by the author. I am sharing this because I believe honesty about the use of AI tools in work is the right posture, and because knowing how to effectively direct and collaborate with AI systems is itself a relevant and increasingly valued skill in modern data science.

---

## Table of Contents

- [What This Project Is](#what-this-project-is)
- [Why This Project Is Unusual](#why-this-project-is-unusual)
- [The Story of the Data](#the-story-of-the-data)
- [Project Architecture](#project-architecture)
- [Pipeline Walkthrough](#pipeline-walkthrough)
- [Key Results and Performance](#key-results-and-performance)
- [Conclusions and Suggestions](#conclusions-and-suggestions)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)

---

## What This Project Is

VERITAS-RISK is an end-to-end, production-hardened actuarial pricing engine for French Motor Third-Party Liability (MTPL) insurance. It is built around a **Forensic-First** philosophy: before any pricing model is trained, the integrity of the underlying data is interrogated using Benford's Law profiling and two unsupervised anomaly detection methods. Anomalous records are not discarded. Instead, they receive a quantified **Forensic Risk Score** that enters the pricing engine as a predictive feature, implementing an implicit uncertainty loading mechanism at the individual policy level.

The pipeline decomposes the pure premium into its two actuarial components. Claim frequency (how often a policyholder claims) is modeled first on the full portfolio using an ensemble of five models. Claim severity (how costly each claim is) is modeled separately on the claimants-only subset using an ensemble of four models. The two predictions are multiplied at the policy level to produce the technical pure premium. This is the two-stage frequency-severity architecture that is standard in institutional actuarial pricing but rarely seen in data science portfolios, where Tweedie regression is often used as a shortcut that sacrifices interpretability.

The project is not a tutorial or an academic exercise. It is designed to replicate the kind of work a pricing actuary or risk data scientist would produce in a commercial insurance context, including forensic data validation, regulatory compliance through monotonic constraints, SHAP-based reason codes, counterfactual fairness analysis, A/E calibration, loss ratio projection, bootstrap confidence intervals, model serialization, latency benchmarking, and a production inference prototype.

> *"A forecast is a guess, but a Technical Premium is a calculation. Veritas-Risk provides the calculation with a forensic audit trail."*

---

## Why This Project Is Unusual

Most actuarial or insurance ML projects in the public domain fall into one of two categories. The first is a tutorial that fits a single GLM or gradient boosting model on the French MTPL dataset and reports a Gini coefficient. The second is a purely academic exercise that uses advanced distributional assumptions but produces nothing that a business could actually use.

This project deliberately occupies a third space. Below is a comparison of what is typically done versus what VERITAS-RISK does.

| Dimension | Typical Project | VERITAS-RISK |
|---|---|---|
| Data integrity | Skip directly to modeling | Benford's Law + Isolation Forest + LOF before any model |
| Anomaly handling | Drop or ignore | Quantify as Forensic Risk Score and price the anomaly |
| Frequency model | Single Poisson GLM | 5-model ensemble with formal overdispersion test |
| Severity model | Ignored or Tweedie shortcut | 4-model ensemble on claimant subset with Gamma GLM baseline |
| Monotonic constraints | Not enforced | Native enforcement in CatBoost, XGBoost, LightGBM, plus post-processing |
| Evaluation metrics | Only Gini | Gini, Poisson/Gamma Deviance, D-Squared, RMSE, MAE, A/E ratio |
| Explainability | None or SHAP only | SHAP, partial dependence profiles, counterfactual fairness analysis |
| Business output | Model only | Loss ratio by region, profitability by risk decile, rate adequacy flags |
| Statistical confidence | Single point estimate | 200-resample bootstrap with 95% confidence intervals |
| Production readiness | None | Latency benchmarking, 17 pipeline assertions, serialized artifacts, inference prototype |
| Regulatory compliance | Not considered | Monotonic enforcement, full audit trail, anomaly justification |

The Forensic-First framing is the most distinctive element. It addresses a real problem in insurance data science: the data used to train pricing models is rarely clean, and ignoring data quality at the modeling stage quietly corrupts the premium for every policyholder in the book.

---

## The Story of the Data

### What Is French MTPL Insurance?

Motor Third-Party Liability (MTPL) insurance is compulsory in France and most of Europe. Every vehicle owner is legally required to hold a policy that covers damage or injury caused to third parties in an accident. Because it is compulsory, the French MTPL market produces one of the largest and most statistically representative insurance datasets in the world, covering a genuinely random cross-section of the driving population rather than a self-selected group of voluntary buyers.

The French MTPL dataset used in this project is maintained on OpenML and is widely regarded as the benchmark dataset for actuarial pricing research. It has been used in academic publications, actuarial working party papers, and industry technical standards. The global actuarial community has collectively adopted this dataset as the reference point against which new pricing methods are evaluated, which makes the Gini coefficients and model comparisons produced here directly comparable to published benchmarks in the actuarial literature.

### How the Data Was Collected

The dataset was originally prepared by actuaries from the French insurance industry and published through a collaboration with academic researchers. It covers a single observation year of French motor insurance policies. Each row represents one policy with its associated rating factors, exposure (the fraction of a year the policy was active), and realized claim experience.

Data was collected through standard insurance operations: policyholders provided vehicle and driver information at the time of quoting, which became the rating factors. Claim events were recorded by claims handlers when notified by policyholders. The two datasets, frequency and severity, are stored separately because they originate from different operational systems (the policy administration system and the claims management system respectively), and the join key (policy ID) must be handled carefully to avoid silent data loss.

### The Two-Dataset Structure

**Frequency Dataset (OpenML ID 41214):** One row per policy. The key target variable is `ClaimNb`, the count of claims made during the policy period. Also contains `Exposure`, the fraction of a year the policy was active, which is used as an offset in all frequency models to normalize predictions to a per-policy-year basis.

**Severity Dataset (OpenML ID 41215):** One row per claim event. Multiple rows can exist for the same policy if multiple claims occurred during the period. The key target variable is `ClaimAmount`, the total cost of each individual claim in EUR.

The merge of these two datasets requires engineering care. Policy IDs were stored as floating-point numbers in the raw OpenML data, meaning that the integer 1 was represented as 1.0 in one dataset and potentially differently formatted in another. Without explicit normalization through Int64 before string conversion, the join produces silent null values for all severity amounts, causing ClaimAmount to read as zero for every policy even though the data exists in the severity file. This was a critical bug in the first version of this pipeline and was resolved by normalizing both datasets through `cast(pl.Int64).cast(pl.Utf8)` before joining, with an assertion that the total ClaimAmount post-join exceeds EUR 50 million (the known portfolio total).

### What the Data Looks Like After Joining

After the precision merge, the combined dataset contains 678,013 policies with 14 columns and zero null values across all fields.

| Actuarial Summary | Value |
|---|---|
| Total policies | 678,013 |
| Claimants (ClaimNb >= 1) | 34,060 (5.02%) |
| Claimants (ClaimAmount > 0) | 24,944 |
| Total exposure | 358,499 policy years |
| Total claims | 36,102 |
| Total claim amount | EUR 59,909,224 |
| Mean claim severity | EUR 2,401.75 |
| Median claim severity | EUR 1,172.00 |
| P95 claim severity | EUR 5,026.20 |
| P99 claim severity | EUR 18,278.38 |

The 95% zero-inflation in claim counts (95% of policies made no claim in the observation year) is the primary reason a single Tweedie regression is actuarially inferior to the two-stage approach. Compressing frequency and severity into one model forces the algorithm to simultaneously explain why most policies have zero cost and why the non-zero costs span six orders of magnitude, which produces suboptimal predictions on both dimensions.

The claim severity distribution is extremely heavy-tailed. The P99 of EUR 18,278 is 7.6 times the mean of EUR 2,402. The top 10% of claimants account for 60.2% of total claim cost, measured by a cost concentration Gini of 0.647. This extreme concentration means that severity modeling is commercially critical: a model that misprices the tail of the severity distribution is financially dangerous regardless of how well it handles the median claim.

### The Rating Factors

All rating factors are observable at policy inception, before any claim occurs. This is the fundamental requirement for a pricing model: it must use only information available at the time the premium is quoted.

| Feature | Type | Description |
|---|---|---|
| BonusMalus | Numeric | Claims history score. Minimum 50 (maximum bonus). Increases by approximately 25% per at-fault claim |
| VehAge | Numeric | Age of the insured vehicle in years |
| DrivAge | Numeric | Age of the primary driver in years |
| VehPower | Numeric | Engine power rating (ordinal integer categories) |
| Density | Numeric | Population density of the driver's residential area per km squared |
| Area | Categorical | Geographic zone A through F (rural to dense urban) |
| VehBrand | Categorical | Vehicle brand code (B1 through B14) |
| VehGas | Categorical | Fuel type (Diesel or Regular) |
| Region | Categorical | Administrative region of France (21 distinct regions) |

### Why BonusMalus Is the Most Important Variable

The BonusMalus (bonus-malus) system is the French actuarial mechanism for adjusting premiums based on individual claims history. A policyholder who makes no at-fault claims for several consecutive years accumulates bonus points and eventually reaches the minimum score of 50, representing the maximum available discount. A policyholder who makes an at-fault claim sees their score increase by approximately 25%, and this malus persists for several years before reverting.

This system means that BonusMalus encodes years of individual driving behavior in a single number. A score of 50 represents a long record of claim-free driving. A score of 150 or higher represents a history of repeated at-fault claims. The SHAP analysis confirms this: BonusMalus has a mean absolute SHAP value of 0.336 log-frequency units, more than twice the contribution of the next most important feature (VehAge at 0.165).

The BonusMalus empirical frequency curve shows a clear increasing trend from approximately 0.05 claims per exposure year at BM=50 to approximately 0.30 at BM=100, confirming it is a highly predictive and actuarially valid pricing factor. Because of this regulatory importance, a monotonic constraint is enforced in all three gradient boosting models: higher BonusMalus scores must never produce lower predicted frequencies. This is both actuarially correct and legally required under European insurance regulatory frameworks.

---

## Project Architecture

```
Part I    Data Ingestion and Schema Validation
Part II   Forensic Integrity Gate (Benford + Isolation Forest + LOF)
Part III  Extended Forensic EDA and Feature Engineering
Part IV   Two-Stage Pricing Engine (Frequency and Severity Modeling)
Part V    Ensemble Architecture and Optuna Optimization (700 trials per study)
Part VI   Regulatory and Interpretability Suite (SHAP, Lorenz, Fairness)
Part VII  Business Analytics Suite (Profitability, Loss Ratio, Bootstrap)
Part VIII Production Hardening and Actuarial Compliance
```

**Technology choices and rationale:**

| Component | Technology | Reason |
|---|---|---|
| Data processing | Polars | Multithreaded, Arrow-native, avoids Pandas memory overhead on 678k rows |
| Statistical baseline | Statsmodels GLM | Actuarially standard, interpretable coefficients with formal p-values |
| Gradient boosting | CatBoost, XGBoost, LightGBM | Each has a distinct inductive bias; ensemble of three outperforms any single model |
| Hyperparameter search | Optuna (TPE + MedianPruner) | Bayesian search outperforms grid search; pruner eliminates unpromising trials early |
| Explainability | SHAP TreeExplainer | Exact Shapley values for tree ensembles; industry standard for regulatory filings |
| Anomaly detection | Isolation Forest + LOF | Global outliers (IF) and local contextual outliers (LOF) are complementary methods |
| Forensic audit | Benford's Law (Nigrini 2012) | Established forensic accounting standard with chi-squared and MAD quantification |

---

## Pipeline Walkthrough

### Part I: Data Ingestion

The frequency and severity datasets are fetched from OpenML and merged via a left join on the normalized policy ID. After the merge, all 678,013 policies have zero null values across all 14 columns. Exposure is confirmed within the range [0.0027, 2.01], consistent with 6-month and 12-month French MTPL policy terms. Memory usage is 40.68 MB in Polars Float32 format.

Two engineered features are created:
- `Freq_Target`: ClaimNb divided by Exposure (annualized claim frequency rate per policy)
- `Forensic_Risk_Score`: computed in Part II and injected back into the master dataset before any modeling

### Part II: Forensic Integrity Gate

#### Benford's Law Profiling

Benford's Law states that in naturally occurring numerical data, the first significant digit d appears with probability log10(1 + 1/d). This produces a distribution where roughly 30% of first digits are 1, 18% are 2, and so on down to approximately 5% being 9. Deviations from this distribution in financial data are a classical forensic indicator of manual data intervention, systematic rounding, or administrative encoding artifacts.

The Mean Absolute Deviation (MAD) is the standard conformity metric from Nigrini (2012): below 0.006 is close conformity, below 0.012 is acceptable, below 0.015 is marginal, and above 0.015 is non-conformity requiring investigation.

| Field | MAD | Chi-squared | Verdict | Structural Explanation |
|---|---|---|---|---|
| ClaimAmount | 0.0619 | 9,486.5 | NON-CONFORMITY | Small claims concentrated in EUR 100-999 range; digit 1 inflated |
| BonusMalus | 0.1464 | 3,535,132.9 | NON-CONFORMITY | Administrative system; digit 5 dominates (BM=50-59 is the majority) |
| VehPower | 0.1123 | 955,699.6 | NON-CONFORMITY | Discrete power categories produce near-uniform digit distribution |

All three NON-CONFORMITY findings are structurally explicable and do not require record exclusion. They are documented and contextualized, as Nigrini's framework requires, rather than used as grounds for data removal.

#### Unsupervised Anomaly Detection

Two complementary methods are applied using only the nine vehicle and driver characteristics observable at policy inception. Claim-derived targets (ClaimNb, ClaimAmount, Freq_Target) are strictly excluded from the forensic feature set to prevent information leakage from realized claim outcomes into the anomaly score.

**Forensic feature set:** `BonusMalus, VehAge, DrivAge, VehPower, Density, Area_enc, VehBrand_enc, VehGas_enc, Region_enc`

| Method | Anomalies Detected | Rate |
|---|---|---|
| Isolation Forest (300 trees, 5% contamination) | 33,901 | 5.00% |
| Local Outlier Factor (k=20, novelty mode) | 33,975 | 5.01% |
| Combined High and Critical tier | 67,802 | 10.00% |

The Forensic Risk Score is the equal-weight average of both normalized anomaly scores, bounded to [0, 1]:

| Tier | Threshold | Policies | Mean ClaimNb | Mean ClaimAmt |
|---|---|---|---|---|
| Low | Below P75 | 508,509 | 0.0495 | EUR 66.39 |
| Medium | P75 to P90 | 101,702 | 0.0600 | EUR 170.19 |
| High | P90 to P97 | 47,461 | 0.0689 | EUR 140.96 |
| Critical | Above P97 | 20,341 | 0.0769 | EUR 105.81 |

The monotonically increasing Mean ClaimNb across tiers (0.0495 to 0.0769) confirms the Forensic Risk Score has genuine predictive content. Anomalous profiles do exhibit higher realized claim frequency.

**Why anomalous records are not removed:** Removing the 10% flagged records would introduce Data Pruning Bias. The removed policies are disproportionately concentrated in extreme BonusMalus and high-density zones. A model trained without them would systematically underprice the highest-risk policyholders. By injecting the Forensic Risk Score as a pricing feature, the model implements an Uncertainty Loading mechanism at the individual policy level. The SHAP analysis in Part VI confirms the score is the 8th most important feature (mean absolute SHAP = 0.034 log-frequency units), proving it contributes genuine pricing information rather than noise.

### Part III: Extended EDA

The extended EDA examines actuarially meaningful relationships beyond standard distribution plots:

**BonusMalus empirical frequency:** The scatter plot of observed claim frequency by BonusMalus bin (sized by policy count) shows a clear increasing trend and confirms the monotonic constraint is actuarially justified. The large bubble at BM=50 reveals that the vast majority of the French MTPL portfolio is at the maximum bonus, meaning most policyholders are in the lowest-risk tier.

**Age risk profiles:** Young drivers (18-25) exhibit empirical claim frequency of approximately 0.22 per exposure year, declining to approximately 0.09 by age 30. This confirms the actuarial validity of the young driver surcharge visible in the fairness analysis. Vehicle Age shows a spike for brand-new vehicles, reflecting adverse selection: buyers of new, high-value vehicles have higher claim frequency in the first policy year.

**Severity distribution:** The normal QQ-plot of log(ClaimAmount) shows reasonable log-normality in the central region with heavy upper-tail deviation, confirming Gamma GLM suitability. The claim cost Lorenz curve (cost concentration Gini: 0.647) shows the top 10% of claimants account for 60.2% of total claim costs, which makes severity tail modeling commercially critical.

**Correlation structure:** The strong negative correlation between DrivAge and BonusMalus (-0.48) is a structural feature of the French MTPL system (older drivers have more years of bonus accumulation). The Forensic_Risk_Score shows positive correlations with BonusMalus (0.48) and Density (0.56), confirming the anomaly algorithm correctly identifies extreme-score, urban-density profiles as unusual relative to the broader population.

### Part IV: Two-Stage Pricing Engine

#### Overdispersion Test

The dispersion ratio (variance / mean of ClaimNb in the training set) is 1.0876, below the 1.10 threshold. No significant overdispersion is detected. The Poisson GLM is the primary statistical baseline. The Negative Binomial GLM is fitted alongside for completeness.

#### Stage 1: Frequency Modeling (678,013 policies)

All three gradient boosting models include `monotone_constraints` enforced at the tree-split level for BonusMalus, ensuring regulatory compliance is built into the model structure rather than applied after the fact.

**Key GLM findings:** The Poisson GLM Forensic_Risk_Score coefficient is +0.6946 (z=5.58, p<0.001), confirming the score is a statistically significant predictor of claim frequency with formal inferential validity.

**Frequency leaderboard (test set):**

| Model | Normalized Gini | Poisson Deviance | D-Squared | RMSE | MAE |
|---|---|---|---|---|---|
| XGBoost | 0.3141 | 0.3062 | 0.0465 | 0.2357 | 0.0966 |
| CatBoost | 0.3086 | 0.3074 | 0.0428 | 0.2362 | 0.0966 |
| LightGBM | 0.2762 | 0.3508 | -0.0922 | 0.2640 | 0.1493 |
| Poisson GLM | 0.2292 | 0.3216 | -0.0013 | 0.2372 | 0.0988 |
| NegBinom GLM | 0.2291 | 0.3216 | -0.0014 | 0.2373 | 0.0994 |

**Generalization gap (train vs. test deviance):**

| Model | Gap | Assessment |
|---|---|---|
| Poisson GLM | +0.30% | Near-zero (expected for parametric model) |
| NegBinom GLM | +0.30% | Near-zero |
| CatBoost | +2.75% | Healthy, well-controlled |
| XGBoost | +4.12% | Acceptable, within normal range |
| LightGBM | +6.74% | Largest gap; benefits from ensemble averaging |

#### Stage 2: Severity Modeling (24,943 claimant policies)

The strict ClaimAmount > 1.0 filter eliminates zero-amount records that would cause Gamma deviance = infinity (from log(y_true / y_pred) diverging at zero). This was a critical fix from v1 of the pipeline.

**Severity leaderboard (claimant test set):**

| Model | Normalized Gini | Gamma Deviance | D-Squared | RMSE (log) |
|---|---|---|---|---|
| XGBoost (reg:gamma) | 0.3178 | 2.5115 | -0.0753 | 1.2447 |
| LightGBM | 0.2480 | 4.2167 | -0.8053 | 1.1233 |
| CatBoost | 0.2161 | 4.1971 | -0.7970 | 1.1219 |
| Gamma GLM | 0.0745 | 2.5121 | -0.0755 | 1.3809 |

#### Tweedie Challenger

A Tweedie XGBoost model (variance power 1.5) is fitted on the full dataset as a combined frequency-severity benchmark. It achieves a Gini of 0.2799 on the pure premium target, confirming that the two-stage decomposition (Gini 0.3176) preserves substantially more actuarial signal than the compressed Tweedie formulation.

### Part V: Ensemble Optimization

#### Optuna Configuration

| Setting | Value |
|---|---|
| Sampler | Tree-structured Parzen Estimator (TPE) |
| Pruner | MedianPruner (n_startup_trials=30) |
| Frequency trials | 700 |
| Severity trials | 700 |
| Tweedie blend trials | 200 |
| Objective | Maximize normalized Gini coefficient |
| Weight constraint | Softmax normalization (all weights sum to 1.0) |

Both optimization studies converge within approximately 50-75 trials. The remaining trials confirm stability without finding improvements, demonstrating the search space is fully explored.

#### Optimal Ensemble Weights

**Frequency ensemble (Gini: 0.3176):**

| Model | Weight |
|---|---|
| Poisson GLM | 10.30% |
| NegBinom GLM | 10.30% |
| CatBoost | 27.99% |
| XGBoost | 27.40% |
| LightGBM | 24.00% |

The GLMs receive a combined 20.6% weight. Optuna's finding that including the GLMs improves ensemble Gini reflects the complementary regularization behavior of linear and non-linear models: the GLMs smooth out high-variance gradient boosting predictions in sparse data regions.

**Severity ensemble (Gini on claimant subset: 0.3090):**

| Model | Weight |
|---|---|
| Gamma GLM | 14.15% |
| CatBoost | 33.33% |
| XGBoost | 38.39% |
| LightGBM | 14.14% |

**Tweedie blend weight:** 40% (the Tweedie challenger receives 40% in the final pure premium blend, regularizing the two-stage product).

### Part VI: Regulatory and Interpretability Suite

#### Lorenz Curves and Normalized Gini

The Lorenz curve plots the cumulative share of policies (sorted ascending by predicted risk) on the x-axis against the cumulative share of actual claim costs on the y-axis. A random model follows the 45-degree diagonal. A model with selection power curves above it.

The VERITAS-RISK ensemble Lorenz curve lies above all individual constituent models across the full distribution, confirming that ensemble stacking adds genuine selection power. At the 80th cumulative percentile, the ensemble has captured approximately 67% of total claim costs versus approximately 60% for the Poisson baseline. This 7-percentage-point difference represents commercially meaningful separation: a portfolio manager using the ensemble premium can write a materially less adverse book of business at any given volume target.

#### SHAP Feature Importance

| Rank | Feature | Mean SHAP | Implication |
|---|---|---|---|
| 1 | BonusMalus | 0.3360 | Claims history dominates all other rating factors |
| 2 | VehAge | 0.1653 | Adverse selection in new vehicles correctly priced |
| 3 | DrivAge | 0.1620 | Young driver surcharge captured non-linearly |
| 4 | Density | 0.0806 | Urban exposure risk correctly incorporated |
| 5 | VehPower | 0.0702 | Higher-powered vehicles are surcharged |
| 6 | VehBrand | 0.0601 | Brand-level risk differentiation exists |
| 7 | Region | 0.0484 | Geographic risk beyond urban density |
| 8 | Forensic_Risk_Score | 0.0340 | Anomaly uncertainty loading confirmed active |
| 9 | VehGas | 0.0316 | Diesel vs. Regular fuel type difference |
| 10 | Area | 0.0164 | Area zone partially collinear with Density |

The Forensic_Risk_Score at rank 8 validates the forensic architecture: the anomaly detection pipeline contributes genuine, statistically significant pricing information.

#### Partial Dependence Profiles

The partial dependence curves (holding all other features at population median or mode while varying a single feature) reveal four key actuarial insights:

- **BonusMalus:** Monotonically increasing from approximately 0.04 predicted frequency at BM=50 to 0.12 at BM=100+. The slope accelerates above BM=90, reflecting the non-linear malus structure.
- **VehAge:** Sharp drop from 0.27 at VehAge=0 (brand-new vehicle adverse selection) to approximately 0.04 by VehAge=3, then a slow decline.
- **DrivAge:** High frequency at ages 20-25, plateau through working years, slight uptick at advanced ages.
- **VehPower:** Step-function increases at power ratings 7-9, where the actuarial surcharge is most concentrated.

#### Monotonic Constraint Validation

4 micro-violations (amplitude below 0.005) remain in the CatBoost partial dependence curve despite native constraint enforcement, caused by sparse data in the BM=78-80 range. These are resolved in Part VIII via post-processing.

#### Counterfactual Fairness Analysis

**Area zone premiums:** EUR 72 (Area A, rural) to EUR 130 (Area E, dense urban). The gradient is actuarially justified by documented claim frequency differentials.

**Driver age premiums:** The 18-25 band faces a median premium of approximately EUR 155, roughly 67% above the portfolio median. This is proportionate to their empirical claim frequency of 0.22 versus the portfolio average of 0.09.

**Counterfactual Area delta:** Mean delta of 0.017 with standard deviation 0.030, indicating Area has a modest pricing effect. The narrow delta distribution confirms the model does not over-weight geographic factors relative to individual risk characteristics.

**Gini by zone:** All six Area zones produce Gini coefficients between 0.28 and 0.37 (mean 0.3203), confirming consistent selection power across geographies.

### Part VII: Business Analytics Suite

#### A/E Calibration by Risk Decile

A/E (Actual vs. Expected) calibration is a metric used in actuarial practice to assess whether predicted values are monetarily accurate, not just well-ranked. While Gini measures rank ordering (does the model correctly identify which policies are higher risk than others), the A/E ratio measures absolute accuracy: does the model predict the right number of claims, not just the right order.

The A/E ratio is computed as: **Total Observed Claims / Total Predicted Claims** within each group.

A well-calibrated model produces A/E ratios near 1.0 across all groups. An A/E below 1.0 means the model over-predicts (conservative). An A/E above 1.0 means it under-predicts (dangerous from a reserving perspective). The test portfolio is segmented into 10 risk deciles by predicted frequency, and A/E is reported per decile.

| Decile | Actual Claims | Predicted Claims | A/E Ratio |
|---|---|---|---|
| D1 (lowest risk) | 198 | 253.2 | 0.782 |
| D2 | 319 | 387.0 | 0.824 |
| D3 | 418 | 505.8 | 0.827 |
| D4 | 424 | 622.1 | 0.682 |
| D5 | 537 | 742.3 | 0.723 |
| D6 | 658 | 874.4 | 0.753 |
| D7 | 726 | 1018.0 | 0.713 |
| D8 | 886 | 1182.4 | 0.749 |
| D9 | 1076 | 1494.3 | 0.720 |
| D10 (highest risk) | 1999 | 2961.1 | 0.675 |
| **Overall** | **7241** | **10040.6** | **0.721** |

The overall A/E ratio of 0.721 indicates systematic over-prediction: the ensemble predicts approximately 39% more claims than actually occurred in the test year. This is expected behavior for a Gini-optimized model (see Conclusions). Critically, the A/E ratios are consistent across all 10 deciles (range 0.68 to 0.83, with no directional trend), confirming there is no systematic directional bias in any risk segment. The model over-predicts uniformly, which means the relative risk ordering is correct even if the absolute level requires a calibration adjustment.

#### Profitability Analysis by Risk Decile

| Decile | Avg Predicted Premium | Observed Freq Rate | Predicted Freq Rate | Loss Ratio |
|---|---|---|---|---|
| D1 | EUR 44.14 | 0.1299 | 0.1643 | 1.033 |
| D2 | EUR 62.59 | 0.1141 | 0.1410 | 0.647 |
| D3 | EUR 72.62 | 0.0996 | 0.1094 | 1.375 |
| D4 | EUR 80.37 | 0.0694 | 0.0939 | 0.721 |
| D5 | EUR 87.85 | 0.0705 | 0.0886 | 0.840 |
| D6 | EUR 97.48 | 0.0663 | 0.0896 | 0.678 |
| D7 | EUR 108.66 | 0.0737 | 0.0927 | 0.698 |
| D8 | EUR 121.61 | 0.0784 | 0.0997 | 0.652 |
| D9 | EUR 157.26 | 0.1030 | 0.1281 | 0.650 |
| D10 | EUR 300.39 | 0.2371 | 0.3057 | 0.780 |

The predicted frequency correctly separates policies by risk: D10 has a predicted rate of 0.3057 versus D5's 0.0886, a 3.5x separation that enables meaningful commercial differentiation. The average predicted premium escalates monotonically from EUR 44 in D1 to EUR 300 in D10, demonstrating that the pricing engine is acting as a risk-proportionate tariff.

#### Loss Ratio Projection by Region

| Region | Policies | Observed Loss | Predicted Premium | Loss Ratio | Action |
|---|---|---|---|---|---|
| R21 | 594 | EUR 400,627 | EUR 65,607 | 6.11 | Urgent rate increase |
| R22 | 1,535 | EUR 271,938 | EUR 176,130 | 1.54 | Rate increase required |
| R82 | 16,921 | EUR 2,282,542 | EUR 2,108,779 | 1.08 | Rate review recommended |
| R93 | 16,022 | EUR 1,796,446 | EUR 2,107,540 | 0.85 | Adequate |
| R24 | 31,923 | EUR 2,559,010 | EUR 3,228,080 | 0.79 | Adequate |

R21 is the most dangerous segment: a loss ratio of 6.11 means the insurer receives EUR 1 for every EUR 6.11 of claim cost. The small policy count (594) makes this estimate volatile, but the magnitude of under-pricing signals a structural hazard not fully captured by the current rating model. R82 is the most commercially significant: with 16,921 policies, a 10% rate increase would generate approximately EUR 210,000 of additional annual premium.

#### Bootstrap Stability Analysis

200 bootstrap resamples of the test set quantify the statistical uncertainty around the headline Gini:

| Metric | Point Estimate | 95% Confidence Interval |
|---|---|---|
| VERITAS-RISK Ensemble Gini | 0.3176 | [0.3049, 0.3337] |
| Poisson GLM Baseline Gini | 0.2292 | [0.2146, 0.2440] |
| Gini Lift | 0.0884 | [0.0795, 0.0987] |
| Statistical significance | CONFIRMED | Lower bound never crosses zero |

The two Gini distributions are completely non-overlapping (ensemble range 0.30-0.34, baseline range 0.21-0.25). There is less than a 2.5% probability that the observed Gini improvement is due to sampling variation.

### Part VIII: Production Hardening

#### Monotonic Post-Processing

The 4 residual BonusMalus violations are resolved by applying `np.maximum.accumulate` to sorted BonusMalus risk buckets, guaranteeing 100% regulatory compliance with zero violations. The mean correction factor is 9.27% deviation from the identity, and the Gini adjusts from 0.3176 to 0.3022 (the cost of full monotonic compliance).

#### Actuarial Premium Capping and Flooring

| Metric | Value |
|---|---|
| Training P99.5 cap | EUR 533.91 |
| Floor value | EUR 1.00 |
| Policies trimmed | 987 (0.728% of test set) |
| Max raw premium (pre-trim) | EUR 8,747.54 |
| Max trimmed premium | EUR 533.91 |

#### Global Reproducibility Audit

RANDOM_SEED = 42 is verified across all 13 stochastic components. All 13 checks pass. The pipeline produces identical results on any run from the same starting data.

```
RANDOM_SEED              : 42
Dataset shape            : (678013, 16)
Headline Gini            : 0.302240
Frequency ensemble Gini  : 0.317646
Bootstrap lift CI lower  : 0.079511
All seeds consistent     : True
```

#### Inference Latency Benchmark

| Model | Per Record | SLA Status (50ms threshold) |
|---|---|---|
| Poisson GLM | 0.34 µs | OK (147,000x under SLA) |
| NegBinom GLM | 0.28 µs | OK |
| CatBoost | 3.99 µs | OK |
| XGBoost | 1.40 µs | OK |
| LightGBM | 7.98 µs | OK |
| Full Ensemble | 15.05 µs | OK (3,317x under SLA) |

The full 5-model frequency ensemble completes inference in 15 microseconds per record, making it suitable for real-time quoting systems with extreme concurrency requirements.

#### Pipeline Integrity Assertions

17 automated assertions are verified before any inference. All 17 pass with 0 failures and 0 warnings. Assertions cover schema consistency, exposure positivity, feature null completeness, target integrity, index non-overlap, seed consistency, and Gini sanity bounds.

#### Production Inference Prototype

The `predict_technical_premium(policy_dict)` function accepts a raw policy dictionary, runs it through the full forensic scoring pipeline, computes the ensemble prediction, applies capping and flooring, and returns a structured output containing:

- Predicted claim frequency (annualized rate)
- Predicted claim severity (EUR)
- Pure premium (frequency times severity)
- Trimmed premium (after actuarial capping)
- Forensic Risk Score and tier
- Individual model contributions for audit traceability
- Ensemble weights and inference latency in the audit trail

---

## Key Results and Performance

### Headline KPIs

| KPI | Target | Result | Status |
|---|---|---|---|
| Gini Lift over Poisson Baseline | 5-10% | +38.57% | KPI MET |
| Overdispersion handling | Tested and addressed | Ratio = 1.0876, NB GLM fitted | Confirmed |
| Monotonic integrity (BonusMalus) | 100% compliance | 0 violations post-enforcement | PASS |
| Statistical significance | Lift CI > 0 | CI [0.0795, 0.0987] | CONFIRMED |
| Pipeline integrity | All assertions pass | 17/17 PASS | CONFIRMED |
| Inference SLA | Under 50ms per record | 15.05 µs (3,317x under SLA) | CONFIRMED |

### Final Model Scorecard

**Frequency Models:**

| Model | Gini | Poisson Dev | D-Squared | RMSE | MAE |
|---|---|---|---|---|---|
| VERITAS-RISK Ensemble | **0.3176** | | | | |
| XGBoost | 0.3141 | 0.3062 | 0.0465 | 0.2357 | 0.0966 |
| CatBoost | 0.3086 | 0.3074 | 0.0428 | 0.2362 | 0.0966 |
| LightGBM | 0.2762 | 0.3508 | -0.0922 | 0.2640 | 0.1493 |
| Poisson GLM (Baseline) | 0.2292 | 0.3216 | -0.0013 | 0.2372 | 0.0988 |
| NegBinom GLM | 0.2291 | 0.3216 | -0.0014 | 0.2373 | 0.0994 |

**Severity Models:**

| Model | Gini | Gamma Dev | D-Squared | RMSE (log) |
|---|---|---|---|---|
| VERITAS-RISK Severity Ensemble | **0.3090** | | | |
| XGBoost (reg:gamma) | 0.3178 | 2.5115 | -0.0753 | 1.2447 |
| LightGBM | 0.2480 | 4.2167 | -0.8053 | 1.1233 |
| CatBoost | 0.2161 | 4.1971 | -0.7970 | 1.1219 |
| Gamma GLM (Baseline) | 0.0745 | 2.5121 | -0.0755 | 1.3809 |

---

## Conclusions and Suggestions

### What This Pipeline Demonstrates

VERITAS-RISK demonstrates that a pricing engine can simultaneously be statistically rigorous, actuarially defensible, regulatorily compliant, and computationally efficient. The +38.57% Gini lift over the Poisson GLM baseline is statistically confirmed by bootstrap analysis and is driven primarily by the gradient boosting models' ability to capture non-linear interactions between rating factors that the GLM linearizes away.

The two-stage architecture proves its superiority over the Tweedie shortcut at every level: quantitatively (Gini 0.3176 vs. 0.2799), practically (separate frequency and severity reason codes for regulatory submission), and commercially (profitability analysis can be decomposed into frequency and severity components for targeted rate action).

The Forensic-First approach adds genuine value, confirmed by three independent lines of evidence: the monotonically increasing Mean_ClaimNb across forensic tiers, the statistically significant Forensic_Risk_Score coefficient in the Poisson GLM (z=5.58), and the SHAP rank of 8 with a mean absolute contribution of 0.034 log-frequency units.

### Limitations

**A/E calibration gap:** The overall A/E ratio of 0.721 means the ensemble over-predicts claim counts by approximately 39%. This is a common and expected behavior in Gini-optimized models. The Gini objective rewards correct rank ordering, not correct absolute prediction. A model that perfectly ranks policyholders by risk can still over-predict the total number of claims. Because the A/E ratio is consistent across all 10 risk deciles (no directional trend), the relative risk ordering is correct. In practice, actuaries apply a calibration loading factor (here approximately 0.721) to bridge the gap between model predictions and observed experience without disturbing the Gini coefficient.

**LightGBM underperformance:** LightGBM has the largest generalization gap (6.74%) and the lowest individual Gini (0.2762) among the gradient boosting models. It still contributes 24% weight in the optimal ensemble because it provides modeling diversity that improves ensemble stability, but its individual performance suggests independent LightGBM hyperparameter tuning through Optuna could close this gap.

**Monotonic constraint trade-off:** Enforcing the BonusMalus monotonic constraint post-processing reduces the headline Gini from 0.3176 to 0.3022. This is the cost of regulatory compliance. In a production setting, this trade-off is non-negotiable: a model that violates monotonicity on a mandatory rating factor cannot be filed with a regulatory authority regardless of its Gini coefficient.

**Single-year observation:** The dataset covers one observation year. The heavy-tailed nature of the severity distribution means that a single catastrophic event in a small region (such as R21's loss ratio of 6.11 on 594 policies) can produce extreme statistics that may not persist. Multi-year training data would produce more stable regional loss ratio projections and more reliable severity model estimates.

### Actionable Suggestions

**Rate correction for underpriced regions:** Regions R21, R22, and R82 require immediate rate action. R21 needs investigation into whether a structural hazard (flood zone, high-crime area, unreported road infrastructure issue) explains the 6.11 loss ratio. R22 and R82 require rate increases of at least 54% and 8% respectively based on the observed loss ratios. A pricing actuary would normally set rates at Loss Ratio / Target Loss Ratio to achieve a target combined ratio.

**Apply the A/E calibration loading:** Before deploying ensemble predictions as live premium rates, multiply all frequency predictions by 0.721 to achieve portfolio-level calibration. Re-calibrate this factor annually by comparing predicted versus actual claims on each year's holdout data.

**Monitor Forensic Risk Score drift:** The Isolation Forest and LOF models are fitted on the training population. As the insured portfolio evolves (new vehicle brands enter the market, driver demographics shift, geographic density patterns change), the definition of "anomalous" changes. The forensic pipeline should be re-fitted at minimum annually, with the Forensic Risk Score distribution monitored for population drift relative to the training baseline.

**Extend to multi-year data:** Adding two additional observation years would nearly triple the severity training set from 24,943 to approximately 75,000 claimant observations, which would substantially improve both Gamma GLM and gradient boosting severity model stability. The low D-Squared scores in severity (XGBoost: -0.075) reflect the inherent difficulty of predicting severity from single-year data rather than a model design flaw.

**Productionize the inference function:** The `predict_technical_premium()` function is a working prototype. Wrapping it in a FastAPI endpoint with input validation, response caching, and a Redis-backed rate limiting layer would produce a production-ready quoting API. The 15µs per-record latency leaves more than 49ms of headroom within the 50ms SLA for all overhead, including network round trips, database lookups, and logging.

---

## How to Run

**Prerequisites:** Python 3.10+, Jupyter Notebook or JupyterLab. No GPU required. All computation is CPU-bound and tested on standard multi-core CPU hardware.

```bash
git clone https://github.com/VictorSunarko/Forensic-Integrated-ML-Pipeline-for-Actuarial-Technical-Pricing.git
cd veritas-risk

pip install polars scikit-learn catboost xgboost lightgbm optuna shap \
            statsmodels matplotlib seaborn scipy numpy pandas openml joblib

jupyter notebook Victor_Sunarko_Projects_Forensic-Integrated_ML_Pipeline_for_Actuarial_Technical_Pricing.ipynb
```

Run all 63 cells sequentially from top to bottom without interruption. Variables produced in earlier parts are consumed by later parts; out-of-order execution will produce NameError exceptions. Total runtime on a standard 8-core CPU machine is approximately 2-3 hours, dominated by the two 700-trial Optuna optimization studies.

**Verifying reproducibility:** After a full run, check the pipeline fingerprints printed in Cell 8.5:

```
RANDOM_SEED              : 42
Dataset shape            : (678013, 16)
Headline Gini            : 0.302240
Frequency ensemble Gini  : 0.317646
Bootstrap lift CI lower  : 0.079511
All seeds consistent     : True
```

If these values match, the run is confirmed reproducible.

---

## Dependencies

| Library | Version | Role |
|---|---|---|
| Polars | 1.39.0 | High-performance data processing (CPU-optimized, Arrow-native) |
| NumPy | 1.26.4 | Numerical computing and array operations |
| Pandas | Latest | GLM design matrix preparation and group-by analytics |
| Statsmodels | 0.14.6 | Poisson GLM, Negative Binomial GLM, Gamma GLM with log link |
| Scikit-Learn | Latest | Isolation Forest, LOF, StandardScaler, train_test_split, metrics |
| CatBoost | Latest | Gradient boosting with native categorical encoding and monotone constraints |
| XGBoost | 2.1.4 | Gradient boosting (Poisson, Gamma, Tweedie objectives; monotone constraints) |
| LightGBM | 4.6.0 | Histogram-based gradient boosting with advanced monotone constraint method |
| Optuna | 4.7.0 | Bayesian hyperparameter optimization (TPE sampler, MedianPruner) |
| SHAP | 0.48.0 | Shapley Additive Explanations via TreeExplainer for actuarial reason codes |
| Matplotlib | Latest | All 22 pipeline visualizations |
| Seaborn | Latest | Statistical visualization theme, heatmaps, and distribution plots |
| SciPy | Latest | Statistical tests (chi-squared, QQ-plots, quantile functions) |
| Joblib | Latest | Model serialization and parallel processing utilities |
| OpenML | Latest | Dataset fetching (French MTPL IDs 41214 and 41215) |

---

*Forensic-Integrated ML Pipeline for Actuarial Technical Pricing | VERITAS-RISK | Victor Sunarko*

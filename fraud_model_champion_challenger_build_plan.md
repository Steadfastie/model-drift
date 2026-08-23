# Fraud Model Champion / Challenger PoC

## 1. Goal

The core question is:

> **Can we safely deploy v2?**

The PoC should simulate a production ML workflow where a new fraud model is trained on fresher data, evaluated independently, compared against the current champion, and only promoted when the evidence supports the change.

The project should demonstrate three separate ideas:

```text
                    Can we safely deploy v2?
                              │
             ┌────────────────┼────────────────┐
             │                │                │
        Model quality     Model change     Data change
             │                │                │
      Precision/Recall   v1 vs v2 agreement   Feature drift
      F1 / PR-AUC        score distributions   PSI, etc.
```

The project is therefore closer to **champion/challenger + model monitoring + safe deployment** than to a simple ML benchmark.

---

## 2. High-level PoC architecture

```text
                    Transaction data
                           │
                           ▼
                  ┌─────────────────┐
                  │    Data Prep    │
                  │ time-based split│
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Feature Engine  │
                  └────────┬────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
       Logistic Regression          LightGBM
          baseline                  candidate
              │                         │
              └────────────┬────────────┘
                           ▼
                      Optuna HPO
                           │
                           ▼
                    Model evaluation
                           │
                           ▼
               Champion / Challenger
                           │
                           ▼
             v1 vs v2 comparison on
                 the same holdout
                           │
                           ▼
                    ClearML tracking
```

A production-like monitoring layer can then expose:

```text
                    MODEL MONITOR
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
    Quality          Behaviour         Data drift
        │                │                │
 Precision             Agreement          PSI
 Recall                Disagreements      JSD*
 F1                    Score shift        Volume
 PR-AUC
 ROC-AUC*
```

`*` JSD and score-distribution drift require model scores/probabilities. They are optional because the original dataset contains only boolean labels.

---

## 3. Input data

Initial dataset:

```text
~7.374M transactions
May 21 → Aug 21
```

Example fields:

```text
SENDER_ID
CREATED
PAYOUT_COUNTRY
PAYOUT_CURRENCY
AMOUNT_USD
FEE_USD
STATUS
ANTI_FRAUD_STATUS
COMPLIANCE_STATUS
```

### Label

`ANTI_FRAUD_STATUS` is treated as the target label:

```text
positive = fraud / anti-fraud hit
negative = non-fraudulent
```

Do **not** use `ANTI_FRAUD_STATUS` as a feature.

Do **not** use `COMPLIANCE_STATUS` as an input feature for the initial PoC. It may represent another downstream decision about the same transaction and can introduce leakage or an unrealistic dependency between systems.

Treat compliance as a separate signal/label or omit it from the first version.

### No pre-existing model scores

The dataset contains only binary labels, which is fine. Train a classifier that outputs a probability:

```text
transaction
     │
     ▼
   model
     │
     ├── probability: 0.83
     └── class: positive when probability >= threshold
```

Use `0.5` as the initial classification threshold for simplicity. Threshold tuning can be added later.

---

## 4. Final data distribution

Use chronological splits rather than random train/test splitting. This prevents future information from leaking into earlier training periods and better simulates production.

### v1: Champion

```text
May 21 ───────────────────── Jul 21 │ Jul 22 ─ Jul 28 │ Jul 29 ─ Aug 4
             TRAIN                  │   VALIDATION   │      TEST
```

Purpose:

```text
Train       → learn parameters
Validation  → HPO / model selection
Test        → unbiased v1 evaluation
```

### v2: Challenger

```text
May 28 ───────────────────────── Jul 28 │ Jul 29 ─ Aug 4 │ Aug 5 ─ Aug 21
             TRAIN                      │   VALIDATION  │      TEST
```

This intentionally gives v2 access to newer data and tests the practical question:

> Does retraining with fresher data produce a better candidate?

### Final v1 vs v2 comparison

```text
Aug 5 ───────────────────────────────────────────────────── Aug 21
                    FINAL HOLDOUT

                  ┌─────────┬─────────┐
                  │   v1    │   v2    │
                  └────┬────┴────┬────┘
                       │         │
                       └────┬────┘
                            ▼
                       compare both
```

Both models must be run on **exactly the same Aug 5–21 transactions** for the final head-to-head comparison.

This final holdout is never used for HPO.

---

## 5. Final agreed model structure

### Stage 1: Logistic Regression baseline

Purpose:

```text
Simple → interpretable → fast → baseline
```

It establishes a minimum reference point before introducing a stronger model.

### Stage 2: LightGBM

Primary candidate model for the PoC.

Why:

```text
Fast on tabular data
Handles nonlinear relationships
Works well with millions of rows
Practical on an M4 MacBook Air
Common production choice for fraud/tabular ML
```

The project should stop short of neural networks. They are unnecessary for this PoC.

---

## 6. Feature engineering progression

Build features in two layers.

### Transaction-level features

Transform raw fields into useful numerical/contextual signals:

```text
AMOUNT_USD
FEE_USD
       │
       ├── fee_ratio = fee / amount
       ├── log_amount
       ├── hour_of_day
       ├── day_of_week
       └── is_weekend
```

### Behavioural features

Use `SENDER_ID` to create history-based features rather than one-hot encoding millions of sender IDs.

```text
SENDER_ID
    │
    ├── sender_tx_count_10m
    ├── sender_tx_count_1h
    ├── sender_tx_count_24h
    ├── sender_amount_1h
    ├── sender_avg_amount
    ├── sender_amount_stddev
    ├── sender_unique_countries
    ├── sender_unique_currencies
    └── sender_previous_fraud_rate*
```

`previous_fraud_rate` must only use information available **before the current transaction** to avoid leakage.

Conceptually:

```text
Current transaction
        │
        ├── current amount
        ├── current country
        └── current currency

        +

Sender history BEFORE this transaction
        │
        ├── recent transaction count
        ├── recent amount
        ├── historical behaviour
        └── historical outcomes
```

This is expected to be more valuable than simply changing ML algorithms.

---

## 7. HPO with Optuna

Use Optuna to tune the LightGBM challenger.

Candidate parameters:

```text
learning_rate
num_leaves
max_depth
min_child_samples
feature_fraction
bagging_fraction
lambda_l1
lambda_l2
```

Start with approximately 30–50 trials on the M4 MacBook Air and adjust based on runtime.

HPO rules:

```text
Training data     → allowed
Validation data   → allowed
Final holdout     → NEVER used
```

The final holdout must remain untouched until model comparison.

---

## 8. Metrics for the PoC

Separate metrics into **model quality** and **model/data change**.

### Model quality

#### Precision

> When the model says “fraud”, how often is it correct?

#### Recall

> Of all real fraud, how much did the model catch?

#### F1

> A single balance between precision and recall.

#### PR-AUC

> How good is the model at ranking/identifying fraud when fraud is relatively rare?

This should be one of the primary metrics for the PoC.

#### ROC-AUC

Optional secondary ranking metric.

### v1 vs v2 behaviour

#### Prediction agreement

```text
same prediction from v1 and v2
--------------------------------
all transactions
```

Track overall agreement and directional disagreements:

```text
v1 = negative, v2 = positive
v1 = positive, v2 = negative
```

These are more informative operationally than agreement alone.

#### Score drift / JSD

Optional because the dataset does not contain model scores. Add once the models expose probabilities.

```text
v1 score distribution
        vs
v2 score distribution
```

### Data drift

Use PSI for important input features when comparing a reference period with a newer period.

Potential features:

```text
PAYOUT_COUNTRY
PAYOUT_CURRENCY
AMOUNT_USD
FEE_USD
hour_of_day
transaction frequency
```

Volume should also be monitored, but it is a traffic/anomaly signal rather than model drift by itself.

---

## 9. Final comparison report

The final Aug 5–21 evaluation should produce something conceptually like:

```text
                 v1 Champion    v2 Challenger
------------------------------------------------
Precision             0.xx           0.xx
Recall                0.xx           0.xx
F1                    0.xx           0.xx
PR-AUC                0.xx           0.xx
ROC-AUC               0.xx           0.xx

Agreement                           xx.xx%
v1 NEG → v2 POS                    xx.xx%
v1 POS → v2 NEG                    xx.xx%

PSI / drift by feature             ...
```

The decision question becomes:

```text
             Is v2 actually better?
                      │
        ┌─────────────┼─────────────┐
        │             │             │
      quality      behaviour      drift
        │             │             │
      F1/PR-AUC    agreement      PSI
        │             │             │
        └─────────────┼─────────────┘
                      ▼
             Safe to promote?
```

---

## 10. ClearML pipeline

Use ClearML as the experiment-tracking and orchestration layer.

Suggested pipeline:

```text
                         ClearML Pipeline
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     │                     │
          ▼                     ▼                     ▼
     Data Preparation     Feature Engineering    Dataset metadata
          │                     │
          └──────────────┬──────┘
                         ▼
                 ┌─────────────────┐
                 │ Train Baseline  │
                 │ Logistic Reg.   │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Train LightGBM  │
                 └────────┬────────┘
                          │
                          ▼
                    Optuna HPO
                          │
                          ▼
                 ┌─────────────────┐
                 │ Evaluate models │
                 └────────┬────────┘
                          │
                          ▼
               Champion / Challenger
                      comparison
                          │
                          ▼
                  Register artifacts
```

Each major stage should be a ClearML Task so that the project records:

```text
code version
parameters
metrics
artifacts
models
experiment lineage
execution environment
```

### Suggested ClearML Tasks

```text
prepare_data
create_features
train_logistic_baseline
train_lightgbm
run_optuna_hpo
evaluate_model
compare_champion_challenger
```

The final comparison task should consume the saved v1/v2 models and the common Aug 5–21 holdout.

---

## 11. Recommended project progression

Build incrementally rather than implementing everything at once.

```text
1. Load + validate 7.374M transactions
          │
2. Chronological splits
          │
3. Logistic Regression baseline
          │
4. LightGBM baseline
          │
5. Transaction features
          │
6. Behavioural features
          │
7. Optuna HPO
          │
8. v1 / v2 training workflow
          │
9. Common Aug 5–21 comparison
          │
10. Agreement + directional disagreements
          │
11. PSI / drift monitoring
          │
12. ClearML pipeline + experiment tracking
```

The final PoC should tell one coherent story:

```text
New data arrives
       │
       ▼
Train challenger on fresher history
       │
       ▼
Tune and validate it
       │
       ▼
Run challenger alongside champion
       │
       ▼
Compare on the same untouched holdout
       │
       ├── Is v2 better?
       ├── How different is v2?
       └── Has the data itself changed?
       │
       ▼
             Can we safely deploy v2?
```

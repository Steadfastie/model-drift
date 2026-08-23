# Model Drift · Champion / Challenger Fraud PoC

A small, production-shaped machine-learning project built around one question:

> **Can we safely deploy v2?**

Instead of asking only whether a new model scores higher, the project treats a model release as a controlled decision. Train a challenger on newer data, compare it with the existing champion, look at prediction quality and behavioural stability, and only then suggest promotion.

The core implementation lives in [`notebooks/12_full_pipeline.ipynb`](notebooks/12_full_pipeline.ipynb). The pipeline is orchestrated with [ClearML](https://clear.ml/) and runs locally, in-process, so a ClearML agent is not required for the PoC.

## The idea in one picture

A new model can be **better**, **worse**, or simply **different**. Those are three different questions.

```text
                           New transaction
                                  │
                         ┌────────┴────────┐
                         │                 │
                        v1                v2
                      champion          challenger
                         │                 │
                         └────────┬────────┘
                                  │
                           compare both
                                  │
             ┌────────────────────┼────────────────────┐
             │                    │                    │
         Quality              Agreement             Drift
             │                    │                    │
      precision/recall       same decision?       JSD / PSI
      F1 / ROC-AUC          where do they flip?   did behaviour move?
             │                    │                    │
             └────────────────────┼────────────────────┘
                                  ▼
                         PROMOTE or REJECT
```

The important distinction is:

```text
Model quality     → "Is v2 correct?"
Model agreement   → "How differently does v2 behave from v1?"
Data drift        → "Has the input population changed?"
```

The goal is not to find a magical single drift number. It is to combine these signals into a practical release decision.

## Architecture

![Pipeline architecture](pipeline.png)

The implementation follows the same flow as the diagram:

```text
prepare_v2_data
       │
       ▼
engineer_v2_features
       │
       ├──────────────► train_v2_logistic_baseline ────┐
       │                                                │
       └──────────────► train_v2_lightgbm_baseline ─────┤
                                                        ▼
                                                 select_v2_model
                                                    (HITL)
                                                        │
                                                        ▼
                                            selected model + HPO
                                                        │
                                                        ▼
                                         compare_v1_champion_v2
                                                        │
                                                        ▼
                                                 promotion
                                                    (HITL)
```

The two baseline models are deliberately independent. Logistic Regression gives a simple reference point; LightGBM is the stronger tabular candidate. The human selection gate prevents the next stage from silently choosing a model just because one metric happened to win.

The current implementation then tunes the selected model with Optuna and finally compares the resulting challenger with the saved v1 champion on the same holdout population.

## Data split

The data covers **2026-05-21 → 2026-08-21**. Splits are chronological rather than random, because a fraud model should not learn from the future.

The split configuration used by the pipeline is:

| Version | Train | Validation | Test / final comparison |
| --- | --- | --- | --- |
| **v1 champion** | May 21 → Jul 21 | Jul 22 → Jul 28 | Jul 29 → Aug 4 |
| **v2 challenger** | May 28 → Jul 28 | Jul 29 → Aug 4 | Aug 5 → Aug 11 |

The v2 test week is also exposed as the pipeline holdout used for the final v1-v2 comparison. The source dataset continues to Aug 21, leaving later transactions unused by the current seven-day test configuration.

Why move v2 forward by a week?

```text
v1:  [========== train ==========][valid][test]
v2:       [========== train ==========][valid][test]
         └──── 7-day freshness shift ────┘
```

This makes v2 a genuine challenger: it has seen a more recent slice of reality before being evaluated.

## What do the metrics mean?

The project uses metrics at three levels.

### Model quality

| Metric | Question it answers | Rough interpretation |
| --- | --- | --- |
| **Precision** | When the model says “fraud”, how often is it right? | 0% = always wrong, 100% = every flagged case is fraud |
| **Recall** | Of all real fraud, how much did we catch? | 0% = caught none, 100% = caught all |
| **F1** | Can we balance precision and recall? | Higher is better; 1.0 is perfect |
| **ROC-AUC** | Can the model rank fraud above legitimate traffic across thresholds? | 0.50 ≈ random, ~0.70 useful, ~0.90+ strong |
| **PR-AUC** | How good is that ranking when fraud is relatively rare? | Higher is better; especially useful for imbalanced fraud data |

### Model behaviour

| Metric | Question it answers | Rough interpretation |
| --- | --- | --- |
| **Agreement** | Do v1 and v2 make the same decision? | 100% = identical decisions; lower means more prediction changes |
| **Score JSD** | Did the score distributions move? | 0 = identical; small values mean very similar; larger values mean increasingly different distributions |
| **Max PSI** | Did an input distribution move away from its reference population? | < 0.10 is usually considered stable; 0.10–0.25 deserves attention; > 0.25 is significant drift |

These are guardrails, not laws. The thresholds below are intentionally simple for a PoC.

## Parameters and why they exist

| Parameter | What question does it answer? | Value / breakdown |
| --- | --- | --- |
| `TRAIN_MONTHS` | How much recent history should a model learn from? | **2 months**; enough context without making old behaviour dominate |
| `VALIDATION_DAYS` | How long do we reserve for tuning? | **7 days**; recent, unseen data for HPO decisions |
| `TEST_DAYS` | How long should the final unseen evaluation window be? | **7 days**; keeps the final comparison clean |
| `V2_SHIFT_DAYS` | How much fresher is v2's training window? | **7 days**; v2 starts one week later |
| Classification threshold | When does a probability become a fraud decision? | **0.5**; simple baseline for the PoC |
| Baseline selection | Which model should continue to HPO? | **F1 + ROC-AUC + PR-AUC**, higher is better |
| HPO budget | How much search is enough for a laptop PoC? | **10 Optuna trials** |
| Agreement gate | When is v2 behaving too differently from v1? | **> 95%** suggested as stable |
| JSD gate | When has the score distribution moved too far? | **< 0.05** suggested as stable |
| PSI gate | When has the feature population moved too far? | **< 0.10** suggested as stable |

For example, ROC-AUC and drift metrics can be read as a scale rather than a binary pass/fail:

```text
ROC-AUC
0.50 ─────────────── 0.70 ─────────────── 0.90 ───────── 1.00
random                useful                    strong       perfect

JSD
0.00 ─────── small change ─────── 0.05 ──────────────── 0.50+
identical             usually stable                 very different

PSI
0.00 ─────── 0.10 ───────────── 0.25 ─────────────────────►
stable      watch closely             significant drift
```

## Final comparison: what actually happened?

The v1 champion and v2 challenger were evaluated on the same holdout population.

```jsonc
{
  "precision": { "v1": 0.702611, "v2": 0.703690 }, // v2 is slightly better
  "recall":    { "v1": 0.333601, "v2": 0.330033 }, // v2 catches slightly less fraud
  "f1":        { "v1": 0.452401, "v2": 0.449329 }, // v1 is slightly better overall balance
  "roc_auc":   { "v1": 0.855241, "v2": 0.855949 }, // v2 has slightly better ranking
  "pr_auc":    { "v1": 0.392633, "v2": 0.392304 }, // essentially unchanged, tiny v1 edge

  "agreement": 0.9993, // 99.93% of decisions are identical: behaviour is very stable
  "score_jsd": 0.0004, // extremely small distribution change
  "max_psi":   0.0239, // comfortably below the 0.10 drift guardrail

  "promotion_suggestion": "PROMOTE v2"
}
```

The interesting part is that **v2 is not a runaway winner**. Its quality metrics are mixed: precision and ROC-AUC improve very slightly, while recall, F1 and PR-AUC are essentially flat or marginally worse.

What is strong is the stability signal:

```text
99.93% agreement       ████████████████████░
JSD = 0.0004           █░
PSI = 0.0239           ████░░░░░░░░░░░░░░░░
```

So the recommendation is **PROMOTE v2**, but the result should be read as **“v2 is a safe, very small change”**, not **“v2 is dramatically better”**. The final promotion remains a human decision in the ClearML pipeline.

## Why ClearML?

ClearML ties the experiment together as a reproducible pipeline rather than a collection of unrelated notebook runs. Each stage can log its parameters, metrics and artifacts, while the pipeline preserves the dependency graph from data preparation to the final promotion decision.

```text
Data → Features → Parallel baselines → Human selection → HPO → Holdout comparison → Human promotion
```

The project is intentionally small enough to run on an M4 MacBook Air while still demonstrating the core production idea: **a model should earn the right to replace the current model.**

## Getting started

```bash
uv sync
docker compose -f build/compose.yml -p cross-model-drift up -d
```

MySQL is at `localhost:3306` (`cross_model_drift` / `drift_user` / `password`).
ClearML UI is at [http://localhost:8080](http://localhost:8080) (API `8008`, files `8081`).
Configuration lives in `configs/local.json`.

Seed the CSV from `notebooks/01_seed_data.ipynb` if the table is empty.

## Notebooks

| Notebook | Stage |
| --- | --- |
| `01_seed_data.ipynb` | schema + CSV load |
| `02_data_distribution.ipynb` | EDA |
| `03_chronological_splits.ipynb` | v1 / v2 / holdout windows |
| `04_transaction_features.ipynb` | row-local features |
| `05_behavioural_features.ipynb` | sender history (no leakage) |
| `06_train_logistic_baseline.ipynb` | logistic baseline |
| `07_train_lightgbm_baseline.ipynb` | LightGBM baseline |
| `08_optuna_hpo.ipynb` | tune on train/valid only |
| `09_train_v1_v2.ipynb` | champion + challenger |
| `10_holdout_comparison.ipynb` | holdout quality |
| `11_clearml_pipeline.ipynb` | earlier ClearML pipeline |
| `12_full_pipeline.ipynb` | current end-to-end ClearML pipeline |

Do not use the final holdout during HPO.

## ClearML on this Mac

The `clearml/` workspace folder is the Python SDK, not the server. Local server compose is `build/compose.clearml.yml`. `clearml/server` is amd64 and runs via Rosetta/QEMU on M4.

Give Docker Desktop at least 8 GB RAM. ClearML MCP is not a Docker service; VS Code starts `uvx clearml-mcp` as a local stdio process and uses `~/.clearml/clearml.conf`.

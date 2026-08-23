# cross-model-drift

Local workspace for [Steadfastie/model-drift](https://github.com/Steadfastie/model-drift).

Champion / challenger fraud-model PoC. Work happens in the Jupyter notebooks.

Local notes and data kept from this workspace:

- `fraud_model_champion_challenger_build_plan.md`
- `transactions.csv` (gitignored; not committed)

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
| `10_holdout_comparison.ipynb` | Aug 5–21 quality |
| `11_agreement_and_disagreements.ipynb` | prediction flips |
| `12_psi_drift.ipynb` | feature PSI |
| `13_clearml_pipeline.ipynb` | experiment tracking |

Do not use the final holdout in notebook 08.

## ClearML on this Mac

The `clearml/` workspace folder is the Python SDK, not the server. Local server compose is `build/compose.clearml.yml` (named volumes, no `/opt/clearml`, no services agent). Give Docker Desktop ≥ 8 GB RAM. `clearml/server` is amd64 and runs via Rosetta/QEMU on M4.
services live in `build/compose.yml` under the same `cross-model-drift` project as MySQL. Give Docker Desktop ≥ 8 GB RAM. `clearml/server` is amd64 and runs via Rosetta/QEMU on M4.

ClearML MCP is **not** a Docker service. VS Code starts `uvx clearml-mcp` as a local stdio process. It needs `~/.clearml/clearml.conf` (copy `configs/clearml.conf.example` and fill credentials from [http://localhost:8080/settings/workspace-configuration](http://localhost:8080/settings/workspace-configuration)). Then set `init=True` in the training notebooks


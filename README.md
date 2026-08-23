# cross-model-drift

Local workspace for [Steadfastie/model-drift](https://github.com/Steadfastie/model-drift).

Champion / challenger fraud-model PoC. Work happens in the Jupyter notebooks.

Local notes and data kept from this workspace:

- `fraud_model_champion_challenger_build_plan.md`
- `transactions.csv` (gitignored; not committed)

## Getting started

```bash
uv sync
docker compose -f build/compose.infra.yml -p cross-model-drift up -d
uv run jupyter lab
```

MySQL is at `localhost:3306` (`cross_model_drift` / `drift_user` / `password`).
Configuration lives in `configs/local.json`.

Seed the CSV from `notebooks/01_seed_data.ipynb`.

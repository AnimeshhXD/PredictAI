# Notebooks

Exploratory analysis and paper-figure notebooks live here.

> **Rule:** Notebooks are for exploration only. Any result used in the paper
> must be reproduced by running `make evaluate` (which writes a CSV to
> `results/`) — never read from a notebook cell's in-memory state.

| Notebook | Purpose |
|---|---|
| `01_eda_m5.ipynb` | Exploratory data analysis of the M5 dataset |
| `02_leakage_sanity.ipynb` | Visual verification of the leakage guard |
| `03_forecast_comparison.ipynb` | ARIMA vs XGBoost vs LLM side-by-side |
| `04_paper_figures.ipynb` | Generates all figures from `results/*.csv` |

> Notebooks are added in later phases.

# Data Directory — PredictAI

This directory holds all dataset files. **Nothing here is committed to git**
(see `.gitignore`). Download instructions are below.

```
data/
├── raw/          ← original, untouched downloads (read-only after download)
│   └── m5/       ← M5 Forecasting competition files
│       ├── sales_train_evaluation.csv
│       ├── sales_train_validation.csv
│       ├── sell_prices.csv
│       ├── calendar.csv
│       └── sample_submission.csv
├── processed/    ← feature-engineered parquet files produced by the pipeline
└── README.md     ← this file
```

---

## Downloading the M5 Forecasting Dataset from Kaggle

The M5 dataset is hosted on Kaggle under the
[m5-forecasting-accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)
competition page.

### Option A — Kaggle CLI (recommended)

1. **Install the Kaggle CLI** (already included in `requirements.txt`):
   ```bash
   pip install kaggle
   ```

2. **Get your API token**
   Go to <https://www.kaggle.com/settings> → *API* → **Create New Token**.
   This downloads `kaggle.json`. Place it at:
   - Linux / macOS: `~/.kaggle/kaggle.json`
   - Windows: `%USERPROFILE%\.kaggle\kaggle.json`

   Set permissions (Linux/macOS only):
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

3. **Accept the competition rules**
   Visit <https://www.kaggle.com/competitions/m5-forecasting-accuracy> and click
   **Join Competition** / accept the rules (required even for the CLI).

4. **Download and unzip**
   ```bash
   # From the repo root
   kaggle competitions download -c m5-forecasting-accuracy -p data/raw/m5/
   cd data/raw/m5
   unzip m5-forecasting-accuracy.zip
   rm m5-forecasting-accuracy.zip   # optional – remove the archive
   ```

   Or with `make` (once the script is wired up in Phase 2):
   ```bash
   make data
   ```

### Option B — Manual download

1. Sign in to Kaggle and navigate to the
   [data tab](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data).
2. Download each file individually and save them under `data/raw/m5/`.

---

## Expected files after download

| File | Size (approx.) | Description |
|---|---|---|
| `sales_train_evaluation.csv` | ~116 MB | Daily unit sales for 30,490 items × 1,941 days |
| `sales_train_validation.csv` | ~113 MB | Same but stops 28 days earlier |
| `sell_prices.csv` | ~141 MB | Weekly sell price per item × store |
| `calendar.csv` | ~60 KB | Date metadata, snap flags, NWBI events |
| `sample_submission.csv` | ~66 MB | Submission template |

---

## Data Dictionary (quick reference)

### `calendar.csv`

| Column | Type | Description |
|---|---|---|
| `date` | string (YYYY-MM-DD) | Calendar date |
| `wm_yr_wk` | int | Walmart week (year × 100 + week) |
| `weekday` | string | Day name |
| `wday` | int | 1 = Saturday … 7 = Friday |
| `month` | int | Month number |
| `year` | int | Year |
| `d` | string | Day column name, e.g. `d_1` |
| `event_name_1/2` | string | NWBI event name if applicable |
| `event_type_1/2` | string | Cultural / National / Religious / Sporting |
| `snap_CA/TX/WI` | int | 1 if SNAP purchases allowed in that state |

### `sell_prices.csv`

| Column | Type | Description |
|---|---|---|
| `store_id` | string | Store identifier (e.g. `CA_1`) |
| `item_id` | string | Item identifier (e.g. `FOODS_1_001`) |
| `wm_yr_wk` | int | Walmart week |
| `sell_price` | float | Weekly price in USD |

### `sales_train_evaluation.csv` (shape: 30,490 × 1,947)

| Columns | Description |
|---|---|
| `id` | Unique series ID |
| `item_id` | Item identifier |
| `dept_id` | Department identifier |
| `cat_id` | Category (FOODS, HOBBIES, HOUSEHOLD) |
| `store_id` | Store identifier |
| `state_id` | State (CA, TX, WI) |
| `d_1 … d_1941` | Daily unit sales |

---

## Leakage policy

> **Do not use any data beyond the scenario `cutoff_date`.**
> The pipeline's `filter_before_cutoff()` and `assert_no_leakage()` functions
> enforce this automatically. Any test that fails the leakage check will block
> the `make test` step.

---

## License

The M5 dataset is provided by Walmart under the
[Kaggle competition rules](https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules).
It is **for non-commercial research use only**.

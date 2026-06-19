# M5 Forecasting — Accuracy

> 28-day unit sales forecasting for Walmart across 10 stores and 3,049 SKUs — **Top 4% on Kaggle** (private leaderboard, 5,533 teams), deployed as a web app on GCP App Engine.

📝 [Read the full write-up on Medium](https://medium.com/@tarun99.m/m5-forecasting-accuracy-56e4c601e69c)

---

## Highlights

**Kaggle Top 4% — WRMSSE 0.62229**
Custom per-state LightGBM ensemble scored 0.62229 WRMSSE on the private leaderboard (rank 186 / 5,533 teams). The key lever: training separate models per state (CA, TX, WI) rather than a single global model, exploiting the distinct sales distributions and price structures observed across states in EDA.

**Feature Engineering from First Principles**
EDA revealed a strong weekly periodicity (autocorrelation peaks at lag 7 and multiples) and meaningful signals from SNAP sale dates, sell price, and day-of-month spending patterns. Features designed from these: lag values at 0/7/14/21/28/35 days, rolling averages over 7–42 days (both with 28-day shift for the forecasting horizon), weekday avg trend features split by SNAP/non-SNAP, a 2-year revenue feature per product-store pair, and an imputation boolean for missing sell prices. Feature importance confirmed these designed features ranked highest across all three state models.

**Principled Sell Price Imputation**
Missing sell prices encode meaningful information — they indicate no-stock or zero-purchase periods. Handled in two steps: added an `imputed` boolean to preserve this signal, then interpolated prices row-wise (pivoted to wide form, interpolated, merged back) using the full merged dataframe to ensure even missing-day entries got a valid price estimate.

**End-to-End Deployment**
Flask web app deployed on GCP App Engine (Flexible). Takes product ID and store ID as input, pulls from the repository, and returns 28-day demand forecasts using the custom model. Input validation with error handling for unknown product IDs.

---

## Problem Statement

Estimate point forecasts of unit sales for Walmart products across 10 US stores for the next 28 days. A Kaggle competition by MOFC using hierarchical sales data (item, department, category, store, state) with explanatory variables: price, promotions, day of week, and special events. Historical data: 2011–01–29 to 2016–06–19.

---

## Dataset

Walmart M5 competition data — 3 CSVs: `sales_train_evaluation`, `calendar`, `sell_prices`. ~60M rows after melting to long format. Source: [Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data).

---

## Data Pipeline

- **Downcasting** — reduced memory ~50% by casting int64/float64 to minimal types and object columns to `category`
- **Melt to long format** — converted 1,941 daily sales columns to rows; added 28 future days with NaN targets
- **Merge** — joined calendar (events, SNAP dates) and sell prices (weekly → daily via week key)
- **Sell price imputation** — pivoted to wide, interpolated row-wise per product-store, merged back; added `imputed` boolean

Final dataframe: 60M rows × 23 columns.

---

## Modelling

Time-series converted to a supervised learning problem. Train/val split: last 28 days of historical data as validation; next 28 days as test.

| Model | RMSSE (val) | Notes |
|---|---|---|
| SGD Regressor | 1.5861 | Below seasonal baseline |
| Random Forest | 0.7709 | Strong but slow |
| LightGBM | Better than RF | Gradient-based one-side sampling |
| **Custom per-state LGBM** | **→ 0.62229 WRMSSE** | Separate model per state — best results |

**Evaluation metric:** WRMSSE (Weighted Root Mean Squared Scaled Error) — scale-independent, handles intermittent/zero-sales series, penalises errors symmetrically.

---

## Deployment

Flask web app deployed on **GCP App Engine (Flexible)**. Input: product ID + store ID (dropdown). Output: 28-day point forecasts from the custom model. Returns an error message for invalid product IDs.

---

## Stack

`Python` · `LightGBM` · `scikit-learn` · `pandas` · `Flask` · `GCP App Engine`

---

## References

- [M5 Competition — MOFC](https://mofc.unic.ac.cy/m5-competition/)
- [1st place solution writeup](https://www.kaggle.com/competitions/m5-forecasting-accuracy/discussion/163684)
- [M5 competition paper — ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0169207021001874)

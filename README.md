# NYC Airbnb Price Prediction

## 1. Project overview

This project predicts nightly Airbnb prices in New York City using tabular listing data. The goal is to understand which factors drive price (location, reviews, availability, minimum nights, etc.) and to build a model that can estimate a fair price for a new listing. [web:228]

The work is organised as a small end‑to‑end pipeline:

- Week 1 – Data loading, cleaning, and saving `nyc_clean.parquet`.
- Week 2 – Feature engineering and saving `features.parquet`.
- Week 3 – Baseline and tree‑based models, plus feature importance.
- Week 4 – Evaluation, residual diagnostics, and model comparison. [cite:180]

---

## 2. Dataset

- **Source:** NYC Airbnb Open Data (listings, calendar, and reviews for New York City). [web:228]
- **Size:** ~48k listings after cleaning, with 23 engineered features and a log‑transformed target `log_price`. [cite:184]
- **Granularity:** One row per listing, with information about location, room type, availability, reviews, and host activity.
- **Time frame:** Single snapshot of listing data, treated as cross‑sectional (no explicit time‑series modelling). [web:228]

The raw CSV files are stored under `data/raw/`, and cleaned/engineered datasets are stored as parquet files under `data/processed/`.

---

## 3. Features

After cleaning, I engineered 23 features that capture location, neighbourhood pricing, demand, and constraints. [cite:184]

Key groups:

- **Location**
  - `latitude`, `longitude`
  - neighbourhood / neighbourhood_group indicators

- **Neighbourhood price context**
  - `neigh_price_mean`: mean price for the listing’s neighbourhood
  - `neigh_price_median`: median price for the listing’s neighbourhood

- **Relative pricing**
  - `price_premium_vs_neigh`: listing’s price divided by (or difference from) its neighbourhood mean, capturing how “cheap” or “premium” it is locally.

- **Demand and host activity**
  - `number_of_reviews`, `reviews_per_month`, `days_since_last_review`
  - `availability_365`
  - host listing counts (e.g. `host_listing_count`, `calculated_host_listings_count`)

- **Constraints**
  - `minimum_nights`, `min_nights_log`

The **target** is:

- `log_price = log(nightly_price)`

Using `log_price` stabilises variance and makes multiplicative effects (e.g. “20% more expensive in this neighbourhood”) closer to additive in the model, which generally improves regression performance. [web:236]

---

## 4. Models

I trained three main models on the engineered feature set, using an 80/20 train‑test split and median imputation for missing values. [web:262][web:275]

- **Ridge Regression**
  - Linear model in log space with L2 regularisation.
  - Serves as an interpretable baseline.

- **HistGradientBoostingRegressor (HistGB)**
  - Gradient‑boosted decision trees (`sklearn.ensemble.HistGradientBoostingRegressor`). [web:268]
  - Handles missing values natively and captures non‑linear interactions between features (e.g. neighbourhood price stats × room type).

- **RandomForestRegressor (RF)**
  - Ensemble of decision trees.
  - In this project, RF achieves almost perfect performance on the test split; residual diagnostics and extremely small dollar errors suggest target leakage or an overly easy setup.
  - RF is therefore used only as an **optimistic upper bound**, not as the final selected model. [web:236]

All models are trained on the same train/test split with the same features and target to keep comparisons fair.

---

## 5. Evaluation results

All metrics below are computed on the **held‑out test set**, reported both in log space and in original dollar units:

| Model  | RMSE (log) | R² (log) | RMSE ($) | MAE ($) |
|--------|-----------:|---------:|---------:|--------:|
| Ridge  | 0.250      | 0.859    | 1257.1   | 58.1    |
| HistGB | 0.020      | 0.999    | 10.0     | 2.1     |
| RF     | 0.013      | 1.000    | 5.2      | 0.9     |

Interpretation:

- **Ridge** provides a realistic, interpretable baseline: it fits well in log space but tends to underestimate very expensive, rare listings, leading to large dollar errors for those cases. [web:236]
- **HistGB** captures non‑linear relationships between price and engineered neighbourhood features, reducing typical dollar error from roughly 58 USD to around 2–10 USD per listing while still leaving some residual noise, which is plausible for messy real‑world pricing. [web:268]
- **RF** appears almost perfect numerically (very small residuals and near‑zero errors), which is usually a red flag; residual plots show the model is essentially reconstructing the target, so these numbers are treated as an optimistic upper bound rather than a trustworthy estimate of generalisation. [web:236]

---

## 6. Diagnostic plots and error analysis

The `notebooks/04_evaluation.ipynb` notebook contains:

- **Residuals vs predicted plots**
  - Ridge residuals show a curved pattern and large negative errors for high predicted log prices, meaning the linear model underfits high‑end listings.
  - Random Forest residuals cluster extremely close to zero across the entire prediction range, reinforcing the “too‑good‑to‑be‑true” performance and suggesting leakage/overfitting.
  - HistGB residuals are tightly centered around zero but still show some spread, which is more realistic. [web:236][web:238]

- **Residual distribution histogram**
  - Ridge has a wide residual distribution with a heavy tail.
  - HistGB’s distribution is sharply peaked near zero with a much smaller spread, showing its improvement in accuracy.

- **Error analysis table**
  - A table of the top 10 listings with the largest absolute dollar errors.
  - Ridge’s biggest misses correspond to very expensive (3–5k+ USD per night) or otherwise unusual listings, where it can be off by hundreds or more.
  - HistGB dramatically reduces these errors, usually bringing them down to tens of dollars on the same listings, indicating that non‑linear models are better suited for luxury and atypical properties. [web:236][web:268]

---

## 7. Key insights and limitations

### Key insights

- **Neighbourhood pricing dominates:** Features such as `price_premium_vs_neigh`, `neigh_price_mean`, and `neigh_price_median` are among the most important features in tree‑based models, confirming that local market context is the strongest driver of price. [web:232][web:238]
- **Location still matters:** Latitude and longitude still carry signal even after controlling for neighbourhood averages, capturing within‑neighbourhood differences.
- **Demand and host activity help refine price:** Number of reviews, recency of reviews, availability, and host listing counts add smaller but meaningful improvements.
- **Log‑transforming price helps:** Working with `log_price` stabilises variance and makes the regression problem easier, especially for linear models. [web:236]

### Limitations

- **Random Forest overperformance / possible leakage:** RF’s near‑perfect performance is not treated as reliable; more work is needed to diagnose the exact leakage source (e.g. inadvertent target‑derived features or data splitting issues).
- **Static snapshot:** The dataset is a snapshot of NYC listings; seasonality, changing demand, and time‑varying effects are not modelled.
- **No text or image features:** Listing descriptions, titles, and photos are not used, even though they likely explain additional variance in price.
- **Single‑city focus:** Models are trained only on NYC data; they might not generalise directly to other cities without retraining.

---

## 8. Repository structure

```text
airbnb-price-prediction-nyc/
├── data/
│   ├── raw/
│   │   ├── AB_NYC_2019.csv
│   │   ├── calendar.csv
│   │   ├── listings.csv
│   │   └── reviews.csv
│   └── processed/
│       ├── nyc_clean.parquet
│       └── features.parquet
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   └── ... (optional helper modules)
├── models/
│   └── ... (saved models / artefacts, if any)
├── requirements.txt
└── README.md


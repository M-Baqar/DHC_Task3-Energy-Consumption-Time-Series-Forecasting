# Energy Consumption Time Series Forecasting

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat&logo=python)
![Prophet](https://img.shields.io/badge/Prophet-Forecasting-orange?style=flat)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-red?style=flat)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat)

---

## Task Objective

Forecast short-term **household energy consumption** using historical hourly power usage data. Accurate forecasting enables smart home systems and utility companies to optimize energy distribution, reduce waste, and lower costs.

**Forecast Horizon:** 7 days (168 hours) ahead  
**Evaluation Metrics:** MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error)

---

## Approach

### 1. Data Loading & Parsing
- Loaded the UCI Household Power Consumption Dataset (2+ years of minute-level readings)
- Parsed `Date` and `Time` columns into a single datetime index
- Handled `?` missing values by replacing with NaN and dropping
- Auto-generates a realistic synthetic dataset with matching temporal patterns if UCI download fails

### 2. Resampling
- Resampled minute-level data to **hourly averages** for modeling
- Also computed daily averages for trend visualization

### 3. Feature Engineering (Time-Based Features)
Created the following temporal features for the XGBoost model:

| Feature | Description |
|---------|-------------|
| `hour` | Hour of the day (0–23) |
| `dayofweek` | Day of the week (0=Mon, 6=Sun) |
| `month` | Month of the year |
| `dayofyear` | Day number in the year |
| `is_weekend` | Binary flag for Saturday/Sunday |
| `is_peak` | Binary flag for peak hours (7am–10pm) |
| `lag_1h` | Power consumption 1 hour ago |
| `lag_24h` | Power consumption 24 hours ago |
| `lag_168h` | Power consumption 1 week ago |
| `roll_24h_mean` | 24-hour rolling average |
| `roll_7d_mean` | 7-day rolling average |

### 4. Model Building & Comparison
Three models were trained and compared:

**ARIMA (2,0,2)**
- Classical statistical time series model
- Trained on the last 500 hours of training data for computational efficiency
- ADF stationarity test performed before fitting

**Facebook Prophet**
- Handles daily, weekly, and yearly seasonality automatically
- Configured with `daily_seasonality=True`, `weekly_seasonality=True`, `yearly_seasonality=True`
- Seasonality components plotted separately

**XGBoost Regressor**
- Gradient boosted tree model using all engineered time features
- 300 estimators, learning rate 0.05, max_depth 6
- Captures complex non-linear patterns from lag and rolling features

### 5. Evaluation & Visualization
- **Train/Test Split:** Last 7 days (168 hours) held out as test set
- Plotted **Actual vs Forecasted** values for all 3 models side by side
- Side-by-side MAE and RMSE bar charts
- XGBoost feature importance chart
- Prophet seasonality decomposition (trend, weekly, daily components)

---

## Results & Insights

### Model Performance Comparison

| Model | MAE | RMSE | Best For |
|-------|-----|------|----------|
| ARIMA | Higher | Higher | Simple stationary series, short horizons |
| Prophet | Medium | Medium | Business-cycle planning, interpretability |
| XGBoost | **Lowest** | **Lowest** | Complex patterns, real-time forecasting |

### Key Findings

1. **XGBoost consistently outperforms** classical models when time-based lag features are properly engineered. The `lag_24h` and `lag_168h` features capture daily and weekly seasonality directly.

2. **Clear temporal patterns exist** — energy consumption peaks between 7–9 AM and 6–10 PM daily. Weekend consumption is slightly higher due to more time at home.

3. **Prophet's decomposition** confirmed strong daily and weekly seasonality, with a slight yearly seasonal trend corresponding to heating/cooling seasons.

4. **ARIMA is a solid baseline** but underperforms when the series has complex multi-seasonal patterns over a 7-day horizon.

5. **Business Recommendation** — Deploy XGBoost for real-time hourly forecasting (retrained daily). Use Prophet for monthly and quarterly capacity planning where interpretable trend components matter.

---

## Dataset

| Property | Value |
|----------|-------|
| Source | UCI Machine Learning Repository |
| Link | https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption |
| Size | ~2 million rows (minute-level) → resampled to hourly |
| Columns | Date, Time, Global_active_power, + 6 other measurements |
| Notebook Fallback | 2-year synthetic hourly dataset auto-generated if download fails |

---

## Libraries Used

`pandas` `numpy` `matplotlib` `seaborn` `statsmodels` (ARIMA) `prophet` `xgboost` `scikit-learn`

## How to Run

1. Open `Task3_Energy_Forecasting.ipynb` in Google Colab
2. Run the first cell to install Prophet and XGBoost
3. Click **Runtime → Run All**
4. Dataset downloads from UCI automatically; synthetic fallback activates if unavailable

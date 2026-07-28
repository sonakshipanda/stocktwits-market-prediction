# Using NLP and StockTwits to Predict Market Data

**AI4ALL Ignite — Group 04**
Jasleen Kaur · David Mora · Sonakshi Panda · Shana Ibatuan

**Repository:** [stocktwits-market-prediction](https://github.com/sonakshipanda/stocktwits-market-prediction)

**Live showcase:** [stocktwits-market-prediction.streamlit.app](https://stocktwits-market-prediction.streamlit.app)

> **Research question:** Can StockTwits sentiment help predict the next-trading-day price direction of AAPL, AMZN, NVDA, and TSLA?

---

## Project Overview

Social media increasingly shapes how people make financial decisions. StockTwits is a platform built specifically for investors, where users can tag their posts as **Bullish** or **Bearish** on a given stock. This project investigates whether that crowd sentiment carries useful predictive signal or is mostly noise and hype.

We focus on four high-attention U.S. stocks — **AAPL, AMZN, NVDA, and TSLA** — using daily stock prices and millions of StockTwits posts. We clean and align the datasets by ticker and date, engineer market and sentiment features, and then apply supervised models to predict next-day direction and an unsupervised model to group ticker-days by behavior. We also clean a financial-news dataset for a planned comparison, but news headlines are not inputs to the current models.

**How the project evolved.** We began by aiming to predict raw price *movement* from sentiment alone. As we explored the data, we narrowed the scope to four tickers with dense post activity, shifted the prediction target to a cleaner binary "will the stock close up tomorrow?", and added K-Means clustering when we realized the most interesting story was not a single prediction but the *pattern* between sentiment extremes and next-day behavior.

> **NLP scope:** The current pipeline uses the Bullish/Bearish labels supplied by StockTwits users; it does not yet train a separate model to classify the text in `Post_Text`. The predictive models use daily features derived from those labels.

---

## Data

| Dataset | Source | Coverage | Role |
|---|---|---|---|
| World Stock Prices (Daily Updating) | Kaggle | 2000–2025 | Daily OHLC + volume for the 4 tickers |
| US Capital Markets News Headlines | Kaggle | 2020–2024 | Cleaned for a future news-vs.-social comparison; not used in the current models |
| StockTwits Post Data | Kaggle export of StockTwits data | 2020–2022 | 4,201,837 posts, including tagged and untagged posts |

After cleaning and filtering to the four tickers, the StockTwits data yielded **1,612,213 Bullish**, **519,413 Bearish**, and **2,070,211 untagged** posts (2,131,626 tagged posts used for sentiment analysis). Cleaned files are written to a shared Drive folder (see *How to Run*); the raw data is not committed to this repo because of its size.

---

## Methods & Algorithms

### 1. Sentiment-based prediction (supervised)
- **Type:** Supervised binary classification — **Logistic Regression** and **Random Forest**, compared against a majority-class baseline.
- **Goal:** Predict whether a stock closes **up** the next day.
- **Inputs:** Market features (daily return, volume change, intraday return, daily range, 5-day volatility, price vs. 5-day moving average) **and** social features (bullish/bearish/unknown ratios, log post volume, average sentiment, labeled ratio, day-over-day sentiment change).
- **Output:** `Target_Up` — 1 if next-day return > 0, else 0. (`Next_Day_Return` is **excluded** from the features to avoid target leakage.)
- **Why these models:** Logistic Regression provides a simple, interpretable baseline, while Random Forest can capture nonlinear relationships and feature interactions.
- **Validation:** A **time-based split** (train on the earlier 80% of dates, test on the most recent 20%) so the model never trains on the test period. The final dataset contains 3,936 ticker-day observations with StockTwits activity: 3,149 train and 787 test.

#### Test-set results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Majority baseline | 0.536 | 0.536 | 1.000 | 0.698 | N/A |
| Logistic Regression | 0.506 | 0.530 | 0.699 | 0.603 | 0.502 |
| Random Forest | 0.526 | 0.544 | 0.709 | 0.616 | 0.505 |

The majority baseline predicts the positive class for every test observation, which explains its perfect recall and relatively high F1 score. Neither trained model beats its accuracy, and both ROC-AUC scores are approximately 0.50. Random Forest feature importance also places the market features above the sentiment features.

### 2. K-Means Clustering (unsupervised)
- **Type:** Unsupervised clustering.
- **Goal:** Group individual trading days by *what StockTwits was saying* and *how the price moved*, to see whether sentiment extremes line up with next-day behavior.
- **Inputs (per ticker-day):** bullish share of posts, log post volume, same-day return, and intraday range (all standardized). **Next-day return is deliberately *not* a clustering input** — it's held out as an outcome measured *after* clustering, so the "sentiment vs. next-day move" comparison isn't circular.
- **Output:** A cluster label per day. `k = 5` was selected by comparing candidate values with the **elbow method and silhouette scores** (silhouette was highest at k = 5, 0.330 vs 0.311 for k = 4), and the fifth cluster is interpretable rather than redundant.
- **Why K-Means:** It's simple, interpretable, and well-suited to finding natural groupings without labels — a good fit for surfacing sentiment/price patterns.

### 3. Data Visualization
Built with **seaborn/matplotlib**: growth-of-$100 price trends, rolling volatility, average trading volume, sentiment counts and bullish share by ticker, price-vs-sentiment dual-axis charts for all four tickers, and the K-Means cluster scatter.

---

## Key Results

- **Sentiment is structurally optimistic.** StockTwits runs **~76% bullish overall (roughly 74–79% per ticker)**, so the raw bullish *level* barely distinguishes stocks or days.
- **Sentiment does not predict next-day direction.** Neither the Logistic Regression (0.506) nor the Random Forest (0.526) beat the majority-class baseline (0.536), and ROC-AUC sits at ~0.50. In the feature-importance ranking, market features (returns, volatility, price-vs-moving-average) dominate while every sentiment feature is near-zero — sentiment adds essentially no predictive power on top of price action.
- **A descriptive contrarian tilt appears at the loud extremes.** With next-day return held out of the clustering, K-Means (k = 5) isolates two high-volume "loud" clusters that reverse the next day: the biggest-surge hype days (+5.8% same-day, heaviest posting) average a next-day *decline* of −0.27%, while the loud panic-selloff days (−4.4% same-day) average the strongest next-day *bounce*, +0.81%. This is an observed association in the analyzed sample, not proof of a profitable or causal trading signal.
- **Takeaway for the research question:** StockTwits sentiment is **not** a reliable predictor of next-day direction; if anything, it leans slightly contrarian at its extremes. This supports caution about relying on social media for investment decisions — consistent with the idea that any easily-found signal would already be priced in.

### Supporting charts

**Growth of $100 invested (2000–2025)**
![Price trends](figures/01_price_trends.png)

_This chart provides long-run market context; model training and evaluation use only the period that overlaps with the StockTwits data._

**StockTwits sentiment counts by ticker**
![Sentiment by ticker](figures/04_sentiment_counts.png)

**Price vs. StockTwits bullish sentiment**
![Price vs sentiment](figures/06_price_vs_sentiment.png)

**K-Means clusters of trading days**
![K-Means clusters](figures/07_kmeans_clusters.png)

_Additional charts (volatility, trading volume, bullish share) are in the [`figures/`](figures/) folder and the notebook._

---

## Societal Impact & Bias

**Positive impact.** If sentiment carries signal, it could give everyday investors — who lack time or tools for deep research — an additional, free input. Even the *negative* finding is useful: showing that crowd hype doesn't predict (and slightly reverses) helps people avoid chasing momentum.

**Negative impact.** Tools like this can reinforce herd behavior and online echo chambers, encourage over-trading, and give a false sense of certainty. Presenting sentiment as "prediction" could mislead inexperienced investors.

**Sources of bias and how the ML process can amplify or mitigate them.**
- **Sampling bias:** the price dataset tracks only famous, currently-successful U.S. brands (survivorship bias). A model trained on it can *amplify* bias by generalizing to "all stocks." We *mitigate* by explicitly scoping conclusions to these four large-cap U.S. tickers.
- **Platform bias:** StockTwits users are not representative of all investors, and the labeled posts are overwhelmingly bullish. A naive model can amplify this imbalance. We mitigate it by using ratios and sentiment changes rather than raw bullish counts and by comparing trained models against a majority-class baseline.
- **Time-zone bias:** posts are currently parsed as UTC and grouped by calendar date rather than aligned to U.S. market sessions. This can assign after-hours posts to the wrong predictive window. A future version should explicitly convert timestamps and apply market-open and market-close cutoffs.

---

## Limitations

- Results are based on four popular U.S. stocks and may not generalize to smaller companies, other sectors, or other markets.
- User-supplied Bullish/Bearish tags may be missing, sarcastic, coordinated, or otherwise noisy.
- K-Means clusters describe patterns in this sample; they were not evaluated as a trading strategy on a separate out-of-sample period.
- The current supervised experiment compares combined market-and-sentiment features with a majority baseline. A stronger follow-up should also compare market-only, sentiment-only, and combined models on the same chronological split.
- The cleaned news headlines are not yet part of the predictive model.

---

## Repository Structure

```
.
├── README.md
├── .gitignore
├── streamlit_app.py             # presentation-ready results dashboard
├── requirements.txt             # Streamlit deployment dependencies
├── .streamlit/
│   └── config.toml              # dashboard theme
├── notebooks/
│   └── Group_04.ipynb          # full pipeline: cleaning → features → visualization → prediction → K-Means
├── figures/                    # exported charts used in the README / presentation
│   ├── 01_price_trends.png
│   ├── 02_volatility.png
│   ├── 03_volume.png
│   ├── 04_sentiment_counts.png
│   ├── 05_bullish_share.png
│   ├── 06_price_vs_sentiment.png
│   └── 07_kmeans_clusters.png
└── data/
    └── README.md               # where to get the data (NOT the data itself)
```

---

## How to Run

### Option 1: Explore the deployed dashboard

Open the [live Streamlit showcase](https://stocktwits-market-prediction.streamlit.app). No installation, dataset download, or Google Drive access is required. The dashboard presents the completed methodology, evaluation results, figures, K-Means finding, and limitations; it does not rerun the multi-gigabyte cleaning pipeline.

### Option 2: Run the dashboard locally

```bash
git clone https://github.com/sonakshipanda/stocktwits-market-prediction.git
cd stocktwits-market-prediction
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Streamlit will print a local URL, normally `http://localhost:8501`.

### Option 3: Reproduce the full analysis in Google Colab

1. Request access to the shared **Group-04** Google Drive folder (contains `raw_data/` and generated `processed_data/`) and add a shortcut to it in your My Drive.
2. Open `notebooks/Group_04.ipynb` in Google Colab. The loader cell auto-locates the project folder, so it works regardless of the exact folder name.
3. Use **Runtime → Restart session and run all** to run everything top to bottom: mount and load → clean stock prices → clean news → clean StockTwits posts → daily sentiment summary → feature merge → visualizations → supervised prediction models → K-Means.
4. Generated files (`cleaned_*.csv`, a gzip-compressed `cleaned_stocktwits_posts.csv`, and `combined_stocktwits_market_data.csv`) are written to `processed_data/` and `model_ready_data/`.

> Raw data is not stored in this repo due to size. See `data/README.md`.

---

## Next Steps

- Test whether *sentiment change* (not level) or a **contrarian** framing improves prediction, based on the reversal pattern K-Means surfaced.
- Try predicting over longer horizons (e.g. weekly return) instead of next-day, where sentiment may carry more signal than daily noise.
- Extend beyond four large-cap U.S. tickers to test how far the conclusions generalize.
- Compare StockTwits sentiment head-to-head with the news-headline data as competing predictors.

---

## Citations

1. Divernois, M. A., & Filipović, D. (2024). ["StockTwits classified sentiment and stock returns."](https://doi.org/10.1007/s42521-023-00102-z) *Digital Finance, 6*, 249–281.
2. Elgiriyewithana, N. ["World Stock Prices (Daily Updating)."](https://www.kaggle.com/datasets/nelgiriyewithana/world-stock-prices-daily-updating) *Kaggle*.
3. Cao, Y. ["StockTwits 2020–2022 Raw."](https://www.kaggle.com/datasets/frankcaoyun/stocktwits-2020-2022-raw) *Kaggle*.
4. Darmanin, A. ["US Capital Markets News Headlines 2020 to 2024."](https://www.kaggle.com/datasets/addarm/us-capital-markets-news-headlines-2020-to-2024) *Kaggle*.

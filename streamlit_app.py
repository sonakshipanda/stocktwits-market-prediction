from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="StockTwits Market Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PROJECT_ROOT = Path(__file__).parent
FIGURES_PATH = PROJECT_ROOT / "figures"

RESULTS = pd.DataFrame(
    {
        "Model": [
            "Majority Baseline",
            "Logistic Regression",
            "Random Forest",
        ],
        "Accuracy": [0.536, 0.506, 0.526],
        "Precision": [0.536, 0.530, 0.544],
        "Recall": [1.000, 0.699, 0.709],
        "F1 Score": [0.698, 0.603, 0.616],
        "ROC-AUC": [None, 0.502, 0.505],
    }
)

FIGURE_DETAILS = {
    "01_price_trends.png": {
        "title": "Long-term price growth",
        "eyebrow": "Market context",
        "insight": (
            "The four stocks grew at very different rates over the full price "
            "history. The prediction models use only dates that overlap with "
            "the 2020–2022 StockTwits sample."
        ),
    },
    "02_volatility.png": {
        "title": "Rolling market volatility",
        "eyebrow": "Risk profile",
        "insight": (
            "Volatility changes over time and differs by ticker, making recent "
            "price behavior an important part of the model-ready feature set."
        ),
    },
    "03_volume.png": {
        "title": "Average trading volume",
        "eyebrow": "Market activity",
        "insight": (
            "Trading activity varies substantially across the four stocks, so "
            "raw volume is transformed into comparable daily features."
        ),
    },
    "04_sentiment_counts.png": {
        "title": "Sentiment counts by ticker",
        "eyebrow": "Crowd mood",
        "insight": (
            "Bullish labels outnumber bearish labels across every ticker. That "
            "imbalance is why the project uses ratios and a majority baseline."
        ),
    },
    "05_bullish_share.png": {
        "title": "Bullish share by ticker",
        "eyebrow": "Platform bias",
        "insight": (
            "StockTwits sentiment is structurally optimistic—roughly three of "
            "every four labeled posts are bullish—so sentiment level alone is "
            "not very discriminating."
        ),
    },
    "06_price_vs_sentiment.png": {
        "title": "Price versus bullish sentiment",
        "eyebrow": "Signal check",
        "insight": (
            "Price and daily bullish share do not move together consistently. "
            "This visual foreshadows the near-chance supervised model results."
        ),
    },
    "07_kmeans_clusters.png": {
        "title": "K-Means behavior clusters",
        "eyebrow": "Unsupervised finding",
        "insight": (
            "The loudest hype and panic days show a small next-day reversal in "
            "this sample. It is a descriptive pattern—not proof of a profitable "
            "or causal trading signal."
        ),
    },
}


st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at 90% 0%, rgba(20, 184, 166, 0.10), transparent 24rem),
                #f7f9fc;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2.4rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3 {
            color: #172033;
            letter-spacing: -0.025em;
        }

        .hero {
            position: relative;
            overflow: hidden;
            padding: 2.2rem 2.4rem;
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 24px;
            background:
                radial-gradient(circle at 92% 20%, rgba(45, 212, 191, 0.35), transparent 16rem),
                linear-gradient(125deg, #101827 0%, #153142 58%, #0f766e 120%);
            color: white;
            box-shadow: 0 22px 55px rgba(15, 23, 42, 0.16);
        }

        .hero-badge {
            display: inline-block;
            margin-bottom: 1rem;
            padding: 0.35rem 0.72rem;
            border: 1px solid rgba(255, 255, 255, 0.22);
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.09);
            color: #ccfbf1;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .hero h1 {
            max-width: 760px;
            margin: 0;
            color: white;
            font-size: clamp(2.1rem, 4.5vw, 4rem);
            line-height: 1.02;
        }

        .hero p {
            max-width: 760px;
            margin: 1rem 0 0;
            color: #dbeafe;
            font-size: 1.08rem;
            line-height: 1.65;
        }

        .hero-answer {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            margin-top: 1.3rem;
            color: #a7f3d0;
            font-weight: 700;
        }

        .hero-answer-dot {
            width: 0.62rem;
            height: 0.62rem;
            border-radius: 50%;
            background: #34d399;
            box-shadow: 0 0 0 6px rgba(52, 211, 153, 0.12);
        }

        .metric-card {
            min-height: 132px;
            padding: 1.15rem 1.2rem;
            border: 1px solid #e5eaf1;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.92);
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        }

        .metric-label {
            margin-bottom: 0.45rem;
            color: #64748b;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }

        .metric-value {
            color: #172033;
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            line-height: 1.05;
        }

        .metric-detail {
            margin-top: 0.45rem;
            color: #64748b;
            font-size: 0.84rem;
        }

        .section-kicker {
            margin-bottom: 0.25rem;
            color: #0f766e;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.09em;
            text-transform: uppercase;
        }

        .insight-card, .question-card, .pipeline-card {
            height: 100%;
            padding: 1.25rem;
            border: 1px solid #e4e9f0;
            border-radius: 18px;
            background: white;
        }

        .question-card {
            border-left: 5px solid #14b8a6;
        }

        .question-card strong {
            display: block;
            margin-bottom: 0.4rem;
            color: #0f766e;
            font-size: 0.78rem;
            letter-spacing: 0.07em;
            text-transform: uppercase;
        }

        .question-card span {
            color: #172033;
            font-size: 1.25rem;
            font-weight: 700;
            line-height: 1.45;
        }

        .pipeline-number {
            display: inline-grid;
            width: 2rem;
            height: 2rem;
            margin-bottom: 0.85rem;
            place-items: center;
            border-radius: 10px;
            background: #ccfbf1;
            color: #0f766e;
            font-weight: 800;
        }

        .pipeline-card h4 {
            margin: 0 0 0.35rem;
            color: #172033;
        }

        .pipeline-card p, .insight-card p {
            margin: 0;
            color: #64748b;
            font-size: 0.91rem;
            line-height: 1.55;
        }

        .result-bar-row {
            display: grid;
            grid-template-columns: minmax(150px, 1.1fr) 3fr 64px;
            gap: 0.85rem;
            align-items: center;
            margin: 0.8rem 0;
        }

        .result-model {
            color: #334155;
            font-weight: 700;
        }

        .result-track {
            height: 0.75rem;
            overflow: hidden;
            border-radius: 999px;
            background: #e8edf3;
        }

        .result-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #14b8a6, #2dd4bf);
        }

        .result-fill.baseline {
            background: linear-gradient(90deg, #64748b, #94a3b8);
        }

        .result-score {
            color: #172033;
            font-weight: 800;
            text-align: right;
        }

        .finding {
            padding: 1.35rem;
            border-radius: 18px;
            color: #f8fafc;
        }

        .finding.hype {
            background: linear-gradient(145deg, #9f1239, #e11d48);
        }

        .finding.panic {
            background: linear-gradient(145deg, #065f46, #0f9f78);
        }

        .finding-label {
            color: rgba(255, 255, 255, 0.75);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .finding-value {
            margin: 0.3rem 0;
            color: white;
            font-size: 2rem;
            font-weight: 800;
        }

        .finding p {
            margin: 0;
            color: rgba(255, 255, 255, 0.88);
        }

        .footer {
            margin-top: 2.5rem;
            padding-top: 1.25rem;
            border-top: 1px solid #e2e8f0;
            color: #64748b;
            font-size: 0.85rem;
            text-align: center;
        }

        div[data-baseweb="tab-list"] {
            gap: 0.35rem;
            margin-top: 1rem;
            padding: 0.35rem;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            background: white;
        }

        button[data-baseweb="tab"] {
            border-radius: 10px;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            overflow: hidden;
        }

        div[data-testid="stButton"] > button {
            min-height: 2.75rem;
            border-radius: 12px;
            font-weight: 700;
        }

        @media (max-width: 760px) {
            .block-container {
                padding-top: 1rem;
            }

            .hero {
                padding: 1.6rem 1.25rem;
                border-radius: 18px;
            }

            .result-bar-row {
                grid-template-columns: 1fr 55px;
            }

            .result-track {
                grid-column: 1 / -1;
                grid-row: 2;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def metric_card(label: str, value: str, detail: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def result_bars(metric: str) -> None:
    rows = []
    for _, row in RESULTS.iterrows():
        score = row[metric]
        if pd.isna(score):
            continue
        bar_class = " baseline" if row["Model"] == "Majority Baseline" else ""
        rows.append(
            f"""
            <div class="result-bar-row">
                <div class="result-model">{row["Model"]}</div>
                <div class="result-track">
                    <div class="result-fill{bar_class}" style="width: {score * 100:.1f}%"></div>
                </div>
                <div class="result-score">{score:.3f}</div>
            </div>
            """
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def select_page(page_name: str) -> None:
    st.session_state.dashboard_page = page_name


st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">AI4ALL Ignite · Group 04</div>
        <h1>Can social sentiment predict the market?</h1>
        <p>
            We combined 4.2 million StockTwits posts with daily market data to
            test whether crowd sentiment helps predict the next trading-day
            direction of AAPL, AMZN, NVDA, and TSLA.
        </p>
        <div class="hero-answer">
            <span class="hero-answer-dot"></span>
            Our answer: not reliably at the next-day horizon
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
metric_columns = st.columns(4)
with metric_columns[0]:
    metric_card("StockTwits posts", "4.2M+", "2020–2022 raw post sample")
with metric_columns[1]:
    metric_card("Stocks analyzed", "4", "AAPL · AMZN · NVDA · TSLA")
with metric_columns[2]:
    metric_card("Model-ready rows", "3,936", "Ticker-day observations")
with metric_columns[3]:
    metric_card("Best trained model", "52.6%", "Random Forest accuracy")

PAGES = [
    "Project Overview",
    "Model Results",
    "K-Means Finding",
    "Limitations",
]
if st.session_state.get("dashboard_page") not in PAGES:
    st.session_state.dashboard_page = PAGES[0]

st.write("")
navigation_columns = st.columns([1.2, 1, 1.05, 0.9])
for column, page_name in zip(navigation_columns, PAGES):
    with column:
        st.button(
            page_name,
            key=f"nav_{page_name}",
            type=(
                "primary"
                if st.session_state.dashboard_page == page_name
                else "secondary"
            ),
            use_container_width=True,
            on_click=select_page,
            args=(page_name,),
        )

active_page = st.session_state.dashboard_page

if active_page == "Project Overview":
    st.write("")
    st.markdown('<div class="section-kicker">Research question</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="question-card">
            <strong>What we tested</strong>
            <span>
                Can daily StockTwits sentiment improve next-day direction
                predictions for four highly discussed U.S. stocks?
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.subheader("From millions of posts to one testable question")
    pipeline_columns = st.columns(4)
    pipeline_steps = [
        (
            "01",
            "Clean",
            "Standardize prices and millions of StockTwits posts while removing invalid records.",
        ),
        (
            "02",
            "Aggregate",
            "Convert tagged posts into daily bullish, bearish, volume, and change features.",
        ),
        (
            "03",
            "Model",
            "Train Logistic Regression and Random Forest with a chronological split.",
        ),
        (
            "04",
            "Evaluate",
            "Compare against a majority baseline and inspect behavior clusters.",
        ),
    ]
    for column, (number, title, description) in zip(pipeline_columns, pipeline_steps):
        with column:
            st.markdown(
                f"""
                <div class="pipeline-card">
                    <div class="pipeline-number">{number}</div>
                    <h4>{title}</h4>
                    <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    st.markdown('<div class="section-kicker">Project visuals</div>', unsafe_allow_html=True)
    st.header("Explore the data story")
    st.caption(
        "Choose a chart for a presentation-friendly view and a short interpretation."
    )

    available_figures = [
        file_name
        for file_name in FIGURE_DETAILS
        if (FIGURES_PATH / file_name).exists()
    ]
    if available_figures:
        selected_figure = st.selectbox(
            "Choose a chart",
            available_figures,
            format_func=lambda file_name: FIGURE_DETAILS[file_name]["title"],
        )
        details = FIGURE_DETAILS[selected_figure]

        chart_column, insight_column = st.columns([2.1, 1])
        with chart_column:
            st.image(
                str(FIGURES_PATH / selected_figure),
                caption=details["title"],
                use_container_width=True,
            )
        with insight_column:
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="section-kicker">{details["eyebrow"]}</div>
                    <h4>{details["title"]}</h4>
                    <p>{details["insight"]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.warning(
            "The dashboard is working, but no project figures were found in "
            "the `figures/` folder."
        )

    st.write("")
    left_column, right_column = st.columns([1.35, 1])
    with left_column:
        st.subheader("What the models saw")
        st.markdown(
            """
            - **Market features:** returns, volume change, price range,
              volatility, and moving-average position
            - **Sentiment features:** bullish/bearish ratios, labeled share,
              post volume, average sentiment, and daily sentiment change
            - **Target:** whether the stock closed up on the next trading day
            - **Validation:** earlier 80% of dates for training; latest 20% for testing
            """
        )
    with right_column:
        st.markdown(
            """
            <div class="insight-card">
                <div class="section-kicker">Important scope</div>
                <h4>Sentiment labels—not a text classifier</h4>
                <p>
                    The current pipeline uses Bullish and Bearish tags supplied
                    by StockTwits users. It does not train a separate NLP model
                    to read the raw post text.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

elif active_page == "Model Results":
    st.write("")
    st.markdown('<div class="section-kicker">Supervised learning</div>', unsafe_allow_html=True)
    st.header("The baseline remained the hardest model to beat")

    result_metrics = st.columns(3)
    with result_metrics[0]:
        metric_card("Majority baseline", "53.6%", "Predicts every test row as up")
    with result_metrics[1]:
        metric_card("Random Forest", "52.6%", "Best trained-model accuracy")
    with result_metrics[2]:
        metric_card("Best ROC-AUC", "0.505", "Approximately random ranking")

    st.write("")
    comparison_column, takeaway_column = st.columns([1.5, 1])
    with comparison_column:
        st.subheader("Compare evaluation metrics")
        selected_metric = st.selectbox(
            "Metric",
            ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
            label_visibility="collapsed",
        )
        result_bars(selected_metric)
    with takeaway_column:
        st.markdown(
            """
            <div class="insight-card">
                <div class="section-kicker">How to read this</div>
                <h4>Accuracy alone is not enough</h4>
                <p>
                    The baseline's high recall and F1 come from predicting every
                    observation as “up.” ROC-AUC near 0.50 shows that the trained
                    models did not reliably separate up days from down days.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.subheader("Full test-set results")
    formatted_results = RESULTS.style.format(
        {
            "Accuracy": "{:.3f}",
            "Precision": "{:.3f}",
            "Recall": "{:.3f}",
            "F1 Score": "{:.3f}",
            "ROC-AUC": lambda value: "N/A" if pd.isna(value) else f"{value:.3f}",
        }
    )
    st.dataframe(
        formatted_results,
        hide_index=True,
        use_container_width=True,
        height=145,
    )

    st.info(
        "Main finding: StockTwits sentiment did not add a reliable next-day "
        "predictive signal in the current combined experiment."
    )

    st.write("")
    st.markdown('<div class="section-kicker">Best way forward</div>', unsafe_allow_html=True)
    st.header("Use a stronger candidate—but require real out-of-sample improvement")
    st.info(
        "Random Forest performed best among the trained models, but it did not "
        "outperform the majority baseline. A better next step is to test "
        "gradient-boosted trees with walk-forward validation and retain them "
        "only if they demonstrate consistent out-of-sample improvement."
    )

    recommendation_columns = st.columns(3)
    with recommendation_columns[0]:
        st.markdown(
            """
            <div class="insight-card">
                <div class="section-kicker">Current winner</div>
                <h4>Random Forest</h4>
                <p>
                    It is the best trained model by accuracy at 52.6%, but it
                    still trails the 53.6% majority baseline and should not be
                    used for live trading.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with recommendation_columns[1]:
        st.markdown(
            """
            <div class="insight-card">
                <div class="section-kicker">Next model to test</div>
                <h4>Gradient-boosted trees</h4>
                <p>
                    XGBoost or LightGBM is the strongest next candidate for
                    nonlinear interactions among market and sentiment features.
                    This is a recommendation—not a validated result.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with recommendation_columns[2]:
        st.markdown(
            """
            <div class="insight-card">
                <div class="section-kicker">Validation rule</div>
                <h4>Walk-forward testing</h4>
                <p>
                    Compare market-only, sentiment-only, and combined models
                    across future time windows. Keep a model only if its
                    improvement is consistent across dates and tickers.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.warning(
        "Best decision today: do not deploy any current model as a trading "
        "system. The next model must beat the baseline on unseen future data, "
        "not just during tuning."
    )

elif active_page == "K-Means Finding":
    st.write("")
    st.markdown('<div class="section-kicker">Unsupervised learning</div>', unsafe_allow_html=True)
    st.header("The loudest days showed a small contrarian tilt")
    st.write(
        "K-Means grouped ticker-days using bullish share, log post volume, "
        "same-day return, and intraday range. Next-day return was held out and "
        "measured only after the clusters were formed."
    )

    finding_columns = st.columns(2)
    with finding_columns[0]:
        st.markdown(
            """
            <div class="finding hype">
                <div class="finding-label">Loud hype days</div>
                <div class="finding-value">−0.27%</div>
                <p>Average next-day return after approximately +5.8% same-day gains.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with finding_columns[1]:
        st.markdown(
            """
            <div class="finding panic">
                <div class="finding-label">Loud panic days</div>
                <div class="finding-value">+0.81%</div>
                <p>Average next-day return after approximately −4.4% same-day declines.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    if (FIGURES_PATH / "07_kmeans_clusters.png").exists():
        st.image(
            str(FIGURES_PATH / "07_kmeans_clusters.png"),
            caption="Five K-Means behavior clusters",
            use_container_width=True,
        )
    else:
        st.info(
            "Add `07_kmeans_clusters.png` to the `figures/` folder to display "
            "the cluster chart here."
        )
    st.warning(
        "This is an association in the analyzed sample. The clusters were not "
        "tested as an out-of-sample trading strategy."
    )

elif active_page == "Limitations":
    st.write("")
    st.markdown('<div class="section-kicker">Responsible interpretation</div>', unsafe_allow_html=True)
    st.header("What this project can—and cannot—claim")

    limitation_column, next_step_column = st.columns(2)
    with limitation_column:
        st.subheader("Current limitations")
        st.markdown(
            """
            - Only four large-cap U.S. stocks are included.
            - User-supplied sentiment labels may be missing, sarcastic, or noisy.
            - Posts are grouped by UTC date rather than U.S. market sessions.
            - The combined model has not yet been compared with controlled
              market-only and sentiment-only experiments.
            - News headlines were cleaned but not used in the current models.
            - No out-of-sample trading strategy or transaction costs were tested.
            """
        )
    with next_step_column:
        st.subheader("Best next experiments")
        st.markdown(
            """
            1. Compare market-only, sentiment-only, and combined features.
            2. Align posts to U.S. market sessions and trading-day cutoffs.
            3. Test longer horizons such as weekly direction.
            4. Expand to more sectors and smaller-cap stocks.
            5. Train a text classifier and compare social sentiment with news.
            """
        )

    st.write("")
    st.markdown('<div class="section-kicker">Bias and mitigation</div>', unsafe_allow_html=True)
    st.header("Where bias can enter the analysis")

    bias_mitigations = pd.DataFrame(
        {
            "Potential bias": [
                "Platform selection",
                "Self-reported labels",
                "Ticker selection",
                "Time-period / market regime",
                "Class imbalance",
                "Timing and leakage",
            ],
            "Why it matters": [
                "StockTwits users may not represent all investors.",
                "Bullish and Bearish tags can be missing, sarcastic, or incorrect.",
                "Four popular technology stocks may not represent the wider market.",
                "The 2020–2022 period includes unusual volatility and retail activity.",
                "Up days and bullish labels are more common, which can inflate accuracy.",
                "UTC dates or future-derived features can mix information across trading sessions.",
            ],
            "Mitigation": [
                "Compare multiple social platforms and news sources.",
                "Validate a hand-labeled sample and test a text-based sentiment model.",
                "Add more sectors, company sizes, and per-ticker reporting.",
                "Use multiple market regimes and rolling walk-forward evaluation.",
                "Report balanced accuracy, ROC-AUC, and class-specific precision/recall.",
                "Align posts to market cutoffs and fit every transformation on training data only.",
            ],
        }
    )
    st.dataframe(
        bias_mitigations,
        hide_index=True,
        use_container_width=True,
        height=320,
    )

    st.success(
        "Responsible takeaway: the negative result is useful. It cautions "
        "against treating crowd sentiment as a dependable next-day trading signal."
    )

st.markdown(
    """
    <div class="footer">
        AI4ALL Ignite · Group 04 · Educational research dashboard ·
        Not financial advice
    </div>
    """,
    unsafe_allow_html=True,
)

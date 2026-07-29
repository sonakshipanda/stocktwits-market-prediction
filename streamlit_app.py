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
        :root {
            --bg: #F5EEE4;
            --ink: #2B2119;
            --accent: #A83B2A;
            --accent-strong: #7F2A20;
            --accent-soft: #C45A48;
            --muted: #7A6E62;
            --rule: #D9CFC2;
            --positive: #2E7D4F;
            --negative: #B3372B;
            --neutral: #9C9081;
            --aapl: #1F77B4;
            --amzn: #FF7F0E;
            --nvda: #2CA02C;
            --tsla: #D62728;
        }

        .stApp {
            background: var(--bg);
            color: var(--ink);
            font-family: Calibri, "Aptos", Arial, sans-serif;
        }

        .stApp p,
        .stApp li,
        .stApp label,
        .stApp div,
        .stApp span {
            color: var(--ink);
        }

        .stApp a {
            color: var(--accent-strong);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2.8rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3 {
            color: var(--accent) !important;
            font-family: Cambria, Georgia, serif !important;
            font-weight: 700 !important;
            letter-spacing: -0.015em;
        }

        h4 {
            color: var(--ink);
            font-family: Cambria, Georgia, serif;
        }

        p, li, label, div[data-testid="stCaptionContainer"] {
            color: var(--ink);
        }

        .hero {
            max-width: 900px;
            padding: 0.4rem 0 2rem 1.4rem;
            border-left: 5px solid var(--accent);
            border-bottom: 1px solid var(--rule);
        }

        .hero-badge {
            display: inline-block;
            margin-bottom: 0.8rem;
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }

        .hero h1 {
            max-width: 760px;
            margin: 0;
            color: var(--accent) !important;
            font-size: clamp(2.1rem, 4.5vw, 4rem);
            line-height: 1.06;
        }

        .hero p {
            max-width: 760px;
            margin: 1rem 0 0;
            color: var(--ink);
            font-size: 1.08rem;
            line-height: 1.65;
        }

        .hero-answer {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            margin-top: 1.3rem;
            color: var(--positive);
            font-weight: 700;
        }

        .hero-answer-dot {
            width: 0.62rem;
            height: 0.62rem;
            border-radius: 50%;
            background: var(--positive);
            box-shadow: 0 0 0 6px rgba(46, 125, 79, 0.12), 0 0 18px rgba(46, 125, 79, 0.38);
        }

        .metric-card {
            min-height: 112px;
            padding: 1rem 1rem 0.9rem;
            border: 1px solid rgba(171, 153, 133, 0.7);
            border-radius: 1rem;
            background: rgba(255, 255, 255, 0.78);
            box-shadow: 0 10px 22px rgba(70, 49, 29, 0.08), 0 2px 5px rgba(70, 49, 29, 0.04);
        }

        .metric-label {
            margin-bottom: 0.45rem;
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }

        .metric-value {
            color: var(--accent-strong);
            font-family: Cambria, Georgia, serif;
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            line-height: 1.05;
        }

        .metric-detail {
            margin-top: 0.45rem;
            color: #5E5449;
            font-size: 0.84rem;
        }

        .section-kicker {
            margin-bottom: 0.25rem;
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.09em;
            text-transform: uppercase;
        }

        .insight-card, .question-card, .pipeline-card {
            height: 100%;
            padding: 1.15rem 1.1rem;
            border: 1px solid rgba(171, 153, 133, 0.68);
            border-radius: 1rem;
            background: rgba(255, 255, 255, 0.74);
            box-shadow: 0 10px 22px rgba(70, 49, 29, 0.08), 0 2px 5px rgba(70, 49, 29, 0.04);
        }

        .question-card {
            padding: 1.2rem 1.2rem 1.35rem;
        }

        .question-card strong {
            display: block;
            margin-bottom: 0.4rem;
            color: var(--accent-strong);
            font-size: 0.78rem;
            letter-spacing: 0.07em;
            text-transform: uppercase;
        }

        .question-card span {
            color: var(--ink);
            font-family: Cambria, Georgia, serif;
            font-size: 1.25rem;
            font-weight: 700;
            line-height: 1.45;
        }

        .pipeline-number {
            display: block;
            margin-bottom: 0.85rem;
            color: var(--accent-strong);
            font-family: Cambria, Georgia, serif;
            font-size: 1.45rem;
            font-weight: 800;
        }

        .pipeline-card h4 {
            margin: 0 0 0.35rem;
            color: #241B15;
        }

        .pipeline-card p, .insight-card p {
            margin: 0;
            color: #5E5449;
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
            color: var(--ink);
            font-weight: 700;
        }

        .result-track {
            height: 0.75rem;
            overflow: hidden;
            background: rgba(217, 207, 194, 0.85);
            border-radius: 999px;
        }

        .result-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--accent-soft), var(--accent));
            border-radius: 999px;
        }

        .result-fill.baseline {
            background: linear-gradient(90deg, #B0A59A, var(--neutral));
        }

        .result-score {
            color: var(--ink);
            font-weight: 800;
            text-align: right;
        }

        .finding {
            padding: 1.15rem 1.1rem;
            border: 1px solid rgba(171, 153, 133, 0.68);
            border-top: 4px solid;
            border-radius: 1rem;
            background: rgba(255, 255, 255, 0.74);
            box-shadow: 0 10px 22px rgba(70, 49, 29, 0.08), 0 2px 5px rgba(70, 49, 29, 0.04);
        }

        .finding.hype {
            border-color: var(--negative);
        }

        .finding.panic {
            border-color: var(--positive);
        }

        .finding-label {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .finding-value {
            margin: 0.3rem 0;
            font-family: Cambria, Georgia, serif;
            font-size: 2rem;
            font-weight: 800;
        }

        .finding.hype .finding-value {
            color: var(--negative);
        }

        .finding.panic .finding-value {
            color: var(--positive);
        }

        .finding p {
            margin: 0;
            color: #5E5449;
        }

        .text-note {
            margin: 1rem 0;
            padding: 0.95rem 1rem;
            border: 1px solid rgba(171, 153, 133, 0.65);
            border-radius: 0.9rem;
            background: rgba(255, 255, 255, 0.68);
            box-shadow: 0 8px 18px rgba(70, 49, 29, 0.06);
            color: var(--ink);
        }

        .text-note strong {
            color: var(--accent-strong);
        }

        .text-note.positive strong {
            color: var(--positive);
        }

        .text-note.negative strong {
            color: var(--negative);
        }

        .footer {
            margin-top: 2.5rem;
            padding-top: 1.25rem;
            border-top: 1px solid var(--rule);
            color: var(--muted);
            font-size: 0.85rem;
            text-align: center;
        }

        div[data-baseweb="tab-list"] {
            gap: 1.2rem;
            margin-top: 1rem;
            border-bottom: 1px solid var(--rule);
        }

        button[data-baseweb="tab"] {
            color: var(--muted);
            font-weight: 700;
            border-radius: 999px 999px 0 0;
            padding: 0.55rem 1rem;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            color: #FFFFFF;
            background: var(--accent-strong);
            box-shadow: 0 8px 18px rgba(127, 42, 32, 0.2);
        }

        div[data-baseweb="select"] > div {
            border-color: rgba(171, 153, 133, 0.82);
            border-radius: 0.9rem;
            background: rgba(255, 255, 255, 0.86);
            box-shadow: 0 6px 16px rgba(70, 49, 29, 0.05);
        }

        div[data-baseweb="select"] input {
            color: var(--ink);
        }

        div[data-baseweb="select"] svg {
            color: var(--accent-strong);
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(171, 153, 133, 0.7);
            border-radius: 1rem;
            background: rgba(255, 255, 255, 0.8);
            box-shadow: 0 10px 22px rgba(70, 49, 29, 0.08), 0 2px 5px rgba(70, 49, 29, 0.04);
            overflow: hidden;
        }

        div[data-testid="stExpander"] {
            border: 1px solid rgba(171, 153, 133, 0.7);
            border-radius: 0.9rem;
            background: rgba(255, 255, 255, 0.68);
            box-shadow: 0 8px 18px rgba(70, 49, 29, 0.05);
        }

        div[data-testid="stExpander"] details > summary {
            border-radius: 0.9rem;
        }

        hr {
            border-color: var(--rule) !important;
        }

        .stButton > button {
            border-radius: 999px;
            border: 1px solid rgba(171, 153, 133, 0.8);
            background: rgba(255, 255, 255, 0.84);
            color: var(--ink);
            box-shadow: 0 6px 14px rgba(70, 49, 29, 0.05);
        }

        .stButton > button:hover {
            border-color: var(--accent-soft);
            color: var(--accent-strong);
        }

        @media (max-width: 760px) {
            .block-container {
                padding-top: 1rem;
            }

            .hero {
                padding: 0.2rem 0 1.5rem 1rem;
            }

            .result-bar-row {
                grid-template-columns: 1fr 55px;
            }

            .result-track {
                grid-column: 1 / -1;
                grid-row: 2;
            }

            .metric-card,
            .insight-card,
            .question-card,
            .pipeline-card,
            .finding,
            .text-note,
            div[data-testid="stDataFrame"],
            div[data-testid="stExpander"] {
                border-radius: 0.8rem;
            }

            button[data-baseweb="tab"] {
                padding: 0.45rem 0.8rem;
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


def text_note(label: str, message: str, tone: str = "") -> None:
    st.markdown(
        f"""
        <div class="text-note {tone}">
            <strong>{label}</strong> {message}
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

overview_tab, results_tab, figures_tab, clusters_tab, limits_tab = st.tabs(
    [
        "Overview",
        "Model Results",
        "Chart Explorer",
        "K-Means Finding",
        "Limitations",
    ]
)

with overview_tab:
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

with results_tab:
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
        width="stretch",
        height=145,
    )

    text_note(
        "Main finding:",
        "StockTwits sentiment did not add a reliable next-day predictive "
        "signal in the current combined experiment.",
    )

with figures_tab:
    st.write("")
    st.markdown('<div class="section-kicker">Interactive gallery</div>', unsafe_allow_html=True)
    st.header("Explore the project visuals")
    st.caption(
        "Choose one chart at a time for a presentation-friendly view and a short interpretation."
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
                width="stretch",
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

        with st.expander("Show all figure files"):
            for file_name in available_figures:
                st.markdown(f"- `{file_name}` — {FIGURE_DETAILS[file_name]['title']}")
    else:
        st.error("No project figures were found in the `figures/` folder.")

with clusters_tab:
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
            width="stretch",
        )
    text_note(
        "Interpret carefully:",
        "This is an association in the analyzed sample. The clusters were not "
        "tested as an out-of-sample trading strategy.",
        "negative",
    )

with limits_tab:
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

    text_note(
        "Responsible takeaway:",
        "The negative result is useful. It cautions against treating crowd "
        "sentiment as a dependable next-day trading signal.",
        "positive",
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

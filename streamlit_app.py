from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="StockTwits Market Prediction",
    page_icon="📈",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).parent
FIGURES_PATH = PROJECT_ROOT / "figures"


st.title("📈 StockTwits Market Prediction")
st.write(
    "Exploring whether StockTwits sentiment can help predict the "
    "next-trading-day direction of AAPL, AMZN, NVDA, and TSLA."
)

# Project summary
col1, col2, col3, col4 = st.columns(4)

col1.metric("StockTwits Posts", "4.2M+")
col2.metric("Stocks Analyzed", "4")
col3.metric("Ticker-Days", "3,936")
col4.metric("Best Model Accuracy", "52.6%")


overview_tab, results_tab, figures_tab, limitations_tab = st.tabs(
    ["Project Overview", "Model Results", "Visualizations", "Limitations"]
)

with overview_tab:
    st.header("Project Overview")
    st.write(
        """
        This project combines historical market data with daily StockTwits
        sentiment features. Logistic Regression and Random Forest models
        were trained to predict whether each stock would rise or fall on
        the next trading day.
        """
    )

    st.info(
        "The current project uses the Bullish and Bearish sentiment labels "
        "provided in the StockTwits dataset."
    )

with results_tab:
    st.header("Supervised Model Results")

    results = pd.DataFrame({
        "Model": [
            "Majority Baseline",
            "Logistic Regression",
            "Random Forest"
        ],
        "Accuracy": [0.536, 0.506, 0.526],
        "Precision": [0.536, 0.530, 0.544],
        "Recall": [1.000, 0.699, 0.709],
        "F1 Score": [0.698, 0.603, 0.616],
        "ROC-AUC": ["N/A", "0.502", "0.505"]
    })

    st.dataframe(
        results,
        hide_index=True,
        use_container_width=True
    )

    st.warning(
        "Neither trained model outperformed the majority-class baseline. "
        "ROC-AUC scores near 0.50 indicate performance close to random guessing."
    )

with figures_tab:
    st.header("Project Visualizations")

    figure_files = sorted(FIGURES_PATH.glob("*.png"))

    if figure_files:
        for figure in figure_files:
            caption = figure.stem.replace("_", " ").title()
            st.subheader(caption)
            st.image(
                str(figure),
                caption=caption,
                use_container_width=True
            )
    else:
        st.error("No PNG figures were found in the figures folder.")

with limitations_tab:
    st.header("Limitations")

    st.write(
        """
        - The analysis covers only four large-cap technology stocks.
        - StockTwits sentiment labels are incomplete and potentially noisy.
        - The models did not reliably outperform the majority baseline.
        - The current analysis does not prove a profitable trading strategy.
        - News headlines were cleaned but were not included in the final models.
        """
    )

st.divider()
st.caption("AI4ALL StockTwits Market Prediction Project")
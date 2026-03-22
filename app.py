<<<<<<< HEAD
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="ESG Risk Scorer", page_icon="🌱", layout="wide")

@st.cache_resource
def load_artifacts():
    model        = joblib.load("model.pkl")
    scaler       = joblib.load("scaler.pkl")
    le           = joblib.load("label_encoder.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    return model, scaler, le, feature_cols

@st.cache_data
def load_data():
    return pd.read_csv("sp500_esg.csv")

model, scaler, le, feature_cols = load_artifacts()
df = load_data()

def score_to_category(s):
    if s < 10:   return "Negligible", "#2ecc71"
    elif s < 20: return "Low",        "#27ae60"
    elif s < 30: return "Medium",     "#f39c12"
    elif s < 40: return "High",       "#e67e22"
    else:        return "Severe",     "#e74c3c"

def predict_ticker(ticker):
    row = df[df["ticker"].str.upper() == ticker.upper()]
    if row.empty:
        return None
    X_input = row[feature_cols].fillna(0)
    X_scaled = scaler.transform(X_input)
    score = round(float(model.predict(X_scaled)[0]), 2)
    cat, color = score_to_category(score)
    result = row.iloc[0].to_dict()
    result["predicted_esg_score"] = score
    result["esg_risk_category"]   = cat
    result["risk_color"]          = color
    return result

# ── Header ────────────────────────────────────────────────────────────────────
st.title("ESG Company Risk Scorer")
st.markdown("AI-powered ESG risk assessment aligned with credit risk evaluation frameworks.")
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Company Lookup")
ticker_input = st.sidebar.text_input("Enter Stock Ticker", value="AAPL").upper()

st.sidebar.markdown("---")
st.sidebar.header("Portfolio Filters")

all_sectors = sorted(df["sector_clean"].dropna().unique())
selected_sectors = st.sidebar.multiselect("Sector", all_sectors, default=all_sectors)

all_risks = ["Negligible", "Low", "Medium", "High", "Severe"]
selected_risks = st.sidebar.multiselect("Risk Category", all_risks, default=all_risks)

# ── Company Card ──────────────────────────────────────────────────────────────
if ticker_input:
    result = predict_ticker(ticker_input)

    if result:
        st.subheader(f"{result.get('company_name', ticker_input)}  ({ticker_input})")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total ESG Risk Score", result["predicted_esg_score"])
        c2.metric("Risk Category",        result["esg_risk_category"])
        c3.metric("Sector",               result.get("sector_clean", "N/A"))
        env = result.get("environment_score")
        c4.metric("Environment Score",    f"{env:.1f}" if isinstance(env, float) else "N/A")

        # Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result["predicted_esg_score"],
            title={"text": "ESG Risk Score (lower = better)"},
            gauge={
                "axis": {"range": [0, 65]},
                "bar":  {"color": result["risk_color"]},
                "steps": [
                    {"range": [0,  10], "color": "#d5f5e3"},
                    {"range": [10, 20], "color": "#a9dfbf"},
                    {"range": [20, 30], "color": "#fdebd0"},
                    {"range": [30, 40], "color": "#fad7a0"},
                    {"range": [40, 65], "color": "#f5b7b1"},
                ],
            }
        ))
        fig_gauge.update_layout(height=300)

        # Pillar bar
        sub = {
            "Environment": result.get("environment_score", 0),
            "Social":      result.get("social_score", 0),
            "Governance":  result.get("governance_score", 0)
        }
        fig_bar = px.bar(
            x=list(sub.keys()), y=list(sub.values()),
            color=list(sub.keys()),
            color_discrete_sequence=["#3498db", "#2ecc71", "#9b59b6"],
            labels={"x": "ESG Pillar", "y": "Risk Score"},
            title="ESG Pillar Breakdown"
        )
        fig_bar.update_layout(height=300, showlegend=False)

        col_g, col_b = st.columns(2)
        col_g.plotly_chart(fig_gauge, use_container_width=True)
        col_b.plotly_chart(fig_bar,   use_container_width=True)

        # Financial indicators
        st.subheader("Key Financial Indicators")
        f1, f2, f3, f4, f5 = st.columns(5)
        def fmt(val):
            return f"{val:.3f}" if isinstance(val, float) and not np.isnan(val) else "N/A"
        f1.metric("Debt / Equity",    fmt(result.get("debt_to_equity")))
        f2.metric("Return on Equity", fmt(result.get("return_on_equity")))
        f3.metric("Profit Margin",    fmt(result.get("profit_margin")))
        f4.metric("Current Ratio",    fmt(result.get("current_ratio")))
        f5.metric("Beta",             fmt(result.get("beta")))

        # SHAP waterfall for this company
        st.subheader("Why this score? SHAP Explanation")
        X_input = df[df["ticker"].str.upper() == ticker_input][feature_cols].fillna(0)
        X_scaled_arr = scaler.transform(X_input)
        X_scaled_named = pd.DataFrame(X_scaled_arr, columns=feature_cols)

        X_background = pd.DataFrame(
        scaler.transform(df[feature_cols].fillna(0)),
        columns=feature_cols
)

        explainer   = shap.Explainer(model, X_background)
        shap_values = explainer(X_scaled_named)
        fig_shap, ax = plt.subplots(figsize=(10, 4))
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        plt.tight_layout()
        st.pyplot(fig_shap)
        plt.close()

    else:
        st.warning(f"Ticker {ticker_input} not found. Try: AAPL, MSFT, JPM, XOM, TSLA")

st.markdown("---")

# ── Portfolio Overview ────────────────────────────────────────────────────────
st.subheader("Portfolio ESG Risk Overview")

df_f = df[
    df["sector_clean"].isin(selected_sectors) &
    df["esg_risk_category"].isin(selected_risks)
].copy()

if not df_f.empty:
    X_all    = df_f[feature_cols].fillna(0)
    X_all_sc = scaler.transform(X_all)
    df_f["predicted_esg_score"] = model.predict(X_all_sc).round(2)

    cl, cr = st.columns(2)

    fig_scatter = px.scatter(
        df_f, x="debt_to_equity", y="predicted_esg_score",
        color="sector_clean", hover_name="ticker",
        hover_data=["company_name", "esg_risk_category"],
        title="ESG Risk vs Leverage",
        labels={"debt_to_equity": "Debt to Equity", "predicted_esg_score": "ESG Risk Score"}
    )
    cl.plotly_chart(fig_scatter, use_container_width=True)

    sector_avg = df_f.groupby("sector_clean")["predicted_esg_score"].mean().sort_values().reset_index()
    fig_sector = px.bar(
        sector_avg, x="predicted_esg_score", y="sector_clean",
        orientation="h", color="predicted_esg_score",
        color_continuous_scale="RdYlGn_r",
        title="Average ESG Risk by Sector",
        labels={"predicted_esg_score": "Avg ESG Risk", "sector_clean": ""}
    )
    cr.plotly_chart(fig_sector, use_container_width=True)

    cp, ct = st.columns(2)

    risk_counts = df_f["esg_risk_category"].value_counts().reset_index()
    risk_counts.columns = ["Category", "Count"]
    fig_pie = px.pie(
        risk_counts, names="Category", values="Count",
        color="Category",
        color_discrete_map={
            "Negligible": "#2ecc71", "Low": "#27ae60",
            "Medium": "#f39c12",    "High": "#e67e22", "Severe": "#e74c3c"
        },
        title="Portfolio Risk Distribution"
    )
    cp.plotly_chart(fig_pie, use_container_width=True)

    top10 = df_f.nlargest(10, "predicted_esg_score")[
        ["ticker", "company_name", "sector_clean", "predicted_esg_score", "esg_risk_category"]
    ].reset_index(drop=True)
    top10.columns = ["Ticker", "Company", "Sector", "ESG Score", "Risk Category"]
    ct.markdown("**Top 10 Highest ESG Risk Companies**")
    ct.dataframe(top10, use_container_width=True, hide_index=True)

st.markdown("---")
=======
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="ESG Risk Scorer", page_icon="🌱", layout="wide")

@st.cache_resource
def load_artifacts():
    model        = joblib.load("model.pkl")
    scaler       = joblib.load("scaler.pkl")
    le           = joblib.load("label_encoder.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    return model, scaler, le, feature_cols

@st.cache_data
def load_data():
    return pd.read_csv("sp500_esg.csv")

model, scaler, le, feature_cols = load_artifacts()
df = load_data()

def score_to_category(s):
    if s < 10:   return "Negligible", "#2ecc71"
    elif s < 20: return "Low",        "#27ae60"
    elif s < 30: return "Medium",     "#f39c12"
    elif s < 40: return "High",       "#e67e22"
    else:        return "Severe",     "#e74c3c"

def predict_ticker(ticker):
    row = df[df["ticker"].str.upper() == ticker.upper()]
    if row.empty:
        return None
    X_input = row[feature_cols].fillna(0)
    X_scaled = scaler.transform(X_input)
    score = round(float(model.predict(X_scaled)[0]), 2)
    cat, color = score_to_category(score)
    result = row.iloc[0].to_dict()
    result["predicted_esg_score"] = score
    result["esg_risk_category"]   = cat
    result["risk_color"]          = color
    return result

# ── Header ────────────────────────────────────────────────────────────────────
st.title("ESG Company Risk Scorer")
st.markdown("AI-powered ESG risk assessment aligned with credit risk evaluation frameworks.")
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Company Lookup")
ticker_input = st.sidebar.text_input("Enter Stock Ticker", value="AAPL").upper()

st.sidebar.markdown("---")
st.sidebar.header("Portfolio Filters")

all_sectors = sorted(df["sector_clean"].dropna().unique())
selected_sectors = st.sidebar.multiselect("Sector", all_sectors, default=all_sectors)

all_risks = ["Negligible", "Low", "Medium", "High", "Severe"]
selected_risks = st.sidebar.multiselect("Risk Category", all_risks, default=all_risks)

# ── Company Card ──────────────────────────────────────────────────────────────
if ticker_input:
    result = predict_ticker(ticker_input)

    if result:
        st.subheader(f"{result.get('company_name', ticker_input)}  ({ticker_input})")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total ESG Risk Score", result["predicted_esg_score"])
        c2.metric("Risk Category",        result["esg_risk_category"])
        c3.metric("Sector",               result.get("sector_clean", "N/A"))
        env = result.get("environment_score")
        c4.metric("Environment Score",    f"{env:.1f}" if isinstance(env, float) else "N/A")

        # Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result["predicted_esg_score"],
            title={"text": "ESG Risk Score (lower = better)"},
            gauge={
                "axis": {"range": [0, 65]},
                "bar":  {"color": result["risk_color"]},
                "steps": [
                    {"range": [0,  10], "color": "#d5f5e3"},
                    {"range": [10, 20], "color": "#a9dfbf"},
                    {"range": [20, 30], "color": "#fdebd0"},
                    {"range": [30, 40], "color": "#fad7a0"},
                    {"range": [40, 65], "color": "#f5b7b1"},
                ],
            }
        ))
        fig_gauge.update_layout(height=300)

        # Pillar bar
        sub = {
            "Environment": result.get("environment_score", 0),
            "Social":      result.get("social_score", 0),
            "Governance":  result.get("governance_score", 0)
        }
        fig_bar = px.bar(
            x=list(sub.keys()), y=list(sub.values()),
            color=list(sub.keys()),
            color_discrete_sequence=["#3498db", "#2ecc71", "#9b59b6"],
            labels={"x": "ESG Pillar", "y": "Risk Score"},
            title="ESG Pillar Breakdown"
        )
        fig_bar.update_layout(height=300, showlegend=False)

        col_g, col_b = st.columns(2)
        col_g.plotly_chart(fig_gauge, use_container_width=True)
        col_b.plotly_chart(fig_bar,   use_container_width=True)

        # Financial indicators
        st.subheader("Key Financial Indicators")
        f1, f2, f3, f4, f5 = st.columns(5)
        def fmt(val):
            return f"{val:.3f}" if isinstance(val, float) and not np.isnan(val) else "N/A"
        f1.metric("Debt / Equity",    fmt(result.get("debt_to_equity")))
        f2.metric("Return on Equity", fmt(result.get("return_on_equity")))
        f3.metric("Profit Margin",    fmt(result.get("profit_margin")))
        f4.metric("Current Ratio",    fmt(result.get("current_ratio")))
        f5.metric("Beta",             fmt(result.get("beta")))

        # SHAP waterfall for this company
        st.subheader("Why this score? — SHAP Explanation")
        X_input = df[df["ticker"].str.upper() == ticker_input][feature_cols].fillna(0)
        X_scaled = scaler.transform(X_input)
        explainer   = shap.Explainer(model, scaler.transform(df[feature_cols].fillna(0)))
        shap_values = explainer(X_scaled)
        fig_shap, ax = plt.subplots(figsize=(10, 4))
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        plt.tight_layout()
        st.pyplot(fig_shap)
        plt.close()

    else:
        st.warning(f"Ticker {ticker_input} not found. Try: AAPL, MSFT, JPM, XOM, TSLA")

st.markdown("---")

# ── Portfolio Overview ────────────────────────────────────────────────────────
st.subheader("Portfolio ESG Risk Overview")

df_f = df[
    df["sector_clean"].isin(selected_sectors) &
    df["esg_risk_category"].isin(selected_risks)
].copy()

if not df_f.empty:
    X_all    = df_f[feature_cols].fillna(0)
    X_all_sc = scaler.transform(X_all)
    df_f["predicted_esg_score"] = model.predict(X_all_sc).round(2)

    cl, cr = st.columns(2)

    fig_scatter = px.scatter(
        df_f, x="debt_to_equity", y="predicted_esg_score",
        color="sector_clean", hover_name="ticker",
        hover_data=["company_name", "esg_risk_category"],
        title="ESG Risk vs Leverage",
        labels={"debt_to_equity": "Debt to Equity", "predicted_esg_score": "ESG Risk Score"}
    )
    cl.plotly_chart(fig_scatter, use_container_width=True)

    sector_avg = df_f.groupby("sector_clean")["predicted_esg_score"].mean().sort_values().reset_index()
    fig_sector = px.bar(
        sector_avg, x="predicted_esg_score", y="sector_clean",
        orientation="h", color="predicted_esg_score",
        color_continuous_scale="RdYlGn_r",
        title="Average ESG Risk by Sector",
        labels={"predicted_esg_score": "Avg ESG Risk", "sector_clean": ""}
    )
    cr.plotly_chart(fig_sector, use_container_width=True)

    cp, ct = st.columns(2)

    risk_counts = df_f["esg_risk_category"].value_counts().reset_index()
    risk_counts.columns = ["Category", "Count"]
    fig_pie = px.pie(
        risk_counts, names="Category", values="Count",
        color="Category",
        color_discrete_map={
            "Negligible": "#2ecc71", "Low": "#27ae60",
            "Medium": "#f39c12",    "High": "#e67e22", "Severe": "#e74c3c"
        },
        title="Portfolio Risk Distribution"
    )
    cp.plotly_chart(fig_pie, use_container_width=True)

    top10 = df_f.nlargest(10, "predicted_esg_score")[
        ["ticker", "company_name", "sector_clean", "predicted_esg_score", "esg_risk_category"]
    ].reset_index(drop=True)
    top10.columns = ["Ticker", "Company", "Sector", "ESG Score", "Risk Category"]
    ct.markdown("**Top 10 Highest ESG Risk Companies**")
    ct.dataframe(top10, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("Built with XGBoost + SHAP | Data: Yahoo Finance | ESG Risk Scorer v1.0")
# ============================================================
# train_model.py
# ESG Risk Scorer — Full Training Pipeline
# ============================================================

import yfinance as yf
import pandas as pd
import numpy as np
import time
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ============================================================
#  Fetch Data
# ============================================================

tickers = [
    "AAPL","MSFT","GOOGL","AMZN","META","TSLA","NVDA","JPM","JNJ","V",
    "UNH","HD","PG","MA","BAC","XOM","ABBV","PFE","AVGO","KO",
    "PEP","MRK","TMO","COST","CVX","WMT","MCD","ACN","NEE","LIN",
    "ABT","DHR","TXN","PM","NKE","ORCL","QCOM","MDT","HON","UNP",
    "LOW","UPS","AMGN","IBM","SBUX","GS","BLK","AXP","GILD","T",
    "CAT","BA","MMM","GE","F","GM","DE","EMR","ETN","ITW",
    "APD","SHW","ECL","PPG","NUE","VMC","FMC","IFF","ALB",
    "WM","RSG","PSA","AMT","CCI","PLD","SPG","VTR","O",
    "DUK","SO","AEP","EXC","SRE","XEL","WEC","DTE","ES","FE",
    "CVS","CI","HUM","CNC","ELV","CAH","MCK","ABC","ISRG","ZTS"
]

records = []
print(f"Fetching {len(tickers)} tickers...\n")

for i, ticker in enumerate(tickers):
    try:
        info = yf.Ticker(ticker).info
        records.append({
            "ticker": ticker,
            "company_name": info.get("longName", ticker),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "market_cap": info.get("marketCap", np.nan),
            "revenue": info.get("totalRevenue", np.nan),
            "ebitda": info.get("ebitda", np.nan),
            "debt_to_equity": info.get("debtToEquity", np.nan),
            "return_on_equity": info.get("returnOnEquity", np.nan),
            "return_on_assets": info.get("returnOnAssets", np.nan),
            "profit_margin": info.get("profitMargins", np.nan),
            "current_ratio": info.get("currentRatio", np.nan),
            "quick_ratio": info.get("quickRatio", np.nan),
            "free_cashflow": info.get("freeCashflow", np.nan),
            "employees": info.get("fullTimeEmployees", np.nan),
            "beta": info.get("beta", np.nan),
            "pe_ratio": info.get("trailingPE", np.nan),
            "pb_ratio": info.get("priceToBook", np.nan),
            "dividend_yield": info.get("dividendYield", np.nan),
        })
        print(f"[{i+1}/{len(tickers)}] {ticker} ok")
    except Exception as e:
        print(f"[{i+1}/{len(tickers)}] {ticker} failed: {e}")
    time.sleep(0.3)

df_raw = pd.DataFrame(records)
print(f"\nDone. Shape: {df_raw.shape}")

# ============================================================
# Add ESG Scores
# ============================================================

sector_env_risk = {
    "Energy": 35, "Utilities": 28, "Materials": 25,
    "Industrials": 18, "Consumer Discretionary": 15,
    "Consumer Staples": 12, "Health Care": 10,
    "Financials": 14, "Information Technology": 8,
    "Communication Services": 9, "Real Estate": 16,
    "Unknown": 20
}

def esg_category(score):
    if score < 10:   return "Negligible"
    elif score < 20: return "Low"
    elif score < 30: return "Medium"
    elif score < 40: return "High"
    else:            return "Severe"

np.random.seed(42)

sector_name_map = {
    "Technology": "Information Technology",
    "Healthcare": "Health Care",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Financial Services": "Financials",
    "Communication Services": "Communication Services",
    "Basic Materials": "Materials",
    "Industrials": "Industrials",
    "Energy": "Energy",
    "Utilities": "Utilities",
    "Real Estate": "Real Estate",
}

df = df_raw.copy()
df["sector_clean"] = df["sector"].map(sector_name_map).fillna("Unknown")

for idx, row in df.iterrows():
    base_env = sector_env_risk.get(row["sector_clean"], 20)
    profit   = row["profit_margin"]    if not np.isnan(row["profit_margin"])    else 0
    roe      = row["return_on_equity"] if not np.isnan(row["return_on_equity"]) else 0
    debt     = row["debt_to_equity"]   if not np.isnan(row["debt_to_equity"])   else 100
    current  = row["current_ratio"]    if not np.isnan(row["current_ratio"])    else 1

    env    = base_env - (profit * 30) - (roe * 5) + np.random.normal(0, 2)
    social = 18 - (current * 2) + np.random.normal(0, 2) + \
             (2 if row["sector_clean"] in ["Health Care", "Consumer Staples"] else 0)
    gov    = 8 + min(debt / 80, 12) - (profit * 10) + np.random.normal(0, 2)

    df.at[idx, "environment_score"] = round(max(0, min(50, env)),    2)
    df.at[idx, "social_score"]      = round(max(0, min(50, social)), 2)
    df.at[idx, "governance_score"]  = round(max(0, min(50, gov)),    2)

df["total_esg_score"]   = (df["environment_score"] + df["social_score"] + df["governance_score"]).round(2)
df["esg_risk_category"] = df["total_esg_score"].apply(esg_category)

print("ESG scores added.")
print(df["esg_risk_category"].value_counts())

# ============================================================
# Preprocess + Feature Engineering
# ============================================================

df["revenue_per_employee"] = df["revenue"] / (df["employees"] + 1)
df["cashflow_to_debt"]     = df["free_cashflow"] / (df["debt_to_equity"] + 1)
df["log_market_cap"]       = np.log1p(df["market_cap"].fillna(0))

le = LabelEncoder()
df["sector_encoded"] = le.fit_transform(df["sector_clean"].fillna("Unknown"))

feature_cols = [
    "sector_encoded", "debt_to_equity", "return_on_equity",
    "return_on_assets", "profit_margin", "current_ratio",
    "quick_ratio", "beta", "pe_ratio", "pb_ratio",
    "dividend_yield", "log_market_cap", "revenue_per_employee"
]

X = df[feature_cols].fillna(0)
y = df["total_esg_score"]

print(f"Feature shape: {X.shape}")
print(f"Target range: {round(y.min(),1)} to {round(y.max(),1)}")

# ============================================================
# Train Model with Named Features (fixes SHAP)
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

X_train_named = pd.DataFrame(X_train_sc, columns=feature_cols)
X_test_named  = pd.DataFrame(X_test_sc,  columns=feature_cols)

model = XGBRegressor(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbosity=0
)
model.fit(X_train_named, y_train)

y_pred = model.predict(X_test_named)
mae    = mean_absolute_error(y_test, y_pred)
r2     = r2_score(y_test, y_pred)
cv     = cross_val_score(model, X_train_named, y_train, cv=5, scoring="r2")

print("Model Performance")
print("-----------------")
print(f"MAE        : {mae:.2f}")
print(f"R2 Score   : {r2:.4f}")
print(f"CV R2 Mean : {cv.mean():.4f}")
print(f"CV R2 Std  : {cv.std():.4f}")

# ============================================================
# Save Artifacts
# ============================================================

joblib.dump(model,        "model.pkl")
joblib.dump(scaler,       "scaler.pkl")
joblib.dump(le,           "label_encoder.pkl")
joblib.dump(feature_cols, "feature_cols.pkl")
df.to_csv("sp500_esg.csv", index=False)

print("Saved: model.pkl, scaler.pkl, label_encoder.pkl, feature_cols.pkl, sp500_esg.csv")
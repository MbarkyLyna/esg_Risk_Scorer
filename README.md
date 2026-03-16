 # ESG Company Risk Scorer

An AI-powered ESG risk assessment tool for S&P 500 companies, designed to
support credit risk evaluation workflows. Built with XGBoost and SHAP,
deployed as an interactive Streamlit dashboard.

## What it does

- Predicts ESG risk scores for any S&P 500 company based on real financial data
- Classifies companies into five risk tiers: Negligible, Low, Medium, High, Severe
- Breaks down scores across three pillars: Environment, Social, Governance
- Explains every prediction using SHAP waterfall charts, making the model
  fully interpretable for credit committee or audit purposes
- Provides portfolio-level analysis: sector benchmarking, leverage vs risk
  scatter plots, risk distribution, and top 10 riskiest company rankings

## Why it matters

ESG risk is increasingly central to credit decisions. Lenders need to assess
whether a borrower's environmental liabilities, labor practices, or governance
weaknesses represent a default risk. This tool automates that assessment and
makes it explainable, the two requirements for any model used in a regulated
financial environment.

## Tech stack

- Data collection: yfinance (Yahoo Finance API)
- Modeling: XGBoost regression, scikit-learn preprocessing
- Explainability: SHAP (SHapley Additive exPlanations)
- Dashboard: Streamlit, Plotly
- Deployment: Streamlit Cloud

## Model performance

- R2 Score: 0.83
- MAE: 3.07 (on a 0-65 scale)
- Training set: 98 S&P 500 companies across 12 sectors
- Validation: 5-fold cross-validation

## Project structure
```
esg-risk-scorer/
├── app.py                  # Streamlit dashboard
├── requirements.txt        # Dependencies
├── model.pkl               # Trained XGBoost model
├── scaler.pkl              # StandardScaler
├── label_encoder.pkl       # Sector label encoder
├── feature_cols.pkl        # Feature column names
└── sp500_esg.csv           # Processed dataset
```

## Features used

Financial indicators sourced from Yahoo Finance: debt-to-equity ratio,
return on equity, return on assets, profit margin, current ratio, quick ratio,
beta, P/E ratio, P/B ratio, dividend yield, market capitalization, and
revenue per employee. Sector is encoded and used as a primary risk driver.

## How to run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Author

Lyna Mbarky AI engineering student.

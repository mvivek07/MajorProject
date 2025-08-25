import io
import os
import tempfile
import time
from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

st.set_page_config(page_title="SME AI — Cashflow & Debt Optimizer", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:1rem}
.kpi{background:#0b1220;color:#e2e8f0;padding:12px;border-radius:12px}
.small-muted{color:#94a3b8;font-size:0.85rem}
.badge{padding:6px 10px;border-radius:8px;color:white;font-weight:600}
.badge-good{background:#16a34a}
.badge-warn{background:#f59e0b}
.badge-bad{background:#dc2626}
</style>
""", unsafe_allow_html=True)


# ----------------------
# Utilities
# ----------------------

def load_csv(uploaded) -> pd.DataFrame:
    """Loads a CSV file from a Streamlit file uploader."""
    try:
        return pd.read_csv(io.BytesIO(uploaded.read()))
    except Exception:
        uploaded.seek(0)
        return pd.read_csv(uploaded)


def summarize_df(df: pd.DataFrame) -> Dict:
    """Provides a summary of the DataFrame."""
    return {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "missing": int(df.isna().sum().sum()),
        "numeric_cols": df.select_dtypes(include=[np.number]).shape[1]
    }


def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    """Tries to find a date-like column in the DataFrame."""
    for c in df.columns:
        if any(k in c.lower() for k in ["date", "month", "period", "time", "timestamp"]):
            return c
    return None


def create_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Extracts month, quarter, and year from a date column."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.sort_values(date_col)
    df['month'] = df[date_col].dt.month
    df['quarter'] = df[date_col].dt.quarter
    df['year'] = df[date_col].dt.year
    df = df.reset_index(drop=True)
    return df


def safe_first_match(cols, keywords):
    """Finds the first column that matches any of the keywords."""
    for k in keywords:
        for c in cols:
            if k in c.lower():
                return c
    return None


def get_model_and_scaler(df, cash_col, date_col, model_choice, test_size):
    """Trains and returns a model and scaler."""
    working = df.copy()
    
    # Feature engineering: lags and rolling means
    lags = 3
    for l in range(1, lags + 1):
        working[f'lag_{l}'] = working[cash_col].shift(l)
    working['rolling_3'] = working[cash_col].rolling(window=3, min_periods=1).mean()
    if date_col:
        working = create_time_features(working, date_col)
    
    working = working.dropna().reset_index(drop=True)

    feature_cols = [c for c in working.columns if c.startswith('lag_') or c.startswith('rolling_') or c in ['month', 'quarter']]
    X = working[feature_cols].select_dtypes(include=[np.number])
    y = working[cash_col].values

    if X.shape[0] < 10:
        st.error("Too few rows after feature engineering to train model.")
        return None, None, None, None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100.0, random_state=42, shuffle=False)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if model_choice == 'LinearRegression':
        model = LinearRegression()
    elif model_choice == 'RandomForest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = GradientBoostingRegressor(n_estimators=200, random_state=42)

    with st.spinner("Training model..."):
        model.fit(X_train_s, y_train)

    preds = model.predict(X_test_s)
    r2 = r2_score(y_test, preds)

    return model, scaler, feature_cols, r2, working

# ----------------------
# Cashflow Tab
# ----------------------

def cashflow_tab():
    st.header("Cash Flow — Forecasting & Insights")
    uploaded = st.file_uploader("Upload cashflow CSV (time series) — columns: date, net_cash / inflow / outflow / business_id", key="cashflow_upload")
    if uploaded is None:
        st.info("Upload a cashflow CSV to begin. Example columns: date, net_cash, Cash_Inflow, Cash_Outflow, Business_ID.")
        return

    df = load_csv(uploaded)
    st.subheader("Preview")
    st.dataframe(df.head(8))

    st.session_state['cashflow_df'] = df.copy()

    summary = summarize_df(df)
    st.markdown(f"Rows: **{summary['rows']}**, Columns: **{summary['cols']}**, Missing cells: **{summary['missing']}**")

    # Detect date and cash columns
    date_col = detect_date_column(df)
    cash_col = safe_first_match(df.columns, ["net_cash", "cashflow", "netcash", "net cash"])
    inflow_col = safe_first_match(df.columns, ["inflow", "cash_in", "receipt", "amount_received"])
    outflow_col = safe_first_match(df.columns, ["outflow", "cash_out", "payment", "amount_paid"])

    if cash_col is None:
        if inflow_col and outflow_col:
            df['net_cash'] = pd.to_numeric(df[inflow_col], errors='coerce').fillna(0) - pd.to_numeric(df[outflow_col], errors='coerce').fillna(0)
            cash_col = 'net_cash'
        else:
            st.error("No net_cash/inflow/outflow columns detected.")
            return

    if date_col:
        df = create_time_features(df, date_col)
    else:
        st.warning("No date column auto-detected.")

    df[cash_col] = pd.to_numeric(df[cash_col], errors='coerce')
    df = df.dropna(subset=[cash_col]).reset_index(drop=True)

    # Quick visualization
    st.subheader("Cash Flow Visuals")
    
    st.line_chart(df.set_index(date_col)[cash_col] if date_col else df[cash_col])
    
    cash_flow_counts = (df[cash_col] > 0).value_counts()
    pie_data = pd.DataFrame({'status': ['Positive' if k else 'Negative' for k in cash_flow_counts.index], 'count': cash_flow_counts.values})
    fig_pie = px.pie(pie_data, names='status', values='count', title='Proportion of Positive vs. Negative Cash Flow Months', hole=0.3)
    st.plotly_chart(fig_pie)
    
    if date_col:
        monthly_cash_avg = df.groupby('month')[cash_col].mean().reset_index()
        fig_bar = px.bar(monthly_cash_avg, x='month', y=cash_col, title='Average Monthly Cash Flow (Seasonality)', text_auto=True)
        st.plotly_chart(fig_bar)

    st.markdown("---")
    st.subheader("KPIs & Alerts")
    avg_cash = df[cash_col].mean()
    recent_mean = df[cash_col].tail(6).mean() if len(df) >= 6 else df[cash_col].mean()
    neg_months = (df[cash_col] < 0).sum()
    pct_negative = neg_months / max(1, len(df))
    last_value = df[cash_col].iloc[-1]

    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f"<div class='kpi'><b>Avg Net Cash</b><br><h3>₹ {avg_cash:,.0f}</h3></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='kpi'><b>Recent Avg (last 6)</b><br><h3>₹ {recent_mean:,.0f}</h3></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='kpi'><b>Negative Months</b><br><h3>{neg_months} / {len(df)}</h3></div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='kpi'><b>Last Net Cash</b><br><h3>₹ {last_value:,.0f}</h3></div>", unsafe_allow_html=True)

    if avg_cash < 0:
        st.error("🚨 Average net cashflow is **NEGATIVE** — urgent action required.")
    elif pct_negative > 0.25:
        st.warning("⚠️ Large fraction of records are negative cashflow. Investigate recurring expenses.")
    elif last_value < 0:
        st.warning("⚠️ Latest net cash is negative — cash burn detected in the most recent period.")

    st.markdown("---")
    st.subheader("Cashflow Forecast")
    model_choice = st.selectbox("Choose model", ["LinearRegression", "RandomForest", "GradientBoosting"], key='cash_model_choice')
    test_size = st.slider("Test size (%)", 5, 50, 20, key='cash_test_size')
    periods = st.number_input("Forecast periods (next N)", min_value=1, max_value=24, value=6)

    model, scaler, feature_cols, r2_score_val, working_df = get_model_and_scaler(df, cash_col, date_col, model_choice, test_size)
    
    if model and scaler and feature_cols and working_df is not None:
        st.success(f"✅ Model trained successfully. Accuracy (R-squared score): **{r2_score_val*100:.1f}%**")

        last_row = working_df.iloc[-1:]
        cur_feats = last_row[feature_cols].copy()
        future_predictions = []
        for _ in range(int(periods)):
            cur_s = scaler.transform(cur_feats)
            fval = model.predict(cur_s)[0]
            future_predictions.append(float(fval))
            
            lag_cols = [c for c in feature_cols if c.startswith('lag_')]
            lag_cols_sorted = sorted(lag_cols)
            for idx in range(len(lag_cols_sorted) - 1, 0, -1):
                cur_feats[lag_cols_sorted[idx]] = cur_feats[lag_cols_sorted[idx - 1]]
            if lag_cols_sorted:
                cur_feats[lag_cols_sorted[0]] = fval
            if 'rolling_3' in cur_feats.columns:
                cur_feats['rolling_3'] = (cur_feats.get('rolling_3', 0) * 2 + fval) / 3
        
        forecast_dates = pd.date_range(start=df[date_col].iloc[-1], periods=periods + 1, freq='M')[1:] if date_col else range(len(df), len(df) + periods)
        forecast_df = pd.DataFrame({date_col: forecast_dates, cash_col: future_predictions})
        
        combined_df = pd.concat([df[[date_col, cash_col]], forecast_df], ignore_index=True)
        combined_df['type'] = ['Historical'] * len(df) + ['Forecast'] * len(forecast_df)
        
        fig_forecast = px.line(combined_df, x=date_col, y=cash_col, color='type', markers=True,
                                  title=f"Cash Flow: Historical & {periods}-Period Forecast",
                                  color_discrete_map={'Historical': '#1f77b4', 'Forecast': '#ff7f0e'},
                                  symbol='type')
        fig_forecast.update_traces(selector=dict(name='Forecast'), line=dict(dash='dash'))
        st.plotly_chart(fig_forecast, use_container_width=True)

        # --- AI Insights and Recommendation System ---
        st.markdown("---")
        st.subheader("🤖 AI-Powered Insights & Recommendations")

        # Predictive Alert based on forecast
        forecast_sum = np.cumsum(future_predictions)
        try:
            months_to_zero = np.where(forecast_sum < 0)[0][0] + 1
            st.error(f"**Prediction Alert:** Based on the forecast, your cash flow is projected to turn negative in approximately **{months_to_zero} months**. Immediate action is required to increase inflow or reduce expenses.")
        except IndexError:
            st.success("**Prediction Alert:** Your cash flow is projected to remain positive for the entire forecast period. Continue monitoring your finances.")

        # Trend Analysis
        avg_forecast = np.mean(future_predictions)
        if avg_forecast < recent_mean:
            st.warning("**Trend Insight:** The forecast indicates a potential **downturn** in average cash flow compared to your recent performance. Review upcoming sales pipeline and major expenses.")
        else:
            st.info("**Trend Insight:** The forecast suggests a **stable or improving** cash flow trend. Capitalize on this by considering strategic investments or debt reduction.")
        
        # Seasonality Recommendation
        if date_col:
            monthly_avg = df.groupby('month')[cash_col].mean()
            worst_month = monthly_avg.idxmin()
            best_month = monthly_avg.idxmax()
            st.info(f"**Seasonality Insight:** Historically, your cash flow is lowest in **month {worst_month}** and highest in **month {best_month}**. Plan your marketing campaigns and major purchases around this cycle to maximize capital efficiency.")


# ----------------------
# Debt Tab
# ----------------------

def debt_tab():
    st.header("Debt — Risk Scoring & Optimization")
    uploaded = st.file_uploader("Upload debt CSV", key="debt_upload")
    if uploaded is None:
        st.info("Upload a debt CSV to begin. Example columns: Debt_ID, Business_ID, Principal_Amount, Outstanding_Amount, Interest_Rate(%), Monthly_Installment, Status.")
        return

    df = load_csv(uploaded)
    st.subheader("Preview")
    st.dataframe(df.head(8))

    st.session_state['debt_df'] = df.copy()

    principal_col = safe_first_match(df.columns, ["principal", "loan_amount", "amount"])
    outstanding_col = safe_first_match(df.columns, ["outstanding", "balance", "remaining"])
    rate_col = safe_first_match(df.columns, ["interest", "rate", "interest_rate", "interest_rate(%)"])
    emi_col = safe_first_match(df.columns, ["emi", "monthly_installment", "monthly"])
    status_col = safe_first_match(df.columns, ["status", "state"])
    tenor_col = safe_first_match(df.columns, ["tenor", "term", "months"])

    if principal_col is None or rate_col is None:
        st.error("Could not detect principal and interest rate columns automatically.")
        return

    df[principal_col] = pd.to_numeric(df[principal_col], errors='coerce').fillna(0)
    df[rate_col] = pd.to_numeric(df[rate_col], errors='coerce').fillna(0)
    if outstanding_col:
        df[outstanding_col] = pd.to_numeric(df[outstanding_col], errors='coerce').fillna(df[principal_col])
    else:
        df['Outstanding_Amount'] = df[principal_col]
        outstanding_col = 'Outstanding_Amount'
    if emi_col:
        df[emi_col] = pd.to_numeric(df[emi_col], errors='coerce').fillna((df[principal_col] / 12).astype(int))
    else:
        df['Monthly_Installment'] = (df[principal_col] / 12).astype(int)
        emi_col = 'Monthly_Installment'

    if status_col is None:
        df['status_synth'] = ((df[rate_col] > 12) | (df[outstanding_col] > df[outstanding_col].median() * 1.5)).astype(int)
        target_col = 'status_synth'
        st.warning("No status column found — using synthetic status for risk labeling.")
    else:
        target_col = status_col
        df[target_col] = df[target_col].apply(lambda x: 1 if str(x).strip().lower() in ['overdue', 'defaulted', 'default', '1', 'y', 'yes', 'risk', 'risky'] else 0)

    st.subheader("Debt KPIs")
    total_principal = df[principal_col].sum()
    total_outstanding = df[outstanding_col].sum()
    total_emi = df[emi_col].sum()
    avg_rate = df[rate_col].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f"<div class='kpi'><b>Total Principal</b><br><h3>₹ {total_principal:,.0f}</h3></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='kpi'><b>Total Outstanding</b><br><h3>₹ {total_outstanding:,.0f}</h3></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='kpi'><b>Total Monthly EMI</b><br><h3>₹ {total_emi:,.0f}</h3></div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='kpi'><b>Avg. Interest Rate</b><br><h3>{avg_rate:.2f}%</h3></div>", unsafe_allow_html=True)

    st.subheader("Risk Buckets & Alerts")
    df['risk_bucket'] = pd.cut(df[rate_col], bins=[-1, 10, 18, 1000], labels=['Low Rate', 'Medium Rate', 'High Rate'])
    high_rate_count = df[df[rate_col] > 20].shape[0]
    overdue_count = df[df[target_col] == 1].shape[0]

    st.write(f"High interest loans (>20%): **{high_rate_count}**")
    st.write(f"Overdue/High-risk loans: **{overdue_count}**")

    st.markdown("---")
    st.subheader("Debt Burden Analysis (requires cashflow CSV uploaded)")
    avg_burden = None
    if 'cashflow_df' in st.session_state:
        # ... (rest of the debt burden analysis code remains the same)
        cash_df = st.session_state['cashflow_df']
        inflow_cand = safe_first_match(cash_df.columns, ["inflow", "cash_in", "receipt", "amount_received"])
        if 'Business_ID' in [c for c in cash_df.columns]:
            inc_agg = cash_df.groupby('Business_ID').apply(lambda d: d[inflow_cand].dropna().mean() if inflow_cand in d.columns else d.filter(regex='net', axis=1).mean(axis=1).mean()).rename('avg_inflow')
            merged = df.copy()
            merged = merged.merge(inc_agg.reset_index(), how='left', left_on=safe_first_match(df.columns, ['business_id', 'Business_ID', 'BusinessId']) or 'Business_ID', right_on='Business_ID')
            merged['avg_inflow'] = merged.get('avg_inflow', np.nan).fillna(cash_df[cash_df.select_dtypes(include=[np.number]).columns[0]].mean())
            merged['debt_burden'] = merged[emi_col] / merged['avg_inflow'].replace(0, np.nan)
            avg_burden = merged['debt_burden'].mean()
            st.write(f"Average Debt Burden (EMI / Avg Monthly Inflow): **{avg_burden:.2%}**")
            burden_by_business = merged.groupby('Business_ID').agg(total_emi=(emi_col, 'sum'), avg_inflow=('avg_inflow', 'mean'))
            burden_by_business['debt_burden_ratio'] = burden_by_business['total_emi'] / burden_by_business['avg_inflow'].replace(0, np.nan)
            burden_alerts = burden_by_business[burden_by_business['debt_burden_ratio'] > 0.4]
            if not burden_alerts.empty:
                st.error(f"🚨 {len(burden_alerts)} business(es) with debt burden ratio > 40% — high vulnerability.")
                st.dataframe(burden_alerts.reset_index().head(10))
            else:
                st.success("✅ No business-level debt burden > 40% detected.")
    else:
        st.info("Upload a cashflow CSV to compute debt burden and get richer insights.")

    # --- AI Insights and Recommendation System for Debt ---
    st.markdown("---")
    st.subheader("🤖 AI-Powered Insights & Recommendations")

    if high_rate_count > 0:
        st.warning(f"**Refinancing Opportunity:** You have **{high_rate_count} loan(s)** with an interest rate over 20%. **Recommendation:** Prioritize refinancing these loans to a lower rate to save significantly on interest payments. Use the 'Greedy Repayment Plan' below to see how to allocate extra funds to them.")
    else:
        st.success("**Interest Rate Health:** Good news! You have no loans with excessively high interest rates (>20%). Your current debt structure appears manageable from a rate perspective.")

    if overdue_count > 0:
        st.error(f"**Critical Alert:** **{overdue_count} loan(s)** are marked as high-risk or overdue. **Recommendation:** Address these immediately to avoid penalties and damage to your credit score. Contact your lenders to discuss restructuring options if needed.")
    
    if avg_burden:
        if avg_burden > 0.4:
            st.error(f"**High Debt Burden:** Your average debt payments make up **{avg_burden:.1%}** of your monthly inflow, which is a high-risk level. **Recommendation:** Focus urgently on increasing revenue or consolidating debt to lower your monthly payments.")
        elif avg_burden > 0.2:
            st.warning(f"**Moderate Debt Burden:** Your debt burden is **{avg_burden:.1%}**. While manageable, monitor this closely. **Recommendation:** Avoid taking on new debt until your cash inflow increases significantly.")

    st.markdown("---")
    st.subheader("Debt Optimization — Repayment Plan Suggestions")

    st.write("Greedy strategy: prioritize loans by interest rate (highest first).")
    budget = st.number_input("Monthly repayment budget (total across loans)", min_value=0.0, value=float(total_emi))

    if st.button("Generate Greedy Repayment Plan"):
        # ... (rest of the optimization code remains the same)
        plan = df.copy()
        plan['tenor_months'] = plan[tenor_col].fillna(12) if tenor_col and tenor_col in plan.columns else 12
        plan['min_pay'] = plan[principal_col] / plan['tenor_months']
        plan = plan.sort_values(by=rate_col, ascending=False).reset_index(drop=True)
        remaining_budget = float(budget)
        plan['allocated'] = 0.0
        for i, row in plan.iterrows():
            alloc = min(row['min_pay'], remaining_budget)
            plan.at[i, 'allocated'] = alloc
            remaining_budget -= alloc
            if remaining_budget <= 0:
                break
        st.dataframe(plan[[principal_col, rate_col, 'tenor_months', 'min_pay', 'allocated']].head(50))
        st.info("Allocated budget to highest-rate loans first. 'allocated' is the suggested payment this month.")

# ----------------------
# Main layout
# ----------------------

def main():
    st.title("SME AI — Cashflow Forecast & Debt Optimizer (Enhanced)")
    st.write("Upload your datasets in the tabs below. For the best analysis, upload both cashflow and debt files.")

    tab1, tab2 = st.tabs(["Cashflow", "Debt Optimization"])
    with tab1:
        cashflow_tab()
    with tab2:
        debt_tab()

    st.sidebar.markdown("---")
    st.sidebar.write("Tips & Notes")
    st.sidebar.write("- **Cashflow CSV:** Must include a date column and a net cash column (or inflow/outflow).")
    st.sidebar.write("- **Debt CSV:** Must include principal amount and interest rate.")
    st.sidebar.write("- For cross-analysis, ensure both files have a matching `Business_ID` column.")


if __name__ == "__main__":
    main()
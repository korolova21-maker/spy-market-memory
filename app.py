import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(
    page_title="SPY Market Memory",
    page_icon="🧠",
    layout="wide"
)

# =========================
# Helper functions
# =========================

@st.cache_data(ttl=3600)
def download_market_data(start="2014-01-01"):
    tickers = ["SPY", "QQQ", "^VIX"]
    raw = yf.download(tickers, start=start, auto_adjust=False, progress=False, group_by="ticker")

    def clean_ticker(ticker):
        df = raw[ticker].copy()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        df = df.dropna()
        return df

    spy = clean_ticker("SPY")
    qqq = clean_ticker("QQQ")
    vix = clean_ticker("^VIX")

    spy.index = pd.to_datetime(spy.index)
    qqq.index = pd.to_datetime(qqq.index)
    vix.index = pd.to_datetime(vix.index)

    return spy, qqq, vix


def rsi(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def atr(df, length=14):
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def add_features(spy, qqq, vix):
    df = spy.copy()

    # Core indicators
    df["rsi_14"] = rsi(df["close"], 14)
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()

    # Bollinger Bands
    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_mid"] = bb_mid
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # Keltner Channels
    df["atr_14"] = atr(df, 14)
    kc_mid = df["ema_20"]
    df["kc_upper"] = kc_mid + 2 * df["atr_14"]
    df["kc_lower"] = kc_mid - 2 * df["atr_14"]
    df["kc_position"] = (df["close"] - df["kc_lower"]) / (df["kc_upper"] - df["kc_lower"])

    # Returns
    df["ret_1d"] = df["close"].pct_change(1)
    df["ret_5d"] = df["close"].pct_change(5)
    df["ret_20d"] = df["close"].pct_change(20)

    # Distance features
    df["dist_ema20"] = (df["close"] / df["ema_20"]) - 1
    df["dist_ema50"] = (df["close"] / df["ema_50"]) - 1
    df["dist_ema200"] = (df["close"] / df["ema_200"]) - 1
    df["dist_bb_upper"] = (df["close"] / df["bb_upper"]) - 1
    df["dist_bb_lower"] = (df["close"] / df["bb_lower"]) - 1
    df["dist_kc_upper"] = (df["close"] / df["kc_upper"]) - 1
    df["dist_kc_lower"] = (df["close"] / df["kc_lower"]) - 1

    # Gap size
    df["gap"] = (df["open"] / df["close"].shift(1)) - 1

    # Day of week
    df["day_of_week"] = df.index.dayofweek

    # VIX
    vix_close = vix["close"].rename("vix_close")
    df = df.join(vix_close, how="left")
    df["vix_ret_5d"] = df["vix_close"].pct_change(5)

    # QQQ/SPY relative strength
    qqq_close = qqq["close"].rename("qqq_close")
    df = df.join(qqq_close, how="left")
    df["qqq_spy_rs"] = (df["qqq_close"].pct_change(5) - df["close"].pct_change(5))

    # Days since pullbacks
    for pct in [0.01, 0.02, 0.03]:
        pullback = df["close"].pct_change() <= -pct
        days = []
        counter = np.nan
        for hit in pullback:
            if hit:
                counter = 0
            elif np.isnan(counter):
                counter = np.nan
            else:
                counter += 1
            days.append(counter)
        df[f"days_since_{int(pct*100)}pct_pullback"] = days

    # Weekly sequence approximation
    weekly = df["close"].resample("W-FRI").last()
    weekly_green = weekly.diff() > 0
    streak = []
    count = 0
    for val in weekly_green:
        count = count + 1 if val else 0
        streak.append(count)
    weekly_streak = pd.Series(streak, index=weekly.index, name="weekly_green_streak")
    df = df.join(weekly_streak.reindex(df.index, method="ffill"), how="left")

    # Forward outcomes
    for horizon in [1, 3, 5, 10]:
        df[f"fwd_ret_{horizon}d"] = df["close"].shift(-horizon) / df["close"] - 1

    # Max favourable/adverse move over next 5 and 10 trading days
    for horizon in [5, 10]:
        future_high = df["high"].shift(-1).rolling(horizon).max().shift(-(horizon - 1))
        future_low = df["low"].shift(-1).rolling(horizon).min().shift(-(horizon - 1))
        df[f"max_favourable_{horizon}d"] = future_high / df["close"] - 1
        df[f"max_adverse_{horizon}d"] = future_low / df["close"] - 1

    return df.dropna()


def classify_state(row):
    close = row["close"]
    above_200 = close > row["ema_200"]
    rsi_val = row["rsi_14"]
    dist_20 = row["dist_ema20"]
    dist_200 = row["dist_ema200"]
    bb_pos = row["bb_position"]
    kc_pos = row["kc_position"]
    vix_val = row["vix_close"]
    ret_5d = row["ret_5d"]

    if above_200 and rsi_val >= 70 and bb_pos > 0.95 and kc_pos > 0.95:
        return "Exhaustion / wait-for-pullback mode"
    if above_200 and rsi_val >= 60 and ret_5d > 0.025 and dist_20 > 0.02:
        return "Late-rally extension / chase-risk mode"
    if above_200 and 45 <= rsi_val <= 60 and abs(dist_20) < 0.015:
        return "Momentum-continuation mode"
    if above_200 and rsi_val < 45 and dist_50 > -0.04 if "dist_50" in row else False:
        return "Pullback-buy mode"
    if not above_200 and vix_val >= 25:
        return "Hedge-watch mode"
    if not above_200 and rsi_val < 45:
        return "Fade-risk / defensive mode"
    if abs(ret_5d) < 0.01 and 45 <= rsi_val <= 55:
        return "Chop/no-trade mode"
    return "Scalp-only / context-dependent mode"


def similar_days(df, selected_date, n=50):
    feature_cols = [
        "rsi_14", "dist_ema20", "dist_ema50", "dist_ema200",
        "bb_position", "kc_position", "atr_14", "ret_1d", "ret_5d", "ret_20d",
        "gap", "vix_close", "vix_ret_5d", "qqq_spy_rs",
        "days_since_1pct_pullback", "days_since_2pct_pullback", "days_since_3pct_pullback",
        "weekly_green_streak", "day_of_week"
    ]

    work = df.copy()
    work = work.dropna(subset=feature_cols)
    work = work.loc[work.index <= selected_date]

    if selected_date not in work.index:
        selected_date = work.index[-1]

    # Avoid matching with the current day itself and avoid recent forward-data leakage
    historical = work.loc[work.index < selected_date - pd.Timedelta(days=15)].copy()
    target = work.loc[[selected_date], feature_cols]

    X = historical[feature_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    target_scaled = scaler.transform(target)

    n_neighbors = min(n, len(historical))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(X_scaled)
    distances, indices = nn.kneighbors(target_scaled)

    matches = historical.iloc[indices[0]].copy()
    matches["similarity_distance"] = distances[0]
    matches["similarity_score"] = 1 / (1 + matches["similarity_distance"])
    return matches.sort_values("similarity_distance")


def outcome_summary(matches):
    summary = {}

    for horizon in [1, 3, 5, 10]:
        col = f"fwd_ret_{horizon}d"
        vals = matches[col].dropna()
        summary[f"median_{horizon}d"] = vals.median()
        summary[f"mean_{horizon}d"] = vals.mean()
        summary[f"prob_positive_{horizon}d"] = (vals > 0).mean()
        summary[f"prob_pullback_{horizon}d"] = (vals < -0.01).mean()

    summary["median_max_favourable_5d"] = matches["max_favourable_5d"].median()
    summary["median_max_adverse_5d"] = matches["max_adverse_5d"].median()
    summary["median_max_favourable_10d"] = matches["max_favourable_10d"].median()
    summary["median_max_adverse_10d"] = matches["max_adverse_10d"].median()

    return summary


def make_playbook(state, summary):
    prob_5d_positive = summary["prob_positive_5d"]
    prob_3d_pullback = summary["prob_pullback_3d"]
    med_5d = summary["median_5d"]
    maf = summary["median_max_favourable_5d"]
    mae = summary["median_max_adverse_5d"]

    if "Exhaustion" in state or "chase-risk" in state:
        if prob_3d_pullback > 0.45:
            return "Avoid chasing. Historical analogues favour waiting for a pullback or using scalp-only longs."
        return "Chase risk is elevated, but analogue pullback risk is not extreme. Prefer smaller size and faster exits."

    if "Momentum-continuation" in state:
        if prob_5d_positive > 0.55 and med_5d > 0:
            return "Continuation has historical support. Pullbacks may be buyable if intraday structure confirms."
        return "Momentum state is present, but historical follow-through is weak. Treat signals carefully."

    if "Pullback-buy" in state:
        if prob_5d_positive > 0.55:
            return "Pullback-buy conditions look historically favourable. Best behaviour is usually buying weakness, not chasing strength."
        return "Pullback setup exists, but historical edge is weak. Wait for confirmation."

    if "Hedge" in state or "defensive" in state:
        return "Defensive conditions. Long signals should be treated as scalp-only unless analogue outcomes improve."

    if "Chop" in state:
        return "Chop/no-trade conditions. Expect false signals and poor follow-through."

    if abs(maf) > abs(mae) and prob_5d_positive > 0.5:
        return "Slightly constructive, but not a clean high-conviction state."
    return "Context is mixed. Treat signals as scalp-only until the state improves."


def pct(x):
    if pd.isna(x):
        return "N/A"
    return f"{x*100:.2f}%"


def prob(x):
    if pd.isna(x):
        return "N/A"
    return f"{x*100:.1f}%"


# =========================
# UI
# =========================

st.title("🧠 SPY Market Memory")
st.caption("Not a signal generator. A historical context engine for SPY/QQQ traders.")

with st.sidebar:
    st.header("Settings")
    start_date = st.date_input("Historical start date", value=pd.to_datetime("2014-01-01"))
    match_count = st.slider("Similar historical days", 20, 150, 50, step=10)
    selected_mode = st.radio("Analysis date", ["Latest available day", "Choose date"])
    st.markdown("---")
    st.caption("MVP: daily SPY state only. QQQ and VIX are used as context features.")

try:
    spy, qqq, vix = download_market_data(start=start_date.strftime("%Y-%m-%d"))
    df = add_features(spy, qqq, vix)

    if selected_mode == "Choose date":
        chosen_date = st.sidebar.date_input(
            "Choose market date",
            value=df.index[-1].date(),
            min_value=df.index[0].date(),
            max_value=df.index[-1].date()
        )
        selected_date = df.index[df.index <= pd.to_datetime(chosen_date)][-1]
    else:
        selected_date = df.index[-1]

    current = df.loc[selected_date]
    state = classify_state(current)
    matches = similar_days(df, selected_date, match_count)
    summary = outcome_summary(matches)
    playbook = make_playbook(state, summary)

    st.subheader("Today's SPY Playbook" if selected_mode == "Latest available day" else f"SPY Playbook for {selected_date.date()}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("SPY close", f"${current['close']:.2f}")
    col2.metric("RSI 14", f"{current['rsi_14']:.1f}")
    col3.metric("VIX", f"{current['vix_close']:.2f}")
    col4.metric("5D SPY return", pct(current["ret_5d"]))

    st.info(f"**Current state:** {state}")
    st.success(f"**Playbook:** {playbook}")

    st.markdown("### Historical analogue outcomes")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("1D positive probability", prob(summary["prob_positive_1d"]))
    c2.metric("3D positive probability", prob(summary["prob_positive_3d"]))
    c3.metric("5D positive probability", prob(summary["prob_positive_5d"]))
    c4.metric("10D positive probability", prob(summary["prob_positive_10d"]))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Median 1D return", pct(summary["median_1d"]))
    c6.metric("Median 3D return", pct(summary["median_3d"]))
    c7.metric("Median 5D return", pct(summary["median_5d"]))
    c8.metric("Median 10D return", pct(summary["median_10d"]))

    c9, c10, c11, c12 = st.columns(4)
    c9.metric("3D pullback probability", prob(summary["prob_pullback_3d"]))
    c10.metric("5D pullback probability", prob(summary["prob_pullback_5d"]))
    c11.metric("Median max favourable 5D", pct(summary["median_max_favourable_5d"]))
    c12.metric("Median max adverse 5D", pct(summary["median_max_adverse_5d"]))

    st.markdown("### SPY chart context")
    chart_df = df.loc[selected_date - pd.Timedelta(days=365):selected_date].copy()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=chart_df.index,
        open=chart_df["open"],
        high=chart_df["high"],
        low=chart_df["low"],
        close=chart_df["close"],
        name="SPY"
    ))
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["ema_20"], name="EMA 20"))
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["ema_50"], name="EMA 50"))
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["ema_200"], name="EMA 200"))
    fig.update_layout(height=550, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Most similar historical days")
    display_cols = [
        "close", "rsi_14", "vix_close", "ret_5d", "ret_20d",
        "fwd_ret_1d", "fwd_ret_3d", "fwd_ret_5d", "fwd_ret_10d",
        "max_favourable_5d", "max_adverse_5d", "similarity_score"
    ]
    table = matches[display_cols].copy()
    for col in ["ret_5d", "ret_20d", "fwd_ret_1d", "fwd_ret_3d", "fwd_ret_5d", "fwd_ret_10d", "max_favourable_5d", "max_adverse_5d"]:
        table[col] = table[col].map(lambda x: f"{x*100:.2f}%")
    table["similarity_score"] = table["similarity_score"].map(lambda x: f"{x:.3f}")
    st.dataframe(table, use_container_width=True)

    st.markdown("### Signal Forensics — placeholder")
    signal = st.selectbox(
        "Choose a signal type to test later",
        [
            "None",
            "Bullish reversal warning",
            "Bearish reversal warning",
            "Trend exhaustion",
            "ORB breakout long",
            "VWAP reclaim",
            "RSI divergence"
        ]
    )

    if signal != "None":
        st.warning(
            "This MVP has not yet imported your TradingView signal history. "
            "Next version should let you upload/export signal dates, then test how this signal behaved in similar market states."
        )

    st.caption(
        "Educational/research prototype only. This does not give financial advice or predict future prices. "
        "It shows how historically similar SPY conditions behaved."
    )

except Exception as e:
    st.error("The app could not complete the analysis.")
    st.exception(e)
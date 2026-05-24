import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go

st.set_page_config(
    page_title="SPY Market Memory",
    page_icon="🧠",
    layout="wide"
)

# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(ttl=3600)
def download_single_ticker(ticker, start):
    """
    Downloads one ticker at a time.
    This is safer on Streamlit Cloud than downloading multiple tickers together.
    """
    data = yf.download(
        ticker,
        start=start,
        progress=False,
        auto_adjust=False,
        threads=False
    )

    if data is None or data.empty:
        return pd.DataFrame()

    # If yfinance returns MultiIndex columns, flatten them
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    data = data.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume"
    })

    data.index = pd.to_datetime(data.index)
    data = data.sort_index()

    required_cols = ["open", "high", "low", "close", "volume"]
    available_cols = [c for c in required_cols if c in data.columns]

    if len(available_cols) < 4:
        return pd.DataFrame()

    return data


@st.cache_data(ttl=3600)
def load_market_data(start="2014-01-01"):
    spy = download_single_ticker("SPY", start)
    qqq = download_single_ticker("QQQ", start)
    vix = download_single_ticker("^VIX", start)

    return spy, qqq, vix


# ============================================================
# INDICATOR FUNCTIONS
# ============================================================

def calculate_rsi(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_atr(df, length=14):
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1 / length, adjust=False).mean()

    return atr


def add_market_features(spy, qqq, vix):
    df = spy.copy()

    # -------------------------
    # Core technical indicators
    # -------------------------
    df["rsi_14"] = calculate_rsi(df["close"], 14)

    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()

    df["atr_14"] = calculate_atr(df, 14)

    # -------------------------
    # Bollinger Bands
    # -------------------------
    df["bb_mid"] = df["close"].rolling(20).mean()
    df["bb_std"] = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]

    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # -------------------------
    # Keltner Channels
    # -------------------------
    df["kc_mid"] = df["ema_20"]
    df["kc_upper"] = df["kc_mid"] + 2 * df["atr_14"]
    df["kc_lower"] = df["kc_mid"] - 2 * df["atr_14"]

    df["kc_position"] = (df["close"] - df["kc_lower"]) / (df["kc_upper"] - df["kc_lower"])

    # -------------------------
    # Returns
    # -------------------------
    df["ret_1d"] = df["close"].pct_change(1)
    df["ret_5d"] = df["close"].pct_change(5)
    df["ret_20d"] = df["close"].pct_change(20)

    # -------------------------
    # Distance from key levels
    # -------------------------
    df["dist_ema20"] = (df["close"] / df["ema_20"]) - 1
    df["dist_ema50"] = (df["close"] / df["ema_50"]) - 1
    df["dist_ema200"] = (df["close"] / df["ema_200"]) - 1

    df["dist_bb_upper"] = (df["close"] / df["bb_upper"]) - 1
    df["dist_bb_lower"] = (df["close"] / df["bb_lower"]) - 1

    df["dist_kc_upper"] = (df["close"] / df["kc_upper"]) - 1
    df["dist_kc_lower"] = (df["close"] / df["kc_lower"]) - 1

    # -------------------------
    # Gap size
    # -------------------------
    df["gap"] = (df["open"] / df["close"].shift(1)) - 1

    # -------------------------
    # Day of week
    # Monday = 0, Friday = 4
    # -------------------------
    df["day_of_week"] = df.index.dayofweek

    # -------------------------
    # VIX context
    # -------------------------
    if vix is not None and not vix.empty and "close" in vix.columns:
        vix_close = vix["close"].rename("vix_close")
        df = df.join(vix_close, how="left")
        df["vix_close"] = df["vix_close"].ffill()
        df["vix_ret_5d"] = df["vix_close"].pct_change(5)
    else:
        df["vix_close"] = np.nan
        df["vix_ret_5d"] = np.nan

    # -------------------------
    # QQQ/SPY relative strength
    # -------------------------
    if qqq is not None and not qqq.empty and "close" in qqq.columns:
        qqq_close = qqq["close"].rename("qqq_close")
        df = df.join(qqq_close, how="left")
        df["qqq_close"] = df["qqq_close"].ffill()
        df["qqq_spy_rs"] = df["qqq_close"].pct_change(5) - df["close"].pct_change(5)
    else:
        df["qqq_close"] = np.nan
        df["qqq_spy_rs"] = np.nan

    # -------------------------
    # Days since pullbacks
    # -------------------------
    for pct in [0.01, 0.02, 0.03]:
        daily_pullback = df["close"].pct_change() <= -pct

        counter = 999
        values = []

        for hit in daily_pullback:
            if hit:
                counter = 0
            else:
                counter += 1

            values.append(counter)

        df[f"days_since_{int(pct * 100)}pct_pullback"] = values

    # -------------------------
    # Weekly green candle streak
    # -------------------------
    weekly_close = df["close"].resample("W-FRI").last()
    weekly_green = weekly_close.diff() > 0

    streak_values = []
    streak = 0

    for is_green in weekly_green:
        if is_green:
            streak += 1
        else:
            streak = 0

        streak_values.append(streak)

    weekly_streak = pd.Series(
        streak_values,
        index=weekly_close.index,
        name="weekly_green_streak"
    )

    df = df.join(
        weekly_streak.reindex(df.index, method="ffill"),
        how="left"
    )

    # -------------------------
    # Forward outcomes
    # These are only for historical analogue testing.
    # Latest rows will naturally have NaN forward values.
    # -------------------------
    for horizon in [1, 3, 5, 10]:
        df[f"fwd_ret_{horizon}d"] = df["close"].shift(-horizon) / df["close"] - 1

    # -------------------------
    # Future max favourable/adverse moves
    # -------------------------
    for horizon in [5, 10]:
        future_high = df["high"].shift(-1).rolling(horizon).max().shift(-(horizon - 1))
        future_low = df["low"].shift(-1).rolling(horizon).min().shift(-(horizon - 1))

        df[f"max_favourable_{horizon}d"] = future_high / df["close"] - 1
        df[f"max_adverse_{horizon}d"] = future_low / df["close"] - 1

    return df


# ============================================================
# MARKET STATE CLASSIFICATION
# ============================================================

def classify_state(row):
    close = row["close"]
    above_200 = close > row["ema_200"]

    rsi_val = row["rsi_14"]
    dist_20 = row["dist_ema20"]
    dist_50 = row["dist_ema50"]
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

    if above_200 and rsi_val < 45 and dist_50 > -0.04:
        return "Pullback-buy mode"

    if not above_200 and pd.notna(vix_val) and vix_val >= 25:
        return "Hedge-watch mode"

    if not above_200 and rsi_val < 45:
        return "Fade-risk / defensive mode"

    if abs(ret_5d) < 0.01 and 45 <= rsi_val <= 55:
        return "Chop/no-trade mode"

    return "Scalp-only / context-dependent mode"


# ============================================================
# HISTORICAL ANALOGUE MATCHING
# ============================================================

def get_feature_columns():
    return [
        "rsi_14",
        "dist_ema20",
        "dist_ema50",
        "dist_ema200",
        "bb_position",
        "kc_position",
        "atr_14",
        "ret_1d",
        "ret_5d",
        "ret_20d",
        "gap",
        "vix_close",
        "vix_ret_5d",
        "qqq_spy_rs",
        "days_since_1pct_pullback",
        "days_since_2pct_pullback",
        "days_since_3pct_pullback",
        "weekly_green_streak",
        "day_of_week"
    ]


def prepare_usable_dataframe(df):
    feature_cols = get_feature_columns()

    needed_cols = feature_cols + [
        "open",
        "high",
        "low",
        "close",
        "ema_20",
        "ema_50",
        "ema_200",
        "fwd_ret_1d",
        "fwd_ret_3d",
        "fwd_ret_5d",
        "fwd_ret_10d",
        "max_favourable_5d",
        "max_adverse_5d",
        "max_favourable_10d",
        "max_adverse_10d"
    ]

    for col in needed_cols:
        if col not in df.columns:
            df[col] = np.nan

    # For current-day classification, we only need current features,
    # not future outcomes.
    current_ready = df.dropna(subset=feature_cols + ["close", "ema_20", "ema_50", "ema_200"])

    return current_ready


def find_similar_days(df, selected_date, n=50):
    feature_cols = get_feature_columns()

    work = prepare_usable_dataframe(df)

    if work.empty:
        return pd.DataFrame(), None

    # Choose nearest available trading day
    if selected_date not in work.index:
        available_dates = work.index[work.index <= selected_date]

        if len(available_dates) == 0:
            selected_date = work.index[-1]
        else:
            selected_date = available_dates[-1]

    target = work.loc[[selected_date], feature_cols]

    # Historical rows must have forward outcome data.
    historical = work.loc[work.index < selected_date - pd.Timedelta(days=15)].copy()

    historical = historical.dropna(subset=feature_cols + [
        "fwd_ret_1d",
        "fwd_ret_3d",
        "fwd_ret_5d",
        "fwd_ret_10d",
        "max_favourable_5d",
        "max_adverse_5d"
    ])

    if historical.empty or len(historical) < 20:
        return pd.DataFrame(), selected_date

    X = historical[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
    historical = historical.loc[X.index]

    if X.empty or len(X) < 20:
        return pd.DataFrame(), selected_date

    target = target.replace([np.inf, -np.inf], np.nan)

    if target.isna().any(axis=1).iloc[0]:
        return pd.DataFrame(), selected_date

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    target_scaled = scaler.transform(target)

    n_neighbors = min(n, len(historical))

    model = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="euclidean"
    )

    model.fit(X_scaled)

    distances, indices = model.kneighbors(target_scaled)

    matches = historical.iloc[indices[0]].copy()
    matches["similarity_distance"] = distances[0]
    matches["similarity_score"] = 1 / (1 + matches["similarity_distance"])

    matches = matches.sort_values("similarity_distance")

    return matches, selected_date


# ============================================================
# OUTCOME SUMMARY
# ============================================================

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
    prob_5d_positive = summary.get("prob_positive_5d", np.nan)
    prob_3d_pullback = summary.get("prob_pullback_3d", np.nan)
    median_5d = summary.get("median_5d", np.nan)
    max_favourable = summary.get("median_max_favourable_5d", np.nan)
    max_adverse = summary.get("median_max_adverse_5d", np.nan)

    if "Exhaustion" in state or "chase-risk" in state:
        if pd.notna(prob_3d_pullback) and prob_3d_pullback > 0.45:
            return "Avoid chasing. Historical analogues favour waiting for a pullback or using scalp-only longs."
        return "Chase risk is elevated. Prefer smaller size, faster exits, or wait for pullback."

    if "Momentum-continuation" in state:
        if pd.notna(prob_5d_positive) and prob_5d_positive > 0.55 and median_5d > 0:
            return "Continuation has historical support. Pullbacks may be buyable if intraday structure confirms."
        return "Momentum state is present, but historical follow-through is not strong. Treat signals carefully."

    if "Pullback-buy" in state:
        if pd.notna(prob_5d_positive) and prob_5d_positive > 0.55:
            return "Pullback-buy conditions look historically favourable. Best behaviour is usually buying weakness, not chasing strength."
        return "Pullback setup exists, but historical edge is weak. Wait for confirmation."

    if "Hedge" in state or "defensive" in state:
        return "Defensive conditions. Long signals should be treated as scalp-only unless analogue outcomes improve."

    if "Chop" in state:
        return "Chop/no-trade conditions. Expect false signals and poor follow-through."

    if pd.notna(max_favourable) and pd.notna(max_adverse):
        if abs(max_favourable) > abs(max_adverse) and prob_5d_positive > 0.5:
            return "Slightly constructive, but not a clean high-conviction state."

    return "Context is mixed. Treat signals as scalp-only until the state improves."


def pct(x):
    if pd.isna(x):
        return "N/A"
    return f"{x * 100:.2f}%"


def prob(x):
    if pd.isna(x):
        return "N/A"
    return f"{x * 100:.1f}%"


# ============================================================
# UI
# ============================================================

st.title("🧠 SPY Market Memory")
st.caption("Not a signal generator. A historical context engine for SPY/QQQ traders.")

with st.sidebar:
    st.header("Settings")

    start_date = st.date_input(
        "Historical start date",
        value=pd.to_datetime("2014-01-01")
    )

    match_count = st.slider(
        "Similar historical days",
        min_value=20,
        max_value=150,
        value=50,
        step=10
    )

    selected_mode = st.radio(
        "Analysis date",
        ["Latest available day", "Choose date"]
    )

    st.markdown("---")
    st.caption("MVP: daily SPY state only. QQQ and VIX are used as context features.")

# ============================================================
# MAIN APP LOGIC
# ============================================================

spy, qqq, vix = load_market_data(start=start_date.strftime("%Y-%m-%d"))

if spy.empty:
    st.error(
        "SPY data did not download. This is usually a temporary yfinance/Yahoo issue. "
        "Refresh the app in a few minutes."
    )
    st.stop()

df = add_market_features(spy, qqq, vix)
usable_df = prepare_usable_dataframe(df)

if usable_df.empty:
    st.error(
        "The app downloaded data, but no usable rows remained after indicator calculations. "
        "Try using an earlier historical start date, such as 2014-01-01."
    )
    st.stop()

latest_date = usable_df.index[-1]

if selected_mode == "Choose date":
    chosen_date = st.sidebar.date_input(
        "Choose market date",
        value=latest_date.date(),
        min_value=usable_df.index[0].date(),
        max_value=latest_date.date()
    )

    selected_date_input = pd.to_datetime(chosen_date)
else:
    selected_date_input = latest_date

matches, selected_date = find_similar_days(df, selected_date_input, match_count)

if selected_date is None:
    st.error("The app could not identify a valid market date.")
    st.stop()

current = usable_df.loc[selected_date]
state = classify_state(current)

if matches.empty:
    st.error(
        "Not enough historical analogue matches were found. "
        "Try an earlier start date or fewer required similar days."
    )
    st.stop()

summary = outcome_summary(matches)
playbook = make_playbook(state, summary)

# ============================================================
# OUTPUT
# ============================================================

if selected_mode == "Latest available day":
    st.subheader("Today's SPY Playbook")
else:
    st.subheader(f"SPY Playbook for {selected_date.date()}")

col1, col2, col3, col4 = st.columns(4)

col1.metric("SPY close", f"${current['close']:.2f}")
col2.metric("RSI 14", f"{current['rsi_14']:.1f}")

if pd.notna(current["vix_close"]):
    col3.metric("VIX", f"{current['vix_close']:.2f}")
else:
    col3.metric("VIX", "N/A")

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

chart_start = selected_date - pd.Timedelta(days=365)
chart_df = df.loc[chart_start:selected_date].copy()

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=chart_df.index,
    open=chart_df["open"],
    high=chart_df["high"],
    low=chart_df["low"],
    close=chart_df["close"],
    name="SPY"
))

fig.add_trace(go.Scatter(
    x=chart_df.index,
    y=chart_df["ema_20"],
    name="EMA 20"
))

fig.add_trace(go.Scatter(
    x=chart_df.index,
    y=chart_df["ema_50"],
    name="EMA 50"
))

fig.add_trace(go.Scatter(
    x=chart_df.index,
    y=chart_df["ema_200"],
    name="EMA 200"
))

fig.update_layout(
    height=550,
    xaxis_rangeslider_visible=False,
    margin=dict(l=10, r=10, t=30, b=10)
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("### Most similar historical days")

display_cols = [
    "close",
    "rsi_14",
    "vix_close",
    "ret_5d",
    "ret_20d",
    "fwd_ret_1d",
    "fwd_ret_3d",
    "fwd_ret_5d",
    "fwd_ret_10d",
    "max_favourable_5d",
    "max_adverse_5d",
    "similarity_score"
]

available_display_cols = [c for c in display_cols if c in matches.columns]

table = matches[available_display_cols].copy()

percent_cols = [
    "ret_5d",
    "ret_20d",
    "fwd_ret_1d",
    "fwd_ret_3d",
    "fwd_ret_5d",
    "fwd_ret_10d",
    "max_favourable_5d",
    "max_adverse_5d"
]

for col in percent_cols:
    if col in table.columns:
        table[col] = table[col].map(lambda x: f"{x * 100:.2f}%" if pd.notna(x) else "N/A")

if "similarity_score" in table.columns:
    table["similarity_score"] = table["similarity_score"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")

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

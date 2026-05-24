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

# =====================================================
# Data + indicator helpers
# =====================================================

@st.cache_data(ttl=3600)
def download_market_data(start="2014-01-01"):
    tickers = ["SPY", "QQQ", "^VIX"]
    raw = yf.download(tickers, start=start, auto_adjust=False, progress=False, group_by="ticker")

    def clean_ticker(ticker):
        df = raw[ticker].copy()
        df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
        df = df.dropna()
        return df

    spy = clean_ticker("SPY")
    qqq = clean_ticker("QQQ")
    vix = clean_ticker("^VIX")

    for d in [spy, qqq, vix]:
        d.index = pd.to_datetime(d.index)

    if spy.empty:
        st.error("SPY data did not download. Refresh the app or try again later.")
        st.stop()

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


def consecutive_true_count(condition):
    out = []
    count = 0
    for value in condition.fillna(False):
        count = count + 1 if bool(value) else 0
        out.append(count)
    return out


def days_since_event(condition):
    out = []
    count = np.nan
    for hit in condition.fillna(False):
        if bool(hit):
            count = 0
        elif pd.isna(count):
            count = np.nan
        else:
            count += 1
        out.append(count)
    return out


def add_features(spy, qqq, vix):
    df = spy.copy()

    # Core indicators
    df["rsi_14"] = rsi(df["close"], 14)
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()
    df["ema20_slope_5d"] = df["ema_20"].pct_change(5)
    df["ema50_slope_10d"] = df["ema_50"].pct_change(10)

    # ATR
    df["atr_14"] = atr(df, 14)
    df["atr_pct"] = df["atr_14"] / df["close"]

    # Bollinger Bands
    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_mid"] = bb_mid
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # Keltner Channels
    df["kc_mid"] = df["ema_20"]
    df["kc_upper"] = df["kc_mid"] + 2 * df["atr_14"]
    df["kc_lower"] = df["kc_mid"] - 2 * df["atr_14"]
    df["kc_position"] = (df["close"] - df["kc_lower"]) / (df["kc_upper"] - df["kc_lower"])

    # Returns
    df["ret_1d"] = df["close"].pct_change(1)
    df["ret_3d"] = df["close"].pct_change(3)
    df["ret_5d"] = df["close"].pct_change(5)
    df["ret_10d"] = df["close"].pct_change(10)
    df["ret_20d"] = df["close"].pct_change(20)

    # Calendar / seasonality features
    df["weekday"] = df.index.day_name()
    df["weekday_num"] = df.index.weekday
    df["month"] = df.index.month_name()
    df["month_num"] = df.index.month
    df["day_of_month"] = df.index.day
    df["week_of_year"] = df.index.isocalendar().week.astype(int)

    # Distance from trend, raw and ATR-normalised
    for ema in [20, 50, 200]:
        df[f"dist_ema{ema}"] = (df["close"] / df[f"ema_{ema}"]) - 1
        df[f"atr_dist_ema{ema}"] = (df["close"] - df[f"ema_{ema}"]) / df["atr_14"]

    df["dist_bb_upper"] = (df["close"] / df["bb_upper"]) - 1
    df["dist_bb_lower"] = (df["close"] / df["bb_lower"]) - 1
    df["dist_kc_upper"] = (df["close"] / df["kc_upper"]) - 1
    df["dist_kc_lower"] = (df["close"] / df["kc_lower"]) - 1

    # Gap and candle/range pressure
    df["gap"] = (df["open"] / df["close"].shift(1)) - 1
    df["gap_atr"] = (df["open"] - df["close"].shift(1)) / df["atr_14"]
    df["range_atr"] = (df["high"] - df["low"]) / df["atr_14"]
    df["body_atr"] = (df["close"] - df["open"]).abs() / df["atr_14"]
    df["close_location"] = (df["close"] - df["low"]) / (df["high"] - df["low"])
    df["close_location"] = df["close_location"].replace([np.inf, -np.inf], np.nan)

    # Duration / trend age
    df["days_above_ema20"] = consecutive_true_count(df["close"] > df["ema_20"])
    df["days_above_ema50"] = consecutive_true_count(df["close"] > df["ema_50"])
    df["days_above_ema200"] = consecutive_true_count(df["close"] > df["ema_200"])
    df["days_below_ema20"] = consecutive_true_count(df["close"] < df["ema_20"])
    df["days_below_ema50"] = consecutive_true_count(df["close"] < df["ema_50"])
    df["days_below_ema200"] = consecutive_true_count(df["close"] < df["ema_200"])

    df["green_day"] = df["close"] > df["open"]
    df["red_day"] = df["close"] < df["open"]
    df["green_streak"] = consecutive_true_count(df["green_day"])
    df["red_streak"] = consecutive_true_count(df["red_day"])
    df["up_days_10"] = (df["close"].diff() > 0).rolling(10).sum()
    df["down_days_10"] = (df["close"].diff() < 0).rolling(10).sum()

    for pct in [0.01, 0.02, 0.03]:
        pullback = df["close"].pct_change() <= -pct
        df[f"days_since_{int(pct*100)}pct_pullback"] = days_since_event(pullback)

    # Weekly streaks
    weekly = df["close"].resample("W-FRI").last()
    weekly_green = weekly.diff() > 0
    weekly_red = weekly.diff() < 0
    weekly_green_streak = pd.Series(consecutive_true_count(weekly_green), index=weekly.index, name="weekly_green_streak")
    weekly_red_streak = pd.Series(consecutive_true_count(weekly_red), index=weekly.index, name="weekly_red_streak")
    df = df.join(weekly_green_streak.reindex(df.index, method="ffill"), how="left")
    df = df.join(weekly_red_streak.reindex(df.index, method="ffill"), how="left")

    # VIX
    vix_close = vix["close"].rename("vix_close")
    df = df.join(vix_close, how="left")
    df["vix_ret_5d"] = df["vix_close"].pct_change(5)

    # QQQ/SPY relative strength
    qqq_close = qqq["close"].rename("qqq_close")
    df = df.join(qqq_close, how="left")
    df["qqq_spy_rs_5d"] = qqq_close.pct_change(5) - df["close"].pct_change(5)
    df["qqq_spy_rs_20d"] = qqq_close.pct_change(20) - df["close"].pct_change(20)

    # Market condition scores
    df = add_market_condition_scores(df)

    # Forward outcomes
    for horizon in [1, 3, 5, 10]:
        # Final close-to-close outcome after N trading days
        df[f"fwd_ret_{horizon}d"] = df["close"].shift(-horizon) / df["close"] - 1

        # Path risk inside the forward window, using the next N daily highs/lows.
        # This is essential because a setup can finish positive after 10D but still
        # suffer a large drawdown first.
        future_high = df["high"].shift(-1).rolling(horizon).max().shift(-(horizon - 1))
        future_low = df["low"].shift(-1).rolling(horizon).min().shift(-(horizon - 1))
        df[f"max_favourable_{horizon}d"] = future_high / df["close"] - 1
        df[f"max_adverse_{horizon}d"] = future_low / df["close"] - 1

    return df.dropna()


def add_market_condition_scores(df):
    # Trend score: directional regime quality, not a trade signal by itself
    df["trend_score"] = 0
    df["trend_score"] += np.where(df["close"] > df["ema_20"], 1, -1)
    df["trend_score"] += np.where(df["close"] > df["ema_50"], 1, -1)
    df["trend_score"] += np.where(df["close"] > df["ema_200"], 2, -2)
    df["trend_score"] += np.where(df["ema_20"] > df["ema_50"], 1, -1)
    df["trend_score"] += np.where(df["ema_50"] > df["ema_200"], 1, -1)
    df["trend_score"] += np.where(df["ema20_slope_5d"] > 0, 1, -1)
    df["trend_score"] += np.where(df["ema50_slope_10d"] > 0, 1, -1)

    # Extension score: positive = upside stretched, negative = downside stretched
    ext = np.zeros(len(df))
    ext += np.where(df["rsi_14"] >= 75, 3, np.where(df["rsi_14"] >= 70, 2, np.where(df["rsi_14"] >= 65, 1, 0)))
    ext += np.where(df["rsi_14"] <= 25, -3, np.where(df["rsi_14"] <= 30, -2, np.where(df["rsi_14"] <= 35, -1, 0)))
    ext += np.where(df["atr_dist_ema20"] >= 2.0, 2, np.where(df["atr_dist_ema20"] >= 1.2, 1, 0))
    ext += np.where(df["atr_dist_ema20"] <= -2.0, -2, np.where(df["atr_dist_ema20"] <= -1.2, -1, 0))
    ext += np.where(df["bb_position"] >= 1.0, 2, np.where(df["bb_position"] >= 0.90, 1, 0))
    ext += np.where(df["bb_position"] <= 0.0, -2, np.where(df["bb_position"] <= 0.10, -1, 0))
    ext += np.where(df["kc_position"] >= 1.0, 1, 0)
    ext += np.where(df["kc_position"] <= 0.0, -1, 0)
    df["extension_score"] = ext

    # Duration/age score: positive = mature upside run, negative = mature downside run
    dur = np.zeros(len(df))
    dur += np.where(df["days_above_ema20"] >= 20, 2, np.where(df["days_above_ema20"] >= 10, 1, 0))
    dur += np.where(df["days_below_ema20"] >= 20, -2, np.where(df["days_below_ema20"] >= 10, -1, 0))
    dur += np.where(df["days_since_2pct_pullback"] >= 40, 2, np.where(df["days_since_2pct_pullback"] >= 20, 1, 0))
    dur += np.where(df["green_streak"] >= 5, 1, 0)
    dur += np.where(df["red_streak"] >= 5, -1, 0)
    dur += np.where(df["weekly_green_streak"] >= 5, 2, np.where(df["weekly_green_streak"] >= 3, 1, 0))
    dur += np.where(df["weekly_red_streak"] >= 5, -2, np.where(df["weekly_red_streak"] >= 3, -1, 0))
    df["duration_score"] = dur

    # Candle pressure: large candles/gaps near extremes add more context
    cp = np.zeros(len(df))
    cp += np.where((df["range_atr"] >= 1.4) & (df["close_location"] >= 0.75), 1, 0)
    cp += np.where((df["range_atr"] >= 1.4) & (df["close_location"] <= 0.25), -1, 0)
    cp += np.where(df["gap_atr"] >= 0.75, 1, 0)
    cp += np.where(df["gap_atr"] <= -0.75, -1, 0)
    df["candle_pressure_score"] = cp

    # VIX context: high VIX strengthens downside/panic context, low VIX strengthens complacent extension context
    vx = np.zeros(len(df))
    vx += np.where((df["vix_close"] < 16) & (df["extension_score"] > 0), 1, 0)
    vx += np.where((df["vix_close"] >= 25) & (df["extension_score"] < 0), -2, 0)
    vx += np.where((df["vix_close"] >= 25) & (df["close"] < df["ema_200"]), -2, 0)
    df["vix_context_score"] = vx

    raw = df["extension_score"] + df["duration_score"] + df["candle_pressure_score"] + df["vix_context_score"]
    df["extremity_score"] = raw.clip(-10, 10)

    return df


def classify_state(row):
    score = row["extremity_score"]
    trend = row["trend_score"]
    close = row["close"]
    above_200 = close > row["ema_200"]
    rsi_val = row["rsi_14"]
    vix_val = row["vix_close"]

    if score >= 7 and above_200:
        return "Extreme upside extension / chase-risk"
    if score >= 4 and above_200:
        return "Late-rally extension / wait-for-pullback"
    if score <= -7 and above_200:
        return "Extreme pullback in uptrend / bounce-watch"
    if score <= -7 and not above_200:
        return "Bear-market oversold / high-volatility bounce risk"
    if score <= -4 and above_200 and rsi_val < 45:
        return "Pullback-buy watch"
    if trend >= 5 and -3 <= score <= 3:
        return "Clean trend continuation"
    if above_200 and 35 <= rsi_val <= 55 and abs(row["atr_dist_ema20"]) <= 1.0:
        return "Healthy reset / possible continuation"
    if not above_200 and vix_val >= 25:
        return "Hedge-watch / defensive regime"
    if abs(score) <= 2 and abs(row["ret_5d"]) < 0.01:
        return "Neutral/chop / no strong edge"
    return "Mixed / context-dependent"


def signal_type_from_row(row):
    score = row["extremity_score"]
    above_200 = row["close"] > row["ema_200"]
    trend = row["trend_score"]
    if score >= 7:
        return "Extreme overbought / upside extension"
    if score <= -7:
        return "Extreme oversold / downside extension"
    if above_200 and row["rsi_14"] < 45 and row["atr_dist_ema20"] < -0.5 and score <= -3:
        return "Pullback in uptrend"
    if trend >= 5 and -3 <= score <= 3:
        return "Clean trend continuation"
    if abs(score) <= 2 and abs(row["ret_5d"]) < 0.01:
        return "Neutral/chop"
    return "Mixed/no clear signal"


def similar_days(df, selected_date, n=50):
    feature_cols = [
        "trend_score", "extension_score", "duration_score", "candle_pressure_score", "vix_context_score", "extremity_score",
        "rsi_14", "atr_dist_ema20", "atr_dist_ema50", "atr_dist_ema200",
        "bb_position", "kc_position", "range_atr", "body_atr", "close_location",
        "ret_1d", "ret_5d", "ret_20d", "gap_atr", "vix_close", "vix_ret_5d",
        "qqq_spy_rs_5d", "days_above_ema20", "days_above_ema50", "days_since_2pct_pullback",
        "weekly_green_streak", "weekly_red_streak"
    ]

    work = df.dropna(subset=feature_cols).copy()
    work = work.loc[work.index <= selected_date]

    if selected_date not in work.index:
        selected_date = work.index[-1]

    historical = work.loc[work.index < selected_date - pd.Timedelta(days=15)].copy()
    target = work.loc[[selected_date], feature_cols]

    if len(historical) < 10:
        st.error("Not enough historical data for analogue matching.")
        st.stop()

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


def outcome_summary(rows):
    summary = {}
    if rows.empty:
        return None
    for horizon in [1, 3, 5, 10]:
        col = f"fwd_ret_{horizon}d"
        vals = rows[col].dropna()
        summary[f"median_{horizon}d"] = vals.median()
        summary[f"mean_{horizon}d"] = vals.mean()
        summary[f"prob_positive_{horizon}d"] = (vals > 0).mean()
        summary[f"prob_pullback_{horizon}d"] = (vals < -0.01).mean()
    for horizon in [1, 3, 5, 10]:
        summary[f"median_max_favourable_{horizon}d"] = rows[f"max_favourable_{horizon}d"].median()
        summary[f"median_max_adverse_{horizon}d"] = rows[f"max_adverse_{horizon}d"].median()
        summary[f"chance_down_1pct_{horizon}d"] = (rows[f"max_adverse_{horizon}d"] <= -0.01).mean()
        summary[f"chance_down_2pct_{horizon}d"] = (rows[f"max_adverse_{horizon}d"] <= -0.02).mean()
        summary[f"chance_down_3pct_{horizon}d"] = (rows[f"max_adverse_{horizon}d"] <= -0.03).mean()
        summary[f"chance_down_5pct_{horizon}d"] = (rows[f"max_adverse_{horizon}d"] <= -0.05).mean()
        summary[f"worst10_adverse_{horizon}d"] = rows[f"max_adverse_{horizon}d"].quantile(0.10)
    summary["sample_size"] = len(rows)
    return summary


def pct(x):
    if pd.isna(x):
        return "N/A"
    return f"{x*100:.2f}%"


def prob(x):
    if pd.isna(x):
        return "N/A"
    return f"{x*100:.1f}%"


def make_playbook(state, row, summary):
    score = row["extremity_score"]
    prob_5d_positive = summary["prob_positive_5d"]
    prob_3d_pullback = summary["prob_pullback_3d"]
    med_5d = summary["median_5d"]

    if score >= 7:
        return "Market is highly extended to the upside. This is not automatically bearish, but historical testing should treat new longs as chase-risk unless pullbacks reset the condition."
    if score <= -7:
        return "Market is highly stretched to the downside. This is a potential bounce-watch zone, but volatility and adverse movement can remain high."
    if "Pullback" in state and prob_5d_positive > 0.55:
        return "Pullback conditions have historical support. Better suited to buying weakness than chasing strength."
    if "Clean trend" in state and prob_5d_positive > 0.55 and med_5d > 0:
        return "Trend continuation has historical support. Signals are more holdable than scalp-only."
    if prob_3d_pullback > 0.45:
        return "Short-term pullback risk is elevated in similar historical states. Treat fresh entries carefully."
    if prob_5d_positive > 0.55:
        return "Slightly constructive historical context, but check whether the current signal is extension, pullback, or neutral."
    return "Mixed context. No clean high-conviction edge from similar historical states."


def render_summary_metrics(summary, prefix=""):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{prefix}1D positive", prob(summary["prob_positive_1d"]))
    c2.metric(f"{prefix}3D positive", prob(summary["prob_positive_3d"]))
    c3.metric(f"{prefix}5D positive", prob(summary["prob_positive_5d"]))
    c4.metric(f"{prefix}10D positive", prob(summary["prob_positive_10d"]))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Median 1D return", pct(summary["median_1d"]))
    c6.metric("Median 3D return", pct(summary["median_3d"]))
    c7.metric("Median 5D return", pct(summary["median_5d"]))
    c8.metric("Median 10D return", pct(summary["median_10d"]))

    c9, c10, c11 = st.columns(3)
    c9.metric("5D pullback probability", prob(summary["prob_pullback_5d"]))
    c10.metric("Median max favourable 5D", pct(summary["median_max_favourable_5d"]))
    c11.metric("Median max adverse 5D", pct(summary["median_max_adverse_5d"]))


def render_path_risk_metrics(summary, horizon=10, prefix=""):
    """Render path-risk metrics: what happened inside the forward window, not only final return."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{prefix}{horizon}D final positive", prob(summary[f"prob_positive_{horizon}d"]))
    c2.metric(f"{prefix}Median {horizon}D final return", pct(summary[f"median_{horizon}d"]))
    c3.metric(f"{prefix}Median max favourable", pct(summary[f"median_max_favourable_{horizon}d"]))
    c4.metric(f"{prefix}Median max adverse", pct(summary[f"median_max_adverse_{horizon}d"]))

    r1, r2, r3, r4 = st.columns(4)
    r1.metric(f"Chance of -1% inside {horizon}D", prob(summary[f"chance_down_1pct_{horizon}d"]))
    r2.metric(f"Chance of -2% inside {horizon}D", prob(summary[f"chance_down_2pct_{horizon}d"]))
    r3.metric(f"Chance of -3% inside {horizon}D", prob(summary[f"chance_down_3pct_{horizon}d"]))
    r4.metric(f"Chance of -5% inside {horizon}D", prob(summary[f"chance_down_5pct_{horizon}d"]))

    p1, p2 = st.columns(2)
    median_mfe = summary[f"median_max_favourable_{horizon}d"]
    median_mae = summary[f"median_max_adverse_{horizon}d"]
    reward_to_pain = np.nan if median_mae == 0 else median_mfe / abs(median_mae)
    p1.metric("Worst 10% adverse move", pct(summary[f"worst10_adverse_{horizon}d"]))
    p2.metric("Reward-to-pain ratio", "N/A" if pd.isna(reward_to_pain) else f"{reward_to_pain:.2f}")


def first_touch_stats(full_df, rows, horizon=10, up_threshold=0.01, down_threshold=0.01):
    """Estimate whether price hit downside pain before upside reward inside the window."""
    if rows.empty:
        return {"drop_before_gain": np.nan, "gain_before_drop": np.nan, "neither": np.nan, "sample_size": 0}

    drop_first = 0
    gain_first = 0
    neither = 0
    usable = 0
    pos_lookup = {date: i for i, date in enumerate(full_df.index)}

    for date, row in rows.iterrows():
        pos = pos_lookup.get(date)
        if pos is None or pos + horizon >= len(full_df):
            continue
        future = full_df.iloc[pos + 1: pos + horizon + 1]
        if future.empty:
            continue
        entry = row["close"]
        down_level = entry * (1 - down_threshold)
        up_level = entry * (1 + up_threshold)

        first_down = None
        first_up = None
        for i, (_, frow) in enumerate(future.iterrows(), start=1):
            if first_down is None and frow["low"] <= down_level:
                first_down = i
            if first_up is None and frow["high"] >= up_level:
                first_up = i
            if first_down is not None and first_up is not None:
                break

        usable += 1
        if first_down is None and first_up is None:
            neither += 1
        elif first_down is not None and (first_up is None or first_down < first_up):
            drop_first += 1
        elif first_up is not None and (first_down is None or first_up < first_down):
            gain_first += 1
        else:
            # Same day touch of both levels. Count as ambiguous pain first for conservative risk reading.
            drop_first += 1

    if usable == 0:
        return {"drop_before_gain": np.nan, "gain_before_drop": np.nan, "neither": np.nan, "sample_size": 0}

    return {
        "drop_before_gain": drop_first / usable,
        "gain_before_drop": gain_first / usable,
        "neither": neither / usable,
        "sample_size": usable,
    }


def risk_label(summary, horizon=10):
    chance_2 = summary[f"chance_down_2pct_{horizon}d"]
    chance_3 = summary[f"chance_down_3pct_{horizon}d"]
    med_final = summary[f"median_{horizon}d"]
    med_adverse = summary[f"median_max_adverse_{horizon}d"]

    if chance_3 >= 0.25 or chance_2 >= 0.45:
        return "High pullback / reversal risk"
    if chance_2 >= 0.30 or med_adverse <= -0.015:
        return "Moderate pullback risk"
    if med_final > 0 and chance_2 < 0.25:
        return "Lower pullback risk"
    return "Mixed / unclear path risk"


def build_risk_patterns(df):
    """Predefined risk-focused market states. These are deliberately broad and testable."""
    patterns = {
        "Late rally extension": (df["close"] > df["ema_200"]) & (df["extremity_score"] >= 4),
        "Extreme upside extension": (df["close"] > df["ema_200"]) & (df["extremity_score"] >= 7),
        "RSI overbought + upper band": (df["rsi_14"] >= 70) & (df["bb_position"] >= 0.90),
        "Far above EMA20 by ATR": (df["atr_dist_ema20"] >= 1.5) & (df["close"] > df["ema_200"]),
        "Long run without 2% pullback": (df["days_since_2pct_pullback"] >= 30) & (df["close"] > df["ema_20"]) & (df["ema_20"] > df["ema_50"]),
        "5D rally + low VIX complacency": (df["ret_5d"] >= 0.02) & (df["vix_close"] < 18) & (df["close"] > df["ema_20"]),
        "SPY up while VIX rising": (df["ret_5d"] > 0) & (df["vix_ret_5d"] > 0.10) & (df["close"] > df["ema_20"]),
        "Gap up into extension": (df["gap_atr"] >= 0.50) & (df["extremity_score"] >= 4),
        "Neutral middle/no edge": (df["extremity_score"].abs() <= 2) & (df["ret_5d"].abs() < 0.01),
    }
    return patterns


def risk_pattern_table(df, horizon=10, min_sample=30):
    patterns = build_risk_patterns(df)
    valid = df.dropna(subset=[f"fwd_ret_{horizon}d", f"max_adverse_{horizon}d", f"max_favourable_{horizon}d"]).copy()
    baseline = outcome_summary(valid)
    rows = []
    for name, mask in patterns.items():
        sample = valid[mask.reindex(valid.index).fillna(False)]
        if len(sample) < min_sample:
            continue
        s = outcome_summary(sample)
        ft = first_touch_stats(valid, sample, horizon=horizon, up_threshold=0.01, down_threshold=0.01)
        median_mfe = s[f"median_max_favourable_{horizon}d"]
        median_mae = s[f"median_max_adverse_{horizon}d"]
        reward_to_pain = np.nan if median_mae == 0 else median_mfe / abs(median_mae)
        rows.append({
            "risk_pattern": name,
            "sample_size": len(sample),
            "final_positive_rate": s[f"prob_positive_{horizon}d"],
            "baseline_positive_rate": baseline[f"prob_positive_{horizon}d"],
            "median_final_return": s[f"median_{horizon}d"],
            "chance_-1%": s[f"chance_down_1pct_{horizon}d"],
            "chance_-2%": s[f"chance_down_2pct_{horizon}d"],
            "chance_-3%": s[f"chance_down_3pct_{horizon}d"],
            "median_max_adverse": s[f"median_max_adverse_{horizon}d"],
            "worst10_adverse": s[f"worst10_adverse_{horizon}d"],
            "reward_to_pain": reward_to_pain,
            "drop_1%_before_gain_1%": ft["drop_before_gain"],
            "risk_label": risk_label(s, horizon),
        })
    if not rows:
        return pd.DataFrame(), baseline
    out = pd.DataFrame(rows)
    out["excess_-2%_risk_vs_baseline"] = out["chance_-2%"] - baseline[f"chance_down_2pct_{horizon}d"]
    return out.sort_values(["chance_-2%", "worst10_adverse"], ascending=[False, True]), baseline


def render_clean_path_risk_table(summary, horizon=10, baseline=False):
    """Readable path-risk table. Avoids misaligned metric cards."""
    prefix = "Baseline " if baseline else ""
    median_mfe = summary[f"median_max_favourable_{horizon}d"]
    median_mae = summary[f"median_max_adverse_{horizon}d"]
    reward_to_pain = np.nan if median_mae == 0 else median_mfe / abs(median_mae)

    rows = [
        {
            "Group": "Final outcome",
            "Metric": f"{prefix}{horizon}D final positive rate",
            "Value": prob(summary[f"prob_positive_{horizon}d"]),
            "Meaning": "How often SPY finished higher after the selected window."
        },
        {
            "Group": "Final outcome",
            "Metric": f"{prefix}Median {horizon}D final return",
            "Value": pct(summary[f"median_{horizon}d"]),
            "Meaning": "Typical close-to-close return after the selected window."
        },
        {
            "Group": "Path risk",
            "Metric": f"Chance of -1% drop inside {horizon}D",
            "Value": prob(summary[f"chance_down_1pct_{horizon}d"]),
            "Meaning": "How often SPY fell at least 1% at any point before the window ended."
        },
        {
            "Group": "Path risk",
            "Metric": f"Chance of -2% drop inside {horizon}D",
            "Value": prob(summary[f"chance_down_2pct_{horizon}d"]),
            "Meaning": "More meaningful pullback risk. This matters more than final return for entries."
        },
        {
            "Group": "Path risk",
            "Metric": f"Chance of -3% drop inside {horizon}D",
            "Value": prob(summary[f"chance_down_3pct_{horizon}d"]),
            "Meaning": "Shows sharper downside reversal risk."
        },
        {
            "Group": "Path risk",
            "Metric": f"Chance of -5% drop inside {horizon}D",
            "Value": prob(summary[f"chance_down_5pct_{horizon}d"]),
            "Meaning": "Rare but serious downside move inside the window."
        },
        {
            "Group": "Pain vs reward",
            "Metric": "Median max favourable move",
            "Value": pct(median_mfe),
            "Meaning": "Typical best upside seen inside the window."
        },
        {
            "Group": "Pain vs reward",
            "Metric": "Median max adverse move",
            "Value": pct(median_mae),
            "Meaning": "Typical worst downside seen inside the window."
        },
        {
            "Group": "Pain vs reward",
            "Metric": "Worst 10% adverse move",
            "Value": pct(summary[f"worst10_adverse_{horizon}d"]),
            "Meaning": "Bad-case path risk. In the worst 10% of cases, downside was around this or worse."
        },
        {
            "Group": "Pain vs reward",
            "Metric": "Reward-to-pain ratio",
            "Value": "N/A" if pd.isna(reward_to_pain) else f"{reward_to_pain:.2f}",
            "Meaning": "Median favourable move divided by median adverse move. Above 1 is cleaner; below 1 means pain dominated reward."
        },
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def score_explainer_table():
    rows = [
        ["Trend score", "How clean the trend is: EMA structure, price vs EMA20/50/200, and EMA slopes.", "Higher = stronger uptrend. Negative = weaker/bearish trend."],
        ["Extension score", "How stretched price is from normal trend levels: ATR distance from EMA20/50 and Bollinger/Keltner position.", "High positive = overextended upward. High negative = oversold/downside stretch."],
        ["Duration score", "How long the move has gone without a reset: days above EMA20, days since 1–3% pullback, green streaks.", "Higher = trend is mature/possibly crowded."],
        ["Candle pressure", "How large and directional the recent candle/range pressure is using ATR, body size, gap, and close location.", "High = strong pressure candle; context decides whether it is ignition or exhaustion."],
        ["VIX context", "Whether volatility is calm, elevated, or rising relative to recent SPY movement.", "High VIX/rising VIX can increase path risk even if SPY is still up."],
        ["Extremity score", "Combined score from trend, extension, duration, candle pressure, and VIX.", "+7 to +10 = upside extension/chase risk. -7 to -10 = downside stretch/bounce watch. Near 0 = middle/no clear edge."],
        ["Deviation score", "A similarity/math score used internally for analogue matching.", "Do not treat it as a trading signal. It only helps find similar historical days."],
        ["Weekly context", "Weekly green/red streak and recent weekly direction.", "Useful for knowing whether a daily setup is early, mature, or stretched across weeks."],
    ]
    return pd.DataFrame(rows, columns=["Label", "What it means", "How to read it"])


def seasonality_condition_mask(df, selected_date, condition_name):
    if condition_name == "All market days":
        return pd.Series(True, index=df.index)
    if condition_name == "Current exact market state":
        current_state = classify_state(df.loc[selected_date])
        return df.apply(classify_state, axis=1) == current_state
    if condition_name == "Bullish regime":
        return (df["close"] > df["ema_200"]) & (df["ema_20"] > df["ema_50"])
    if condition_name == "Bearish regime":
        return (df["close"] < df["ema_200"]) | (df["ema_20"] < df["ema_50"])
    if condition_name == "Upside extension":
        return df["extremity_score"] >= 4
    if condition_name == "Extreme upside extension":
        return df["extremity_score"] >= 7
    if condition_name == "Oversold / downside stretch":
        return df["extremity_score"] <= -4
    if condition_name == "Pullback in uptrend":
        return (df["close"] > df["ema_200"]) & (df["ema_20"] > df["ema_50"]) & (df["rsi_14"].between(35, 55)) & (df["close"] <= df["ema_20"])
    return pd.Series(True, index=df.index)


def seasonality_table(df, selected_date, horizon=5, condition_name="All market days", group_by="Weekday", min_sample=20):
    valid = df.loc[df.index <= selected_date].dropna(
        subset=[f"fwd_ret_{horizon}d", f"max_adverse_{horizon}d", f"max_favourable_{horizon}d"]
    ).copy()

    mask = seasonality_condition_mask(valid, selected_date, condition_name)
    filtered = valid[mask.reindex(valid.index).fillna(False)].copy()

    if group_by == "Weekday":
        group_col = "weekday"
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    elif group_by == "Month":
        group_col = "month"
        order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    elif group_by == "Day of month":
        group_col = "day_of_month"
        order = None
    else:
        group_col = "week_of_year"
        order = None

    baseline = outcome_summary(filtered)
    rows = []
    for label, group in filtered.groupby(group_col):
        if len(group) < min_sample:
            continue
        s = outcome_summary(group)
        rows.append({
            "period": label,
            "sample_size": len(group),
            "positive_rate": s[f"prob_positive_{horizon}d"],
            "baseline_positive_rate": baseline[f"prob_positive_{horizon}d"],
            "edge_vs_condition": s[f"prob_positive_{horizon}d"] - baseline[f"prob_positive_{horizon}d"],
            "median_return": s[f"median_{horizon}d"],
            "chance_-2%_inside": s[f"chance_down_2pct_{horizon}d"],
            "median_max_adverse": s[f"median_max_adverse_{horizon}d"],
            "worst10_adverse": s[f"worst10_adverse_{horizon}d"],
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out, baseline, len(filtered)

    if order is not None:
        out["_order"] = out["period"].apply(lambda x: order.index(x) if x in order else 999)
        out = out.sort_values("_order").drop(columns=["_order"])
    else:
        out = out.sort_values("period")
    return out, baseline, len(filtered)


# =====================================================
# UI
# =====================================================

st.title("🧠 SPY Market Memory")
st.caption("Historical market-state memory for SPY. Daily-data MVP with Signal Forensics, Risk Discovery and Seasonality.")

with st.sidebar:
    st.header("Settings")
    start_date = st.date_input("Historical start date", value=pd.to_datetime("2014-01-01"), help="Earliest date used for the historical sample.")
    match_count = st.slider("Similar historical days", 20, 150, 50, step=10, help="How many closest historical analogue days to compare with the selected day.")
    selected_mode = st.radio("Analysis date", ["Latest available day", "Choose date"])
    st.markdown("---")
    st.caption("This version generates signals inside Python. No TradingView connection is needed.")

try:
    spy, qqq, vix = download_market_data(start=start_date.strftime("%Y-%m-%d"))
    df = add_features(spy, qqq, vix)

    if df.empty:
        st.error("No usable data after indicator calculations. Try an earlier start date or refresh the app.")
        st.stop()

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
    current_signal = signal_type_from_row(current)
    matches = similar_days(df, selected_date, match_count)
    summary = outcome_summary(matches)
    playbook = make_playbook(state, current, summary)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Today’s Playbook", "Signal Forensics", "Similar Historical Days", "Risk Discovery", "Seasonality"])

    with tab1:
        st.subheader("Today's SPY Playbook" if selected_mode == "Latest available day" else f"SPY Playbook for {selected_date.date()}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("SPY close", f"${current['close']:.2f}")
        col2.metric("RSI 14", f"{current['rsi_14']:.1f}")
        col3.metric("VIX", f"{current['vix_close']:.2f}")
        col4.metric("Extremity score", f"{current['extremity_score']:.1f} / 10")

        st.info(f"**Current state:** {state}")
        st.info(f"**Current signal type:** {current_signal}")
        st.success(f"**Playbook:** {playbook}")

        st.markdown("### Score breakdown")
        st.caption("These are context scores, not buy/sell signals. Use them to understand whether the market is clean, stretched, mature, or risky.")
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Trend score", f"{current['trend_score']:.1f}", help="EMA structure and trend direction. Higher = cleaner uptrend; negative = weaker/bearish trend.")
        s2.metric("Extension score", f"{current['extension_score']:.1f}", help="How far price is stretched from EMA/bands using ATR. High positive = upside extension/chase risk.")
        s3.metric("Duration score", f"{current['duration_score']:.1f}", help="How long the move has continued without a meaningful reset or pullback.")
        s4.metric("Candle pressure", f"{current['candle_pressure_score']:.1f}", help="Range/body/gap pressure measured with ATR. Context decides whether it is ignition or exhaustion.")
        s5.metric("VIX context", f"{current['vix_context_score']:.1f}", help="Volatility context. Rising/elevated VIX can increase path risk.")

        with st.expander("What do these labels mean?"):
            st.dataframe(score_explainer_table(), use_container_width=True, hide_index=True)

        st.markdown("### Key context features")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("ATR distance from EMA20", f"{current['atr_dist_ema20']:.2f}", help="How many ATRs price is above/below EMA20. This is more useful than raw percent distance.")
        k2.metric("Days above EMA20", f"{int(current['days_above_ema20'])}", help="Trend age above the fast EMA. Very high values can mean mature/extended trend.")
        k3.metric("Days since 2% pullback", f"{int(current['days_since_2pct_pullback'])}", help="How long since SPY had a one-day 2% pullback.")
        k4.metric("Weekly green streak", f"{int(current['weekly_green_streak'])}", help="How many consecutive weekly closes were higher.")

        st.markdown("### Historical analogue outcomes")
        render_summary_metrics(summary)

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

    with tab2:
        st.subheader("Signal Forensics")
        st.caption("This section tests Python-generated signal types. It does not need TradingView exports.")

        signal_choice = st.selectbox(
            "Choose signal type to investigate",
            [
                "Current signal type",
                "Extreme overbought / upside extension",
                "Extreme oversold / downside extension",
                "Pullback in uptrend",
                "Clean trend continuation",
                "Neutral/chop",
            ]
        )

        chosen_signal = current_signal if signal_choice == "Current signal type" else signal_choice
        forensic_df = df.copy()
        forensic_df["signal_type"] = forensic_df.apply(signal_type_from_row, axis=1)
        signal_rows = forensic_df[forensic_df["signal_type"] == chosen_signal].copy()

        # Avoid rows where forward data is unavailable
        signal_rows = signal_rows.dropna(subset=["fwd_ret_1d", "fwd_ret_3d", "fwd_ret_5d", "fwd_ret_10d"])

        st.markdown(f"### Tested signal: {chosen_signal}")
        st.metric("Historical signal count", len(signal_rows))

        if len(signal_rows) < 10:
            st.warning("Too few historical examples for a stable read. Treat this as a weak sample.")
        if len(signal_rows) > 0:
            signal_summary = outcome_summary(signal_rows)
            render_summary_metrics(signal_summary)

            st.markdown("### Performance by market state")
            signal_rows["market_state"] = signal_rows.apply(classify_state, axis=1)
            grouped = []
            for market_state, group in signal_rows.groupby("market_state"):
                if len(group) < 5:
                    continue
                s = outcome_summary(group)
                grouped.append({
                    "market_state": market_state,
                    "count": len(group),
                    "prob_positive_5d": s["prob_positive_5d"],
                    "median_5d": s["median_5d"],
                    "prob_pullback_5d": s["prob_pullback_5d"],
                    "median_max_adverse_5d": s["median_max_adverse_5d"],
                })

            if grouped:
                gdf = pd.DataFrame(grouped).sort_values("prob_positive_5d", ascending=False)
                show = gdf.copy()
                for col in ["prob_positive_5d", "median_5d", "prob_pullback_5d", "median_max_adverse_5d"]:
                    show[col] = show[col].map(lambda x: f"{x*100:.2f}%")
                st.dataframe(show, use_container_width=True)
            else:
                st.info("Not enough grouped samples yet for state-by-state breakdown.")

            st.markdown("### Recent examples of this signal")
            recent_cols = [
                "close", "rsi_14", "vix_close", "extremity_score", "trend_score",
                "atr_dist_ema20", "days_above_ema20", "days_since_2pct_pullback",
                "fwd_ret_1d", "fwd_ret_3d", "fwd_ret_5d", "fwd_ret_10d"
            ]
            recent = signal_rows[recent_cols].tail(30).copy()
            for col in ["fwd_ret_1d", "fwd_ret_3d", "fwd_ret_5d", "fwd_ret_10d"]:
                recent[col] = recent[col].map(lambda x: f"{x*100:.2f}%")
            st.dataframe(recent, use_container_width=True)

    with tab3:
        st.subheader("Most similar historical days")
        display_cols = [
            "close", "rsi_14", "vix_close", "trend_score", "extremity_score",
            "atr_dist_ema20", "days_above_ema20", "days_since_2pct_pullback",
            "ret_5d", "ret_20d", "fwd_ret_1d", "fwd_ret_3d", "fwd_ret_5d", "fwd_ret_10d",
            "max_favourable_5d", "max_adverse_5d", "max_favourable_10d", "max_adverse_10d", "similarity_score"
        ]
        table = matches[display_cols].copy()
        for col in ["ret_5d", "ret_20d", "fwd_ret_1d", "fwd_ret_3d", "fwd_ret_5d", "fwd_ret_10d", "max_favourable_5d", "max_adverse_5d", "max_favourable_10d", "max_adverse_10d"]:
            table[col] = table[col].map(lambda x: f"{x*100:.2f}%")
        table["similarity_score"] = table["similarity_score"].map(lambda x: f"{x:.3f}")
        st.dataframe(table, use_container_width=True)

    with tab4:
        st.subheader("Risk Discovery")
        st.caption("This tab focuses on downside path risk: what happened inside the next few days, not only where SPY finished.")

        risk_horizon = st.selectbox(
            "Forward risk window",
            [5, 10],
            index=1,
            help="Number of trading days to inspect after the selected day. 5–10D is usually cleaner for this MVP; 30D can mix too many unrelated events."
        )

        st.markdown("### Path risk for current similar historical states")
        st.write("This uses the same analogue matches as Today’s Playbook, but asks: **how often did SPY drop first or suffer a meaningful drawdown inside the window?**")

        current_risk_summary = outcome_summary(
            matches.dropna(subset=[f"max_adverse_{risk_horizon}d", f"max_favourable_{risk_horizon}d", f"fwd_ret_{risk_horizon}d"])
        )

        c1, c2, c3 = st.columns(3)
        c1.metric(f"{risk_horizon}D final positive", prob(current_risk_summary[f"prob_positive_{risk_horizon}d"]), help="How often similar historical days finished higher after the selected window.")
        c2.metric(f"Chance of -2% inside {risk_horizon}D", prob(current_risk_summary[f"chance_down_2pct_{risk_horizon}d"]), help="How often price dropped at least 2% at any point inside the selected window.")
        c3.metric("Worst 10% adverse move", pct(current_risk_summary[f"worst10_adverse_{risk_horizon}d"]), help="Bad-case path risk. In the worst 10% of cases, downside was around this or worse.")

        st.warning(f"**Risk label:** {risk_label(current_risk_summary, risk_horizon)}")

        with st.expander("Show full path-risk table"):
            render_clean_path_risk_table(current_risk_summary, horizon=risk_horizon)

        ft = first_touch_stats(df, matches, horizon=risk_horizon, up_threshold=0.01, down_threshold=0.01)
        st.markdown("### First move test")
        st.caption("This checks whether SPY usually hit -1% first or +1% first after similar historical states.")
        ft_df = pd.DataFrame([
            {"First event inside window": "-1% drop before +1% gain", "Probability": prob(ft["drop_before_gain"]), "Meaning": "Pain first. Bad for chasing entries."},
            {"First event inside window": "+1% gain before -1% drop", "Probability": prob(ft["gain_before_drop"]), "Meaning": "Reward first. Cleaner entry path."},
        ])
        st.dataframe(ft_df, use_container_width=True, hide_index=True)

        st.markdown("### Downside risk pattern discovery")
        st.write("These are broad predefined risk states. The goal is to find where upside-looking markets historically had poor path quality or higher pullback/reversal risk.")
        min_sample = st.slider(
            "Minimum historical occurrences",
            20,
            100,
            30,
            step=10,
            help="Minimum number of historical times a risk pattern must have appeared before it is shown. Higher = more reliable, but fewer patterns."
        )
        risk_table, baseline_summary = risk_pattern_table(df.loc[df.index <= selected_date].copy(), horizon=risk_horizon, min_sample=min_sample)

        with st.expander("Baseline SPY path risk for comparison"):
            render_clean_path_risk_table(baseline_summary, horizon=risk_horizon, baseline=True)

        if risk_table.empty:
            st.info("No risk patterns met the minimum historical occurrence filter. Try lowering the minimum.")
        else:
            show_risk = risk_table.copy()
            show_risk = show_risk.rename(columns={
                "risk_pattern": "risk pattern",
                "sample_size": "historical occurrences",
                "final_positive_rate": f"{risk_horizon}D positive rate",
                "baseline_positive_rate": "baseline positive rate",
                "median_final_return": f"median {risk_horizon}D return",
                "chance_-1%": f"chance -1% inside {risk_horizon}D",
                "chance_-2%": f"chance -2% inside {risk_horizon}D",
                "chance_-3%": f"chance -3% inside {risk_horizon}D",
                "median_max_adverse": "median max adverse",
                "worst10_adverse": "worst 10% adverse",
                "reward_to_pain": "reward/pain",
                "drop_1%_before_gain_1%": "-1% before +1%",
                "risk_label": "risk label",
                "excess_-2%_risk_vs_baseline": "extra -2% risk vs baseline",
            })
            percent_cols = [
                f"{risk_horizon}D positive rate", "baseline positive rate", f"median {risk_horizon}D return",
                f"chance -1% inside {risk_horizon}D", f"chance -2% inside {risk_horizon}D", f"chance -3% inside {risk_horizon}D",
                "median max adverse", "worst 10% adverse", "-1% before +1%", "extra -2% risk vs baseline"
            ]
            for col in percent_cols:
                if col in show_risk.columns:
                    show_risk[col] = show_risk[col].map(lambda x: "N/A" if pd.isna(x) else f"{x*100:.2f}%")
            if "reward/pain" in show_risk.columns:
                show_risk["reward/pain"] = show_risk["reward/pain"].map(lambda x: "N/A" if pd.isna(x) else f"{x:.2f}")
            st.dataframe(show_risk, use_container_width=True, hide_index=True)

        st.markdown("### How to read this tab")
        st.info("A setup can finish positive after 10 days and still be a bad entry if it usually drops hard first. This tab exposes that hidden path risk.")

    with tab5:
        st.subheader("Seasonality by market condition")
        st.caption("This checks whether calendar tendencies still appear after filtering by market condition. It is not meant to use old seasonality blindly.")

        s_col1, s_col2, s_col3, s_col4 = st.columns(4)
        season_horizon = s_col1.selectbox("Forward window", [1, 3, 5, 10], index=2, help="Return/path-risk window after each calendar date.")
        group_by = s_col2.selectbox("Group by", ["Weekday", "Month", "Day of month", "Week of year"], index=0)
        condition_name = s_col3.selectbox(
            "Market condition filter",
            [
                "All market days",
                "Current exact market state",
                "Bullish regime",
                "Bearish regime",
                "Upside extension",
                "Extreme upside extension",
                "Oversold / downside stretch",
                "Pullback in uptrend",
            ],
            index=0,
            help="Lets you test whether day/month seasonality changes in bullish, bearish, extended, or pullback conditions."
        )
        season_min_sample = s_col4.slider("Minimum occurrences", 10, 100, 20, step=10, help="Minimum sample count for each weekday/month/day bucket.")

        season_table, season_baseline, filtered_count = seasonality_table(
            df, selected_date, horizon=season_horizon, condition_name=condition_name, group_by=group_by, min_sample=season_min_sample
        )

        st.markdown("### Baseline inside selected condition")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Filtered historical days", f"{filtered_count}")
        b2.metric(f"Baseline {season_horizon}D positive", prob(season_baseline[f"prob_positive_{season_horizon}d"]))
        b3.metric(f"Baseline median {season_horizon}D return", pct(season_baseline[f"median_{season_horizon}d"]))
        b4.metric(f"Baseline -2% path risk", prob(season_baseline[f"chance_down_2pct_{season_horizon}d"]))

        if season_table.empty:
            st.info("No seasonality buckets met the minimum occurrence filter. Try lowering the minimum or using a broader market condition.")
        else:
            display = season_table.copy()
            percent_cols = ["positive_rate", "baseline_positive_rate", "edge_vs_condition", "median_return", "chance_-2%_inside", "median_max_adverse", "worst10_adverse"]
            for col in percent_cols:
                display[col] = display[col].map(lambda x: "N/A" if pd.isna(x) else f"{x*100:.2f}%")
            display = display.rename(columns={
                "period": "period",
                "sample_size": "historical occurrences",
                "positive_rate": f"{season_horizon}D positive rate",
                "baseline_positive_rate": "condition baseline",
                "edge_vs_condition": "edge vs same condition",
                "median_return": f"median {season_horizon}D return",
                "chance_-2%_inside": f"chance -2% inside {season_horizon}D",
                "median_max_adverse": "median max adverse",
                "worst10_adverse": "worst 10% adverse",
            })
            st.dataframe(display, use_container_width=True, hide_index=True)

        with st.expander("How to read Seasonality"):
            st.write("""
Seasonality here means: **does a weekday/month behave differently after controlling for market condition?**

The important column is **edge vs same condition**.  
Example: if Thursdays are bearish overall, this checks whether Thursdays are still bearish inside bullish regimes, bearish regimes, upside extensions, etc.

Do not treat a calendar pattern as meaningful just because one bucket is high. Check:
- historical occurrences
- edge vs same condition
- median return
- path risk
- whether the sample is large enough
""")

    st.caption(
        "Research prototype only. It shows historical analogue outcomes; it does not predict future prices or provide financial advice."
    )

except Exception as e:
    st.error("The app could not complete the analysis.")
    st.exception(e)

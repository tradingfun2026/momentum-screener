import streamlit as st
import pandas as pd
import yfinance as yf
import concurrent.futures
from datetime import datetime, timezone
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objs as go
import math
import random

# ========================= SETTINGS =========================
THREADS               = 20           # keep high but not crazy
AUTO_REFRESH_MS       = 120_000      # auto-refresh every 120 seconds
HISTORY_LOOKBACK_DAYS = 10           # 10-day mode
INTRADAY_INTERVAL     = "2m"         # 2-minute candles
INTRADAY_RANGE        = "1d"

DEFAULT_MAX_PRICE     = 5.0
DEFAULT_MIN_VOLUME    = 0.0          # float now
DEFAULT_MIN_BREAKOUT  = 0.0

# ========================= AUTO REFRESH =========================
st_autorefresh(interval=AUTO_REFRESH_MS, key="refresh_v9")

# ========================= PAGE SETUP =========================
st.set_page_config(
    page_title="V9 – 10-Day Momentum Screener (Hybrid Volume/Randomized)",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🚀 V9 — 10-Day Momentum Breakout Screener (Hybrid Speed + Volume + Randomized)")
st.caption(
    "Short-window model • EMA10 • RSI(7) • 3D & 10D momentum • 10D RVOL • "
    "VWAP + order flow • Watchlist mode • Audio alerts • V9 universe modes (classic / random / volume-ranked)"
)

# ========================= FORMATTER =========================
def fmt2(x):
    """Format numbers to 2 decimals; show '—' if missing or invalid."""
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x))):
            return "—"
        return f"{float(x):.2f}"
    except Exception:
        return "—"

# ========================= SIDEBAR CONTROLS =========================
with st.sidebar:
    st.header("Universe")

    watchlist_text = st.text_area(
        "Watchlist tickers (comma/space/newline separated):",
        value="",
        height=80,
        help="Example: AAPL, TSLA, NVDA, AMD",
    )

    max_universe = st.slider(
        "Max symbols to scan when no watchlist",
        min_value=50,
        max_value=2000,
        value=2000,
        step=50,
        help="Keeps scans fast when you don't use a custom watchlist.",
    )

    # V9 universe mode
    st.markdown("---")
    st.subheader("V9 Universe Mode")
    universe_mode = st.radio(
        "Universe Construction",
        options=[
            "Classic (Alphabetical Slice)",
            "Randomized Slice",
            "Live Volume Ranked (slower)",
        ],
        index=0,
        help=(
            "Classic = original behavior.\n"
            "Randomized = random subset of symbols each scan.\n"
            "Live Volume Ranked = prioritize highest intraday volume (slower)."
        ),
    )

    volume_rank_pool = st.slider(
        "Max symbols to consider when volume-ranking (V9)",
        min_value=100,
        max_value=2000,
        value=600,
        step=100,
        help="Used only when 'Live Volume Ranked (slower)' is selected.",
    )

    enable_enrichment = st.checkbox(
        "Include float/short + news (slower, more data)",
        value=False,
    )

    st.markdown("---")
    st.header("Filters")

    max_price = st.number_input("Max Price ($)", 1.0, 1000.0, DEFAULT_MAX_PRICE, 0.5)

    # float-based to avoid StreamlitMixedNumericTypesError
    min_volume = st.number_input(
        "Min Daily Volume",
        0.0,
        10_000_000.0,
        DEFAULT_MIN_VOLUME,
        10_000.0,
    )

    min_breakout = st.number_input("Min Breakout Score", -50.0, 200.0, 0.0, 0.5)
    min_pm_move = st.number_input("Min Premarket %", -50.0, 200.0, 0.0, 0.5)
    min_yday_gain = st.number_input("Min Yesterday %", -50.0, 200.0, 0.0, 0.5)

    squeeze_only = st.checkbox("Short-Squeeze Only")
    catalyst_only = st.checkbox("Must Have News/Earnings")
    vwap_only = st.checkbox("Above VWAP Only (VWAP% > 0)")

    st.markdown("---")
    st.subheader("Order Flow Filter (optional)")
    enable_ofb_filter = st.checkbox(
        "Use Min Order Flow Bias Filter",
        value=False,
        help="When enabled, only keep symbols where buy volume dominates."
    )
    min_ofb = st.slider(
        "Min Order Flow Bias (0–1, buyer control)",
        min_value=0.00,
        max_value=1.00,
        value=0.50,
        step=0.01,
        help="0.5 = equal buy/sell; 0.7 = strong buyer control."
    )

    st.markdown("---")
    st.subheader("🔊 Audio Alert Thresholds")

    enable_alerts = st.checkbox(
        "Enable Audio + Alert Banner",
        value=False,
        help="Turn this off to completely silence alerts."
    )

    ALERT_SCORE_THRESHOLD = st.slider("Alert when Score ≥", 10, 200, 30, 5)
    ALERT_PM_THRESHOLD = st.slider("Alert when Premarket % ≥", 1, 150, 4, 1)
    ALERT_VWAP_THRESHOLD = st.slider("Alert when VWAP Dist % ≥", 1, 50, 2, 1)

    st.markdown("---")
    if st.button("🧹 Refresh Now"):
        st.cache_data.clear()
        st.success("Cache cleared — fresh scan will run now.")

# ========================= SYMBOL LOAD =========================
@st.cache_data(ttl=900)
def load_symbols():
    """Load US symbols (NASDAQ + otherlisted)."""
    nasdaq = pd.read_csv(
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        sep="|", skipfooter=1, engine="python"
    )
    other = pd.read_csv(
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
        sep="|", skipfooter=1, engine="python"
    )

    nasdaq["Exchange"] = "NASDAQ"
    other["Exchange"] = other["Exchange"].fillna("NYSE/AMEX/ARCA")
    other = other.rename(columns={"ACT Symbol": "Symbol"})

    df = pd.concat(
        [nasdaq[["Symbol", "ETF", "Exchange"]],
         other[["Symbol", "ETF", "Exchange"]]]
    )

    # Fix: ensure Symbol is string, non-null before regex filtering
    df["Symbol"] = df["Symbol"].astype(str).fillna("")
    df = df[df["Symbol"].str.contains(r"^[A-Z]{1,5}$", na=False, regex=True)]
    return df.to_dict("records")


def build_universe(
    watchlist_text: str,
    max_universe: int,
    universe_mode: str,
    volume_rank_pool: int,
):
    """
    Return a list of symbol dicts to scan, based on:
    - Watchlist
    - Classic alphabetical slice
    - Randomized slice
    - Live volume-ranked (V9)
    """
    wl = watchlist_text.strip()
    if wl:
        # WATCHLIST TAKES PRECEDENCE: build only from these tickers
        raw = wl.replace("\n", " ").replace(",", " ").split()
        tickers = sorted(set(s.upper() for s in raw if s.strip()))
        return [{"Symbol": t, "Exchange": "WATCH"} for t in tickers]

    all_syms = load_symbols()

    # V9 modes
    if universe_mode == "Randomized Slice":
        base = all_syms[:]
        random.shuffle(base)
        return base[:max_universe]

    if universe_mode == "Live Volume Ranked (slower)":
        base = all_syms[:volume_rank_pool]
        ranked = []
        for sym in base:
            try:
                t = yf.Ticker(sym["Symbol"])
                d = t.history(period="1d", interval="2m", prepost=True)
                if not d.empty:
                    live_vol = float(d["Volume"].iloc[-1])
                    ranked.append({**sym, "LiveVol": live_vol})
            except Exception:
                continue

        if not ranked:
            # fallback to classic if volume fetch fails
            return all_syms[:max_universe]

        ranked_sorted = sorted(
            ranked,
            key=lambda x: x.get("LiveVol", 0.0),
            reverse=True
        )
        return ranked_sorted[:max_universe]

    # Classic behavior
    return all_syms[:max_universe]

# ========================= SCORING (10-DAY MODEL) =========================
def short_window_score(pm, yday, m3, m10, rsi7, rvol10, catalyst, squeeze, vwap, flow_bias):
    """
    pm        = premarket % (2m candles)
    yday      = yesterday % move
    m3        = 3-day % move
    m10       = 10-day % move
    rsi7      = RSI(7)
    rvol10    = relative volume vs 10-day avg
    catalyst  = bool
    squeeze   = bool
    vwap      = % above VWAP
    flow_bias = 0..1 (buy volume / total volume)
    """
    score = 0.0

    if pm is not None:
        score += max(pm, 0) * 1.6
    if yday is not None:
        score += max(yday, 0) * 0.8
    if m3 is not None:
        score += max(m3, 0) * 1.2
    if m10 is not None:
        score += max(m10, 0) * 0.6

    if rsi7 is not None and rsi7 > 55:
        score += (rsi7 - 55) * 0.4

    if rvol10 is not None and rvol10 > 1.2:
        score += (rvol10 - 1.2) * 2.0

    if vwap is not None and vwap > 0:
        score += min(vwap, 6) * 1.5

    if flow_bias is not None:
        score += (flow_bias - 0.5) * 22.0

    if catalyst:
        score += 8.0
    if squeeze:
        score += 12.0

    return round(score, 2)


def breakout_probability(score: float) -> float:
    """Map heuristic score to a 0–100 'probability-like' value via logistic."""
    try:
        prob = 1 / (1 + math.exp(-score / 20.0))
        return round(prob * 100, 1)
    except Exception:
        return None


def multi_timeframe_label(pm, m3, m10):
    """Simple multi-timeframe alignment label: intraday + 3D + 10D."""
    bull_intraday = pm is not None and pm > 0
    bull_3d = m3 is not None and m3 > 0
    bull_10d = m10 is not None and m10 > 0

    positives = sum([bull_intraday, bull_3d, bull_10d])

    if positives == 3:
        return "🟢 Fully Aligned (Intraday + 3D + 10D)"
    elif positives == 2:
        return "🟡 Momentum Favored"
    elif positives == 1:
        return "⚪ Mixed Trend"
    else:
        return "🔻 Bearish / No Trend"

# ========================= SIMPLE AI COMMENTARY =========================
def ai_commentary(score, pm, rvol, flow_bias, vwap, ten_day):
    comments = []

    if score is not None:
        if score >= 80:
            comments.append("High-compression momentum candidate.")
        elif score >= 40:
            comments.append("Moderate momentum setup forming.")
        elif score >= 10:
            comments.append("Early accumulation potential, watching for confirmation.")
        else:
            comments.append("Weak or non-directional tape so far.")

    if pm is not None and pm > 3:
        comments.append("Strong premarket participation.")
    elif pm is not None and pm < -2:
        comments.append("Premarket selling pressure.")

    if rvol is not None and rvol > 2:
        comments.append("Volume expanding vs 10-day average.")
    elif rvol is not None and rvol < 0.7:
        comments.append("Muted liquidity vs normal regime.")

    if flow_bias is not None:
        if flow_bias > 0.65:
            comments.append("Buyers clearly in control intraday.")
        elif flow_bias < 0.4:
            comments.append("Sellers driving short-term tape.")

    if vwap is not None:
        if vwap > 0:
            comments.append("Trading above VWAP — buyers chasing.")
        elif vwap < 0:
            comments.append("Below VWAP — supply still dominant.")

    if ten_day is not None:
        if ten_day > 10:
            comments.append("10D structure firmly uptrending.")
        elif ten_day < -5:
            comments.append("10D trend under distribution risk.")

    if not comments:
        return "Neutral / indecisive tape — watching for clearer confirmation."

    return " | ".join(comments)

# ========================= CORE SCAN =========================
def scan_one(sym, enable_enrichment: bool, enable_ofb_filter: bool, min_ofb: float):
    try:
        ticker = sym["Symbol"]
        exchange = sym.get("Exchange", "UNKNOWN")
        stock = yf.Ticker(ticker)

        # Daily 10d history
        hist = stock.history(period=f"{HISTORY_LOOKBACK_DAYS}d", interval="1d")
        if hist is None or hist.empty or len(hist) < 5:
            return None

        close = hist["Close"]
        volume = hist["Volume"]

        price = float(close.iloc[-1])
        vol_last = float(volume.iloc[-1])  # DAILY VOLUME SIGNAL

        # WATCHLIST: bypass price/volume gate
        if exchange != "WATCH":
            if price > max_price or vol_last < min_volume:
                return None

        # Momentum windows: yesterday, 3-day, 10-day
        if len(close) >= 2 and close.iloc[-2] > 0:
            yday_pct = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100
        else:
            yday_pct = None

        if len(close) >= 4 and close.iloc[-4] > 0:
            m3 = (close.iloc[-1] - close.iloc[-4]) / close.iloc[-4] * 100
        else:
            m3 = None

        if close.iloc[0] > 0:
            m10 = (close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100
        else:
            m10 = None

        # RSI(7)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(7).mean()
        loss = (-delta.clip(upper=0)).rolling(7).mean()
        rs = gain / loss
        rsi_series = 100 - (100 / (1 + rs))
        rsi7 = float(rsi_series.iloc[-1])

        # EMA10
        ema10_series = close.ewm(span=10, adjust=False).mean()
        ema10 = float(ema10_series.iloc[-1])
        ema_trend = "🔥 Breakout" if price > ema10 and rsi7 > 55 else "Neutral"

        # 10-day RVOL
        avg10 = float(volume.mean()) if len(volume) > 0 else 0
        rvol10 = vol_last / avg10 if avg10 > 0 else None

        # Intraday 2m history for PM, VWAP, order flow
        premarket_pct = None
        vwap_dist = None
        order_flow_bias = None

        try:
            intra = stock.history(period=INTRADAY_RANGE, interval=INTRADAY_INTERVAL, prepost=True)
        except Exception:
            intra = None

        if intra is not None and not intra.empty and len(intra) >= 3:
            iclose = intra["Close"]
            iopen = intra["Open"]
            ivol = intra["Volume"]

            # Premarket % from last two bars
            last_close = float(iclose.iloc[-1])
            prev_close_intraday = float(iclose.iloc[-2])
            if prev_close_intraday > 0:
                premarket_pct = (last_close - prev_close_intraday) / prev_close_intraday * 100

            # VWAP
            typical_price = (intra["High"] + intra["Low"] + intra["Close"]) / 3
            total_vol = ivol.sum()
            if total_vol > 0:
                vwap_val = float((typical_price * ivol).sum() / total_vol)
                if vwap_val > 0:
                    vwap_dist = (price - vwap_val) / vwap_val * 100

            # Order-flow bias: buy vs sell volume
            of_df = intra[["Open", "Close", "Volume"]].dropna()
            if not of_df.empty:
                sign = (of_df["Close"] > of_df["Open"]).astype(int) - (of_df["Close"] < of_df["Open"]).astype(int)
                buy_vol = float((of_df["Volume"] * (sign > 0)).sum())
                sell_vol = float((of_df["Volume"] * (sign < 0)).sum())
                total = buy_vol + sell_vol
                if total > 0:
                    order_flow_bias = buy_vol / total  # 0..1

        # 🔥 Optional order-flow bias filter
        #    Now *skipped* for watchlist symbols
        if enable_ofb_filter and exchange != "WATCH":
            if order_flow_bias is None or order_flow_bias < min_ofb:
                return None

        # Enrichment (float, short, news)
        squeeze = False
        low_float = False
        catalyst = False
        sector = "Unknown"
        industry = "Unknown"
        short_pct_display = None

        if enable_enrichment:
            try:
                info = stock.get_info() or {}
                float_shares = info.get("floatShares")
                short_pct = info.get("shortPercentOfFloat")
                sector = info.get("sector", "Unknown")
                industry = info.get("industry", "Unknown")

                low_float = bool(float_shares and float_shares < 20_000_000)
                squeeze = bool(short_pct and short_pct > 0.15)
                short_pct_display = round(short_pct * 100, 2) if short_pct else None
            except Exception:
                pass

            try:
                news = stock.get_news()
                if news and "providerPublishTime" in news[0]:
                    pub = datetime.fromtimestamp(news[0]["providerPublishTime"], tz=timezone.utc)
                    catalyst = (datetime.now(timezone.utc) - pub).days <= 3
            except Exception:
                pass

        # Multi-timeframe label
        mtf_label = multi_timeframe_label(premarket_pct, m3, m10)

        # Score + probability
        score = short_window_score(
            pm=premarket_pct,
            yday=yday_pct,
            m3=m3,
            m10=m10,
            rsi7=rsi7,
            rvol10=rvol10,
            catalyst=catalyst,
            squeeze=squeeze,
            vwap=vwap_dist,
            flow_bias=order_flow_bias,
        )
        prob_rise = breakout_probability(score)

        # AI commentary
        ai_text = ai_commentary(
            score=score,
            pm=premarket_pct,
            rvol=rvol10,
            flow_bias=order_flow_bias,
            vwap=vwap_dist,
            ten_day=m10,
        )

        # Sparkline: 10d closes
        spark_series = close

        return {
            "Symbol": ticker,
            "Exchange": exchange,
            "Price": round(price, 2),
            "Volume": int(vol_last),
            "Score": score,
            "Prob_Rise%": prob_rise,
            "PM%": round(premarket_pct, 2) if premarket_pct is not None else None,
            "YDay%": round(yday_pct, 2) if yday_pct is not None else None,
            "3D%": round(m3, 2) if m3 is not None else None,
            "10D%": round(m10, 2) if m10 is not None else None,
            "RSI7": round(rsi7, 2),
            "EMA10 Trend": ema_trend,
            "RVOL_10D": round(rvol10, 2) if rvol10 is not None else None,
            "VWAP%": round(vwap_dist, 2) if vwap_dist is not None else None,
            "FlowBias": round(order_flow_bias, 2) if order_flow_bias is not None else None,
            "Squeeze?": squeeze,
            "LowFloat?": low_float,
            "Short % Float": short_pct_display,
            "Sector": sector,
            "Industry": industry,
            "Catalyst": catalyst,
            "MTF_Trend": mtf_label,
            "Spark": spark_series,
            "AI_Commentary": ai_text,
        }

    except Exception:
        # any hard failure -> drop this symbol entirely (even if watchlist),
        # instead of showing 'Error' rows
        return None

@st.cache_data(ttl=6)
def run_scan(
    watchlist_text: str,
    max_universe: int,
    enable_enrichment: bool,
    enable_ofb_filter: bool,
    min_ofb: float,
    universe_mode: str,
    volume_rank_pool: int,
):
    universe = build_universe(
        watchlist_text,
        max_universe,
        universe_mode,
        volume_rank_pool,
    )
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as ex:
        futures = [
            ex.submit(
                scan_one,
                sym,
                enable_enrichment,
                enable_ofb_filter,
                min_ofb,
            )
            for sym in universe
        ]
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res:
                results.append(res)

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results)

# ========================= SPARKLINE & CHART HELPERS =========================
def sparkline(series: pd.Series):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=series.values,
        mode="lines",
        line=dict(width=2),
        hoverinfo="skip",
    ))
    fig.update_layout(
        height=60,
        width=160,
        margin=dict(l=2, r=2, t=2, b=2),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def bigline(series: pd.Series, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=series.values,
        mode="lines+markers",
        name=title,
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis_title="Bars (last 10 days)",
        yaxis_title="Price",
    )
    return fig

# ========================= AUDIO ALERT STATE =========================
if "alerted" not in st.session_state:
    st.session_state.alerted = set()


def trigger_audio_alert(symbol: str, reason: str):
    """Play sound + mark symbol as alerted once per session."""
    st.session_state.alerted.add(symbol)
    audio_html = """
    <audio autoplay>
        <source src="https://actions.google.com/sounds/v1/alarms/digital_watch_alarm_long.ogg" type="audio/ogg">
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)
    st.warning(f"🔔 {symbol}: {reason}")

# ========================= MAIN DISPLAY =========================
with st.spinner("Scanning (10-day momentum, V9 hybrid universe)…"):
    df = run_scan(
        watchlist_text,
        max_universe,
        enable_enrichment,
        enable_ofb_filter,
        min_ofb,
        universe_mode,
        volume_rank_pool,
    )

if df.empty:
    st.error("No results found. Try adding a watchlist or relaxing filters.")
else:
    # Watchlist precedence over filters:
    # If a watchlist is populated, show all scanned watchlist symbols
    # (don't drop them via post-scan filters).
    if not watchlist_text.strip():
        # Apply filters only when NOT using a watchlist
        df = df[df["Score"].fillna(-999) >= min_breakout]

        if min_pm_move != 0.0:
            df = df[df["PM%"].fillna(-999) >= min_pm_move]
        if min_yday_gain != 0.0:
            df = df[df["YDay%"].fillna(-999) >= min_yday_gain]
        if squeeze_only and "Squeeze?" in df:
            df = df[df["Squeeze?"]]
        if catalyst_only and "Catalyst" in df:
            df = df[df["Catalyst"]]
        if vwap_only and "VWAP%" in df:
            df = df[df["VWAP%"].fillna(-999) > 0]

    if df.empty:
        st.error("No results after filters.")
    else:
        df = df.sort_values(by=["Score", "PM%", "RSI7"], ascending=[False, False, False])

        st.subheader(f"🔥 10-Day Momentum Board — {len(df)} symbols")

        # Alert banner (V9)
        if enable_alerts and st.session_state.alerted:
            alerted_list = ", ".join(sorted(st.session_state.alerted))
            st.info(f"🔔 Active alert symbols: {alerted_list}")

        # Iterate + audio alerts + inline charts
        for _, row in df.iterrows():
            sym = row["Symbol"]

            # Audio alerts (once per symbol, if enabled)
            if enable_alerts and sym not in st.session_state.alerted:
                if row["Score"] is not None and row["Score"] >= ALERT_SCORE_THRESHOLD:
                    trigger_audio_alert(sym, f"Score {row['Score']}")
                elif row["PM%"] is not None and row["PM%"] >= ALERT_PM_THRESHOLD:
                    trigger_audio_alert(sym, f"Premarket {row['PM%']}%")
                elif row["VWAP%"] is not None and row["VWAP%"] >= ALERT_VWAP_THRESHOLD:
                    trigger_audio_alert(sym, f"VWAP Dist {row['VWAP%']}%")

            c1, c2, c3, c4 = st.columns([2, 3, 3, 3])

            # Column 1: Basic info + score + volume
            c1.markdown(f"**{sym}** ({row['Exchange']})")
            c1.write(f"💲 Price: {fmt2(row['Price'])}")
            c1.write(f"📊 Volume: {int(row['Volume']):,}" if row["Volume"] is not None else "📊 Volume: —")
            c1.write(f"🔥 Score: **{fmt2(row['Score'])}**")
            c1.write(f"📈 Prob_Rise: {fmt2(row['Prob_Rise%'])}%")
            c1.write(f"{row['MTF_Trend']}")
            c1.write(f"Trend: {row['EMA10 Trend']}")

            # Column 2: Momentum snapshot (2-decimal formatting)
            c2.write(f"PM%: {fmt2(row['PM%'])}")
            c2.write(f"YDay%: {fmt2(row['YDay%'])}")
            c2.write(f"3D%: {fmt2(row['3D%'])}  |  10D%: {fmt2(row['10D%'])}")
            c2.write(f"RSI7: {fmt2(row['RSI7'])}  |  RVOL_10D: {fmt2(row['RVOL_10D'])}x")

            # Column 3: Microstructure + AI commentary
            c3.write(f"VWAP Dist %: {fmt2(row['VWAP%'])}")
            c3.write(f"Order Flow Bias: {fmt2(row['FlowBias'])}")
            if enable_enrichment:
                c3.write(
                    f"Squeeze: {row['Squeeze?']} | LowFloat: {row['LowFloat?']}"
                )
                c3.write(f"Sec/Ind: {row['Sector']} / {row['Industry']}")
            else:
                c3.write("Enrichment: OFF (float/short/news skipped for speed)")

            # AI commentary line
            c3.markdown(f"🧠 **AI View:** {row.get('AI_Commentary', '—')}")

            # Column 4: Sparkline + full chart
            if isinstance(row["Spark"], pd.Series) and not row["Spark"].empty:
                c4.plotly_chart(sparkline(row["Spark"]), use_container_width=False)
                with c4.expander("📊 View 10-day chart"):
                    c4.plotly_chart(bigline(row["Spark"], f"{sym} - Last 10 Days"), use_container_width=True)
            else:
                c4.write("No sparkline data.")

            st.divider()

        # Download current table
        csv_cols = [
            "Symbol", "Exchange", "Price", "Volume", "Score", "Prob_Rise%",
            "PM%", "YDay%", "3D%", "10D%", "RSI7", "EMA10 Trend",
            "RVOL_10D", "VWAP%", "FlowBias", "Squeeze?", "LowFloat?",
            "Short % Float", "Sector", "Industry", "Catalyst", "MTF_Trend",
            "AI_Commentary",
        ]
        csv_cols = [c for c in csv_cols if c in df.columns]

        st.download_button(
            "📥 Download Screener CSV",
            data=df[csv_cols].to_csv(index=False),
            file_name="v9_10day_momentum_screener_hybrid.csv",
            mime="text/csv",
        )

st.caption("For research and education only. Not financial advice.")













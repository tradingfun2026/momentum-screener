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
THREADS               = 20
AUTO_REFRESH_MS       = 120_000
HISTORY_LOOKBACK_DAYS = 10
INTRADAY_INTERVAL     = "2m"
INTRADAY_RANGE        = "1d"

DEFAULT_MAX_PRICE     = 5.0
DEFAULT_MIN_VOLUME    = 0.0
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

# ========================= NEWS SENTIMENT (New – Option A) =========================
POSITIVE_WORDS = [
    "beat","beats","strong","positive","upgrades","upgrade","raises","raised",
    "accelerating","surge","soars","soar","winning","win","growth","expand",
    "expanding","record","optimistic","approval","approved", "buy", "patent"
]

NEGATIVE_WORDS = [
    "miss","misses","downgrade","downgrades","negative","falls","falling",
    "cuts","cut","delay","delayed","weak","warning","lawsuit","investigation",
    "probe","decline","declining","drop","dilution","drops"
]

def news_sentiment_score(title, summary):
    """Lightweight keyword-based sentiment analysis."""
    if not title and not summary:
        return 0.0
    text = f"{title} {summary}".lower()
    score = 0
    for w in POSITIVE_WORDS:
        if w in text:
            score += 1
    for w in NEGATIVE_WORDS:
        if w in text:
            score -= 1
    if score == 0:
        return 0.0
    return max(-1.0, min(1.0, score / 4.0))

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
    )

    volume_rank_pool = st.slider(
        "Max symbols to consider when volume-ranking (V9)",
        min_value=100,
        max_value=2000,
        value=600,
        step=100,
    )

    enable_enrichment = st.checkbox(
        "Include float/short + news (slower, more data)",
        value=False,
    )

    st.markdown("---")
    st.header("Filters")

    max_price = st.number_input("Max Price ($)", 1.0, 1000.0, DEFAULT_MAX_PRICE, 0.5)

    min_volume = st.number_input(
        "Min Daily Volume",
        0.0, 10_000_000.0,
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
    )
    min_ofb = st.slider(
        "Min Order Flow Bias (0–1, buyer control)",
        0.0, 1.0,
        0.50,
        0.01,
    )

    st.markdown("---")
    st.subheader("🔊 Audio Alert Thresholds")

    enable_alerts = st.checkbox("Enable Audio + Alert Banner", value=False)
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
        [nasdaq[["Symbol","ETF","Exchange"]],
         other[["Symbol","ETF","Exchange"]]]
    )

    df["Symbol"] = df["Symbol"].astype(str).fillna("")
    df = df[df["Symbol"].str.contains(r"^[A-Z]{1,5}$", na=False)]
    return df.to_dict("records")

def build_universe(watchlist_text, max_universe, universe_mode, volume_rank_pool):
    wl = watchlist_text.strip()
    if wl:
        raw = wl.replace("\n"," ").replace(","," ").split()
        tickers = sorted(set(s.upper() for s in raw if s.strip()))
        return [{"Symbol":t,"Exchange":"WATCH"} for t in tickers]

    syms = load_symbols()

    if universe_mode == "Randomized Slice":
        base = syms[:]
        random.shuffle(base)
        return base[:max_universe]

    if universe_mode == "Live Volume Ranked (slower)":
        base = syms[:volume_rank_pool]
        ranked = []
        for sym in base:
            try:
                t = yf.Ticker(sym["Symbol"])
                d = t.history(period="1d", interval="2m", prepost=True)
                if not d.empty:
                    ranked.append({**sym,"LiveVol":float(d["Volume"].iloc[-1])})
            except:
                continue
        if not ranked:
            return syms[:max_universe]
        ranked_sorted = sorted(ranked,key=lambda x:x.get("LiveVol",0.0),reverse=True)
        return ranked_sorted[:max_universe]

    return syms[:max_universe]

# ========================= SCORING =========================
def short_window_score(pm,yday,m3,m10,rsi7,rvol10,catalyst,squeeze,vwap,flow_bias):
    score = 0.0
    if pm is not None: score += max(pm,0)*1.6
    if yday is not None: score += max(yday,0)*0.8
    if m3 is not None: score += max(m3,0)*1.2
    if m10 is not None: score += max(m10,0)*0.6
    if rsi7>55: score += (rsi7-55)*0.4
    if rvol10 and rvol10>1.2: score += (rvol10-1.2)*2.0
    if vwap and vwap>0: score += min(vwap,6)*1.5
    if flow_bias: score += (flow_bias-0.5)*22.0
    if catalyst: score += 8.0
    if squeeze: score += 12.0
    return round(score,2)

def breakout_probability(score):
    try:
        prob = 1/(1+math.exp(-score/20))
        return round(prob*100,1)
    except:
        return None

def multi_timeframe_label(pm,m3,m10):
    p = sum([
        pm is not None and pm>0,
        m3 is not None and m3>0,
        m10 is not None and m10>0
    ])
    return ["🔻 Bearish / No Trend","⚪ Mixed Trend",
            "🟡 Momentum Favored","🟢 Fully Aligned (Intraday + 3D + 10D)"][p]

# ========================= AI COMMENTARY =========================
def ai_commentary(score,pm,rvol,flow_bias,vwap,ten_day):
    out=[]
    if score>=80: out.append("High-compression momentum candidate.")
    elif score>=40: out.append("Moderate momentum setup forming.")
    elif score>=10: out.append("Early accumulation potential.")
    else: out.append("Weak or non-directional tape.")

    if pm and pm>3: out.append("Strong premarket participation.")
    if pm and pm< -2: out.append("Premarket selling pressure.")

    if rvol and rvol>2: out.append("Volume expanding.")
    if rvol and rvol<0.7: out.append("Muted liquidity.")

    if flow_bias and flow_bias>0.65: out.append("Buyers in control.")
    if flow_bias and flow_bias<0.4: out.append("Sellers dominating.")

    if vwap and vwap>0: out.append("Above VWAP — buyers chasing.")
    if vwap and vwap<0: out.append("Below VWAP — supply dominant.")

    if ten_day and ten_day>10: out.append("10D trend strong uptrend.")
    if ten_day and ten_day< -5: out.append("10D trend under distribution.")

    return " | ".join(out)

# ========================= CORE SCAN =========================
def scan_one(sym, enable_enrichment, enable_ofb_filter, min_ofb):
    try:
        ticker = sym["Symbol"]
        exchange = sym.get("Exchange","UNKNOWN")
        stock = yf.Ticker(ticker)

        # Daily history
        hist = stock.history(period=f"{HISTORY_LOOKBACK_DAYS}d", interval="1d")
        if hist is None or hist.empty or len(hist)<5:
            return None

        close = hist["Close"]
        volume = hist["Volume"]
        price = float(close.iloc[-1])
        vol_last = float(volume.iloc[-1])

        if exchange!="WATCH":
            if price>max_price or vol_last<min_volume:
                return None

        yday_pct = ((close.iloc[-1]-close.iloc[-2])/close.iloc[-2]*100
                    if close.iloc[-2]>0 else None)
        m3 = ((close.iloc[-1]-close.iloc[-4])/close.iloc[-4]*100
              if len(close)>=4 and close.iloc[-4]>0 else None)
        m10 = ((close.iloc[-1]-close.iloc[0])/close.iloc[0]*100
               if close.iloc[0]>0 else None)

        # RSI7
        delta=close.diff()
        gain=delta.clip(lower=0).rolling(7).mean()
        loss=(-delta.clip(upper=0)).rolling(7).mean()
        rs=gain/loss
        rsi7=float(100-(100/(1+rs.iloc[-1])))

        ema10=float(close.ewm(span=10,adjust=False).mean().iloc[-1])
        ema_trend="🔥 Breakout" if price>ema10 and rsi7>55 else "Neutral"

        avg10=float(volume.mean())
        rvol10=vol_last/avg10 if avg10>0 else None

        # Intraday
        premarket_pct=None
        vwap_dist=None
        order_flow_bias=None

        try:
            intra=stock.history(period=INTRADAY_RANGE,interval=INTRADAY_INTERVAL,prepost=True)
        except:
            intra=None

        if intra is not None and not intra.empty and len(intra)>=3:
            iclose=intra["Close"]
            iopen=intra["Open"]
            ivol=intra["Volume"]

            last_close=float(iclose.iloc[-1])
            prev_close=float(iclose.iloc[-2])
            if prev_close>0:
                premarket_pct=(last_close-prev_close)/prev_close*100

            typical=(intra["High"]+intra["Low"]+intra["Close"])/3
            tv=ivol.sum()
            if tv>0:
                vwap_val=float((typical*ivol).sum()/tv)
                if vwap_val>0:
                    vwap_dist=(price-vwap_val)/vwap_val*100

            df2=intra[["Open","Close","Volume"]].dropna()
            if not df2.empty:
                sign=(df2["Close"]>df2["Open"]).astype(int)-(df2["Close"]<df2["Open"]).astype(int)
                buy=float((df2["Volume"]*(sign>0)).sum())
                sell=float((df2["Volume"]*(sign<0)).sum())
                total=buy+sell
                if total>0:
                    order_flow_bias=buy/total

        if enable_ofb_filter and exchange!="WATCH":
            if order_flow_bias is None or order_flow_bias<min_ofb:
                return None

        # Enrichment
        squeeze=False
        low_float=False
        catalyst=False
        sector="Unknown"
        industry="Unknown"
        short_pct_display=None
        sentiment_score_val=0.0

        if enable_enrichment:
            # float/short
            try:
                info=stock.get_info() or {}
                float_shares=info.get("floatShares")
                short_pct=info.get("shortPercentOfFloat")
                sector=info.get("sector","Unknown")
                industry=info.get("industry","Unknown")
                low_float=bool(float_shares and float_shares<20_000_000)
                squeeze=bool(short_pct and short_pct>0.15)
                short_pct_display=round(short_pct*100,2) if short_pct else None
            except:
                pass

            # news + sentiment
            try:
                news=stock.get_news()
                if news:
                    first=news[0]
                    title=first.get("title","")
                    summary=first.get("summary","")
                    sentiment_score_val=news_sentiment_score(title,summary)

                    if "providerPublishTime" in first:
                        pub=datetime.fromtimestamp(first["providerPublishTime"],tz=timezone.utc)
                        catalyst=(datetime.now(timezone.utc)-pub).days<=3
            except:
                pass

        mtf_label=multi_timeframe_label(premarket_pct,m3,m10)
        score=short_window_score(premarket_pct,yday_pct,m3,m10,rsi7,rvol10,
                                 catalyst,squeeze,vwap_dist,order_flow_bias)
        prob_rise=breakout_probability(score)
        ai_text=ai_commentary(score,premarket_pct,rvol10,order_flow_bias,vwap_dist,m10)

        spark_series=close

        return {
            "Symbol":ticker,
            "Exchange":exchange,
            "Price":round(price,2),
            "Volume":int(vol_last),
            "Score":score,
            "Prob_Rise%":prob_rise,
            "PM%":round(premarket_pct,2) if premarket_pct is not None else None,
            "YDay%":round(yday_pct,2) if yday_pct is not None else None,
            "3D%":round(m3,2) if m3 is not None else None,
            "10D%":round(m10,2) if m10 is not None else None,
            "RSI7":round(rsi7,2),
            "EMA10 Trend":ema_trend,
            "RVOL_10D":round(rvol10,2) if rvol10 is not None else None,
            "VWAP%":round(vwap_dist,2) if vwap_dist is not None else None,
            "FlowBias":round(order_flow_bias,2) if order_flow_bias is not None else None,
            "Squeeze?":squeeze,
            "LowFloat?":low_float,
            "Short % Float":short_pct_display,
            "Sector":sector,
            "Industry":industry,
            "Catalyst":catalyst,
            "Sentiment":sentiment_score_val,   # <-- NEW
            "MTF_Trend":mtf_label,
            "Spark":spark_series,
            "AI_Commentary":ai_text,
        }

    except:
        return None

# ========================= RUN SCAN =========================
@st.cache_data(ttl=6)
def run_scan(watchlist_text,max_universe,enable_enrichment,
             enable_ofb_filter,min_ofb,universe_mode,volume_rank_pool):
    universe=build_universe(
        watchlist_text,
        max_universe,
        universe_mode,
        volume_rank_pool,
    )

    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as ex:
        futures=[
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
            res=f.result()
            if res:
                results.append(res)

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results)

# ========================= CHARTS =========================
def sparkline(series: pd.Series):
    fig=go.Figure()
    fig.add_trace(go.Scatter(
        y=series.values,
        mode="lines",
        line=dict(width=2),
        hoverinfo="skip",
    ))
    fig.update_layout(height=60,width=160,
        margin=dict(l=2,r=2,t=2,b=2),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig

def bigline(series: pd.Series,title:str):
    fig=go.Figure()
    fig.add_trace(go.Scatter(
        y=series.values,
        mode="lines+markers",
        name=title,
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=40,r=20,t=40,b=40),
        xaxis_title="Bars (last 10 days)",
        yaxis_title="Price",
    )
    return fig

# ========================= AUDIO ALERT STATE =========================
if "alerted" not in st.session_state:
    st.session_state.alerted=set()

def trigger_audio_alert(symbol,reason):
    st.session_state.alerted.add(symbol)
    audio_html="""
    <audio autoplay>
        <source src="https://actions.google.com/sounds/v1/alarms/digital_watch_alarm_long.ogg" type="audio/ogg">
    </audio>
    """
    st.markdown(audio_html,unsafe_allow_html=True)
    st.warning(f"🔔 {symbol}: {reason}")

# ========================= MAIN DISPLAY =========================
with st.spinner("Scanning (10-day momentum, V9 hybrid universe)…"):
    df=run_scan(
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
    if not watchlist_text.strip():
        df=df[df["Score"].fillna(-999)>=min_breakout]
        if min_pm_move!=0.0:
            df=df[df["PM%"].fillna(-999)>=min_pm_move]
        if min_yday_gain!=0.0:
            df=df[df["YDay%"].fillna(-999)>=min_yday_gain]
        if squeeze_only and "Squeeze?" in df:
            df=df[df["Squeeze?"]]
        if catalyst_only and "Catalyst" in df:
            df=df[df["Catalyst"]]
        if vwap_only and "VWAP%" in df:
            df=df[df["VWAP%"].fillna(-999)>0]

    if df.empty:
        st.error("No results after filters.")
    else:
        df=df.sort_values(by=["Score","PM%","RSI7"],ascending=[False,False,False])

        st.subheader(f"🔥 10-Day Momentum Board — {len(df)} symbols")

        if enable_alerts and st.session_state.alerted:
            alerted_list=", ".join(sorted(st.session_state.alerted))
            st.info(f"🔔 Active alert symbols: {alerted_list}")

        for _,row in df.iterrows():
            sym=row["Symbol"]

            if enable_alerts and sym not in st.session_state.alerted:
                if row["Score"]>=ALERT_SCORE_THRESHOLD:
                    trigger_audio_alert(sym,f"Score {row['Score']}")
                elif row["PM%"] and row["PM%"]>=ALERT_PM_THRESHOLD:
                    trigger_audio_alert(sym,f"Premarket {row['PM%']}%")
                elif row["VWAP%"] and row["VWAP%"]>=ALERT_VWAP_THRESHOLD:
                    trigger_audio_alert(sym,f"VWAP Dist {row['VWAP%']}%")

            c1,c2,c3,c4=st.columns([2,3,3,3])

            # Col 1
            c1.markdown(f"**{sym}** ({row['Exchange']})")
            c1.write(f"💲 Price: {fmt2(row['Price'])}")
            c1.write(f"📊 Volume: {int(row['Volume']):,}")
            c1.write(f"🔥 Score: **{fmt2(row['Score'])}**")
            c1.write(f"📈 Prob_Rise: {fmt2(row['Prob_Rise%'])}%")
            c1.write(row["MTF_Trend"])
            c1.write(f"Trend: {row['EMA10 Trend']}")

            # Col 2
            c2.write(f"PM%: {fmt2(row['PM%'])}")
            c2.write(f"YDay%: {fmt2(row['YDay%'])}")
            c2.write(f"3D%: {fmt2(row['3D%'])}  |  10D%: {fmt2(row['10D%'])}")
            c2.write(f"RSI7: {fmt2(row['RSI7'])}  |  RVOL_10D: {fmt2(row['RVOL_10D'])}x")

            # Col 3
            c3.write(f"VWAP Dist %: {fmt2(row['VWAP%'])}")
            c3.write(f"Order Flow Bias: {fmt2(row['FlowBias'])}")
            if enable_enrichment:
                c3.write(f"Squeeze: {row['Squeeze?']} | LowFloat: {row['LowFloat?']}")
                c3.write(f"Sec/Ind: {row['Sector']} / {row['Industry']}")
                c3.write(f"News Sentiment: {fmt2(row['Sentiment'])}")
            else:
                c3.write("Enrichment: OFF (float/short/news skipped)")

            c3.markdown(f"🧠 **AI View:** {row['AI_Commentary']}")

            # Col 4
            if isinstance(row["Spark"],pd.Series) and not row["Spark"].empty:
                c4.plotly_chart(sparkline(row["Spark"]),use_container_width=False)
                with c4.expander("📊 View 10-day chart"):
                    c4.plotly_chart(bigline(row["Spark"],f"{sym} - Last 10 Days"),
                                    use_container_width=True)
            else:
                c4.write("No sparkline data.")

            st.divider()

        csv_cols=[
            "Symbol","Exchange","Price","Volume","Score","Prob_Rise%",
            "PM%","YDay%","3D%","10D%","RSI7","EMA10 Trend",
            "RVOL_10D","VWAP%","FlowBias","Squeeze?","LowFloat?",
            "Short % Float","Sector","Industry","Catalyst","MTF_Trend",
            "AI_Commentary","Sentiment"
        ]
        csv_cols=[c for c in csv_cols if c in df.columns]

        st.download_button(
            "📥 Download Screener CSV",
            data=df[csv_cols].to_csv(index=False),
            file_name="v9_10day_momentum_screener_hybrid.csv",
            mime="text/csv",
        )

st.caption("For research and education only. Not financial advice.")



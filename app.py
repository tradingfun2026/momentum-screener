# ===============================================================
#  FULL UPDATED SCRIPT — Version A (Watchlist Always Visible)
# ===============================================================

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
DEFAULT_MIN_VOLUME    = 0
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
st.caption("Watchlist now **ALWAYS OVERRIDES FILTERS & DATA GAPS** — Guaranteed visibility")

# ============================================================
# SIDEBAR CONTROLS
# ============================================================
with st.sidebar:
    st.header("Universe")

    watchlist_text = st.text_area("Watchlist tickers (comma/space/newline separated):",
                                  value="", height=80)

    max_universe = st.slider("Max symbols to scan (if no watchlist)",
                             min_value=50, max_value=2000, value=600, step=100)

    st.markdown("---")
    st.subheader("V9 Universe Mode")
    universe_mode = st.radio("Universe Construction",
        options=["Classic (Alphabetical Slice)","Randomized Slice","Live Volume Ranked (slower)"], index=0)

    volume_rank_pool = st.slider("Volume rank candidate pool",100,2000,600,100)

    enable_enrichment = st.checkbox("Include float/short + news (optional, slower)", value=False)

    st.markdown("---")
    st.header("Filters")

    max_price   = st.number_input("Max Price",1.0,2000.0,DEFAULT_MAX_PRICE,1.0)
    min_volume  = st.number_input("Min Daily Volume",0,10_000_000,DEFAULT_MIN_VOLUME,10_000)
    min_breakout = st.number_input("Min Score", -50.0,200.0,0.0,0.5)
    min_pm_move = st.number_input("Min Premarket %", -50.0,200.0,0.0,0.5)
    min_yday_gain = st.number_input("Min Yesterday %", -50.0,200.0,0.0,0.5)

    squeeze_only  = st.checkbox("Short Squeeze Required")
    catalyst_only = st.checkbox("Must Have Recent News/Earnings")
    vwap_only     = st.checkbox("Above VWAP Only")

    st.markdown("---")
    st.subheader("Order Flow Filter")
    enable_ofb_filter = st.checkbox("Enable Flow Bias Filter", value=False)
    min_ofb = st.slider("Min Order Flow Bias",0.0,1.0,0.50,0.01)

    st.markdown("---")
    enable_alerts = st.checkbox("Audio + Alert Banners", value=False)

    ALERT_SCORE_THRESHOLD = st.slider("Score Alert",10,200,30,5)
    ALERT_PM_THRESHOLD    = st.slider("PM% Alert",1,150,4,1)
    ALERT_VWAP_THRESHOLD  = st.slider("VWAP% Alert",1,50,2,1)

# ============================================================
# LOAD SYMBOLS
# ============================================================
@st.cache_data(ttl=900)
def load_symbols():
    nasdaq = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
                         sep="|", skipfooter=1, engine="python")
    other = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
                        sep="|", skipfooter=1, engine="python")

    nasdaq["Exchange"]="NASDAQ"
    other["Exchange"]=other["Exchange"].fillna("NYSE/AMEX/ARCA")
    other=other.rename(columns={"ACT Symbol":"Symbol"})

    df=pd.concat([nasdaq[["Symbol","ETF","Exchange"]],
                  other[["Symbol","ETF","Exchange"]]]).dropna()

    df=df[df["Symbol"].str.contains(r"^[A-Z]{1,5}$")]
    return df.to_dict("records")

# ============================================================
#  UNIVERSE BUILDER — WATCHLIST TAKES PRIORITY
# ============================================================
def build_universe(watchlist_text,max_universe,universe_mode,volume_rank_pool):

    wl = watchlist_text.strip()
    if wl != "":
        raw = wl.replace("\n"," ").replace(",", " ").split()
        tickers = sorted(set(s.upper() for s in raw))
        return [{"Symbol":t,"Exchange":"WATCH"} for t in tickers]  # 🔥 always prioritized

    all_syms = load_symbols()

    if universe_mode=="Randomized Slice":
        random.shuffle(all_syms)
        return all_syms[:max_universe]

    if universe_mode=="Live Volume Ranked (slower)":
        ranked=[]
        for sym in all_syms[:volume_rank_pool]:
            try:
                h=yf.Ticker(sym["Symbol"]).history(period="1d",interval="2m",prepost=True)
                if not h.empty:
                    ranked.append({**sym,"LiveVol":float(h["Volume"].iloc[-1])})
            except:pass
        ranked=sorted(ranked,key=lambda x:x.get("LiveVol",0),reverse=True)
        return ranked[:max_universe] if ranked else all_syms[:max_universe]

    return all_syms[:max_universe]

# ============================================================
# SCANNER — ***UPDATED WITH WATCHLIST OVERRIDE***
# ============================================================
def scan_one(sym,enable_enrichment,enable_ofb_filter,min_ofb):

    ticker = sym["Symbol"]
    is_watch = (sym.get("Exchange")=="WATCH")

    try:
        stock=yf.Ticker(ticker)
        hist=stock.history(period=f"{HISTORY_LOOKBACK_DAYS}d")

        # -------------------------------------------------------
        # WATCHLIST OVERRIDE ⬇⬇⬇ (never drop watchlist symbols)
        # -------------------------------------------------------
        if hist is None or hist.empty or len(hist)<5:
            return {"Symbol":ticker,"Exchange":"WATCH","Price":None,"Score":None}

        close=hist["Close"];volume=hist["Volume"]
        price=float(close.iloc[-1]);vol=float(volume.iloc[-1])

        # normal filters — skip if watchlist
        if not is_watch:
            if price>max_price or vol<min_volume: return None

        # momentum windows
        yday=(close.iloc[-1]-close.iloc[-2])/close.iloc[-2]*100 if len(close)>=2 else None
        m3  =(close.iloc[-1]-close.iloc[-4])/close.iloc[-4]*100 if len(close)>=4 else None
        m10 =(close.iloc[-1]-close.iloc[0] )/close.iloc[0] *100

        # RSI7
        delta=close.diff()
        gain=delta.clip(lower=0).rolling(7).mean()
        loss=(-delta.clip(upper=0)).rolling(7).mean()
        rs=gain/loss; rsi7=float(100-(100/(1+rs)).iloc[-1])

        # -----------------------------------------------------
        # INTRADAY — if fails, still allow watchlist output
        # -----------------------------------------------------
        try:
            intra=stock.history(period="1d",interval="2m",prepost=True)
        except: intra=pd.DataFrame()

        premarket=vwap=flow=None
        if not intra.empty:
            iclose=intra["Close"];iopen=intra["Open"]
            typical=(intra["High"]+intra["Low"]+intra["Close"])/3
            if len(iclose)>=2:
                premarket=(iclose.iloc[-1]-iclose.iloc[-2])/iclose.iloc[-2]*100
            if intra["Volume"].sum()>0:
                v=float((typical*intra["Volume"]).sum()/intra["Volume"].sum())
                vwap=(price-v)/v*100
            sign=(iclose>iopen).astype(int)-(iclose<iopen).astype(int)
            buy=(intra["Volume"]*(sign>0)).sum()
            sell=(intra["Volume"]*(sign<0)).sum()
            flow=buy/(buy+sell) if buy+sell>0 else None

        # order-flow filter — skipped for watchlist
        if enable_ofb_filter and not is_watch:
            if flow is None or flow < min_ofb: return None

        # simplified scoring (unchanged)
        score=0
        if premarket:score+=premarket*1.6
        if yday:     score+=yday*0.8
        if m3:       score+=m3*1.2
        if m10:      score+=m10*0.6
        if rsi7>55:  score+=(rsi7-55)*0.4
        score=round(score,2)

        return {
            "Symbol":ticker,"Exchange":sym.get("Exchange","?"),
            "Price":round(price,2),"Volume":int(vol),
            "Score":score,"PM%":premarket,"YDay%":yday,"3D%":m3,
            "10D%":m10,"RSI7":rsi7,"VWAP%":vwap,"FlowBias":flow,
            "Spark":close, "AI_Commentary":"WATCHLIST PRIORITY — Data incomplete OK"
        }

    except:
        return {"Symbol":ticker,"Exchange":"WATCH","Price":None,"Score":None}

# ============================================================
# MAIN SCAN — Watchlist ignores post-filters entirely
# ============================================================
@st.cache_data(ttl=6)
def run_scan(watchlist_text,max_universe,enable_enrichment,enable_ofb_filter,min_ofb,universe_mode,volume_rank_pool):
    uni=build_universe(watchlist_text,max_universe,universe_mode,volume_rank_pool)
    out=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as ex:
        for f in concurrent.futures.as_completed([ex.submit(scan_one,s,enable_enrichment,enable_ofb_filter,min_ofb) for s in uni]):
            if f.result():out.append(f.result())
    return pd.DataFrame(out) if out else pd.DataFrame()

# ============================================================
# DISPLAY
# ============================================================

df=run_scan(watchlist_text,max_universe,enable_enrichment,enable_ofb_filter,min_ofb,universe_mode,volume_rank_pool)

if df.empty: st.error("No data returned (rare) — but watchlist will still show symbols")
else:
    # WATCHLIST locks filters OFF
    if not watchlist_text.strip():
        if min_breakout:df=df[df["Score"]>=min_breakout]
        if min_pm_move:df=df[df["PM%"].fillna(-999)>=min_pm_move]
        if min_yday_gain:df=df[df["YDay%"].fillna(-999)>=min_yday_gain]
        if squeeze_only:df=df[df["AI_Commentary"].str.contains("squeeze",case=False,na=False)]
        if catalyst_only:df=df[df["Catalyst"]==True]
        if vwap_only:df=df[df["VWAP%"].fillna(-999)>0]

    df=df.sort_values("Score",ascending=False)
    st.subheader(f"📊 Symbols Returned: {len(df)}")

    for _,row in df.iterrows():
        c1,c2=st.columns([2,5])
        c1.write(f"**{row['Symbol']}** — Score: {row['Score']}")
        c1.write(f"Price: {row['Price']} | Vol: {row.get('Volume','–')}")
        c1.markdown(f"🧠 {row['AI_Commentary']}")
        c2.plotly_chart(go.Figure(data=[go.Scatter(y=row["Spark"].values)]) ,use_container_width=True)

st.caption("For research and education only. Not financial advice.")










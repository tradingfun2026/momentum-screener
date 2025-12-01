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
    page_title="V9 – 10-Day Momentum Screener",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🚀 V9 — 10-Day Momentum Breakout Screener")
st.caption("Watchlist override restored — no filters applied when watchlist is present")

# ========================= FORMATTER =========================
def fmt2(x):
    try:
        if x is None or (isinstance(x,float) and math.isnan(x)):
            return "—"
        return f"{float(x):.2f}"
    except:
        return "—"

# ========================= SIDEBAR =========================
with st.sidebar:

    watchlist_text = st.text_area(
        "Watchlist tickers:",
        value="",
        height=80,
        help="Watchlist now bypasses *every filter* fully"
    )

    max_universe = st.slider("Max symbols (no watchlist)",50,2000,2000,50)

    st.markdown("---")
    universe_mode = st.radio("Universe",
        ["Classic (Alphabetical Slice)","Randomized Slice","Live Volume Ranked (slower)"]
    )
    volume_rank_pool = st.slider("Volume Ranking Pool",100,2000,600,100)

    enable_enrichment = st.checkbox("Enable float/short/news enrichment",value=False)

    st.markdown("---")
    st.header("Filters (ignored when watchlist present)")

    max_price = st.number_input("Max Price",1.0,1000.0,DEFAULT_MAX_PRICE,0.5)
    min_volume = st.number_input("Min Volume",0.0,10_000_000.0,DEFAULT_MIN_VOLUME,10_000.0)

    min_breakout = st.number_input("Min Score",-50.0,200.0,0.0,0.5)
    min_pm_move  = st.number_input("Min Premarket %",-50.0,200.0,0.0,0.5)
    min_yday_gain= st.number_input("Min Yesterday %",-50.0,200.0,0.0,0.5)

    squeeze_only = st.checkbox("Short Squeeze Only")
    catalyst_only= st.checkbox("Must Have Catalyst")
    vwap_only    = st.checkbox("Above VWAP only")

    st.markdown("---")
    enable_ofb_filter=st.checkbox("Order Flow Bias Filter",value=False)
    min_ofb = st.slider("Min FlowBias",0.0,1.0,0.50,0.01)

# ========================= LOAD SYMBOLS =========================
@st.cache_data(ttl=900)
def load_symbols():
    nasdaq=pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
                       sep="|",skipfooter=1,engine="python")
    other =pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
                       sep="|",skipfooter=1,engine="python")

    nasdaq["Exchange"]="NASDAQ"
    other=other.rename(columns={"ACT Symbol":"Symbol"})
    df=pd.concat([nasdaq[["Symbol","ETF","Exchange"]],
                  other[["Symbol","ETF","Exchange"]]])
    df["Symbol"]=df["Symbol"].astype(str).fillna("")
    df=df[df["Symbol"].str.contains(r"^[A-Z]{1,5}$",na=False)]
    return df.to_dict("records")

# ========================= BUILD UNIVERSE =========================
def build_universe(watch,maxu,mode,pool):
    wl=watch.strip()
    if wl:
        tickers=set(wl.replace("\n"," ").replace(","," ").split())
        return [{"Symbol":t.upper(),"Exchange":"WATCH"} for t in tickers]

    syms=load_symbols()

    if mode=="Randomized Slice":
        random.shuffle(syms)
        return syms[:maxu]

    if mode=="Live Volume Ranked (slower)":
        ranked=[]
        for s in syms[:pool]:
            try:
                d=yf.Ticker(s["Symbol"]).history(period="1d",interval="2m",prepost=True)
                if not d.empty:
                    ranked.append({**s,"LiveVol":d["Volume"].iloc[-1]})
            except: pass
        return sorted(ranked,key=lambda x:x.get("LiveVol",0),reverse=True)[:maxu] if ranked else syms[:maxu]

    return syms[:maxu]

# ========================= SCAN ONE =========================
def scan_one(sym, enrich, ofb_filter, min_ofb):
    try:
        ticker=sym["Symbol"]
        ex=sym.get("Exchange")
        stock=yf.Ticker(ticker)

        hist=stock.history(period="10d",interval="1d")
        if hist.empty or len(hist)<5: return None

        close=hist["Close"];vol=hist["Volume"]
        price=close.iloc[-1];v=vol.iloc[-1]

        # 🔥 WATCHLIST OVERRIDE — only skip filters outside this fn
        if ex!="WATCH":
            if price>max_price or v<min_volume: return None

        # compute signals
        yday =(close.iloc[-1]-close.iloc[-2])/close.iloc[-2]*100
        m3   =(close.iloc[-1]-close.iloc[-4])/close.iloc[-4]*100 if len(close)>=4 else None
        m10  =(close.iloc[-1]-close.iloc[0])/close.iloc[0]*100
        rsi7 =float((100-(100/(1+close.diff().clip(lower=0).rolling(7).mean()/
                              (-close.diff().clip(upper=0)).rolling(7).mean())))
                              .iloc[-1])
        ema10=float(close.ewm(span=10).mean().iloc[-1])
        emaTrend="🔥 Breakout" if price>ema10 and rsi7>55 else "Neutral"
        rvol10=v/vol.mean()

        # Intraday
        pm=vwap=flow=None
        intra=stock.history(period="1d",interval="2m",prepost=True)
        if len(intra)>=3:
            ic=intra["Close"];io=intra["Open"]
            pm=(ic.iloc[-1]-ic.iloc[-2])/ic.iloc[-2]*100
            typical=(intra["High"]+intra["Low"]+intra["Close"])/3
            tot=intra["Volume"].sum()
            if tot>0:vwap=((typical*intra["Volume"]).sum()/tot-price)/price*100
            sign=(ic>io).astype(int)-(ic<io).astype(int)
            buys =(intra["Volume"]*(sign>0)).sum()
            sells=(intra["Volume"]*(sign<0)).sum()
            if buys+sells>0: flow=buys/(buys+sells)

        # 🔥 OrderFlow filter DOES NOT block watchlist now
        if ex!="WATCH" and ofb_filter and (flow is None or flow<min_ofb):
            return None

        # score + trend alignment
        score=(max(pm or 0,0)*1.6 + max(yday or 0,0)*0.8 +
               max(m3 or 0,0)*1.2 + max(m10 or 0,0)*0.6 +
               max(rsi7-55,0)*0.4 + (rvol10-1.2)*2 if rvol10>1.2 else 0)
        prob=round(100/(1+math.exp(-score/20)),1)
        trend=("🟢 Fully Aligned" if (pm>0 and m3>0 and m10>0) else
               "🟡 Momentum Favored" if ((pm>0)+(m3>0)+(m10>0))==2 else
               "⚪ Mixed" if ((pm>0)+(m3>0)+(m10>0))==1 else "🔻 Bearish")

        return {"Symbol":ticker,"Exchange":ex,
                "Price":price,"Volume":v,
                "Score":round(score,2),"Prob_Rise%":prob,
                "PM%":pm,"YDay%":yday,"3D%":m3,"10D%":m10,
                "RSI7":rsi7,"RVOL_10D":rvol10,"VWAP%":vwap,"FlowBias":flow,
                "EMA10 Trend":emaTrend,"MTF_Trend":trend,
                "Spark":close,
                "Sector":"—","Industry":"—",
                "AI_Commentary":" | ".join([t for t in [
                    "High momentum" if score>=80 else None,
                    "Moderate trend" if score>=40 else None,
                    "Early rotation" if score>=10 else None
                ] if t])
               }
    except:return None

# ========================= RUN SCAN =========================
@st.cache_data(ttl=6)
def run_scan(w,maxu,enrich,ofb,minofb,mode,pool):
    uni=build_universe(w,maxu,mode,pool);out=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as ex:
        for f in concurrent.futures.as_completed(
                [ex.submit(scan_one,s,enrich,ofb,minofb) for s in uni]):
            r=f.result()
            if r:out.append(r)
    return pd.DataFrame(out)

df=run_scan(watchlist_text,max_universe,enable_enrichment,
            enable_ofb_filter,min_ofb,universe_mode,volume_rank_pool)

# ========================= WATCHLIST FIX APPLIED HERE 🔥 =========================
using_watchlist = bool(watchlist_text.strip())

# Only filter when watchlist is *NOT* used
if not using_watchlist and not df.empty:
    df=df[df["Score"].fillna(-999)>=min_breakout]
    df=df[df["PM%"].fillna(-999)>=min_pm_move]
    df=df[df["YDay%"].fillna(-999)>=min_yday_gain]
    if squeeze_only and "Squeeze?" in df: df=df[df["Squeeze?"]]
    if catalyst_only and "Catalyst" in df: df=df[df["Catalyst"]]
    if vwap_only and "VWAP%" in df: df=df[df["VWAP%"].fillna(-999)>0]

# ========================= DISPLAY =========================
if df.empty:
    st.error("No results found")
else:
    df=df.sort_values(by=["Score","PM%","RSI7"],ascending=[False,False,False])
    st.subheader(f"Results {len(df)} — Watchlist Mode: {using_watchlist}")

    for _,row in df.iterrows():
        c1,c2,c3,c4=st.columns([2,3,3,3])
        sym=row["Symbol"]

        c1.markdown(f"### {sym} ({row['Exchange']})")
        c1.write(f"Price: {fmt2(row['Price'])}")
        c1.write(f"Volume: {int(row['Volume']):,}")
        c1.write(f"Score: **{fmt2(row['Score'])}** | Rise {fmt2(row['Prob_Rise%'])}%")
        c1.write(f"{row['MTF_Trend']} | {row['EMA10 Trend']}")

        c2.write(f"PM {fmt2(row['PM%'])}% | YDay {fmt2(row['YDay%'])}%")
        c2.write(f"3D {fmt2(row['3D%'])}% | 10D {fmt2(row['10D%'])}%")
        c2.write(f"RSI7 {fmt2(row['RSI7'])} | RVOL {fmt2(row['RVOL_10D'])}x")

        c3.write(f"VWAP {fmt2(row['VWAP%'])}% | FlowBias {fmt2(row['FlowBias'])}")
        c3.write(row["AI_Commentary"] or "No commentary")

        c4.plotly_chart(go.Figure(go.Scatter(y=row["Spark"].values)),use_container_width=True)
        st.divider()


st.caption("For research and education only. Not financial advice.")










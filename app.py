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
DEFAULT_MIN_VOLUME    = 0          # 🔥 Changed to 0
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

# ========================= SIDEBAR CONTROLS =========================
with st.sidebar:
    st.header("Universe")

    watchlist_text = st.text_area(
        "Watchlist tickers (comma/space/newline separated):",
        value="",
        height=80,
        help="Example: AAPL, TSLA, NVDA, AMD",
    )

    # 🔥 New Toggle — No UI removed
    ignore_filters_watchlist = st.checkbox(
        "Ignore All Filters When Watchlist Is Used",
        value=True,
        help="If enabled, watchlist symbols bypass price/volume/momentum filters."
    )

    max_universe = st.slider(
        "Max symbols to scan when no watchlist",
        min_value=50,
        max_value=600,
        value=2000,
        step=50,
        help="Keeps scans fast when you don't use a custom watchlist.",
    )

    st.markdown("---")
    st.subheader("V9 Universe Mode")
    universe_mode = st.radio(
        "Universe Construction",
        options=[
            "Classic (Alphabetical Slice)",
            "Randomized Slice",
            "Live Volume Ranked (slower)",
        ],
        index=0
    )

    volume_rank_pool = st.slider("Max symbols for Volume Rank (V9)",100,2000,600,100)
    enable_enrichment = st.checkbox("Include float/short + news (slower)",False)

    st.markdown("---")
    st.header("Filters")

    max_price = st.number_input("Max Price ($)",1.0,1000.0,DEFAULT_MAX_PRICE,1.0)
    min_volume = st.number_input("Min Daily Volume",0,10_000_000,DEFAULT_MIN_VOLUME,10_000)  # 🔥 changed to 0 baseline
    min_breakout = st.number_input("Min Breakout Score",-50.0,200.0,0.0,1.0)
    min_pm_move = st.number_input("Min Premarket %",-50.0,200.0,0.0,0.5)
    min_yday_gain = st.number_input("Min Yesterday %",-50.0,200.0,0.0,0.5)

    squeeze_only = st.checkbox("Short-Squeeze Only")
    catalyst_only = st.checkbox("Must Have News/Earnings")
    vwap_only = st.checkbox("Above VWAP Only (VWAP% > 0)")

    st.markdown("---")
    st.subheader("Order Flow Filter (optional)")
    enable_ofb_filter = st.checkbox("Use Flow Bias", False)
    min_ofb = st.slider("Min Buyer Control (0–1)",0.00,1.00,0.50,0.01)

    st.markdown("---")
    st.subheader("🔊 Audio Alerts")
    enable_alerts = st.checkbox("Enable Audio + Banner", value=False)
    ALERT_SCORE_THRESHOLD = st.slider("Alert Score ≥",10,200,30,5)
    ALERT_PM_THRESHOLD = st.slider("Alert Premarket ≥%",1,150,4,1)
    ALERT_VWAP_THRESHOLD = st.slider("Alert VWAP Dist ≥%",1,50,2,1)

    st.markdown("---")
    if st.button("🧹 Refresh Now"):
        st.cache_data.clear()
        st.success("Cache cleared — rescanning.")

# ========================= SYMBOL LOAD =========================
@st.cache_data(ttl=900)
def load_symbols():
    nasdaq = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",sep="|",skipfooter=1,engine="python")
    other  = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",sep="|",skipfooter=1,engine="python")

    nasdaq["Exchange"]="NASDAQ"
    other["Exchange"]=other["Exchange"].fillna("NYSE/AMEX/ARCA")
    other=other.rename(columns={"ACT Symbol":"Symbol"})

    df=pd.concat([nasdaq[["Symbol","ETF","Exchange"]],other[["Symbol","ETF","Exchange"]]]).dropna(subset=["Symbol"])
    return df[df["Symbol"].str.contains(r"^[A-Z]{1,5}$",na=False)].to_dict("records")

# ========================= UNIVERSE BUILDER =========================
def build_universe(watchlist_text,max_universe,universe_mode,volume_rank_pool):
    wl=watchlist_text.strip()
   
    # 🔥 WATCHLIST ALWAYS WINS FIRST
    if wl:
        raw=wl.replace("\n"," ").replace(","," ").split()
        tickers=sorted(set(s.upper() for s in raw if s.strip()))
        return [{"Symbol":t,"Exchange":"WATCH"} for t in tickers]  

    # otherwise universe logic unchanged
    all_syms=load_symbols()

    if universe_mode=="Randomized Slice":
        base=all_syms[:]
        random.shuffle(base)
        return base[:max_universe]

    if universe_mode=="Live Volume Ranked (slower)":
        ranked=[]
        base=all_syms[:volume_rank_pool]
        for s in base:
            try:
                d=yf.Ticker(s["Symbol"]).history(period="1d",interval="2m",prepost=True)
                if not d.empty: ranked.append({**s,"LiveVol":float(d["Volume"].iloc[-1])})
            except: pass
        ranked=sorted(ranked,key=lambda x:x.get("LiveVol",0),reverse=True)
        return ranked[:max_universe] if ranked else all_syms[:max_universe]
    
    return all_syms[:max_universe]

# ========================= SCORING =========================
def short_window_score(pm,yday,m3,m10,rsi7,rvol10,catalyst,squeeze,vwap,flow):
    score=0
    if pm: score+=max(pm,0)*1.6
    if yday: score+=max(yday,0)*0.8
    if m3: score+=max(m3,0)*1.2
    if m10: score+=max(m10,0)*0.6
    if rsi7>55: score+=(rsi7-55)*0.4
    if rvol10 and rvol10>1.2: score+=(rvol10-1.2)*2.0
    if vwap and vwap>0: score+=min(vwap,6)*1.5
    if flow: score+=(flow-0.5)*22
    if catalyst: score+=8
    if squeeze: score+=12
    return round(score,2)

def breakout_probability(score):
    try:return round((1/(1+math.exp(-score/20)))*100,1)
    except:return None

def multi_timeframe_label(pm,m3,m10):
    p=sum([pm>0 if pm else False,m3>0 if m3 else False,m10>0 if m10 else False])
    return["🔻 Not Aligned","🟡 Mixed","🟢 Leaning Bullish","✅ Aligned Bullish"][p]

def ai_commentary(s,pm,r,flow,v,m10):
    out=[]
    if s>=80:out.append("High compression trend.")
    elif s>=40:out.append("Momentum forming.")
    if pm and pm>3:out.append("Premarket strength.")
    if r and r>2:out.append("Volume expanding 10D.")
    if flow and flow>0.65:out.append("Buyers dominant.")
    if v and v>0:out.append("Trading above VWAP.")
    if m10 and m10>10:out.append("Sustained 10D strength.")
    return " | ".join(out) if out else "Neutral — waiting for structure."

# ========================= SCAN ONE =========================
def scan_one(sym,enrich,of_filter,min_ofb):
    try:
        t=sym["Symbol"]; stock=yf.Ticker(t)
        hist=stock.history(period="10d",interval="1d")
        if hist.empty:return None
        price=float(hist["Close"].iloc[-1])
        vol=float(hist["Volume"].iloc[-1])

        # FILTERS ONLY APPLY IF WATCHLIST NOT OVERRIDING
        if not ignore_filters_watchlist:
            if price>max_price or vol<min_volume:return None

        delta=hist["Close"].diff()
        rsi=100-(100/(1+(delta.clip(lower=0).rolling(7).mean()/(-delta.clip(upper=0)).rolling(7).mean())))
        rsi7=float(rsi.iloc[-1])
        
        yday=(hist["Close"].iloc[-1]-hist["Close"].iloc[-2])/hist["Close"].iloc[-2]*100 if len(hist)>=2 else None
        m3=(hist["Close"].iloc[-1]-hist["Close"].iloc[-4])/hist["Close"].iloc[-4]*100 if len(hist)>=4 else None
        m10=(hist["Close"].iloc[-1]-hist["Close"].iloc[0])/hist["Close"].iloc[0]*100 if hist["Close"].iloc[0]>0 else None
        
        ema10=float(hist["Close"].ewm(span=10,adjust=False).mean().iloc[-1])
        trend="🔥 Breakout" if price>ema10 and rsi7>55 else "Neutral"

        avg10=float(hist["Volume"].mean()); rvol=vol/avg10 if avg10 else None
        
        pm=vwap=flow=None
        try:
            intra=stock.history(period="1d",interval="2m",prepost=True)
            if len(intra)>=3:
                pm=(intra["Close"].iloc[-1]-intra["Close"].iloc[-2])/intra["Close"].iloc[-2]*100
                tp=(intra["High"]+intra["Low"]+intra["Close"])/3
                vwap=float((tp*intra["Volume"]).sum()/intra["Volume"].sum())
                vwap=(price-vwap)/vwap*100 if vwap>0 else None
                df=intra[["Open","Close","Volume"]]
                bv=(df[df["Close"]>df["Open"]]["Volume"].sum())
                sv=(df[df["Close"]<df["Open"]]["Volume"].sum())
                flow=bv/(bv+sv) if bv+sv>0 else None
        except: pass
        
        if of_filter and (flow is None or flow<min_ofb):return None
        
        score=short_window_score(pm,yday,m3,m10,rsi7,rvol,False,False,vwap,flow)
        prob=breakout_probability(score)

        return dict(
    Symbol=t,Exchange=sym.get("Exchange","UNK"),
    Price=round(price,2),Volume=int(vol),
    Score=score,Prob_Rise%=prob,
    PM%=round(pm,2) if pm else None,
    YDay%=round(yday,2) if yday else None,
    "3D%": round(m3,2) if m3 else None,        # ← FIXED
    "10D%": round(m10,2) if m10 else None,     # ← FIXED
    RSI7=round(rsi7,2),EMA10_Trend=trend,
    RVOL_10D=round(rvol,2) if rvol else None,
    VWAP%=round(vwap,2) if vwap else None,
    FlowBias=round(flow,2) if flow else None,
    MTF_Trend=multi_timeframe_label(pm,m3,m10),
    AI_Commentary=ai_commentary(score,pm,rvol,flow,vwap,m10),
    Spark=hist["Close"]
)
    except:return None

# ========================= RUN SCAN =========================
@st.cache_data(ttl=6)
def run_scan():
    universe=build_universe(watchlist_text,max_universe,universe_mode,volume_rank_pool)
    out=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS)as ex:
        for f in concurrent.futures.as_completed([ex.submit(scan_one,s,enable_enrichment,enable_ofb_filter,min_ofb)for s in universe]):
            r=f.result(); 
            if r:out.append(r)
    df=pd.DataFrame(out)

    # 🔥 FILTERS BYPASSED IF WATCHLIST OVERRIDE TOGGLE = ON
    if watchlist_text and ignore_filters_watchlist:
        return df  

    return df

# ========================= UI RENDER =========================
with st.spinner("Scanning…"):
    df=run_scan()

if df.empty:
    st.error("No matches — try watchlist or lower filters.")
else:
    df=df[df["Score"]>=min_breakout]

    if not (watchlist_text and ignore_filters_watchlist):
        if min_pm_move!=0.0:df=df[df["PM%"].fillna(-999)>=min_pm_move]
        if min_yday_gain!=0.0:df=df[df["YDay%"].fillna(-999)>=min_yday_gain]
        if squeeze_only:df=df[df["Squeeze?"]]
        if catalyst_only:df=df[df["Catalyst"]]
        if vwap_only:df=df[df["VWAP%"].fillna(-999)>0]

    df=df.sort_values(by=["Score","PM%","RSI7"],ascending=[False,False,False])
    st.subheader(f"🔥 Results — {len(df)} symbols")

    for _,r in df.iterrows():
        c1,c2,c3,c4=st.columns([2,3,3,3])
        sym=r.Symbol

        c1.markdown(f"**{sym}** ({r.Exchange})")
        c1.write(f"💲 {r.Price}  |  📊 Vol {r.Volume:,}")
        c1.write(f"🔥 Score {r.Score}  |  Prob {r['Prob_Rise%']}%")
        c1.write(r.MTF_Trend)
        c1.write(f"Trend: {r.EMA10_Trend}")

        c2.write(f"PM% {r['PM%']} | YDay {r['YDay%']}")
        c2.write(f"3D {r['3D%']} | 10D {r['10D%']}")
        c2.write(f"RSI7 {r.RSI7} | RVOL {r.RVOL_10D}x")

        c3.write(f"VWAP% {r['VWAP%']} | Flow {r.FlowBias}")
        c3.write(f"🧠 {r.AI_Commentary}")

        c4.plotly_chart(go.Figure(data=[go.Scatter(y=r.Spark.values,mode="lines")]),use_container_width=True)

    st.download_button("📥 Export CSV",df.to_csv(index=False),file_name="scan_v9_watchlist_override.csv")

st.caption("Research use only. Not financial advice.")








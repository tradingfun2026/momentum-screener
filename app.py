import streamlit as st
import pandas as pd
import yfinance as yf
import concurrent.futures
from datetime import datetime, timezone
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objs as go
import math
import random

# =======================================================================
# AUTO REFRESH + PAGE CONFIG
# =======================================================================
THREADS = 20
AUTO_REFRESH_MS = 120_000
HISTORY_LOOKBACK_DAYS = 10
INTRADAY_INTERVAL = "2m"
INTRADAY_RANGE = "1d"

st_autorefresh(interval=AUTO_REFRESH_MS, key="refresh")
st.set_page_config(page_title="V9 Momentum Screener", layout="wide")
st.title("🚀 V9 Momentum Scanner — Watchlist Override + Clean Formatting")
st.caption("All values rounded | Watchlist bypasses filters | AI commentary active")

# =======================================================================
# SIDEBAR — *FLOAT SAFE* (prevents MixedNumericTypesError)
# =======================================================================
with st.sidebar:
    st.header("Watchlist")
    watchlist_text = st.text_area("Tickers:", "", height=90)

    max_universe = st.slider("Max symbols (no watchlist)", 50, 2000, 600, 50)

    universe_mode = st.radio("Universe",
        ["Classic (Alphabetical Slice)", "Randomized Slice", "Live Volume Ranked (slower)"]
    )
    volume_rank_pool = st.slider("Volume rank pool", 200, 2000, 600, 100)

    st.header("Filters (ignored when watchlist is used)")
    max_price = st.number_input("Max Price", 0.0, 10000.0, 5.0, 0.5)
    min_volume = st.number_input("Min Volume", 0.0, 50_000_000.0, 0.0, 50_000.0)

    # ALL DECIMALS — FIXES MIXED NUMERIC TYPE CRASH
    min_breakout = st.number_input("Min Score", -50.0, 200.0, 0.0, 0.5)
    min_pm_move = st.number_input("Min Premarket %", -100.0, 200.0, 0.0, 0.5)
    min_yday_gain = st.number_input("Min Yesterday %", -100.0, 200.0, 0.0, 0.5)

    squeeze_only = st.checkbox("Short Squeeze Only")
    catalyst_only = st.checkbox("Require Catalyst/News")
    vwap_only = st.checkbox("Must be above VWAP")

    enable_ofb_filter = st.checkbox("Min OrderFlow Bias")
    min_ofb = st.slider("FlowBias ≥", 0.00, 1.00, 0.50, 0.01)

# =======================================================================
# FORMATTER — prevents RSI/RVOLUME long floats + handles None safely
# =======================================================================
def fmt2(x):
    try:
        return f"{float(x):.2f}"
    except:
        return "—"

# =======================================================================
# LOAD SYMBOLS SAFELY (Fixes NaN crash)
# =======================================================================
@st.cache_data(ttl=900)
def load_symbols():
    nasdaq = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
                         sep="|", skipfooter=1, engine="python")
    other  = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
                         sep="|", skipfooter=1, engine="python")
    nasdaq["Exchange"]="NASDAQ"
    other=other.rename(columns={"ACT Symbol":"Symbol"})

    df = pd.concat([nasdaq[["Symbol","ETF","Exchange"]],
                    other[["Symbol","ETF","Exchange"]]])

    df["Symbol"]=df["Symbol"].astype(str).fillna("")
    df=df[df["Symbol"].str.match(r"^[A-Z]{1,5}$",na=False)]
    return df.to_dict("records")

# =======================================================================
# UNIVERSE BUILDER — WATCHLIST OVERRIDES
# =======================================================================
def build_universe(watch,maxu,mode,pool):
    wl=watch.strip()
    if wl:
        raw=wl.replace("\n"," ").replace(","," ").split()
        return [{"Symbol":s.upper(),"Exchange":"WATCH"} for s in set(raw)]

    syms=load_symbols()
    if mode=="Randomized Slice":
        random.shuffle(syms)
        return syms[:maxu]

    if mode=="Live Volume Ranked (slower)":
        ranked=[]
        for s in syms[:pool]:
            try:
                h=yf.Ticker(s["Symbol"]).history(period="1d",interval="2m",prepost=True)
                if not h.empty:
                    ranked.append({**s,"LiveVol":float(h["Volume"].iloc[-1])})
            except: pass
        ranked=sorted(ranked,key=lambda x:x.get("LiveVol",0),reverse=True)
        return ranked[:maxu] if ranked else syms[:maxu]

    return syms[:maxu]

# =======================================================================
# AI NARRATIVE — restored to real trend evaluation
# =======================================================================
def ai_commentary(score,pm,rvol,flow,vwap,trend10):
    out=[]
    if score>=80: out.append("🔥 High momentum breakout setup")
    elif score>=40: out.append("🟢 Strong upside pressure building")
    elif score>=10: out.append("🟡 Early setup — needs continuation")
    else: out.append("⚪ No strong trend yet")

    if pm and pm>3: out.append("Premarket buyers active")
    if pm and pm<0: out.append("Premarket selling pressure")
    if rvol and rvol>2: out.append("High volume expansion")
    if rvol and rvol<0.7: out.append("Low participation volume")
    if flow and flow>0.65: out.append("Buyer control 📈")
    if flow and flow<0.4: out.append("Seller dominance 📉")
    if vwap and vwap>0: out.append("Above VWAP → bullish")
    if vwap and vwap<0: out.append("Below VWAP → fading strength")
    if trend10 and trend10>10: out.append("10D uptrend accelerating")
    if trend10 and trend10<-5: out.append("10D trend weakening")

    return " | ".join(out)

# =======================================================================
# CORE SCAN — watchlist ALWAYS returns even if data is light
# =======================================================================
def scan_one(sym):
    t=sym["Symbol"]
    watch=(sym.get("Exchange")=="WATCH")

    try:
        stock=yf.Ticker(t)
        hist=stock.history(period=f"{HISTORY_LOOKBACK_DAYS}d",interval="1d")

        if hist.empty or len(hist)<5:
            return {"Symbol":t,"Exchange":"WATCH","Score":None,
                    "AI_Commentary":"⚠ Insufficient data"}

        close=hist["Close"];vols=hist["Volume"]
        price=float(close.iloc[-1]);vol=float(vols.iloc[-1])

        if not watch and (price>max_price or vol<min_volume):
            return None

        # Returns
        yday=(close.iloc[-1]-close.iloc[-2])/close.iloc[-2]*100
        m3=(close.iloc[-1]-close.iloc[-4])/close.iloc[-4]*100 if len(close)>=4 else None
        m10=(close.iloc[-1]-close.iloc[0])/close.iloc[0]*100

        # RSI7
        delta=close.diff()
        gain=delta.clip(lower=0).rolling(7).mean()
        loss=(-delta.clip(upper=0)).rolling(7).mean()
        rs=gain/loss
        rsi7=float(100-(100/(1+rs)).iloc[-1])

        rvol10=vol/vols.mean() if vols.mean()>0 else None

        # Intraday microstructure
        pm=vwap=flow=None
        try:
            intra=stock.history(period=INTRADAY_RANGE,interval=INTRADAY_INTERVAL,prepost=True)
            if not intra.empty and len(intra)>=3:
                ic=intra["Close"];io=intra["Open"]
                pm=(ic.iloc[-1]-ic.iloc[-2])/ic.iloc[-2]*100

                typical=(intra["High"]+intra["Low"]+intra["Close"])/3
                tot=intra["Volume"].sum()
                if tot>0:
                    vw=(typical*intra["Volume"]).sum()/tot
                    vwap=(price-vw)/vw*100

                sign=(ic>io).astype(int)-(ic<io).astype(int)
                buy=(intra["Volume"]*(sign>0)).sum()
                sell=(intra["Volume"]*(sign<0)).sum()
                if buy+sell>0: flow=buy/(buy+sell)
        except: pass

        if enable_ofb_filter and not watch:
            if flow is None or flow<min_ofb: return None

        score=(
            (pm or 0)*1.6 + (yday or 0)*0.8 + (m3 or 0)*1.2 + (m10 or 0)*0.6 +
            (max(rsi7-55,0)*0.4) +
            ((rvol10-1.2)*2 if rvol10 and rvol10>1.2 else 0) +
            ((min(vwap,6)*1.5) if vwap and vwap>0 else 0) +
            ((flow-0.5)*22 if flow else 0)
        )

        score=round(score,2)
        prob=round(100/(1+math.exp(-score/20)),1)

        comment=ai_commentary(score,pm,rvol10,flow,vwap,m10)
        if watch: comment+=" | 🟡 Watchlist Priority"

        return {"Symbol":t,"Exchange":sym.get("Exchange"),
                "Price":price,"Volume":vol,
                "Score":score,"Prob_Rise%":prob,
                "PM%":pm,"YDay%":yday,"3D%":m3,"10D%":m10,
                "RSI7":rsi7,"RVOL_10D":rvol10,"VWAP%":vwap,
                "FlowBias":flow,"Spark":close,
                "AI_Commentary":comment}
    except:
        return {"Symbol":t,"Exchange":"WATCH","Score":None,"AI_Commentary":"⚠ Scan error"}

@st.cache_data(ttl=6)
def run_scan(w,mode,pool):
    uni=build_universe(w,max_universe,mode,pool)
    out=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as ex:
        for f in concurrent.futures.as_completed([ex.submit(scan_one,s) for s in uni]):
            r=f.result()
            if r: out.append(r)
    return pd.DataFrame(out)

df=run_scan(watchlist_text,universe_mode,volume_rank_pool)

# =======================================================================
# FILTERS — Only apply if **no watchlist**
# =======================================================================
if not df.empty and not watchlist_text.strip():

    if "Score" in df:
        df=df[df["Score"].fillna(-999)>=min_breakout]
    if "PM%" in df:
        df=df[df["PM%"].fillna(-999)>=min_pm_move]
    if "YDay%" in df:
        df=df[df["YDay%"].fillna(-999)>=min_yday_gain]
    if squeeze_only and "Squeeze?" in df:
        df=df[df["Squeeze?"]==True]
    if catalyst_only and "Catalyst" in df:
        df=df[df["Catalyst"]==True]
    if vwap_only and "VWAP%" in df:
        df=df[df["VWAP%"].fillna(-999)>0]

# =======================================================================
# DISPLAY — All stats now 2 decimals
# =======================================================================
if df.empty:
    st.warning("⚠ No results — watchlist still displays even if no matches.")
else:
    df = df.sort_values(by=["Score","PM%","RSI7"],ascending=[False,False,False])

    st.subheader(f"🔥 Returned {len(df)} symbols")

    for _, row in df.iterrows():
        c1,c2,c3,c4 = st.columns([2,3,3,3])

        c1.markdown(f"### **{row['Symbol']}**")
        c1.write(f"Price **{fmt2(row['Price'])}** | Vol **{int(row['Volume']):,}**")
        c1.write(f"Score **{fmt2(row['Score'])}**  | RiseProb {fmt2(row['Prob_Rise%'])}%")

        c2.write(f"PM {fmt2(row['PM%'])}% | YDay {fmt2(row['YDay%'])}%")
        c2.write(f"3D {fmt2(row['3D%'])}% | 10D {fmt2(row['10D%'])}%")
        c2.write(f"RSI7 {fmt2(row['RSI7'])} | RVOL {fmt2(row['RVOL_10D'])}x")

        c3.markdown("### 🧠 Commentary")
        c3.write(row["AI_Commentary"])

        fig=go.Figure(go.Scatter(y=row["Spark"].values,mode="lines"))
        fig.update_layout(height=180,margin=dict(l=5,r=5,t=10,b=10))
        c4.plotly_chart(fig,use_container_width=True)

    st.download_button("📥 Export CSV", df.to_csv(index=False), "momentum.csv")





st.caption("For research and education only. Not financial advice.  All original UI restored — watchlist now fully overridden.")












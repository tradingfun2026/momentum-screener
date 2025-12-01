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
# CONFIG
# =======================================================================
THREADS = 20
AUTO_REFRESH_MS = 120_000
HISTORY_LOOKBACK_DAYS = 10
INTRADAY_INTERVAL = "2m"
INTRADAY_RANGE = "1d"

DEFAULT_MAX_PRICE = 5.0
DEFAULT_MIN_VOLUME = 0.0
DEFAULT_MIN_BREAKOUT = 0.0

# =======================================================================
# PAGE + REFRESH
# =======================================================================
st_autorefresh(interval=AUTO_REFRESH_MS, key="refresh")
st.set_page_config(page_title="Momentum Screener", layout="wide")
st.title("🚀 Momentum Screener — V9 Optimized + Watchlist Override")
st.subheader("Watchlist tickers bypass filters | AI commentary restored")

# =======================================================================
# SIDEBAR (FLOAT-FIXED)
# =======================================================================
with st.sidebar:
    st.header("Watchlist")
    watchlist_text = st.text_area("Tickers (comma or newline separated):", "")

    st.header("Universe Size (if no watchlist)")
    max_universe = st.slider("Max symbols", 50, 2000, 600, 50)

    universe_mode = st.radio("Universe Mode",[
        "Classic (Alphabetical Slice)",
        "Randomized Slice",
        "Live Volume Ranked (slower)"
    ])

    volume_rank_pool = st.slider("Volume rank pool", 200, 2000, 600, 100)

    st.header("Filters (NOT applied to watchlist)")
    max_price = st.number_input("Max Price", 0.0, 5000.0, 5.0, 0.5)
    min_volume = st.number_input("Min Volume", 0.0, 20_000_000.0, 0.0, 50_000.0)

    # 🔥 ALL FLOAT — FIXES mixed numeric type Streamlit crash
    min_breakout = st.number_input("Min Score", -50.0, 200.0, 0.0, 0.5)
    min_pm_move = st.number_input("Min Premarket %", -100.0, 200.0, 0.0, 0.5)
    min_yday_gain = st.number_input("Min Yesterday %", -100.0, 200.0, 0.0, 0.5)

    squeeze_only = st.checkbox("Short squeeze only")
    catalyst_only = st.checkbox("Require news/catalyst")
    vwap_only = st.checkbox("Must be above VWAP")

    enable_ofb_filter = st.checkbox("Order Flow Bias Min")
    min_ofb = st.slider("Min Flow Bias", 0.0, 1.0, 0.50, 0.01)

# =======================================================================
# LOAD SYMBOLS — FIXED NAs
# =======================================================================
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
    other = other.rename(columns={"ACT Symbol": "Symbol"})

    df = pd.concat([nasdaq[["Symbol","ETF","Exchange"]],
                    other[["Symbol","ETF","Exchange"]]])

    df["Symbol"] = df["Symbol"].astype(str).fillna("")
    df = df[df["Symbol"].str.match(r"^[A-Z]{1,5}$", na=False)]
    return df.to_dict("records")


# =======================================================================
# UNIVERSE BUILDER WITH WATCHLIST PRIORITY
# =======================================================================
def build_universe(watch,maxu,mode,pool):
    wl=watch.strip()
    if wl:
        raw=wl.replace("\n"," ").replace(","," ").split()
        return [{"Symbol":s.upper(),"Exchange":"WATCH"} for s in sorted(set(raw))]

    syms=load_symbols()
    if mode=="Randomized Slice": random.shuffle(syms);return syms[:maxu]

    if mode=="Live Volume Ranked (slower)":
        ranked=[]
        for s in syms[:pool]:
            try:
                df=yf.Ticker(s["Symbol"]).history(period="1d",interval="2m",prepost=True)
                if not df.empty:
                    ranked.append({**s,"LiveVol": float(df["Volume"].iloc[-1])})
            except: pass
        ranked=sorted(ranked,key=lambda x:x.get("LiveVol",0),reverse=True)
        return ranked[:maxu] if ranked else syms[:maxu]

    return syms[:maxu]


# =======================================================================
# AI COMMENTARY (RESTORED)
# =======================================================================
def ai_commentary(score,pm,rvol,ob,vwap,trend10):
    c=[]
    if score>=80: c.append("🔥 High momentum breakout potential")
    elif score>=40: c.append("🟢 Building bullish pressure")
    elif score>=10: c.append("🟡 Early setup forming — needs continuation")
    else: c.append("⚪ Low conviction / no breakout yet")

    if pm and pm>3: c.append("Premarket buyers active")
    if pm and pm<0: c.append("Premarket selling pressure")

    if rvol and rvol>2: c.append("Volume expanding vs avg")
    if rvol and rvol<0.7: c.append("Weak volume flow")

    if ob and ob>0.65: c.append("Buy-flow control 📈")
    if ob and ob<0.4: c.append("Seller dominance 📉")

    if vwap and vwap>0: c.append("Above VWAP → bullish")
    if vwap and vwap<0: c.append("Below VWAP → overhead selling")

    if trend10 and trend10>10: c.append("10D trend accelerating")
    if trend10 and trend10<-5: c.append("10D trend weakening")

    return " | ".join(c)


# =======================================================================
# SCAN ONE — WATCHLIST ALWAYS RETURNS
# =======================================================================
def scan_one(sym):
    t=sym["Symbol"];watch=(sym.get("Exchange")=="WATCH")

    try:
        stock=yf.Ticker(t)
        hist=stock.history(period=f"{HISTORY_LOOKBACK_DAYS}d",interval="1d")

        if hist.empty or len(hist)<5:
            return {"Symbol":t,"Exchange":"WATCH","Score":None,
                    "AI_Commentary":"⚠ No market data available"}

        close=hist["Close"];vols=hist["Volume"]
        price=float(close.iloc[-1]);vol=float(vols.iloc[-1])

        # WATCHLIST BYPASSES FILTERS
        if not watch and (price>max_price or vol<min_volume):
            return None

        # BREAKDOWN CALCULATIONS
        yday = (close.iloc[-1]-close.iloc[-2])/close.iloc[-2]*100 if len(close)>=2 else None
        m3 = (close.iloc[-1]-close.iloc[-4])/close.iloc[-4]*100 if len(close)>=4 else None
        m10 = (close.iloc[-1]-close.iloc[0])/close.iloc[0]*100

        # RSI(7)
        delta=close.diff()
        gain=delta.clip(lower=0).rolling(7).mean()
        loss=(-delta.clip(upper=0)).rolling(7).mean()
        rsi7=float(100-(100/(1+(gain/loss))).iloc[-1])

        rvol10 = vol/vols.mean() if vols.mean()>0 else None

        # INTRADAY MICROSTRUCTURE
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
        except:
            pass

        if enable_ofb_filter and not watch:
            if flow is None or flow<min_ofb: return None

        # SCORE MODEL (FLOAT SAFE)
        score=(
            (pm or 0)*1.6 + (yday or 0)*0.8 + (m3 or 0)*1.2 + (m10 or 0)*0.6 +
            (max(rsi7-55,0)*0.4) +
            ((rvol10-1.2)*2 if rvol10 and rvol10>1.2 else 0) +
            ((min(vwap,6)*1.5) if vwap and vwap>0 else 0) +
            ((flow-0.5)*22 if flow else 0)
        )
        score=round(score,2)
        rise=round(100/(1+math.exp(-score/20)),1)

        commentary=ai_commentary(score,pm,rvol10,flow,vwap,m10)
        if watch: commentary+="  | 🟡 Watchlist priority"

        return {"Symbol":t,"Exchange":sym.get("Exchange"),
                "Price":round(price,2),"Volume":int(vol),
                "Score":score,"Prob_Rise%":rise,
                "PM%":pm,"YDay%":yday,"3D%":m3,"10D%":m10,
                "RSI7":rsi7,"RVOL_10D":rvol10,"VWAP%":vwap,
                "FlowBias":flow,"Spark":close,
                "AI_Commentary":commentary}

    except:
        return {"Symbol":t,"Exchange":"WATCH",
                "Score":None,"AI_Commentary":"⚠ Scan failed"}

# =======================================================================
# RUN SCAN
# =======================================================================
@st.cache_data(ttl=6)
def run_scan(w,mode,pool):
    uni=build_universe(w,max_universe,mode,pool)
    res=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as ex:
        for f in concurrent.futures.as_completed([ex.submit(scan_one,s) for s in uni]):
            r=f.result()
            if r: res.append(r)
    return pd.DataFrame(res)


df=run_scan(watchlist_text,universe_mode,volume_rank_pool)

# =======================================================================
# FILTERS — SAFE (NO KEYERRORS)
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
# DISPLAY RESULTS
# =======================================================================
if df.empty:
    st.warning("⚠ No results — but watchlist tickers still appear even with missing data.")
else:
    df=df.sort_values(by=["Score","PM%","RSI7"],ascending=[False,False,False])

    st.subheader(f"🔥 Returned: {len(df)} symbols")

    for _,row in df.iterrows():
        c1,c2,c3,c4 = st.columns([2,3,3,3])
        c1.markdown(f"### **{row['Symbol']}**")
        c1.write(f"Price **{row['Price']}** | Vol **{row['Volume']:,}**")
        c1.write(f"Score **{row['Score']:.2f}** | RiseProb {row['Prob_Rise%']:.2f}%")


        c2.write(f"PM {row['PM%']:.2f}%  |  YDay {row['YDay%']:.2f}%")
        c2.write(f"3D {row['3D%']:.2f}% | 10D {row['10D%']:.2f}%")
        c2.write(f"RSI7 {row['RSI7']} | RVOL {row['RVOL_10D']}x")

        c3.markdown("### 🧠 AI Commentary")
        c3.write(row["AI_Commentary"])

        fig=go.Figure(go.Scatter(y=row["Spark"].values,mode="lines"))
        fig.update_layout(height=180,margin=dict(l=5,r=5,t=10,b=10))
        c4.plotly_chart(fig,use_container_width=True)

    st.download_button(
        "📥 Export CSV",
        df.to_csv(index=False),
        "momentum_screener.csv"
    )




st.caption("For research and education only. Not financial advice.  All original UI restored — watchlist now fully overridden.")












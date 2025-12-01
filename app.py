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
THREADS=20
AUTO_REFRESH_MS=120_000
HISTORY_LOOKBACK_DAYS=10
INTRADAY_INTERVAL="2m"
INTRADAY_RANGE="1d"
DEFAULT_MAX_PRICE=5.0
DEFAULT_MIN_VOLUME=0
DEFAULT_MIN_BREAKOUT=0.0

# ========================= AUTO REFRESH =========================
st_autorefresh(interval=AUTO_REFRESH_MS,key="refresh_v9")

# ========================= PAGE SETUP =========================
st.set_page_config(page_title="V9 Momentum Screener",layout="wide")
st.title("🚀 V9 — Momentum Screener (Watchlist Override Enabled)")
st.caption("All outputs preserved • AI Commentary restored • Watchlist bypasses ALL filters")

# ========================= SIDEBAR =========================
with st.sidebar:
    watchlist_text=st.text_area("Watchlist tickers:",value="",height=80)
    max_universe=st.slider("Max symbols if no watchlist",50,2000,600,200)

    universe_mode=st.radio("Universe Mode",[
        "Classic (Alphabetical Slice)",
        "Randomized Slice",
        "Live Volume Ranked (slower)"
    ],index=0)
    volume_rank_pool=st.slider("Volume rank size",100,2000,600,100)

    enable_enrichment=st.checkbox("Float/Short/News Enrichment",value=False)
    st.header("Filters (ignored w/ watchlist)")

    max_price=st.number_input("Max Price",1.0,2000.0,DEFAULT_MAX_PRICE,1.0)
    min_volume=st.number_input("Min Volume",0,10_000_000,DEFAULT_MIN_VOLUME,10_000)
    min_breakout=st.number_input("Min Score",-50,200,0.0,0.5)
    min_pm_move=st.number_input("Min Premarket %",-50,200,0.0,0.5)
    min_yday_gain=st.number_input("Min Yesterday %",-50,200,0.0,0.5)

    squeeze_only=st.checkbox("Short Squeeze Only")
    catalyst_only=st.checkbox("Must Have Catalyst")
    vwap_only=st.checkbox("Must be Above VWAP")

    st.subheader("Order Flow Filter")
    enable_ofb_filter=st.checkbox("Enable Orderflow Bias Filter",value=False)
    min_ofb=st.slider("Min Flow Bias",0.0,1.0,0.50,0.01)

# ========================= LOAD SYMBOLS =========================
@st.cache_data(ttl=900)
def load_symbols():
    nasdaq=pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
                       sep="|",skipfooter=1,engine="python")
    other=pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
                       sep="|",skipfooter=1,engine="python")
    nasdaq["Exchange"]="NASDAQ"
    other=other.rename(columns={"ACT Symbol":"Symbol"})
    df=pd.concat([nasdaq[["Symbol","ETF","Exchange"]],
                  other[["Symbol","ETF","Exchange"]]])

    df["Symbol"]=df["Symbol"].astype(str).fillna("")  # fix for ValueError
    df=df[df["Symbol"].str.match(r"^[A-Z]{1,5}$",na=False)]
    return df.to_dict("records")

# ========================= UNIVERSE =========================
def build_universe(watch,maxu,mode,pool):
    wl=watch.strip()
    if wl:
        raw=wl.replace("\n"," ").replace(",", " ").split()
        return [{"Symbol":s.upper(),"Exchange":"WATCH"} for s in sorted(set(raw))]
    syms=load_symbols()
    if mode=="Randomized Slice":
        random.shuffle(syms);return syms[:maxu]
    if mode=="Live Volume Ranked (slower)":
        ranked=[]
        for s in syms[:pool]:
            try:
                h=yf.Ticker(s["Symbol"]).history(period="1d",interval="2m",prepost=True)
                if not h.empty: ranked.append({**s,"LiveVol":float(h["Volume"].iloc[-1])})
            except: pass
        ranked=sorted(ranked,key=lambda x:x.get("LiveVol",0),reverse=True)
        return ranked[:maxu] if ranked else syms[:maxu]
    return syms[:maxu]

# ========================= AI COMMENTARY =========================
def ai_commentary(score, pm, rvol, flow_bias, vwap, ten_day):
    parts=[]
    if score>=80: parts.append("🔥 High momentum expansion potential")
    elif score>=40: parts.append("🟢 Constructively building momentum")
    elif score>=10: parts.append("🟡 Early build – requires continuation")
    else: parts.append("⚪ Weak momentum – no clear breakout yet")

    if pm and pm>3: parts.append("Premarket buyers active")
    if pm and pm<0: parts.append("Premarket selling pressure")
    if rvol and rvol>2: parts.append("Volume expanding strongly vs avg")
    if rvol and rvol<0.7: parts.append("Low volume participation")
    if flow_bias and flow_bias>0.65: parts.append("Buy-volume lead 📈")
    if flow_bias and flow_bias<0.4: parts.append("Seller controlled 📉")
    if vwap and vwap>0: parts.append("Above VWAP – bullish control")
    if vwap and vwap<0: parts.append("Below VWAP supply overhead")
    if ten_day and ten_day>10: parts.append("10D trend favoring strength")
    if ten_day and ten_day<-5: parts.append("10D trend weakening")

    return " | ".join(parts)

# ========================= SCAN ONE (WATCHLIST SAFE) =========================
def scan_one(sym,enrich,ofb,min_ofb):
    t=sym["Symbol"];watch=(sym.get("Exchange")=="WATCH")

    try:
        stock=yf.Ticker(t)
        hist=stock.history(period=f"{HISTORY_LOOKBACK_DAYS}d",interval="1d")

        # Watchlist ALWAYS returns even if incomplete
        if hist is None or hist.empty or len(hist)<5:
            return {"Symbol":t,"Exchange":"WATCH","Score":None,
                    "AI_Commentary":"⚠ No data – Watchlist override"}

        close=hist["Close"];volume=hist["Volume"]
        price=float(close.iloc[-1]);vol=float(volume.iloc[-1])

        if not watch and (price>max_price or vol<min_volume): return None

        # Momentum math
        yday=(close.iloc[-1]-close.iloc[-2])/close.iloc[-2]*100 if len(close)>=2 else None
        m3=(close.iloc[-1]-close.iloc[-4])/close.iloc[-4]*100 if len(close)>=4 else None
        m10=(close.iloc[-1]-close.iloc[0])/close.iloc[0]*100

        # RSI7
        delta=close.diff()
        gain=delta.clip(lower=0).rolling(7).mean()
        loss=(-delta.clip(upper=0)).rolling(7).mean()
        rs=gain/loss
        rsi7=float(100-(100/(1+rs)).iloc[-1])
        rvol10=vol/volume.mean() if volume.mean()>0 else None

        # Intraday
        pm=vwap=flow=None
        try:
            intra=stock.history(period=INTRADAY_RANGE,interval=INTRADAY_INTERVAL,prepost=True)
        except: intra=None

        if intra is not None and not intra.empty and len(intra)>=3:
            ic=intra["Close"]; io=intra["Open"]
            pm=(ic.iloc[-1]-ic.iloc[-2])/ic.iloc[-2]*100
            typical=(intra["High"]+intra["Low"]+intra["Close"])/3
            tot=intra["Volume"].sum()
            if tot>0:
                v=(typical*intra["Volume"]).sum()/tot
                vwap=(price-v)/v*100
            sign=(ic>io).astype(int)-(ic<io).astype(int)
            buy=(intra["Volume"]*(sign>0)).sum(); sell=(intra["Volume"]*(sign<0)).sum()
            if buy+sell>0: flow=buy/(buy+sell)

        if ofb and not watch:
            if flow is None or flow<min_ofb: return None

        score=(
            (pm or 0)*1.6 + (yday or 0)*0.8 + (m3 or 0)*1.2 + (m10 or 0)*0.6
            + (max(rsi7-55,0)*0.4) + ((rvol10-1.2)*2 if rvol10 and rvol10>1.2 else 0)
            + ((min(vwap,6)*1.5) if vwap and vwap>0 else 0)
            + ((flow-0.5)*22 if flow else 0)
        )
        score=round(score,2)
        prob=round(100/(1+math.exp(-score/20)),1)

        ai=ai_commentary(score,pm,rvol10,flow,vwap,m10)
        if watch: ai+=" | 🟡 Watchlist Priority"

        return {
            "Symbol":t,"Exchange":sym.get("Exchange","?"),
            "Price":round(price,2),"Volume":int(vol),
            "Score":score,"Prob_Rise%":prob,
            "PM%":pm,"YDay%":yday,"3D%":m3,"10D%":m10,"RSI7":rsi7,
            "RVOL_10D":rvol10,"VWAP%":vwap,"FlowBias":flow,
            "Spark":close,"AI_Commentary":ai
        }
    except:
        return {"Symbol":t,"Exchange":"WATCH","Score":None,"AI_Commentary":"⚠ Scan failed"}

# ========================= RUN SCAN =========================
@st.cache_data(ttl=6)
def run_scan(w,maxu,en,ofb,min_ofb,mode,pool):
    uni=build_universe(w,maxu,mode,pool)
    out=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as ex:
        for f in concurrent.futures.as_completed([ex.submit(scan_one,s,en,ofb,min_ofb) for s in uni]):
            r=f.result()
            if r: out.append(r)
    return pd.DataFrame(out)

# ========================= MAIN UI =========================
df=run_scan(watchlist_text,max_universe,enable_enrichment,enable_ofb_filter,min_ofb,universe_mode,volume_rank_pool)

if not df.empty:
    # ---------------- SAFE FILTERING (fixes KeyError on PM%) ---------------- #
    if not watchlist_text.strip():

        if "Score" in df: df=df[df["Score"].fillna(-999)>=min_breakout]
        if "PM%" in df: df=df[df["PM%"].fillna(-999)>=min_pm_move]
        if "YDay%" in df: df=df[df["YDay%"].fillna(-999)>=min_yday_gain]
        if squeeze_only and "Squeeze?" in df: df=df[df["Squeeze?"]==True]
        if catalyst_only and "Catalyst" in df: df=df[df["Catalyst"]==True]
        if vwap_only and "VWAP%"] in df: df=df[df["VWAP%"].fillna(-999)>0]

    df=df.sort_values(by=["Score","PM%","RSI7"],ascending=[False,False,False])
    st.subheader(f"Results Returned: {len(df)}")

    for _,row in df.iterrows():
        c1,c2,c3,c4=st.columns([2,3,3,3])
        c1.write(f"**{row['Symbol']}**")
        c1.write(f"Price: {row['Price']} | Vol: {row['Volume']:,}")
        c1.write(f"Score: {row['Score']} | RiseProb: {row['Prob_Rise%']}%")

        c2.write(f"PM%: {row['PM%']} | YDay%: {row['YDay%']}")
        c2.write(f"3D:{row['3D%']} | 10D:{row['10D%']}")
        c2.write(f"RSI7:{row['RSI7']} | RVOL:{row['RVOL_10D']}x")

        c3.markdown("### 🧠 AI Commentary")
        c3.write(row["AI_Commentary"])

        c4.plotly_chart(go.Figure(data=[go.Scatter(y=row["Spark"].values)]),
                        use_container_width=True)

    # CSV
    cols=[c for c in ["Symbol","Exchange","Price","Volume","Score","Prob_Rise%","PM%","YDay%","3D%","10D%","RSI7","RVOL_10D","VWAP%","FlowBias","AI_Commentary"] if c in df.columns]
    st.download_button("📥 Download CSV",df[cols].to_csv(index=False),"screener.csv")

else:
    st.error("No results – watchlist symbols still output even if no data")


st.caption("For research and education only. Not financial advice.  All original UI restored — watchlist now fully overridden.")












# =====================================================================
# V9 — FULL UI RESTORED + WATCHLIST OVERRIDE FIX (PRESERVES ALL OUTPUTS)
# =====================================================================

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
st.set_page_config(page_title="V9 Momentum Screener",layout="wide",initial_sidebar_state="expanded")
st.title("🚀 V9 — Momentum Screener (Watchlist Override Fixed)")
st.caption("Full UI + data restored — watchlist now ALWAYS supersedes filters")

# ========================= SIDEBAR =========================
with st.sidebar:
    st.header("Universe")

    watchlist_text=st.text_area("Watchlist tickers:",value="",height=80)
    max_universe = st.slider("Max symbols if NO watchlist",50,2000,600,200)

    st.subheader("Universe Mode")
    universe_mode=st.radio("Mode",["Classic (Alphabetical)","Randomized Slice","Live Volume Ranked (slower)"],index=0)
    volume_rank_pool=st.slider("Volume rank pool",100,2000,600,100)

    enable_enrichment=st.checkbox("Float/Short/News Enrichment",value=False)

    st.header("Filters")
    max_price=st.number_input("Max Price",1.0,1000.0,DEFAULT_MAX_PRICE,1.0)
    min_volume=st.number_input("Min Volume",0,10_000_000,DEFAULT_MIN_VOLUME,10_000)
    min_breakout=st.number_input("Min Score",-50.0,200.0,0.0,0.5)
    min_pm_move=st.number_input("Min Premarket %",-50.0,200.0,0.0,0.5)
    min_yday_gain=st.number_input("Min Yesterday %",-50.0,200.0,0.0,0.5)

    squeeze_only=st.checkbox("Short Squeeze Only")
    catalyst_only=st.checkbox("News/Earnings Only")
    vwap_only=st.checkbox("Above VWAP Only")

    st.subheader("Order flow filter")
    enable_ofb_filter=st.checkbox("Enable Flow Bias Filter",value=False)
    min_ofb=st.slider("Min Flow Bias",0.00,1.00,0.50,0.01)

    enable_alerts=st.checkbox("Enable Alerts",value=False)
    ALERT_SCORE_THRESHOLD=st.slider("Alert Score",10,200,30,5)
    ALERT_PM_THRESHOLD=st.slider("Alert Premarket %",1,150,4,1)
    ALERT_VWAP_THRESHOLD=st.slider("Alert VWAP %",1,50,2,1)

# ========================= LOAD SYMBOLS =========================
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

    df = pd.concat([
        nasdaq[["Symbol", "ETF", "Exchange"]],
        other[["Symbol", "ETF", "Exchange"]]
    ])

    # FIX: ensure symbols are clean before filtering
    df["Symbol"] = df["Symbol"].astype(str).fillna("")

    # Only keep valid ticker formats A–Z length 1–5
    df = df[df["Symbol"].str.match(r"^[A-Z]{1,5}$", na=False)]

    return df.to_dict("records")


# ========================= UNIVERSE =========================
def build_universe(watch,limit,mode,pool):
    wl=watch.strip()
    if wl:
        raw=wl.replace("\n"," ").replace(",", " ").split()
        return [{"Symbol":s.upper(),"Exchange":"WATCH"} for s in sorted(set(raw))]
    all_syms=load_symbols()
    if mode=="Randomized Slice":
        random.shuffle(all_syms);return all_syms[:limit]
    if mode=="Live Volume Ranked (slower)":
        ranked=[]
        for sym in all_syms[:pool]:
            try:
                d=yf.Ticker(sym["Symbol"]).history(period="1d",interval="2m",prepost=True)
                if not d.empty:
                    ranked.append({**sym,"LiveVol":float(d["Volume"].iloc[-1])})
            except:pass
        ranked=sorted(ranked,key=lambda x:x.get("LiveVol",0),reverse=True)
        return ranked[:limit] if ranked else all_syms[:limit]
    return all_syms[:limit]

# ========================= SHORT WINDOW SCORE (unchanged) =========================
def short_window_score(pm,yday,m3,m10,rsi7,rvol10,catalyst,squeeze,vwap,flow_bias):
    score=0
    if pm:score+=pm*1.6
    if yday:score+=yday*0.8
    if m3:score+=m3*1.2
    if m10:score+=m10*0.6
    if rsi7>55:score+=(rsi7-55)*0.4
    if rvol10 and rvol10>1.2:score+=(rvol10-1.2)*2
    if vwap and vwap>0:score+=min(vwap,6)*1.5
    if flow_bias:score+=(flow_bias-0.5)*22
    if catalyst:score+=8
    if squeeze:score+=12
    return round(score,2)

def breakout_probability(score):
    return round(100/(1+math.exp(-score/20)),1)

# ========================= 🟢 FIXED SCAN — WATCHLIST OVERRIDE =========================
def scan_one(sym,enable_enrichment,enable_ofb_filter,min_ofb):

    t=sym["Symbol"]
    is_watch=(sym.get("Exchange")=="WATCH")

    try:
        stock=yf.Ticker(t)
        hist=stock.history(period=f"{HISTORY_LOOKBACK_DAYS}d",interval="1d")

        # If watchlist → NEVER DROP — create blank output if data missing
        if hist is None or hist.empty or len(hist)<5:
            return {"Symbol":t,"Exchange":"WATCH","Price":None,"Volume":None,"Score":None,
                    "Prob_Rise%":None,"PM%":None,"YDay%":None,"3D%":None,"10D%":None,
                    "RSI7":None,"EMA10 Trend":None,"RVOL_10D":None,"VWAP%":None,
                    "FlowBias":None,"Squeeze?":None,"LowFloat?":None,"Short % Float":None,
                    "Sector":None,"Industry":None,"Catalyst":None,"MTF_Trend":"No Data",
                    "Spark":pd.Series([0]),"AI_Commentary":"⚠ No data — Watchlist enforced"}

        close=hist["Close"];volume=hist["Volume"]
        price=float(close.iloc[-1]);vol=float(volume.iloc[-1])

        # filters apply only if NOT watchlist
        if not is_watch:
            if price>max_price or vol<min_volume:return None

        # Momentum windows
        yday=(close.iloc[-1]-close.iloc[-2])/close.iloc[-2]*100 if len(close)>=2 else None
        m3  =(close.iloc[-1]-close.iloc[-4])/close.iloc[-4]*100 if len(close)>=4 else None
        m10 =(close.iloc[-1]-close.iloc[0])/close.iloc[0]*100

        # RSI(7)
        delta=close.diff()
        gain=delta.clip(lower=0).rolling(7).mean()
        loss=(-delta.clip(upper=0)).rolling(7).mean()
        rs=gain/loss
        rsi7=float(100-(100/(1+rs)).iloc[-1])

        # RVOL
        rvol10=vol/volume.mean() if volume.mean()>0 else None

        # Intraday 2m
        try:
            intra=stock.history(period=INTRADAY_RANGE,interval=INTRADAY_INTERVAL,prepost=True)
        except:intra=None

        pm=vwap=flowbias=None
        if intra is not None and not intra.empty and len(intra)>=3:
            iclose=intra["Close"];iopen=intra["Open"]
            pm=(iclose.iloc[-1]-iclose.iloc[-2])/iclose.iloc[-2]*100
            typical=(intra["High"]+intra["Low"]+intra["Close"])/3
            total_vol=intra["Volume"].sum()
            if total_vol>0:
                v=(typical*intra["Volume"]).sum()/total_vol
                vwap=(price-v)/v*100
            of=intra[["Open","Close","Volume"]]
            sign=(of["Close"]>of["Open"]).astype(int)-(of["Close"]<of["Open"]).astype(int)
            buy=(of["Volume"]*(sign>0)).sum();sell=(of["Volume"]*(sign<0)).sum()
            if buy+sell>0:flowbias=buy/(buy+sell)

        # order-flow filter — disabled for watchlist
        if enable_ofb_filter and not is_watch:
            if flowbias is None or flowbias<min_ofb:return None

        # EMA Trend
        ema10=float(close.ewm(span=10,adjust=False).mean().iloc[-1])
        ema_trend="🔥 Breakout" if price>ema10 and rsi7>55 else "Neutral"

        # Catalyst + Float + Short (unchanged)
        squeeze=False;low_float=False;short_pct_display=None
        sector="Unknown";industry="Unknown";catalyst=False
        if enable_enrichment:
            try:
                info=stock.get_info() or {}
                fs=info.get("floatShares");sp=info.get("shortPercentOfFloat")
                sector=info.get("sector","Unknown");industry=info.get("industry","Unknown")
                low_float=fs and fs<20_000_000;squeeze=sp and sp>0.15
                short_pct_display=round(sp*100,2) if sp else None
            except:pass
            try:
                news=stock.get_news()
                if news and "providerPublishTime" in news[0]:
                    pub=datetime.fromtimestamp(news[0]["providerPublishTime"],tz=timezone.utc)
                    catalyst=(datetime.now(timezone.utc)-pub).days<=3
            except:pass

        # Multi-timeframe label
        mtf="🔻 Not Aligned"
        positives=sum([pm and pm>0 ,m3 and m3>0 ,m10 and m10>0 ])
        if positives==3:mtf="✅ Bullish Multi-TF"
        elif positives==2:mtf="🟢 Lean Bullish"
        elif positives==1:mtf="🟡 Mixed"

        score=short_window_score(pm,yday,m3,m10,rsi7,rvol10,catalyst,squeeze,vwap,flowbias)
        prob=breakout_probability(score)

        return {
            "Symbol":t,"Exchange":sym.get("Exchange","?"),"Price":round(price,2),"Volume":int(vol),
            "Score":score,"Prob_Rise%":prob,"PM%":round(pm,2) if pm else None,
            "YDay%":round(yday,2) if yday else None,"3D%":round(m3,2) if m3 else None,
            "10D%":round(m10,2) if m10 else None,"RSI7":round(rsi7,2),
            "EMA10 Trend":ema_trend,"RVOL_10D":round(rvol10,2) if rvol10 else None,
            "VWAP%":round(vwap,2) if vwap else None,"FlowBias":round(flowbias,2) if flowbias else None,
            "Squeeze?":squeeze,"LowFloat?":low_float,"Short % Float":short_pct_display,
            "Sector":sector,"Industry":industry,"Catalyst":catalyst,"MTF_Trend":mtf,
            "Spark":close,"AI_Commentary":"WATCHLIST PRIORITY" if is_watch else "Algo-scan normal"
        }

    except:
        return {"Symbol":t,"Exchange":"WATCH","Price":None,"Score":None}

# ========================= RUN SCAN — WATCHLIST SKIPS FILTERING =========================
@st.cache_data(ttl=6)
def run_scan(watch,maxu,en,enf,min_ofb,mode,pool):
    uni=build_universe(watch,maxu,mode,pool)
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as ex:
        for f in concurrent.futures.as_completed([ex.submit(scan_one,s,en,enf,min_ofb) for s in uni]):
            if f.result():results.append(f.result())
    return pd.DataFrame(results)

# ========================= MAIN UI =========================
df=run_scan(watchlist_text,max_universe,enable_enrichment,enable_ofb_filter,min_ofb,universe_mode,volume_rank_pool)

if df.empty:
    st.error("No results — but watchlist tickers now ALWAYS appear")
else:
    if not watchlist_text.strip(): # filters OFF when watchlist used
        df=df[df["Score"]>=min_breakout]
        df=df[df["PM%"].fillna(-999)>=min_pm_move]
        df=df[df["YDay%"].fillna(-999)>=min_yday_gain]
        if squeeze_only:df=df[df["Squeeze?"]]
        if catalyst_only:df=df[df["Catalyst"]]
        if vwap_only:df=df[df["VWAP%"].fillna(-999)>0]

    df=df.sort_values(by=["Score","PM%","RSI7"],ascending=[False,False,False])

    st.subheader(f"🔥 Final Output — {len(df)} symbols (full data restored)")
    for _,row in df.iterrows():
        c1,c2,c3,c4=st.columns([2,3,3,3])
        c1.markdown(f"**{row['Symbol']}**");
        c1.write(f"Price: {row['Price']} | Vol: {row['Volume']:,}")
        c1.write(f"Score {row['Score']} | RiseProb {row['Prob_Rise%']}%")
        c1.markdown(row["MTF_Trend"]);c1.write(row["EMA10 Trend"])

        c2.write(f"PM% {row['PM%']}  • YDay% {row['YDay%']}")
        c2.write(f"3D% {row['3D%']}  • 10D% {row['10D%']}")
        c2.write(f"RSI7 {row['RSI7']} • RVOL {row['RVOL_10D']}x")

        c3.write(f"VWAP {row['VWAP%']}%  • Flow {row['FlowBias']}")
        c3.write(f"Squeeze {row['Squeeze?']} • LowFloat {row['LowFloat?']}")
        c3.write(f"{row['Sector']} {row['Industry']}")
        c3.markdown(f"🧠 {row['AI_Commentary']}")

        c4.plotly_chart(go.Figure(data=[go.Scatter(y=row["Spark"].values)]),use_container_width=True)

    # ================= CSV EXPORT =================
    cols=[c for c in ["Symbol","Exchange","Price","Volume","Score","Prob_Rise%","PM%","YDay%","3D%","10D%","RSI7","EMA10 Trend","RVOL_10D","VWAP%","FlowBias","Squeeze?","LowFloat?","Short % Float","Sector","Industry","Catalyst","MTF_Trend","AI_Commentary"] if c in df.columns]
    st.download_button("📥 Download CSV",df[cols].to_csv(index=False),"v9_screen.csv")

st.caption("For research and education only. Not financial advice.  All original UI restored — watchlist now fully overridden.")












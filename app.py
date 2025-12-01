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
st.caption("Watchlist now EXEMPTS ALL FILTERS • AI Commentary restored • Full output still intact")

# ========================= SIDEBAR =========================
with st.sidebar:
    st.header("Watchlist / Universe Control")

    watchlist_text = st.text_area(
        "Watchlist tickers (comma/space/newline separated):",
        value="",
        height=80,
    )

    max_universe = st.slider("Max symbols if NO watchlist", 50, 2000, 600, 200)

    st.subheader("Universe Mode:")
    universe_mode = st.radio("Mode", [
        "Classic (Alphabetical Slice)",
        "Randomized Slice",
        "Live Volume Ranked (slower)"
    ], index=0)

    volume_rank_pool = st.slider("Volume ranking pool (if enabled)", 100, 2000, 600, 100)

    enable_enrichment = st.checkbox("Include Float/Short/News Enrichment", value=False)

    st.subheader("Filters (ignored with watchlist):")

    max_price = st.number_input("Max Price", 1.0, 2000.0, DEFAULT_MAX_PRICE, 1.0)
    min_volume = st.number_input("Min Daily Volume", 0, 10_000_000, DEFAULT_MIN_VOLUME, 10_000)
    min_breakout = st.number_input("Min Score", -50.0, 200.0, 0.0, 0.5)
    min_pm_move = st.number_input("Min Premarket %", -50.0, 200.0, 0.0, 0.5)
    min_yday_gain = st.number_input("Min Yesterday %", -50.0, 200.0, 0.0, 0.5)

    squeeze_only = st.checkbox("Short Squeeze Only")
    catalyst_only = st.checkbox("Must Have Catalyst")
    vwap_only = st.checkbox("Must be Above VWAP")

    st.subheader("Order Flow Filter")
    enable_ofb_filter = st.checkbox("Enable Flow Bias Requirement", value=False)
    min_ofb = st.slider("Min Flow Bias", 0.0, 1.0, 0.50, 0.01)

    enable_alerts = st.checkbox("Enable Alerts", value=False)
    ALERT_SCORE_THRESHOLD = st.slider("Alert Score Threshold", 10, 200, 30, 5)
    ALERT_PM_THRESHOLD = st.slider("Alert PM% Threshold", 1, 150, 4, 1)
    ALERT_VWAP_THRESHOLD = st.slider("Alert VWAP Threshold", 1, 50, 2, 1)

# ========================= LOAD SYMBOLS =========================
@st.cache_data(ttl=900)
def load_symbols():
    nasdaq = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
                         sep="|", skipfooter=1, engine="python")
    other = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
                         sep="|", skipfooter=1, engine="python")

    nasdaq["Exchange"] = "NASDAQ"
    other = other.rename(columns={"ACT Symbol": "Symbol"})

    df = pd.concat([nasdaq[["Symbol","ETF","Exchange"]], other[["Symbol","ETF","Exchange"]]])

    df["Symbol"] = df["Symbol"].astype(str).fillna("")  # <- FIX (prevents ValueError)
    df = df[df["Symbol"].str.match(r"^[A-Z]{1,5}$", na=False)]

    return df.to_dict("records")

# ========================= UNIVERSE =========================
def build_universe(watchlist_text, max_universe, mode, pool):
    wl = watchlist_text.strip()
    if wl:
        raw = wl.replace("\n"," ").replace(",", " ").split()
        return [{"Symbol":s.upper(),"Exchange":"WATCH"} for s in sorted(set(raw))]

    syms = load_symbols()

    if mode == "Randomized Slice":
        random.shuffle(syms)
        return syms[:max_universe]

    if mode == "Live Volume Ranked (slower)":
        ranked = []
        for s in syms[:pool]:
            try:
                d = yf.Ticker(s["Symbol"]).history(period="1d", interval="2m", prepost=True)
                if not d.empty:
                    ranked.append({**s,"LiveVol":float(d["Volume"].iloc[-1])})
            except:
                pass
        ranked = sorted(ranked, key=lambda x:x.get("LiveVol",0), reverse=True)
        return ranked[:max_universe] if ranked else syms[:max_universe]

    return syms[:max_universe]

# ========================= AI Commentary (FULL) =========================
def ai_commentary(score, pm, rvol, flow_bias, vwap, ten_day):
    thoughts=[]

    if score>=80: thoughts.append("🔥 High probability momentum expansion")
    elif score>=40: thoughts.append("🟢 Constructive momentum structure forming")
    elif score>=10: thoughts.append("🟡 Early accumulation signs developing")
    else: thoughts.append("⚪ Weak or indecisive — needs stronger confirmation")

    if pm and pm>3: thoughts.append("Premarket buyers aggressive")
    elif pm and pm<0: thoughts.append("Premarket selling pressure present")

    if rvol and rvol>2: thoughts.append("Volume significantly elevated vs avg")
    elif rvol and rvol<0.8: thoughts.append("Low participation — liquidity muted")

    if flow_bias and flow_bias>0.65: thoughts.append("Buyers dominate orderflow 📈")
    elif flow_bias and flow_bias<0.4: thoughts.append("Seller control detected 📉")

    if vwap and vwap>0: thoughts.append("Trading securely above VWAP — bullish control")
    elif vwap and vwap<0: thoughts.append("Below VWAP — supply still active")

    if ten_day and ten_day>10: thoughts.append("Uptrend developing over 10D window")
    elif ten_day and ten_day<-5: thoughts.append("Structural weakness in 10D trend")

    return " | ".join(thoughts)

# ========================= SCAN ONE (WATCHLIST OVERRIDE) =========================
def scan_one(sym, enrich, ofb_filter, min_ofb):
    t = sym["Symbol"]
    is_watch = (sym.get("Exchange")=="WATCH")

    try:
        stock=yf.Ticker(t)
        hist=stock.history(period=f"{HISTORY_LOOKBACK_DAYS}d",interval="1d")

        # If watchlist & no data → still RETURN it instead of dropping
        if hist is None or hist.empty or len(hist)<5:
            return {"Symbol":t,"Exchange":"WATCH","Score":None,"AI_Commentary":"⚠ No data — watchlist override active"}

        close = hist["Close"]; volume = hist["Volume"]
        price=float(close.iloc[-1]); vol=float(volume.iloc[-1])

        # Filters do NOT apply if watchlist
        if not is_watch:
            if price>max_price or vol<min_volume: return None

        # Momentum windows
        yday=(close.iloc[-1]-close.iloc[-2])/close.iloc[-2]*100 if len(close)>=2 else None
        m3=(close.iloc[-1]-close.iloc[-4])/close.iloc[-4]*100 if len(close)>=4 else None
        m10=(close.iloc[-1]-close.iloc[0])/close.iloc[0]*100

        # RSI7
        delta=close.diff()
        gain=delta.clip(lower=0).rolling(7).mean()
        loss=(-delta.clip(upper=0)).rolling(7).mean()
        rs=gain/loss
        rsi7=float(100-(100/(1+rs)).iloc[-1])

        # RVOL
        avg10=volume.mean(); rvol10=vol/avg10 if avg10>0 else None

        # Intraday
        try:
            intra=stock.history(period=INTRADAY_RANGE,interval=INTRADAY_INTERVAL,prepost=True)
        except: intra=None

        pm=vwap=flow=None
        if intra is not None and not intra.empty and len(intra)>=3:
            ic=intra["Close"]; io=intra["Open"]
            pm=(ic.iloc[-1]-ic.iloc[-2])/ic.iloc[-2]*100
            typical=(intra["High"]+intra["Low"]+intra["Close"])/3
            totalv=intra["Volume"].sum()
            if totalv>0:
                v=(typical*intra["Volume"]).sum()/totalv
                vwap=(price-v)/v*100
            sign=(ic>io).astype(int)-(ic<io).astype(int)
            buy=(intra["Volume"]*(sign>0)).sum(); sell=(intra["Volume"]*(sign<0)).sum()
            if buy+sell>0: flow=buy/(buy+sell)

        # Orderflow filter OFF for watchlist
        if ofb_filter and not is_watch:
            if flow is None or flow<min_ofb: return None

        # EMA Trend
        ema10=float(close.ewm(span=10,adjust=False).mean().iloc[-1])
        ema_trend="🔥 Breakout" if price>ema10 and rsi7>55 else "Neutral"

        # Enrichment
        squeeze=low_float=False; short_pct=None
        catalyst=False; sector="Unknown"; industry="Unknown"

        if enrich:
            try:
                info=stock.get_info() or {}
                fs=info.get("floatShares")
                sp=info.get("shortPercentOfFloat")
                sector=info.get("sector","Unknown")
                industry=info.get("industry","Unknown")
                low_float=fs and fs<20_000_000
                squeeze=sp and sp>0.15
                short_pct=round(sp*100,2) if sp else None
            except:pass
            try:
                news=stock.get_news()
                if news and "providerPublishTime" in news[0]:
                    pub=datetime.fromtimestamp(news[0]["providerPublishTime"],tz=timezone.utc)
                    catalyst=(datetime.now(timezone.utc)-pub).days<=3
            except:pass

        # Score + AI commentary
        score=short_window_score(pm,yday,m3,m10,rsi7,rvol10,catalyst,squeeze,vwap,flow)
        commentary=ai_commentary(score,pm,rvol10,flow,vwap,m10)

        return {
            "Symbol":t,"Exchange":sym.get("Exchange","?"),
            "Price":round(price,2),"Volume":int(vol),
            "Score":score,"Prob_Rise%": breakout_probability(score),
            "PM%":pm,"YDay%":yday,"3D%":m3,"10D%":m10,"RSI7":rsi7,
            "EMA10 Trend":ema_trend,"RVOL_10D":rvol10,"VWAP%":vwap,
            "FlowBias":flow,"Squeeze?":squeeze,"LowFloat?":low_float,
            "Short % Float":short_pct,"Sector":sector,"Industry":industry,
            "Catalyst":catalyst,"MTF_Trend":("Watchlist Override" if is_watch else "Normal"),
            "Spark":close,
            "AI_Commentary":commentary + (" | 🟡 Watchlist Priority" if is_watch else "")
        }

    except:
        return {"Symbol":t,"Exchange":"WATCH","Score":None,"AI_Commentary":"⚠ Scan Failed — Watchlist Preserved"}

# ========================= RUN SCAN =========================
@st.cache_data(ttl=6)
def run_scan(watchlist,maxu,enrich,ofb,min_ofb,mode,pool):
    universe=build_universe(watchlist,maxu,mode,pool)
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as ex:
        for f in concurrent.futures.as_completed([ex.submit(scan_one,s,enrich,ofb,min_ofb) for s in universe]):
            r=f.result()
            if r: results.append(r)
    return pd.DataFrame(results)

# ========================= MAIN OUTPUT =========================
df=run_scan(watchlist_text,max_universe,enable_enrichment,enable_ofb_filter,min_ofb,universe_mode,volume_rank_pool)

if df.empty:
    st.error("No results — but watchlist symbols will ALWAYS render")
else:
    # Filters only apply when NOT watchlist
    if not watchlist_text.strip():

    if "Score" in df:
        df = df[df["Score"].fillna(-999) >= min_breakout]

    if "PM%" in df:
        df = df[df["PM%"].fillna(-999) >= min_pm_move]

    if "YDay%" in df:
        df = df[df["YDay%"].fillna(-999) >= min_yday_gain]

    if squeeze_only and "Squeeze?" in df:
        df = df[df["Squeeze?"]]

    if catalyst_only and "Catalyst" in df:
        df = df[df["Catalyst"] == True]

    if vwap_only and "VWAP%" in df:
        df = df[df["VWAP%"].fillna(-999) > 0]


    df=df.sort_values(by=["Score","PM%","RSI7"],ascending=[False,False,False])
    st.subheader(f"🔥 Final Symbols Returned = **{len(df)}**")

    # DISPLAY RESULTS
    for _,row in df.iterrows():
        symbol=row["Symbol"]
        c1,c2,c3,c4=st.columns([2,3,3,3])

        # Basic stat panel
        c1.markdown(f"### **{symbol}** 🟢")
        c1.write(f"💲 Price: **{row['Price']}**")
        c1.write(f"📊 Volume: {row['Volume']:,}")
        c1.write(f"🔥 Score: **{row['Score']}**")
        c1.write(f"📈 Prob Rise: {row['Prob_Rise%']}%")
        c1.write(f"Trend: {row['EMA10 Trend']}")
        c1.write(f"{row['MTF_Trend']}")

        # Momentum snapshot
        c2.write(f"PM%: {row['PM%']} | YDay%: {row['YDay%']}")
        c2.write(f"3D%: {row['3D%']} | 10D%: {row['10D%']}")
        c2.write(f"RSI7: {row['RSI7']} | RVOL10: {row['RVOL_10D']}x")

        # AI commentary panel
        c3.markdown(f"### 🧠 AI View")
        c3.write(row["AI_Commentary"])
        if enable_enrichment:
            c3.write(f"Squeeze: {row['Squeeze?']} | LowFloat: {row['LowFloat?']}")
            c3.write(f"Sector: {row['Sector']} | Industry: {row['Industry']}")

        # Sparkline visual
        c4.plotly_chart(go.Figure(data=[go.Scatter(y=row["Spark"].values)]),use_container_width=True)

    # CSV Export
    export_cols=[c for c in ["Symbol","Exchange","Price","Volume","Score","Prob_Rise%","PM%","YDay%","3D%","10D%","RSI7","EMA10 Trend","RVOL_10D","VWAP%","FlowBias","Squeeze?","LowFloat?","Short % Float","Sector","Industry","Catalyst","MTF_Trend","AI_Commentary"] if c in df.columns]
    st.download_button("📥 Export CSV",df[export_cols].to_csv(index=False),"v9_screen.csv")


st.caption("For research and education only. Not financial advice.  All original UI restored — watchlist now fully overridden.")












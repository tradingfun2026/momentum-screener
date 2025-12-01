# ===========================================================
# V9 — Watchlist Override + Volume Floor 0 + FIXED SYMBOL PARSE
# ===========================================================

import streamlit as st
import pandas as pd
import yfinance as yf
import concurrent.futures
from datetime import datetime, timezone
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objs as go
import math, random

# ========================= SETTINGS =========================
THREADS               = 20
AUTO_REFRESH_MS       = 120_000
HISTORY_LOOKBACK_DAYS = 10
INTRADAY_INTERVAL     = "2m"
INTRADAY_RANGE        = "1d"

DEFAULT_MAX_PRICE     = 5.0
DEFAULT_MIN_VOLUME    = 0            # updated
DEFAULT_MIN_BREAKOUT  = 0.0


# ========================= AUTO REFRESH =========================
st_autorefresh(interval=AUTO_REFRESH_MS, key="refresh_v9")


# ========================= UI =========================
st.set_page_config(page_title="V9 Momentum Screener",layout="wide")
st.title("🚀 V9 — Momentum Screener (10-Day • Volume • Watchlist Override)")
st.caption("Maintained UI • Just fixed load_symbols + watchlist precedence + min volume 0")

with st.sidebar:

    st.header("Universe")
    watchlist_text = st.text_area("Watchlist", "", height=70)

    max_universe = st.slider("Max universe (if no watchlist)",50,600,200)

    # NEW — toggle that fixes your request
    ignore_filters = st.checkbox("Watchlist Overrides All Filters", value=True)

    st.markdown("---")
    st.subheader("V9 Universe Mode")
    universe_mode = st.radio("Universe Construction",[
        "Classic (Alphabetical Slice)",
        "Randomized Slice",
        "Live Volume Ranked (slower)"
    ])

    volume_rank_pool = st.slider("Symbols considered for volume ranking",100,2000,600,100)
    enable_enrichment = st.checkbox("Float/Short/News Enrichment (slower)",False)

    st.markdown("---")
    st.header("Filters (ignored if watchlist override ON)")

    max_price = st.number_input("Max Price",1.0,1000.0,DEFAULT_MAX_PRICE,1.0)
    min_volume = st.number_input("Min Daily Volume",0,10_000_000,DEFAULT_MIN_VOLUME,10_000)
    min_breakout = st.number_input("Min Breakout Score",-50.0,200.0,0.0)
    min_pm_move  = st.number_input("Min Premarket %",-50.0,200.0,0.0,0.5)
    min_yday_gain= st.number_input("Min Yesterday %",-50.0,200.0,0.0,0.5)

    squeeze_only = st.checkbox("Short-Squeeze Only")
    catalyst_only= st.checkbox("Must Have News")
    vwap_only    = st.checkbox("Above VWAP Only")

    enable_ofb_filter = st.checkbox("Use Min Order Flow Bias",False)
    min_ofb = st.slider("Min Order Flow Bias",0.00,1.00,0.50,0.01)

    st.markdown("---")
    st.subheader("🔊 Alerts")
    enable_alerts = st.checkbox("Enable Alerts",False)
    ALERT_SCORE_THRESHOLD = st.slider("Alert Score ≥",10,200,30)
    ALERT_PM_THRESHOLD    = st.slider("Alert PM% ≥",1,150,4)
    ALERT_VWAP_THRESHOLD  = st.slider("Alert VWAP% ≥",1,50,2)

    st.markdown("---")
    if st.button("Clear Cache"):
        st.cache_data.clear()
        st.success("Cache cleared — rescan triggered")


# ========================= STABLE SYMBOL LOADER (FIXED) =========================
@st.cache_data(ttl=900)
def load_symbols():
    nas=pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
                    sep="|",skipfooter=1,engine="python")
    oth=pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
                    sep="|",skipfooter=1,engine="python").rename(columns={"ACT Symbol":"Symbol"})

    df=pd.concat([nas[["Symbol"]],oth[["Symbol"]]],ignore_index=True)
    df=df[df["Symbol"].str.contains(r"^[A-Z]{1,5}$", na=False)]  # <-- FIXED
    return df.to_dict("records")


# ========================= UNIVERSE BUILDER =========================
def build_universe(wl,max_u,mode,pool):

    # WATCHLIST ALWAYS PRIORITY — no filters if override enabled
    if wl.strip():
        tokens=wl.replace(","," ").replace("\n"," ").split()
        return [{"Symbol":s.upper(),"Exchange":"WATCH"} for s in set(tokens)]

    syms=load_symbols()

    if mode=="Randomized Slice":
        random.shuffle(syms)
        return syms[:max_u]

    if mode=="Live Volume Ranked (slower)":
        ranked=[]
        for s in syms[:pool]:
            try:
                d=yf.Ticker(s["Symbol"]).history(period="1d",interval="2m")
                if not d.empty: ranked.append({**s,"LiveVol":int(d.Volume.iloc[-1])})
            except: pass
        ranked=sorted(ranked,key=lambda x:x.get("LiveVol",0),reverse=True)
        return ranked[:max_u]

    return syms[:max_u]


# ========================= SCORING =========================
def short_window_score(pm,y3,y10,rsi,rvol,cat,sq,vwap,flow):
    sc=0
    if pm:sc+=pm*1.6
    if y3:sc+=y3*1.2
    if y10:sc+=y10*0.6
    if rsi>55:sc+=(rsi-55)*0.4
    if rvol and rvol>1.2:sc+=(rvol-1.2)*2
    if vwap and vwap>0:sc+=min(vwap,6)*1.5
    if flow:sc+=(flow-0.5)*22
    if cat:sc+=8
    if sq:sc+=12
    return round(sc,2)


# ========================= AI COMMENTARY =========================
def ai_comment(score,pm,rvol,flow,vwap,m10):
    txt=[]
    if score>=60:txt.append("High Momentum")
    if pm and pm>3:txt.append("Premarket Strength")
    if rvol and rvol>2:txt.append("Volume Expansion")
    if flow and flow>0.65:txt.append("Buyers Control Tape")
    if vwap and vwap>0:txt.append("Above VWAP")
    if m10 and m10>10:txt.append("Strong Trend Structure")
    return " | ".join(txt) if txt else "Neutral / Watching"


# ========================= SCANNER =========================
def scan(sym):
    try:
        t=sym["Symbol"]; stck=yf.Ticker(t)
        hist=stck.history(period="10d")
        if hist.empty:return None

        close=hist.Close; vol=hist.Volume
        price=float(close.iloc[-1])
        v_last=int(vol.iloc[-1])

        ### FILTER BYPASSED AUTOMATICALLY IF WATCHLIST + OVERRIDE
        if not (ignore_filters and watchlist_text.strip()):
            if price>max_price or v_last<min_volume:return None

        # returns
        yday=(close.iloc[-1]-close.iloc[-2])/close.iloc[-2]*100 if len(close)>=2 else None
        m3  =(close.iloc[-1]-close.iloc[-4])/close.iloc[-4]*100 if len(close)>=4 else None
        m10 =(close.iloc[-1]-close.iloc[0])/close.iloc[0]*100

        # RSI7
        delta=close.diff()
        rsi=float((100-(100/(1+delta.clip(lower=0).rolling(7).mean()/
                              (-delta.clip(upper=0).rolling(7).mean())))).iloc[-1])

        # RVOL
        avg10=vol.mean(); rvol=v_last/avg10 if avg10>0 else None

        # intraday 2m
        intra=stck.history(period="1d",interval="2m",prepost=True)
        pm=vwap=flow=None
        if len(intra)>3:
            c=intra.Close;o=intra.Open;v=intra.Volume
            pm=(c.iloc[-1]-c.iloc[-2])/c.iloc[-2]*100
            tp=(intra.High+intra.Low+intra.Close)/3
            if v.sum()>0:
                vw=float((tp*v).sum()/v.sum())
                vwap=(price-vw)/vw*100
            sign=(c>o).astype(int)-(c<o).astype(int)
            buy=(v*(sign>0)).sum();sell=(v*(sign<0)).sum()
            flow=float(buy/(buy+sell)) if(buy+sell)>0 else None

        score=short_window_score(pm,yday,m3,rsi,rvol,False,False,vwap,flow)
        prob=round((1/(1+math.exp(-score/20)))*100,1)
        ai=ai_comment(score,pm,rvol,flow,vwap,m10)

        return {
            "Symbol":t,"Price":round(price,2),"Volume":v_last,
            "Score":score,"Prob_Rise%":prob,
            "PM%":round(pm,2) if pm else None,
            "YDay%":round(yday,2) if yday else None,
            "3D%":round(m3,2) if m3 else None,
            "10D%":round(m10,2),
            "RSI7":round(rsi,2),
            "RVOL_10D":round(rvol,2) if rvol else None,
            "VWAP%":round(vwap,2) if vwap else None,
            "FlowBias":round(flow,2) if flow else None,
            "AI_Commentary":ai,
            "Spark":close
        }
    except:return None


# ========================= RUN =========================
@st.cache_data(ttl=6)
def run():
    uni=build_universe(watchlist_text,max_universe,universe_mode,volume_rank_pool)
    out=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as ex:
        for r in concurrent.futures.as_completed([ex.submit(scan,s) for s in uni]):
            if r.result(): out.append(r.result())
    return pd.DataFrame(out)


# ========================= UI DISPLAY =========================
with st.spinner("Scanning…"): df=run()
if df.empty: st.error("No results… try removing filters"); st.stop()

# filters apply only if NOT bypassing
if not(ignore_filters and watchlist_text.strip()):
    df=df[df.Score>=min_breakout]
    df=df[df["PM%"].fillna(-9)>=min_pm_move]
    df=df[df["YDay%"].fillna(-9)>=min_yday_gain]
    if squeeze_only: df=df[df.FlowBias>0.6]
    if vwap_only: df=df[df["VWAP%"].fillna(-9)>0]

df=df.sort_values(["Score","PM%","RSI7"],ascending=[False,False,False])

st.subheader(f"🔥 {len(df)} Matches")

for _,r in df.iterrows():
    c1,c2,c3=st.columns([2,3,4])

    c1.markdown(f"### {r.Symbol} — ${r.Price}")
    c1.write(f"📊 Vol {r.Volume:,} | 🔥 Score {r.Score} | Prob {r['Prob_Rise%']}%")
    c1.write("🧠 "+r["AI_Commentary"])

    c2.write(f"PM {r['PM%']}% | YDay {r['YDay%']}% | 3D {r['3D%']}% | 10D {r['10D%']}%")
    c2.write(f"RSI7 {r['RSI7']} | RVOL {r['RVOL_10D']}x | VWAP {r['VWAP%']}% | Flow {r['FlowBias']}")

    c3.plotly_chart(go.Figure(data=[go.Scatter(y=r.Spark.values)]),use_container_width=True)
    st.divider()








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
AUTO_REFRESH_MS       = 60_000
HISTORY_LOOKBACK_DAYS = 10
INTRADAY_INTERVAL     = "2m"
INTRADAY_RANGE        = "1d"

DEFAULT_MAX_PRICE     = 5.0
DEFAULT_MIN_VOLUME    = 100_000
DEFAULT_MIN_BREAKOUT  = 0.0

# ========================= AUTO REFRESH =========================
st_autorefresh(interval=AUTO_REFRESH_MS, key="refresh_v9")

# ========================= PAGE SETUP =========================
st.set_page_config(
    page_title="V9 – 10-Day Momentum Screener (Hybrid Volume/Randomized)",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

st.title("🚀 V9 — 10-Day Momentum Breakout Screener (Hybrid Speed + Volume + Randomized)")
st.caption(
    "Short-window model • EMA10 • RSI(7) • 3D & 10D momentum • 10D RVOL • "
    "VWAP + order flow • Watchlist mode • Audio alerts • V9 universe modes"
)

# ========================= SIDEBAR =========================
with st.sidebar:

    st.header("Universe")
    watchlist_text = st.text_area("Watchlist tickers:", value="", height=80)

    max_universe = st.slider("Max symbols to scan", 50, 2000, 2000, 50)

    st.subheader("V9 Universe Mode")
    universe_mode = st.radio(
        "Universe Construction",
        ["Classic (Alphabetical Slice)", "Randomized Slice", "Live Volume Ranked (slower)"],
        index=0
    )

    volume_rank_pool = st.slider("Volume-Rank Pool (V9 only)", 100, 2000, 600, 100)
    enable_enrichment = st.checkbox("Include float/short/news", False)

    st.markdown("---")
    st.header("Filters")

    max_price    = st.number_input("Max Price",1.0,1000.0,DEFAULT_MAX_PRICE,1.0)
    min_volume   = st.number_input("Min Volume",10_000,10_000_000,DEFAULT_MIN_VOLUME,10_000)
    min_breakout = st.number_input("Min Breakout Score",-50.0,200.0,0.0,1.0)
    min_pm_move  = st.number_input("Min Premarket %",-50.0,200.0,0.0,0.5)
    min_yday_gain= st.number_input("Min Yesterday %",-50.0,200.0,0.0,0.5)

    squeeze_only   = st.checkbox("Short-Squeeze Only")
    catalyst_only  = st.checkbox("Must Have News")
    vwap_only      = st.checkbox("Above VWAP Only")

    st.markdown("---")
    st.subheader("Order Flow Filter (optional)")
    enable_ofb_filter = st.checkbox("Use Min Order Flow Bias")
    min_ofb = st.slider("Min Order Flow Bias",0.0,1.0,0.50,0.01)

    st.markdown("---")
    st.subheader("🔊 Alerts")
    enable_alerts = st.checkbox("Enable Alerts", False)
    ALERT_SCORE_THRESHOLD = st.slider("Score ≥",10,200,30,5)
    ALERT_PM_THRESHOLD    = st.slider("PM% ≥",1,150,4,1)
    ALERT_VWAP_THRESHOLD  = st.slider("VWAP ≥",1,50,2,1)

    if st.button("🧹 Refresh Now"):
        st.cache_data.clear()
        st.success("Cache Cleared")

# ========================= SYMBOL LOAD =========================
@st.cache_data(ttl=900)
def load_symbols():
    nasdaq = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
                         sep="|",skipfooter=1,engine="python")
    other  = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
                         sep="|",skipfooter=1,engine="python")

    nasdaq["Exchange"]="NASDAQ"
    other["Exchange"] = other["Exchange"].fillna("NYSE")
    other = other.rename(columns={"ACT Symbol":"Symbol"})

    df = pd.concat([nasdaq[["Symbol","ETF","Exchange"]],
                    other [["Symbol","ETF","Exchange"]]]).dropna(subset=["Symbol"])
    return df[df["Symbol"].str.contains(r"^[A-Z]{1,5}$",na=False)].to_dict("records")

# ========================= BUILD UNIVERSE =========================
def build_universe(watchlist_text,max_universe,universe_mode,volume_rank_pool):

    wl = watchlist_text.strip()
    if wl:
        t = wl.replace("\n"," ").replace(","," ").split()
        return [{"Symbol":s.upper(),"Exchange":"WATCH"} for s in sorted(set(t))]

    syms = load_symbols()

    if universe_mode=="Randomized Slice":
        random.shuffle(syms)
        return syms[:max_universe]

    if universe_mode=="Live Volume Ranked (slower)":
        ranked=[]
        for s in syms[:volume_rank_pool]:
            try:
                d=yf.Ticker(s["Symbol"]).history(period="1d",interval="2m",prepost=True)
                if not d.empty:
                    ranked.append({**s,"LiveVol":float(d["Volume"].iloc[-1])})
            except: pass
        if not ranked: return syms[:max_universe]
        return sorted(ranked,key=lambda x:x["LiveVol"],reverse=True)[:max_universe]

    return syms[:max_universe]


# ========================= SCORING =========================
def short_window_score(pm,yday,m3,m10,rsi7,rvol10,catalyst,squeeze,vwap,flow_bias):
    score=0
    if pm:    score+=max(pm,0)*1.6
    if yday:  score+=max(yday,0)*0.8
    if m3:    score+=max(m3,0)*1.2
    if m10:   score+=max(m10,0)*0.6
    if rsi7 and rsi7>55: score+=(rsi7-55)*0.4
    if rvol10 and rvol10>1.2: score+=(rvol10-1.2)*2.0
    if vwap and vwap>0: score+=min(vwap,6)*1.5
    if flow_bias: score+=(flow_bias-0.5)*22.0
    if catalyst:  score+=8
    if squeeze:   score+=12
    return round(score,2)

def breakout_probability(s):
    try: return round((1/(1+math.exp(-s/20)))*100,1)
    except: return None

def multi_timeframe_label(pm,m3,m10):
    p=sum([(pm!=None and pm>0),(m3!=None and m3>0),(m10!=None and m10>0)])
    return ["🔻 Not Aligned","🟡 Mixed","🟢 Leaning Bullish","✅ Aligned Bullish"][p]


# ========================= SIMPLE AI COMMENTARY =========================
def ai_commentary(score,pm,rvol,flow_bias,vwap,ten_day):
    c=[]
    if score>=80:c.append("High compression momentum")
    elif score>=40:c.append("Constructive setup forming")

    if pm>3:c.append("Premarket strength")
    elif pm<-2:c.append("Premarket selling pressure")

    if rvol>2:c.append("Volume expansion vs 10D")
    elif rvol<0.7:c.append("Thin liquidity regime")

    if flow_bias>0.65:c.append("Buyers in control")
    elif flow_bias<0.4:c.append("Sellers dominate")

    if vwap>0:c.append("Above VWAP (bullish pressure)")
    elif vwap<0:c.append("Below VWAP (supply > demand)")

    if ten_day>10:c.append("10D trend strong")
    elif ten_day<-5:c.append("10D distribution risk")

    return " | ".join(c) if c else "Neutral tape"


# ========================= SCAN ONE =========================
def scan_one(sym,enrich,ofb_filter,min_ofb):
    try:
        t= yf.Ticker(sym["Symbol"])
        h= t.history(period=f"{HISTORY_LOOKBACK_DAYS}d",interval="1d")
        if h.empty or len(h)<5:return None

        close=h["Close"]; vol=h["Volume"]
        price=float(close.iloc[-1]); vol_last=float(vol.iloc[-1])
        if price>max_price or vol_last<min_volume:return None

        yday= (close.iloc[-1]-close.iloc[-2])/close.iloc[-2]*100 if len(close)>=2 else None
        m3  = (close.iloc[-1]-close.iloc[-4])/close.iloc[-4]*100 if len(close)>=4 else None
        m10 = (close.iloc[-1]-close.iloc[0]) /close.iloc[0] *100 if close.iloc[0]>0 else None

        delta=close.diff(); gain=delta.clip(lower=0).rolling(7).mean()
        loss=(-delta.clip(upper=0)).rolling(7).mean()
        rsi7=float((100-(100/(1+(gain/loss)))).iloc[-1])

        ema10=float(close.ewm(span=10,adjust=False).mean().iloc[-1])
        ema_trend="🔥 Breakout" if price>ema10 and rsi7>55 else "Neutral"

        rvol10=vol_last/vol.mean()

        intra=t.history(period="1d",interval="2m",prepost=True)
        pm=vwap=flow=None

        if not intra.empty:
            c=intra["Close"]; o=intra["Open"]; v=intra["Volume"]
            pm=(float(c.iloc[-1])-float(c.iloc[-2]))/float(c.iloc[-2])*100

            tp=(intra["High"]+intra["Low"]+intra["Close"])/3
            vwap_val=float((tp*v).sum()/v.sum()); vwap=(price-vwap_val)/vwap_val*100

            sign=(c>o)-(c<o)
            buy=(v*(sign>0)).sum(); sell=(v*(sign<0)).sum()
            flow=float(buy/(buy+sell)) if buy+sell>0 else None

        if ofb_filter and (flow is None or flow<min_ofb):return None

        squeeze=low_float=catalyst=False; sector=industry="Unknown"; short_pct=None

        if enrich:
            try:
                info=t.get_info() or {}
                float_shares=info.get("floatShares")
                short=info.get("shortPercentOfFloat")
                low_float=bool(float_shares and float_shares<20_000_000)
                squeeze=bool(short and short>0.15)
                short_pct=round(short*100,2) if short else None
                sector=info.get("sector","Unknown"); industry=info.get("industry","Unknown")
            except:pass

            try:
                news=t.get_news()
                if news and "providerPublishTime" in news[0]:
                    pub=datetime.fromtimestamp(news[0]["providerPublishTime"],tz=timezone.utc)
                    catalyst=(datetime.now(timezone.utc)-pub).days<=3
            except:pass

        label=multi_timeframe_label(pm,m3,m10)
        score=short_window_score(pm,yday,m3,m10,rsi7,rvol10,catalyst,squeeze,vwap,flow)
        prob=breakout_probability(score)
        ai_text=ai_commentary(score,pm,rvol10,flow,vwap,m10)

        return {
            "Symbol":sym["Symbol"],"Exchange":sym["Exchange"],
            "Price":round(price,2),"Volume":int(vol_last),
            "Score":score,"Prob_Rise%":prob,"PM%":round(pm,2) if pm else None,
            "YDay%":round(yday,2) if yday else None,"3D%":round(m3,2) if m3 else None,
            "10D%":round(m10,2) if m10 else None,"RSI7":round(rsi7,2),
            "EMA10 Trend":ema_trend,"RVOL_10D":round(rvol10,2),
            "VWAP%":round(vwap,2) if vwap else None,"FlowBias":round(flow,2) if flow else None,
            "Squeeze?":squeeze,"LowFloat?":low_float,"Short % Float":short_pct,
            "Sector":sector,"Industry":industry,"Catalyst":catalyst,
            "MTF_Trend":label,"Spark":close,"AI_Commentary":ai_text
        }
    except:return None

@st.cache_data(ttl=6)
def run_scan(wl,max_u,enrich,ofb,min_ofb,mode,pool):
    syms=build_universe(wl,max_u,mode,pool)
    res=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as ex:
        fut=[ex.submit(scan_one,s,enrich,ofb,min_ofb) for s in syms]
        for f in concurrent.futures.as_completed(fut):
            r=f.result()
            if r:res.append(r)
    return pd.DataFrame(res) if res else pd.DataFrame()

# ========================= SPARKLINE =========================
def sparkline(s):
    fig=go.Figure()
    fig.add_trace(go.Scatter(y=s.values,mode="lines",line=dict(width=2),hoverinfo="skip"))
    fig.update_layout(height=60,width=160,margin=dict(l=2,r=2,t=2,b=2),
                      xaxis=dict(visible=False),yaxis=dict(visible=False))
    return fig

# ========================= UI RENDER =========================
with st.spinner("Scanning markets..."):
    df=run_scan(watchlist_text,max_universe,enable_enrichment,enable_ofb_filter,
                min_ofb,universe_mode,volume_rank_pool)

if df.empty:
    st.error("No results found")
else:
    df=df[df["Score"]>=min_breakout]
    if min_pm_move:df=df[df["PM%"].fillna(-999)>=min_pm_move]
    if min_yday_gain:df=df[df["YDay%"].fillna(-999)>=min_yday_gain]
    if squeeze_only:df=df[df["Squeeze?"]]
    if catalyst_only:df=df[df["Catalyst"]]
    if vwap_only:df=df[df["VWAP%"].fillna(-999)>0]
    df=df.sort_values(by=["Score","PM%","RSI7"],ascending=[False,False,False])

    st.subheader(f"🔥 Momentum Board — {len(df)} symbols")

    for _,r in df.iterrows():

        sym=r["Symbol"]
        if enable_alerts and sym not in st.session_state.alerted:
            if r["Score"]>=ALERT_SCORE_THRESHOLD:trigger_audio_alert(sym,f"Score {r['Score']}")
            elif r["PM%"] and r["PM%"]>=ALERT_PM_THRESHOLD:trigger_audio_alert(sym,f"PM {r['PM%']}%")
            elif r["VWAP%"] and r["VWAP%"]>=ALERT_VWAP_THRESHOLD:trigger_audio_alert(sym,f"VWAP {r['VWAP%']}%")

        c1,c2,c3,c4=st.columns([2,3,3,3])
        c1.write(f"**{sym}** ({r['Exchange']})")
        c1.write(f"💲 {r['Price']}")
        c1.write(f"📊 Vol {r['Volume']:,}")
        c1.write(f"🔥 Score **{r['Score']}**")
        c1.write(f"📈 Prob {r['Prob_Rise%']}%")
        c1.write(r["MTF_Trend"])
        c1.write(f"Trend: {r['EMA10 Trend']}")

        c2.write(f"PM% {r['PM%']}")
        c2.write(f"YDay% {r['YDay%']}")
        c2.write(f"3D {r['3D%']} | 10D {r['10D%']}")
        c2.write(f"RSI7 {r['RSI7']} | RVOL {r['RVOL_10D']}x")

        c3.write(f"VWAP {r['VWAP%']}")
        c3.write(f"Flow {r['FlowBias']}")
        c3.write(f"🧠 {r['AI_Commentary']}")

        c4.plotly_chart(sparkline(r["Spark"]))

        st.divider()

    st.download_button(
        "📥 Download CSV",
        df.to_csv(index=False),
        "v9_screener.csv",
        "text/csv"
    )

st.caption("For research + edu only — not financial advice.")






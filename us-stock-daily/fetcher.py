import time, logging; from typing import Optional, Tuple
import pandas as pd, yfinance as yf, requests
from config import TICKERS, YFINANCE_VIX_TICKER, CNN_FNG_URL, FETCH_RETRY_COUNT, FETCH_RETRY_DELAYS
logger = logging.getLogger(__name__)
def _retry(fn, label, max_retries=FETCH_RETRY_COUNT):
    for i in range(max_retries):
        try: return fn()
        except Exception as e:
            d = FETCH_RETRY_DELAYS[min(i, len(FETCH_RETRY_DELAYS)-1)]
            logger.warning(f"{label} attempt {i+1}/{max_retries}: {e}")
            if i < max_retries-1: time.sleep(d)
    return None
def fetch_stock(ticker):
    def _f():
        s = yf.Ticker(ticker); h = s.history(period="3mo")
        if h.empty: raise ValueError("no data")
        l, p = h.iloc[-1], h.iloc[-2] if len(h)>=2 else h.iloc[-1]
        ma20 = h["Close"].rolling(20).mean().iloc[-1]
        ma50 = h["Close"].rolling(50).mean().iloc[-1]
        return {"ticker":ticker,"close":float(l["Close"]),"open":float(l["Open"]),
                "high":float(l["High"]),"low":float(l["Low"]),"volume":int(l["Volume"]),
                "ma20":float(ma20) if pd.notna(ma20) else None,
                "ma50":float(ma50) if pd.notna(ma50) else None,
                "prev_close":float(p["Close"]),
                "change_pct":float((l["Close"]-p["Close"])/p["Close"]*100)}
    return _retry(_f, ticker)
def fetch_vix():
    return _retry(lambda: float(yf.Ticker(YFINANCE_VIX_TICKER).history(period="5d").iloc[-1]["Close"]), "VIX")
def fetch_fear_greed():
    def _f():
        r = requests.get(CNN_FNG_URL, timeout=15); r.raise_for_status()
        d = r.json(); s = d.get("fear_and_greed",{}).get("score",{}).get("value")
        if s is None:
            h = d.get("fear_and_greed_historical",{}).get("data",[])
            if h: s = h[0].get("value")
        if s is None: raise ValueError("parse failed")
        return float(s)
    return _retry(_f, "F&G")
def fetch_history(ticker, days=30):
    def _f():
        h = yf.Ticker(ticker).history(period="2mo")
        if h.empty or len(h)<5: raise ValueError("insufficient history")
        c = h["Close"].tail(days); c.index = c.index.strftime("%Y-%m-%d"); return c
    return _retry(_f, f"{ticker}_hist")
def fetch_all():
    rows, failed = [], []
    for t in TICKERS:
        r = fetch_stock(t); (rows.append(r) if r else failed.append(t))
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    hists = {t: s for t in TICKERS if (s:=fetch_history(t)) is not None}
    hist_df = pd.DataFrame(hists) if hists else pd.DataFrame()
    if not hist_df.empty: hist_df.index.name = "date"
    return df, hist_df, fetch_vix(), fetch_fear_greed(), failed

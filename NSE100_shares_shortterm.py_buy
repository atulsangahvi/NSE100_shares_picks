import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf


# -----------------------------
# Config / Utilities
# -----------------------------
st.set_page_config(page_title="NIFTY 100 Technical Scanner", layout="wide")

NSE_INDICES_NIFTY100_CSV = "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv"

# Fallback list (not guaranteed always up-to-date)
# You can edit this list anytime.
FALLBACK_NIFTY100 = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BPCL", "BHARTIARTL",
    "BRITANNIA", "CIPLA", "COALINDIA", "DIVISLAB", "DRREDDY", "EICHERMOT",
    "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HINDALCO",
    "HINDUNILVR", "ICICIBANK", "ITC", "INDUSINDBK", "INFY", "JSWSTEEL",
    "KOTAKBANK", "LT", "M&M", "MARUTI", "NESTLEIND", "NTPC",
    "ONGC", "POWERGRID", "RELIANCE", "SBIN", "SUNPHARMA",
    "TATAMOTORS", "TATASTEEL", "TCS", "TECHM", "TITAN", "ULTRACEMCO",
    "WIPRO",
    # Add more if you want; live fetch is preferred.
]


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


# -----------------------------
# Technical Indicators
# -----------------------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)

    # Wilder's smoothing = EMA with alpha=1/period
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()

    rs = roll_up / roll_down.replace(0, np.nan)
    rsi_ = 100 - (100 / (1 + rs))
    return rsi_


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


# -----------------------------
# Universe fetch
# -----------------------------
@st.cache_data(ttl=6 * 60 * 60)
def fetch_nifty100_symbols() -> List[str]:
    """
    Try to fetch NIFTY 100 constituents from NSE Indices CSV.
    If it fails, return fallback list.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/csv,application/csv",
        }
        resp = requests.get(NSE_INDICES_NIFTY100_CSV, headers=headers, timeout=20)
        resp.raise_for_status()
        df = pd.read_csv(pd.compat.StringIO(resp.text)) if hasattr(pd, "compat") else pd.read_csv(pd.io.common.StringIO(resp.text))  # compatibility
        # Common column name in that CSV: "Symbol"
        if "Symbol" in df.columns:
            syms = sorted(df["Symbol"].dropna().astype(str).unique().tolist())
            return syms
    except Exception:
        pass

    return sorted(list(set(FALLBACK_NIFTY100)))


def to_yf_ticker(nse_symbol: str) -> str:
    """Convert NSE symbol to Yahoo Finance ticker."""
    return f"{nse_symbol}.NS"


# -----------------------------
# Data download
# -----------------------------
@st.cache_data(ttl=60 * 60)
def download_history(tickers: List[str], period: str = "6mo", interval: str = "1d") -> Dict[str, pd.DataFrame]:
    """
    Download OHLCV for each ticker using yfinance.
    Returns dict[ticker] = dataframe with columns: Open, High, Low, Close, Volume
    """
    out = {}
    # yfinance batch download
    # group_by="ticker" returns MultiIndex columns (ticker, field)
    data = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )

    if isinstance(data.columns, pd.MultiIndex):
        # Multi-ticker result
        for t in tickers:
            if t in data.columns.get_level_values(0):
                df = data[t].dropna(how="all")
                if not df.empty:
                    out[t] = df.copy()
    else:
        # Single ticker
        if not data.empty and len(tickers) == 1:
            out[tickers[0]] = data.dropna(how="all").copy()

    return out


# -----------------------------
# Scoring model (simple + explainable)
# -----------------------------
@dataclass
class ScanParams:
    rsi_buy_min: float = 35.0
    rsi_buy_max: float = 65.0
    min_history_bars: int = 120
    vol_boost: float = 1.2  # volume / 20d avg volume
    lookback_days_for_breakout: int = 20


def compute_features(df: pd.DataFrame) -> Optional[Dict[str, float]]:
    """
    Compute last value of key indicators.
    """
    if df is None or df.empty or "Close" not in df.columns:
        return None

    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(index=df.index, data=np.nan)

    if close.dropna().shape[0] < 60:
        return None

    r = rsi(close, 14)
    macd_line, sig, hist = macd(close, 12, 26, 9)
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()

    vol20 = vol.rolling(20).mean()

    last = df.index.max()
    # Last valid rows
    last_close = safe_float(close.loc[last])
    last_rsi = safe_float(r.loc[last])
    last_macdh = safe_float(hist.loc[last])
    last_sma20 = safe_float(sma20.loc[last])
    last_sma50 = safe_float(sma50.loc[last])
    last_vol = safe_float(vol.loc[last])
    last_vol20 = safe_float(vol20.loc[last])

    # Simple breakout signal: close above max close of last N days (excluding today)
    n = 20
    if close.shape[0] > n + 1:
        prev_window = close.iloc[-(n+1):-1]
        breakout = 1.0 if last_close > float(prev_window.max()) else 0.0
    else:
        breakout = np.nan

    return {
        "close": last_close,
        "rsi14": last_rsi,
        "macd_hist": last_macdh,
        "sma20": last_sma20,
        "sma50": last_sma50,
        "vol": last_vol,
        "vol20": last_vol20,
        "vol_ratio": (last_vol / last_vol20) if (last_vol20 and not math.isnan(last_vol20) and last_vol20 > 0) else np.nan,
        "breakout20": breakout,
    }


def score_short_term(feat: Dict[str, float], p: ScanParams) -> Tuple[float, List[str]]:
    """
    Produce an explainable score. Higher = better.
    Criteria:
      + Trend: close > SMA20, SMA20 > SMA50
      + Momentum: MACD hist > 0
      + RSI: in a "healthy" band (avoid too overbought/oversold)
      + Volume: vol ratio > threshold
      + Breakout: optional bonus
    """
    reasons = []
    score = 0.0

    c = feat["close"]
    r = feat["rsi14"]
    mh = feat["macd_hist"]
    sma20 = feat["sma20"]
    sma50 = feat["sma50"]
    vr = feat["vol_ratio"]
    br = feat["breakout20"]

    # Trend
    if np.isfinite(c) and np.isfinite(sma20) and c > sma20:
        score += 1.0
        reasons.append("Close > SMA20")
    else:
        reasons.append("Close <= SMA20")

    if np.isfinite(sma20) and np.isfinite(sma50) and sma20 > sma50:
        score += 1.0
        reasons.append("SMA20 > SMA50")
    else:
        reasons.append("SMA20 <= SMA50")

    # Momentum
    if np.isfinite(mh) and mh > 0:
        score += 1.0
        reasons.append("MACD histogram > 0")
    else:
        reasons.append("MACD histogram <= 0")

    # RSI band
    if np.isfinite(r) and (p.rsi_buy_min <= r <= p.rsi_buy_max):
        score += 1.0
        reasons.append(f"RSI in {p.rsi_buy_min:.0f}-{p.rsi_buy_max:.0f}")
    else:
        reasons.append("RSI outside preferred band")

    # Volume confirmation
    if np.isfinite(vr) and vr >= p.vol_boost:
        score += 1.0
        reasons.append(f"Volume spike (>{p.vol_boost:.1f}x 20D avg)")
    else:
        reasons.append("No strong volume spike")

    # Breakout bonus
    if np.isfinite(br) and br > 0:
        score += 0.5
        reasons.append("20D breakout")
    else:
        reasons.append("No 20D breakout")

    return score, reasons


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("NIFTY 100 Technical Scanner (Short-term)")

with st.sidebar:
    st.header("Scan settings")
    period = st.selectbox("History period", ["3mo", "6mo", "1y", "2y"], index=1)
    interval = st.selectbox("Interval", ["1d", "1h"], index=0)
    top_n = st.slider("Show top N", 5, 50, 20)

    st.subheader("Signal preferences")
    rsi_min = st.slider("RSI min", 10, 60, 35)
    rsi_max = st.slider("RSI max", 40, 90, 65)
    vol_boost = st.slider("Volume boost vs 20D avg", 1.0, 3.0, 1.2, 0.1)

    p = ScanParams(rsi_buy_min=float(rsi_min), rsi_buy_max=float(rsi_max), vol_boost=float(vol_boost))

    st.caption("Tip: 1h interval is heavier and may hit rate-limits. Start with 1d.")

symbols = fetch_nifty100_symbols()
tickers = [to_yf_ticker(s) for s in symbols]

st.write(f"Universe: **NIFTY 100** constituents loaded: **{len(symbols)}**")

run = st.button("Run scan", type="primary")

if run:
    with st.spinner("Downloading data & computing indicators..."):
        data_map = download_history(tickers, period=period, interval=interval)

        rows = []
        for sym in symbols:
            t = to_yf_ticker(sym)
            df = data_map.get(t)
            if df is None or df.empty:
                continue
            if df.shape[0] < p.min_history_bars and interval == "1d":
                # Not enough history for stable 50/200, etc.
                # Still allow but you can tighten this if you want.
                pass

            feat = compute_features(df)
            if not feat:
                continue

            score, reasons = score_short_term(feat, p)

            rows.append({
                "Symbol": sym,
                "Score": score,
                "Close": feat["close"],
                "RSI14": feat["rsi14"],
                "MACD_hist": feat["macd_hist"],
                "SMA20": feat["sma20"],
                "SMA50": feat["sma50"],
                "VolRatio": feat["vol_ratio"],
                "Breakout20": feat["breakout20"],
                "Why": " | ".join(reasons[:5]),
            })

        if not rows:
            st.error("No results. Try 1d interval, larger period, or check connectivity.")
            st.stop()

        res = pd.DataFrame(rows)
        res = res.sort_values(["Score", "RSI14"], ascending=[False, True]).reset_index(drop=True)

    st.subheader("Top candidates (by score)")
    st.dataframe(res.head(top_n), use_container_width=True)

    st.subheader("Quick chart + details")
    pick = st.selectbox("Select a symbol to inspect", res["Symbol"].head(top_n).tolist())
    t = to_yf_ticker(pick)
    df = data_map.get(t)

    if df is not None and not df.empty:
        close = df["Close"].astype(float)
        r = rsi(close, 14)
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        macd_line, sig, hist = macd(close, 12, 26, 9)

        chart_df = pd.DataFrame({
            "Close": close,
            "SMA20": sma20,
            "SMA50": sma50,
        }).dropna()

        st.line_chart(chart_df)

        ind_df = pd.DataFrame({
            "RSI14": r,
            "MACD_hist": hist,
        }).dropna()

        st.line_chart(ind_df)

        st.caption("Reminder: This is a technical filter, not a guarantee. Use risk controls (stop-loss, position sizing).")

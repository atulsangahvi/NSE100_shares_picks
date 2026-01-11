import time
import math
import io
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf


st.set_page_config(page_title="NIFTY 100 Technical Scanner", layout="wide")

NSE_INDICES_NIFTY100_CSV = "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv"

# Fallback list (partial). You can expand it, but we try to fetch live first.
FALLBACK_NIFTY100 = [
    "RELIANCE","TCS","HDFCBANK","ICICIBANK","INFY","LT","ITC","SBIN","AXISBANK",
    "BHARTIARTL","HINDUNILVR","KOTAKBANK","BAJFINANCE","ASIANPAINT","HCLTECH",
    "SUNPHARMA","TITAN","MARUTI","M&M","TATAMOTORS","WIPRO","ULTRACEMCO","TECHM",
    "NTPC","ONGC","POWERGRID","COALINDIA","DIVISLAB","DRREDDY","EICHERMOT",
    "HINDALCO","JSWSTEEL","TATASTEEL","CIPLA","BAJAJFINSV","BAJAJ-AUTO","BPCL",
    "BRITANNIA","GRASIM","HEROMOTOCO","NESTLEIND","HDFCLIFE","ADANIPORTS","ADANIENT",
    "APOLLOHOSP",
]


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


# -----------------------------
# Indicators
# -----------------------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


# -----------------------------
# Universe
# -----------------------------
@st.cache_data(ttl=6*60*60)
def fetch_nifty100_symbols() -> Tuple[List[str], str]:
    """
    Returns (symbols, source_string)
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "text/csv"}
        resp = requests.get(NSE_INDICES_NIFTY100_CSV, headers=headers, timeout=20)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        if "Symbol" in df.columns:
            syms = sorted(df["Symbol"].dropna().astype(str).unique().tolist())
            if len(syms) >= 80:
                return syms, "niftyindices.com CSV"
    except Exception:
        pass

    return sorted(list(set(FALLBACK_NIFTY100))), "fallback list"


def to_yf_ticker(nse_symbol: str) -> str:
    return f"{nse_symbol}.NS"


# -----------------------------
# Robust downloader
# -----------------------------
def chunked(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i:i+n] for i in range(0, len(lst), n)]


@st.cache_data(ttl=60*60)
def download_history_chunked(
    tickers: List[str],
    period: str = "6mo",
    interval: str = "1d",
    chunk_size: int = 20,
    max_retries: int = 2,
    sleep_s: float = 0.8,
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    Downloads in chunks to reduce Yahoo failures/rate limits.
    Returns: (data_map, failed_tickers)
    """
    data_map: Dict[str, pd.DataFrame] = {}
    failed: List[str] = []

    for group in chunked(tickers, chunk_size):
        ok_this_group = False

        for attempt in range(max_retries + 1):
            try:
                raw = yf.download(
                    tickers=group,
                    period=period,
                    interval=interval,
                    group_by="ticker",
                    threads=True,
                    progress=False,
                    auto_adjust=False,
                )

                if raw is None or getattr(raw, "empty", True):
                    raise RuntimeError("Empty response from yfinance")

                # MultiIndex if multiple tickers succeed
                if isinstance(raw.columns, pd.MultiIndex):
                    for t in group:
                        if t in raw.columns.get_level_values(0):
                            df = raw[t].dropna(how="all")
                            if df is not None and not df.empty and "Close" in df.columns:
                                data_map[t] = df
                    ok_this_group = True
                else:
                    # single ticker case
                    if len(group) == 1 and "Close" in raw.columns and not raw.empty:
                        data_map[group[0]] = raw.dropna(how="all")
                        ok_this_group = True

                if ok_this_group:
                    break

            except Exception:
                if attempt < max_retries:
                    time.sleep(sleep_s * (attempt + 1))
                else:
                    pass

        # Mark failures for this group (only those not in map)
        for t in group:
            if t not in data_map:
                failed.append(t)

        time.sleep(sleep_s)

    # De-dup
    failed = sorted(list(set(failed)))
    return data_map, failed


# -----------------------------
# Features + scoring
# -----------------------------
@dataclass
class ScanParams:
    rsi_buy_min: float = 35.0
    rsi_buy_max: float = 65.0
    vol_boost: float = 1.2
    breakout_lookback: int = 20


def compute_features(df: pd.DataFrame, breakout_lookback: int = 20) -> Optional[Dict[str, float]]:
    if df is None or df.empty or "Close" not in df.columns:
        return None

    close = df["Close"].astype(float).dropna()
    if close.shape[0] < 60:
        return None

    vol = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(index=df.index, data=np.nan)

    r = rsi(close, 14)
    _, _, hist = macd(close, 12, 26, 9)
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    vol20 = vol.rolling(20).mean()

    last_idx = close.index[-1]
    last_close = safe_float(close.loc[last_idx])
    last_rsi = safe_float(r.loc[last_idx])
    last_hist = safe_float(hist.loc[last_idx])
    last_sma20 = safe_float(sma20.loc[last_idx])
    last_sma50 = safe_float(sma50.loc[last_idx])

    last_vol = safe_float(vol.loc[last_idx]) if last_idx in vol.index else np.nan
    last_vol20 = safe_float(vol20.loc[last_idx]) if last_idx in vol20.index else np.nan
    vol_ratio = (last_vol / last_vol20) if (np.isfinite(last_vol) and np.isfinite(last_vol20) and last_vol20 > 0) else np.nan

    # Breakout (close > max previous N closes)
    br = np.nan
    if close.shape[0] > breakout_lookback + 1:
        prev = close.iloc[-(breakout_lookback + 1):-1]
        br = 1.0 if last_close > float(prev.max()) else 0.0

    return {
        "close": last_close,
        "rsi14": last_rsi,
        "macd_hist": last_hist,
        "sma20": last_sma20,
        "sma50": last_sma50,
        "vol_ratio": vol_ratio,
        "breakout": br,
    }


def score_short_term(feat: Dict[str, float], p: ScanParams) -> Tuple[float, List[str]]:
    reasons = []
    score = 0.0

    c = feat["close"]
    r = feat["rsi14"]
    h = feat["macd_hist"]
    sma20 = feat["sma20"]
    sma50 = feat["sma50"]
    vr = feat["vol_ratio"]
    br = feat["breakout"]

    # Trend
    if np.isfinite(c) and np.isfinite(sma20) and c > sma20:
        score += 1.0; reasons.append("Close > SMA20")
    else:
        reasons.append("Close <= SMA20")

    if np.isfinite(sma20) and np.isfinite(sma50) and sma20 > sma50:
        score += 1.0; reasons.append("SMA20 > SMA50")
    else:
        reasons.append("SMA20 <= SMA50")

    # Momentum
    if np.isfinite(h) and h > 0:
        score += 1.0; reasons.append("MACD hist > 0")
    else:
        reasons.append("MACD hist <= 0")

    # RSI band
    if np.isfinite(r) and p.rsi_buy_min <= r <= p.rsi_buy_max:
        score += 1.0; reasons.append(f"RSI in {int(p.rsi_buy_min)}–{int(p.rsi_buy_max)}")
    else:
        reasons.append("RSI out of band")

    # Volume
    if np.isfinite(vr) and vr >= p.vol_boost:
        score += 1.0; reasons.append(f"Vol > {p.vol_boost:.1f}×20D")
    else:
        reasons.append("No vol confirmation")

    # Breakout bonus
    if np.isfinite(br) and br > 0:
        score += 0.5; reasons.append("Breakout")
    else:
        reasons.append("No breakout")

    return score, reasons


# -----------------------------
# UI
# -----------------------------
st.title("NIFTY 100 Technical Scanner (Short-term)")

with st.sidebar:
    st.header("Settings")
    period = st.selectbox("History period", ["3mo", "6mo", "1y", "2y"], index=1)
    interval = st.selectbox("Interval", ["1d"], index=0)  # keep MVP stable
    chunk_size = st.slider("Download chunk size", 5, 50, 20)
    top_n = st.slider("Top N", 5, 50, 20)

    st.subheader("Signal filters")
    rsi_min = st.slider("RSI min", 10, 60, 35)
    rsi_max = st.slider("RSI max", 40, 90, 65)
    vol_boost = st.slider("Volume boost vs 20D avg", 1.0, 3.0, 1.2, 0.1)
    breakout_lb = st.slider("Breakout lookback days", 10, 60, 20)

    p = ScanParams(
        rsi_buy_min=float(rsi_min),
        rsi_buy_max=float(rsi_max),
        vol_boost=float(vol_boost),
        breakout_lookback=int(breakout_lb),
    )

symbols, src = fetch_nifty100_symbols()
tickers = [to_yf_ticker(s) for s in symbols]
st.write(f"Universe loaded: **{len(symbols)}** symbols (source: **{src}**)")

run = st.button("Run scan", type="primary")

if run:
    with st.spinner("Downloading data in chunks (with retries)..."):
        data_map, failed = download_history_chunked(
            tickers,
            period=period,
            interval=interval,
            chunk_size=chunk_size,
            max_retries=2,
            sleep_s=0.6,
        )

    st.write(f"Downloaded: **{len(data_map)}** tickers | Failed: **{len(failed)}**")

    if len(data_map) == 0:
        st.error(
            "Still got zero data from Yahoo Finance. This usually means yfinance is blocked/rate-limited from your environment.\n\n"
            "Quick fixes:\n"
            "- Try running locally (works more reliably than Streamlit Cloud for yfinance).\n"
            "- Or switch to an NSE data source (see note below).\n"
        )
        st.stop()

    if failed:
        with st.expander("Show failed tickers"):
            st.write(failed[:200])

    rows = []
    for sym in symbols:
        t = to_yf_ticker(sym)
        df = data_map.get(t)
        if df is None or df.empty:
            continue

        feat = compute_features(df, breakout_lookback=p.breakout_lookback)
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
            "Breakout": feat["breakout"],
            "Why": " | ".join(reasons[:6]),
        })

    if not rows:
        st.error(
            "Downloaded data, but no symbols passed indicator computation.\n"
            "Try:\n"
            "- Increase period to 1y\n"
            "- Widen RSI band\n"
            "- Lower volume boost threshold\n"
        )
        st.stop()

    res = pd.DataFrame(rows).sort_values(["Score", "RSI14"], ascending=[False, True]).reset_index(drop=True)
    st.subheader("Top candidates")
    st.dataframe(res.head(top_n), use_container_width=True)

    st.subheader("Inspect a symbol")
    pick = st.selectbox("Symbol", res["Symbol"].head(top_n).tolist())
    t = to_yf_ticker(pick)
    df = data_map.get(t)

    if df is not None and not df.empty:
        close = df["Close"].astype(float)
        chart = pd.DataFrame({
            "Close": close,
            "SMA20": close.rolling(20).mean(),
            "SMA50": close.rolling(50).mean(),
        }).dropna()
        st.line_chart(chart)

        ind = pd.DataFrame({
            "RSI14": rsi(close, 14),
            "MACD_hist": macd(close, 12, 26, 9)[2],
        }).dropna()
        st.line_chart(ind)

    st.caption("This is a technical filter, not a guarantee. Use position sizing + stop discipline.")

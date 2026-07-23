
import math
import os
import re
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

# =========================
# API KEYS: EDIT THESE TWO LINES
# =========================
ALPHAVANTAGE_API_KEY = "PT0LGXXYRI62PO7B"
FINNHUB_API_KEY = "d8cguj1r01qidic7thjgd8cguj1r01qidic7thk0"
# =========================

st.set_page_config(page_title="FCN Pricer", layout="wide")
st.title("FCN Pricer")
st.caption("Worst-of FCN / autocallable pricer with a two-step market data confirmation flow.")

ALPHAV_URL = "https://www.alphavantage.co/query"
FINNHUB_URL = "https://finnhub.io/api/v1"


def to_date(x):
    if isinstance(x, date):
        return x
    if isinstance(x, datetime):
        return x.date()
    return pd.to_datetime(x).date()


def year_frac(d1, d2):
    return (to_date(d2) - to_date(d1)).days / 365.0


def yf_spot(ticker: str) -> float:
    hist = yf.Ticker(ticker).history(period="10d", auto_adjust=False)
    if hist.empty:
        raise ValueError(f"No price data for {ticker}")
    return float(hist["Close"].dropna().iloc[-1])


def alpha_vantage_spot(ticker: str) -> float:
    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": ticker,
        "apikey": ALPHAVANTAGE_API_KEY,
    }
    r = requests.get(ALPHAV_URL, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    q = data.get("Global Quote", {})
    price = q.get("05. price")
    if price is None:
        raise ValueError(f"Alpha Vantage spot unavailable for {ticker}")
    return float(price)


def alpha_vantage_dividend_yield(ticker: str):
    params = {
        "function": "OVERVIEW",
        "symbol": ticker,
        "apikey": ALPHAVANTAGE_API_KEY,
    }
    r = requests.get(ALPHAV_URL, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    if not data or "DividendYield" not in data:
        raise ValueError(f"Alpha Vantage overview unavailable for {ticker}")
    dy = data.get("DividendYield")
    if dy in (None, "", "None"):
        raise ValueError(f"DividendYield missing for {ticker}")
    return float(dy)


def alpha_vantage_iv(ticker: str, as_of: str | None = None):
    # Best-effort: Alpha Vantage historical options chain endpoint availability depends on account/features.
    # This block tries a documented-style query pattern and then falls back to a local estimate if needed.
    candidates = []
    if as_of:
        candidates.append(
            {
                "function": "HISTORICAL_OPTIONS",
                "symbol": ticker,
                "date": as_of,
                "apikey": ALPHAVANTAGE_API_KEY,
            }
        )
    candidates.append(
        {
            "function": "HISTORICAL_OPTIONS",
            "symbol": ticker,
            "apikey": ALPHAVANTAGE_API_KEY,
        }
    )

    for params in candidates:
        try:
            r = requests.get(ALPHAV_URL, params=params, timeout=20)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict):
                for key in data.keys():
                    if isinstance(data[key], list) and data[key]:
                        rows = data[key]
                        ivs = []
                        for row in rows:
                            for iv_key in ("impliedVolatility", "iv", "implied_volatility"):
                                if iv_key in row and row[iv_key] not in (None, "", "None"):
                                    try:
                                        ivs.append(float(row[iv_key]))
                                        break
                                    except Exception:
                                        pass
                        if ivs:
                            iv = float(np.nanmedian(ivs))
                            return iv / 100.0 if iv > 5 else iv
        except Exception:
            pass

    raise ValueError(f"Alpha Vantage IV unavailable for {ticker}")


def finnhub_quote(ticker: str):
    url = f"{FINNHUB_URL}/quote"
    params = {"symbol": ticker, "token": FINNHUB_API_KEY}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def finnhub_profile2(ticker: str):
    url = f"{FINNHUB_URL}/stock/profile2"
    params = {"symbol": ticker, "token": FINNHUB_API_KEY}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def finnhub_dividend_yield(ticker: str):
    prof = finnhub_profile2(ticker)
    for k in ("dividendYield", "dividend_yield"):
        if k in prof and prof[k] not in (None, "", "None"):
            return float(prof[k])
    raise ValueError(f"Finnhub dividend yield unavailable for {ticker}")


def finnhub_spot(ticker: str):
    q = finnhub_quote(ticker)
    c = q.get("c")
    if c is None:
        raise ValueError(f"Finnhub spot unavailable for {ticker}")
    return float(c)


def fetch_market_data(ticker: str):
    ticker = ticker.upper().strip()
    row = {"Ticker": ticker, "Spot": np.nan, "Vol": np.nan, "Dividend Yield": np.nan}
    status = {"Ticker": ticker, "Spot Source": "manual", "IV Source": "manual", "Div Source": "manual"}

    # Spot: prefer Finnhub, fallback Alpha Vantage, fallback Yahoo
    try:
        row["Spot"] = finnhub_spot(ticker)
        status["Spot Source"] = "Finnhub"
    except Exception:
        try:
            row["Spot"] = alpha_vantage_spot(ticker)
            status["Spot Source"] = "Alpha Vantage"
        except Exception:
            try:
                row["Spot"] = yf_spot(ticker)
                status["Spot Source"] = "yfinance"
            except Exception:
                pass

    # Dividend yield: prefer Finnhub, fallback Alpha Vantage
    try:
        row["Dividend Yield"] = finnhub_dividend_yield(ticker)
        status["Div Source"] = "Finnhub"
    except Exception:
        try:
            row["Dividend Yield"] = alpha_vantage_dividend_yield(ticker)
            status["Div Source"] = "Alpha Vantage"
        except Exception:
            pass

    # IV: prefer Alpha Vantage historical options, fallback local estimate from yfinance history if needed
    try:
        row["Vol"] = alpha_vantage_iv(ticker)
        status["IV Source"] = "Alpha Vantage"
    except Exception:
        try:
            hist = yf.Ticker(ticker).history(period="1y", auto_adjust=False)
            if not hist.empty:
                r = np.log(hist["Close"].dropna()).diff().dropna()
                if not r.empty:
                    row["Vol"] = float(max(0.05, min(2.0, r.std() * np.sqrt(252))))
                    status["IV Source"] = "yfinance estimate"
        except Exception:
            pass

    return row, status


def hist_corr(tickers):
    frames = []
    for tk in tickers:
        try:
            s = yf.Ticker(tk).history(period="1y", auto_adjust=False)["Close"].rename(tk)
            frames.append(s)
        except Exception:
            pass
    df = pd.concat(frames, axis=1).dropna()
    if df.shape[1] < 2 or df.empty:
        return pd.DataFrame(np.eye(len(tickers)), index=tickers, columns=tickers)
    ret = np.log(df).diff().dropna()
    corr = ret.corr().reindex(index=tickers, columns=tickers).fillna(0.0)
    corr = corr.to_numpy(copy=True)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=tickers, columns=tickers)


def cholesky_with_fallback(corr):
    try:
        return np.linalg.cholesky(corr)
    except Exception:
        eigvals, eigvecs = np.linalg.eigh(corr)
        eigvals = np.clip(eigvals, 1e-8, None)
        corr2 = eigvecs @ np.diag(eigvals) @ eigvecs.T
        np.fill_diagonal(corr2, 1.0)
        return np.linalg.cholesky(corr2)


def simulate_paths(spot, vol, div, corr, maturity, n_paths=12000, steps=252, seed=11):
    rng = np.random.default_rng(seed)
    n = len(spot)
    dt = maturity / steps
    chol = cholesky_with_fallback(corr)
    paths = np.zeros((n_paths, steps + 1, n), dtype=float)
    paths[:, 0, :] = spot
    drift = np.array([(0.0 - d - 0.5 * v * v) for d, v in zip(div, vol)], dtype=float)
    diff = np.array(vol, dtype=float)

    for t in range(1, steps + 1):
        z = rng.standard_normal((n_paths, n))
        zc = z @ chol.T
        for i in range(n):
            paths[:, t, i] = paths[:, t - 1, i] * np.exp(drift[i] * dt + diff[i] * math.sqrt(dt) * zc[:, i])
    return paths


def worst_of_ratio(paths, spot):
    return (paths / np.array(spot)[None, None, :]).min(axis=2)


def price_fcn(
    spot,
    vol,
    div,
    corr,
    valuation_date,
    maturity_date,
    obs_months,
    funding_cost,
    notional,
    call_barrier,
    coupon_barrier,
    knock_in_barrier,
    n_paths=12000,
    seed=11,
):
    maturity = max(year_frac(valuation_date, maturity_date), 0.01)
    steps = max(252, int(252 * maturity))
    paths = simulate_paths(spot, vol, div, corr, maturity, n_paths=n_paths, steps=steps, seed=seed)

    obs_times = [year_frac(valuation_date, valuation_date + timedelta(days=int(30.4375 * m))) for m in obs_months]
    obs_idx = [min(steps, max(1, int(round(t / maturity * steps)))) for t in obs_times]

    worst = worst_of_ratio(paths, spot)
    terminal = worst[:, -1]

    pv = np.zeros(n_paths)
    called = np.zeros(n_paths, dtype=bool)
    call_time_bucket = np.full(n_paths, np.nan)

    for idx in obs_idx:
        alive = ~called
        trigger = alive & (worst[:, idx] >= call_barrier)
        if np.any(trigger):
            t = idx / steps * maturity
            cash = notional + notional * funding_cost * t
            pv[trigger] += cash
            called[trigger] = True
            call_time_bucket[trigger] = t

    alive = ~called
    if np.any(alive):
        coupon = notional * funding_cost * maturity
        redemption = np.where(
            terminal[alive] >= coupon_barrier,
            notional + coupon,
            np.where(terminal[alive] >= knock_in_barrier, notional + coupon, notional * terminal[alive]),
        )
        pv[alive] += redemption

    note_pv = pv.mean()
    call_prob = float(np.mean(called))
    expected_call_time = float(np.nanmean(call_time_bucket)) if np.any(~np.isnan(call_time_bucket)) else float("nan")
    option_value = max(0.0, float(note_pv / notional - 1.0))
    coupon_rate = funding_cost + option_value

    return {
        "note_pv": note_pv,
        "coupon_rate": coupon_rate,
        "funding_cost": funding_cost,
        "option_value": option_value,
        "call_prob": call_prob,
        "expected_call_time": expected_call_time,
    }


def build_schedule(valuation_date, obs_months, maturity_date):
    rows = [{"Type": "Valuation", "Date": str(valuation_date)}]
    rows += [{"Type": "Call/Coupon Obs", "Date": str(valuation_date + timedelta(days=int(30.4375 * m)))} for m in obs_months]
    rows += [{"Type": "Maturity", "Date": str(maturity_date)}]
    return pd.DataFrame(rows)


def format_underlying_table(inp, call_barrier, coupon_barrier, knock_in_barrier):
    out = inp.copy()
    out["Knock-in strike"] = out["Spot"] * knock_in_barrier
    out["Coupon strike"] = out["Spot"] * coupon_barrier
    out["Call strike"] = out["Spot"] * call_barrier
    out["Spot"] = out["Spot"].map(lambda x: f"{x:.2f}")
    out["Vol"] = out["Vol"].map(lambda x: f"{x:.2%}")
    out["Dividend Yield"] = out["Dividend Yield"].map(lambda x: f"{x:.2%}")
    out["Knock-in strike"] = out["Knock-in strike"].map(lambda x: f"{x:.2f}")
    out["Coupon strike"] = out["Coupon strike"].map(lambda x: f"{x:.2f}")
    out["Call strike"] = out["Call strike"].map(lambda x: f"{x:.2f}")
    return out


def corr_long_df(corr_df):
    out = corr_df.copy()
    out.insert(0, "Underlying", out.index)
    return out.reset_index(drop=True)


st.markdown(
    """
<style>
.block-container { max-width: 1200px !important; margin-left: auto; margin-right: auto; padding-top: 1.25rem; }
</style>
""",
    unsafe_allow_html=True,
)

if "step" not in st.session_state:
    st.session_state.step = 1
if "market_df" not in st.session_state:
    st.session_state.market_df = None
if "status_df" not in st.session_state:
    st.session_state.status_df = None

st.header("Inputs and assumptions")

col1, col2 = st.columns([2, 1])
with col1:
    n_names = st.selectbox("Number of underlyings", [1, 2, 3, 4], index=2)
    default_tickers = ["DRAM", "SOXX", "FXI", "XLK"]
    tickers = [st.text_input(f"Ticker {i+1}", value=default_tickers[i]) for i in range(n_names)]
    valuation_date = st.date_input("Valuation date", value=date.today())
    maturity_date = st.date_input("Maturity date", value=date.today() + timedelta(days=540))
    notional = st.number_input("Notional", min_value=1.0, value=100000.0, step=1000.0)
with col2:
    st.markdown("### Funding")
    funding_cost = st.number_input("Funding cost (annual)", min_value=0.0, value=0.035, step=0.005, format="%.4f")

c1, c2, c3 = st.columns(3)
with c1:
    call_barrier = st.number_input("Call barrier", min_value=0.0, value=1.0, step=0.01, format="%.4f")
    coupon_barrier = st.number_input("Coupon barrier", min_value=0.0, value=0.5, step=0.01, format="%.4f")
with c2:
    knock_in_barrier = st.number_input("Knock-in barrier", min_value=0.0, value=0.5, step=0.01, format="%.4f")
    n_paths = st.number_input("Simulation paths", min_value=1000, value=12000, step=1000)
with c3:
    seed = st.number_input("Random seed", min_value=1, value=11, step=1)
    obs_months = st.multiselect("Months from valuation", options=list(range(1, 61)), default=[6])

st.markdown("### Step 1")
st.write("Enter the product terms above.")

if st.button("Load / Confirm market data", type="primary", use_container_width=True):
    rows, stats = [], []
    for t in tickers:
        md, stt = fetch_market_data(t)
        rows.append(md)
        stats.append(stt)
    st.session_state.market_df = pd.DataFrame(rows)
    st.session_state.status_df = pd.DataFrame(stats)
    st.session_state.step = 2

if st.session_state.step >= 2 and st.session_state.market_df is not None:
    st.markdown("### Step 2")
    st.write("Review the market data below. Edit anything unusual, then press Calculate.")
    if st.session_state.status_df is not None:
        st.dataframe(st.session_state.status_df, use_container_width=True, hide_index=True)

    editor_df = st.session_state.market_df.copy()
    edited = st.data_editor(
        editor_df,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "Spot": st.column_config.NumberColumn("Spot"),
            "Vol": st.column_config.NumberColumn("Implied Vol", help="Decimal form, e.g. 0.615 = 61.5%"),
            "Dividend Yield": st.column_config.NumberColumn("Dividend Yield", help="Decimal form, e.g. 0.002 = 0.2%"),
        },
        key="market_editor",
    )

    if st.button("Calculate", type="primary", use_container_width=True):
        try:
            inp = edited.copy()
            for col in ["Spot", "Vol", "Dividend Yield"]:
                inp[col] = pd.to_numeric(inp[col], errors="coerce")
            if inp[["Spot", "Vol", "Dividend Yield"]].isna().any().any():
                raise ValueError("Missing market data. Please fill all Spot, Vol, and Dividend Yield values before calculating.")

            spot = inp["Spot"].astype(float).tolist()
            vol = inp["Vol"].astype(float).tolist()
            div = inp["Dividend Yield"].astype(float).tolist()
            corr = hist_corr([t.strip().upper() for t in tickers])

            result = price_fcn(
                spot=spot,
                vol=vol,
                div=div,
                corr=corr,
                valuation_date=valuation_date,
                maturity_date=maturity_date,
                obs_months=obs_months,
                funding_cost=funding_cost,
                notional=notional,
                call_barrier=call_barrier,
                coupon_barrier=coupon_barrier,
                knock_in_barrier=knock_in_barrier,
                n_paths=int(n_paths),
                seed=int(seed),
            )

            st.subheader("Results")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Note PV", f"{result['note_pv']:,.2f}")
            r2.metric("Funding cost", f"{result['funding_cost']:.2%}")
            r3.metric("Option value", f"{result['option_value']:.2%}")
            r4.metric("Final coupon", f"{result['coupon_rate']:.2%}")

            r5, r6 = st.columns(2)
            r5.metric("Call probability", f"{result['call_prob']:.2%}")
            r6.metric("Expected call time", "N/A" if np.isnan(result["expected_call_time"]) else f"{result['expected_call_time']:.2f}y")

            st.subheader("Underlying details")
            st.dataframe(format_underlying_table(inp, call_barrier, coupon_barrier, knock_in_barrier), use_container_width=True, hide_index=True)

            st.subheader("Correlation matrix")
            st.dataframe(corr.round(4), use_container_width=True)
            st.dataframe(corr_long_df(corr).round(4), use_container_width=True, hide_index=True)

            st.subheader("Schedule")
            st.dataframe(build_schedule(valuation_date, obs_months, maturity_date), use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(str(e))

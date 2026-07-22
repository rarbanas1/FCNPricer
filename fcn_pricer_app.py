
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from openpyxl import load_workbook

st.set_page_config(page_title="FCN Pricer", layout="wide")
st.title("FCN Pricer")
st.caption("Callable worst-of FCN pricer with Yahoo Finance inputs.")

TEMPLATE = Path(__file__).with_name("fcn_pricer_template.xlsx")


def get_spot(ticker: str) -> float:
    hist = yf.Ticker(ticker).history(period="10d", auto_adjust=False)
    if hist.empty:
        raise ValueError(f"No price data for {ticker}")
    return float(hist["Close"].dropna().iloc[-1])


def get_iv_proxy(ticker: str) -> float:
    t = yf.Ticker(ticker)
    try:
        expiries = list(t.options or [])
    except Exception:
        expiries = []
    for exp in expiries[:2]:
        try:
            chain = t.option_chain(exp).puts
            if not chain.empty and "impliedVolatility" in chain.columns:
                iv = float(chain["impliedVolatility"].dropna().median())
                if 0 < iv < 5:
                    return iv
        except Exception:
            pass
    hist = t.history(period="1y", auto_adjust=False)
    if hist.empty:
        return 0.30
    r = np.log(hist["Close"].dropna()).diff().dropna()
    if r.empty:
        return 0.30
    return float(r.std() * np.sqrt(252))


def get_dividend_yield(ticker: str) -> float:
    try:
        info = yf.Ticker(ticker).info or {}
        return float(info.get("dividendYield") or 0.0)
    except Exception:
        return 0.0


def hist_corr(tickers):
    frames = []
    for tk in tickers:
        try:
            s = yf.Ticker(tk).history(period="1y", auto_adjust=False)["Close"].rename(tk)
            frames.append(s)
        except Exception:
            pass
    if len(frames) < 2:
        return pd.DataFrame(np.eye(len(tickers)), index=tickers, columns=tickers)
    df = pd.concat(frames, axis=1).dropna()
    if df.shape[1] < 2 or df.empty:
        return pd.DataFrame(np.eye(len(tickers)), index=tickers, columns=tickers)
    ret = np.log(df).diff().dropna()
    corr_df = ret.corr().reindex(index=tickers, columns=tickers).fillna(0.0)
    corr_df = corr_df.clip(-1.0, 1.0)
    corr_values = corr_df.values.astype(float, copy=True)
    if corr_values.shape[0] == corr_values.shape[1] and corr_values.size > 0:
        corr_values = (corr_values + corr_values.T) / 2.0
        for i in range(corr_values.shape[0]):
            corr_values[i, i] = 1.0
    else:
        corr_values = np.eye(len(tickers), dtype=float)
    return pd.DataFrame(corr_values, index=tickers, columns=tickers)


def price_note(spots, vols, divs, corr, obs_months, maturity, funding, knock_in, call_coupon, notional, n_paths=50000, seed=7):
    spots = np.asarray(spots, dtype=float)
    vols = np.asarray(vols, dtype=float)
    divs = np.asarray(divs, dtype=float)
    n = len(spots)
    L = np.linalg.cholesky(corr)
    obs_years = sorted([m / 12.0 for m in obs_months if 0 < m <= maturity * 12 + 1e-12])
    obs_years = obs_years if obs_years else []
    rng = np.random.default_rng(seed)

    discounted_payoffs = []
    call_count = 0
    total_paths = n_paths
    survival = np.ones(n_paths, dtype=bool)

    prev_t = 0.0
    for t in obs_years:
        dt = t - prev_t
        if dt <= 0:
            continue
        z = rng.standard_normal((n_paths, n)) @ L.T
        terminal = spots * np.exp((funding - divs - 0.5 * vols**2) * t + vols * np.sqrt(t) * z)
        worst_ratio = (terminal / spots).min(axis=1)
        call_now = survival & (worst_ratio >= 1.0)
        if call_now.any():
            discounted_payoffs.append(np.exp(-funding * t) * (1.0 + call_coupon) * call_now.mean())
            call_count += int(call_now.sum())
            survival[call_now] = False
        prev_t = t

    z = rng.standard_normal((n_paths, n)) @ L.T
    terminal = spots * np.exp((funding - divs - 0.5 * vols**2) * maturity + vols * np.sqrt(maturity) * z)
    worst_ratio = (terminal / spots).min(axis=1)

    maturity_payoff = np.where(worst_ratio >= 1.0, 1.0, np.where(worst_ratio >= knock_in, 1.0, worst_ratio))
    maturity_pv = np.exp(-funding * maturity) * maturity_payoff.mean() * survival.mean()

    pv = float(sum(discounted_payoffs) + maturity_pv)
    call_probability = float(call_count / total_paths)
    expected_call_time = float(np.mean(obs_years)) if obs_years else maturity
    return pv, call_probability, expected_call_time


def load_template(path: Path):
    wb = load_workbook(path, data_only=True)
    ws = wb["Inputs"]
    vals = {ws[f"A{i}"].value: ws[f"B{i}"].value for i in range(2, ws.max_row + 1)}
    return vals


with st.sidebar:
    st.header("Inputs")
    uploaded = st.file_uploader("Upload input workbook", type=["xlsx"])
    use_template = st.checkbox("Use bundled template if no upload", value=True)
    vals = {}
    if uploaded is not None:
        tmp = Path("uploaded_inputs.xlsx")
        tmp.write_bytes(uploaded.getvalue())
        vals = load_template(tmp)
    elif use_template and TEMPLATE.exists():
        vals = load_template(TEMPLATE)

n_under = st.sidebar.slider("Number of underlyings", 1, 4, 3)
maturity = st.sidebar.number_input("Maturity (years)", 0.25, 5.0, float(vals.get("Maturity (years)", 1.0)), 0.25)
funding = st.sidebar.number_input("Funding rate", 0.0, 0.25, float(vals.get("Funding rate", 0.0375)), 0.0025, format="%.4f")
knock_in = st.sidebar.number_input("Knock-in level %", 0.05, 1.0, float(vals.get("Knock-in %", 0.5)), 0.05, format="%.2f")
call_coupon = st.sidebar.number_input("Call coupon %", 0.0, 1.0, float(vals.get("Call coupon %", 0.0)), 0.01, format="%.2f")
notional = st.sidebar.number_input("Notional", 1000.0, 10000000.0, float(vals.get("Notional", 100000.0)), 1000.0)
auto_pull = st.sidebar.checkbox("Auto-pull Yahoo Finance data", True)
use_corr = st.sidebar.checkbox("Use historical correlation", True)
obs_count = st.sidebar.slider("Number of observation dates", 0, 3, 1)
obs_months = []
for i in range(obs_count):
    default_months = [3, 6, 9][i]
    obs_months.append(st.sidebar.number_input(f"Observation date {i+1} (months)", 1, max(1, int(maturity * 12)), default_months, 1))

base_defaults = [str(vals.get(f"Underlying {i}", "")) for i in range(1, 5)]
base_defaults = [b if b else d for b, d in zip(base_defaults, ["SOXX", "DRAM", "FXI", "AAPL"])]

st.subheader("Underlying setup")
cols = st.columns(n_under)
rows = []
for i in range(n_under):
    with cols[i]:
        tk = st.text_input(f"Ticker {i+1}", value=base_defaults[i], key=f"tk_{i}")
        if auto_pull and tk:
            try:
                spot_auto = get_spot(tk)
            except Exception:
                spot_auto = 100.0
            try:
                vol_auto = get_iv_proxy(tk)
            except Exception:
                vol_auto = 0.30
            try:
                div_auto = get_dividend_yield(tk)
            except Exception:
                div_auto = 0.0
        else:
            spot_auto, vol_auto, div_auto = 100.0, 0.30, 0.0
        st.caption("Pulled from Yahoo Finance when available")
        st.metric(f"Spot {tk}", f"{spot_auto:.2f}")
        st.metric(f"Vol {tk}", f"{vol_auto:.4f}")
        st.metric(f"Dividend yield {tk}", f"{div_auto:.4f}")
        rows.append((tk, float(spot_auto), float(vol_auto), float(div_auto)))

if st.button("Price FCN"):
    tickers = [r[0] for r in rows]
    spots = [r[1] for r in rows]
    vols = [r[2] for r in rows]
    divs = [r[3] for r in rows]
    corr = hist_corr(tickers) if use_corr and len(tickers) > 1 else pd.DataFrame(np.eye(len(tickers)), index=tickers, columns=tickers)
    try:
        pv, call_prob, exp_call_time = price_note(spots, vols, divs, corr.values, obs_months, maturity, funding, knock_in, call_coupon, notional)
        option_cost = pv * notional
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Embedded option value", f"{pv:.4%}")
        c2.metric("Option cost ($)", f"${option_cost:,.2f}")
        c3.metric("Call probability", f"{call_prob:.2%}")
        c4.metric("Expected call time", f"{exp_call_time:.2f}y")
        st.divider()
        inp = pd.DataFrame(rows, columns=["Ticker", "Spot", "Vol", "Dividend Yield"])
        inp["Knock-in strike"] = inp["Spot"] * knock_in
        st.dataframe(inp, use_container_width=True, hide_index=True)
        st.subheader("Observation dates")
        st.write(obs_months if obs_months else ["None"])
    except Exception as e:
        st.error(str(e))

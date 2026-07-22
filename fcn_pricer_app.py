
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from openpyxl import load_workbook

st.set_page_config(page_title="FCN Pricer", layout="wide")
st.title("FCN Pricer")
st.caption("Worst-of FCN pricer using Yahoo Finance data and correlated Monte Carlo.")

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
    df = pd.concat(frames, axis=1).dropna()
    if df.shape[1] < 2 or df.empty:
        return pd.DataFrame(np.eye(len(tickers)), index=tickers, columns=tickers)
    ret = np.log(df).diff().dropna()
    corr = ret.corr().reindex(index=tickers, columns=tickers).fillna(0.0)
    corr = corr.to_numpy()
    if corr.size:
        np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=tickers, columns=tickers)


def mc_worst_of(spots, vols, divs, corr, barrier, maturity, funding, n_paths=30000, seed=7):
    rng = np.random.default_rng(seed)
    n = len(spots)
    L = np.linalg.cholesky(corr)
    z = rng.standard_normal((n_paths, n)) @ L.T
    spots = np.array(spots, dtype=float)
    vols = np.array(vols, dtype=float)
    divs = np.array(divs, dtype=float)
    drift = (funding - divs - 0.5 * vols**2) * maturity
    shock = vols * np.sqrt(maturity)
    terminal = spots * np.exp(drift + z * shock)
    strikes = spots * barrier
    worst_terminal = terminal.min(axis=1)
    worst_strike = strikes.min()
    payoff = np.maximum(0.0, 1.0 - worst_terminal / worst_strike)
    pv = np.exp(-funding * maturity) * payoff.mean()
    breach_prob = float((worst_terminal < worst_strike).mean())
    return pv, breach_prob, terminal


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
barrier = st.sidebar.number_input("Barrier %", 0.05, 1.0, float(vals.get("Barrier %", 0.5)), 0.05, format="%.2f")
notional = st.sidebar.number_input("Notional", 1000.0, 10000000.0, float(vals.get("Notional", 100000.0)), 1000.0)
auto_pull = st.sidebar.checkbox("Auto-pull Yahoo Finance data", True)
use_corr = st.sidebar.checkbox("Use historical correlation", True)

base_defaults = [str(vals.get(f"Underlying {i}", "")) for i in range(1, 5)]
base_defaults = [b if b else d for b, d in zip(base_defaults, ["SOXX", "DRAM", "FXI", "AAPL"])]

st.subheader("Underlying setup")
cols = st.columns(n_under)
rows = []
for i in range(n_under):
    with cols[i]:
        tk = st.text_input(f"Ticker {i+1}", value=base_defaults[i], key=f"tk_{i}")
        if auto_pull:
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
        spot = st.number_input(f"Spot {tk}", value=float(spot_auto), step=0.01, format="%.2f", key=f"spot_{i}")
        vol = st.number_input(f"Vol {tk}", value=float(vol_auto), step=0.01, format="%.4f", key=f"vol_{i}")
        div = st.number_input(f"Dividend yield {tk}", value=float(div_auto), step=0.001, format="%.4f", key=f"div_{i}")
        rows.append((tk, spot, vol, div))

if st.button("Price FCN"):
    tickers = [r[0] for r in rows]
    spots = [r[1] for r in rows]
    vols = [r[2] for r in rows]
    divs = [r[3] for r in rows]
    strikes = [s * barrier for s in spots]

    if use_corr and len(tickers) > 1:
        corr = hist_corr(tickers)
    else:
        corr = pd.DataFrame(np.eye(len(tickers)), index=tickers, columns=tickers)

    try:
        pv, breach_prob, terminal = mc_worst_of(spots, vols, divs, corr.values, barrier, maturity, funding)
        option_cost = pv * notional
        coupon = funding + pv

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Embedded option cost", f"{pv:.4%}")
        c2.metric("Option cost ($)", f"${option_cost:,.2f}")
        c3.metric("Indicative coupon", f"{coupon:.4%}")
        c4.metric("Coupon (bps)", f"{coupon*10000:.1f}")

        st.divider()
        st.subheader("Pricing summary")
        summary = pd.DataFrame({
            "Metric": ["Breach probability", "Worst-of strike", "Maturity", "Funding rate", "Notional"],
            "Value": [f"{breach_prob:.2%}", f"{barrier:.2%}", maturity, funding, f"${notional:,.0f}"]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

        st.subheader("Inputs used")
        inp = pd.DataFrame(rows, columns=["Ticker", "Spot", "Vol", "Dividend Yield"])
        inp["Strike"] = inp["Spot"] * barrier
        st.dataframe(inp, use_container_width=True, hide_index=True)

        st.subheader("Correlation matrix")
        st.dataframe(corr, use_container_width=True)

        st.subheader("Scenario sensitivity")
        sens = []
        for b in [0.40, 0.50, 0.60]:
            pv_s, bp_s, _ = mc_worst_of(spots, vols, divs, corr.values, b, maturity, funding, n_paths=12000, seed=11)
            sens.append([f"{b:.0%}", f"{pv_s:.2%}", f"{(funding + pv_s):.2%}"])
        st.dataframe(pd.DataFrame(sens, columns=["Barrier", "Embedded cost", "Coupon"]), use_container_width=True, hide_index=True)

        csv = inp.to_csv(index=False).encode("utf-8")
        st.download_button("Download inputs CSV", csv, file_name="fcn_pricing_inputs.csv", mime="text/csv")
    except Exception as e:
        st.error(str(e))

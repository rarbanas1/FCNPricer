
import math
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from openpyxl import load_workbook

st.set_page_config(page_title="FCN Pricer", layout="wide")
st.title("FCN Pricer")
st.caption("Term-sheet-based FCN / autocallable pricer using Yahoo Finance inputs.")

TEMPLATE = Path(__file__).with_name("fcn_pricer_template.xlsx")


def to_date(x):
    if isinstance(x, date):
        return x
    if isinstance(x, datetime):
        return x.date()
    return pd.to_datetime(x).date()


def year_frac(d1, d2):
    return (to_date(d2) - to_date(d1)).days / 365.0


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


def cholesky_with_fallback(corr):
    try:
        return np.linalg.cholesky(corr)
    except Exception:
        eigvals, eigvecs = np.linalg.eigh(corr)
        eigvals = np.clip(eigvals, 1e-8, None)
        corr2 = eigvecs @ np.diag(eigvals) @ eigvecs.T
        np.fill_diagonal(corr2, 1.0)
        return np.linalg.cholesky(corr2)


def simulate_paths(spot, vol, div, corr, maturity, n_paths=20000, steps=252, seed=11):
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
            paths[:, t, i] = paths[:, t - 1, i] * np.exp((drift[i] * dt) + diff[i] * math.sqrt(dt) * zc[:, i])
    return paths


def worst_of_ratio(paths, spot):
    rel = paths / np.array(spot)[None, None, :]
    return rel.min(axis=2)


def price_fcn(
    spot,
    vol,
    div,
    corr,
    valuation_date,
    maturity_date,
    call_dates,
    coupon_dates,
    obs_months,
    coupon_rate,
    notional,
    call_barrier,
    coupon_barrier,
    knock_in_barrier,
    n_paths=20000,
    seed=11,
):
    maturity = max(year_frac(valuation_date, maturity_date), 0.01)
    steps = max(252, int(252 * maturity))
    paths = simulate_paths(spot, vol, div, corr, maturity, n_paths=n_paths, steps=steps, seed=seed)

    call_times = [year_frac(valuation_date, d) for d in call_dates if to_date(d) > to_date(valuation_date)]
    call_idx = [min(steps, max(1, int(round(t / maturity * steps)))) for t in call_times]
    maturity_idx = steps

    worst_call = worst_of_ratio(paths, spot)
    worst_maturity = worst_call[:, maturity_idx]

    discount_rate = 0.0
    dfs = {i: math.exp(-discount_rate * (i / steps) * maturity) for i in range(steps + 1)}

    pv = np.zeros(n_paths)
    called = np.zeros(n_paths, dtype=bool)
    call_time_bucket = np.full(n_paths, np.nan)

    for idx in call_idx:
        alive = ~called
        trigger = alive & (worst_call[:, idx] >= call_barrier)
        if np.any(trigger):
            t = idx / steps * maturity
            pv[trigger] += (1.0 + coupon_rate * t) * notional * dfs[idx]
            called[trigger] = True
            call_time_bucket[trigger] = t

    alive = ~called
    if np.any(alive):
        t = maturity
        terminal = worst_maturity[alive]
        payoff = np.where(
            terminal >= coupon_barrier,
            1.0 + coupon_rate * t,
            np.where(terminal >= knock_in_barrier, 1.0 + coupon_rate * t, terminal),
        )
        pv[alive] += payoff * notional * dfs[maturity_idx]

    note_pv = pv.mean()
    call_prob = float(np.mean(called))
    expected_call_time = float(np.nanmean(call_time_bucket)) if np.any(~np.isnan(call_time_bucket)) else float("nan")
    expected_payoff = float(pv.mean() / notional)

    coupon_value = coupon_rate
    option_value = max(0.0, 1.0 - expected_payoff)

    return {
        "note_pv": note_pv,
        "coupon_value": coupon_value,
        "option_value": option_value,
        "call_prob": call_prob,
        "expected_call_time": expected_call_time,
        "expected_payoff": expected_payoff,
    }


with st.sidebar:
    st.header("Inputs")
    n_names = st.selectbox("Number of underlyings", [1, 2, 3, 4], index=2)
    tickers = []
    for i in range(n_names):
        tickers.append(st.text_input(f"Ticker {i+1}", value=["C", "JPM", "BAC", "WFC"][i]))
    valuation_date = st.date_input("Valuation date", value=date.today())
    maturity_date = st.date_input("Maturity date", value=date.today() + timedelta(days=540))
    coupon_rate = st.number_input("Coupon rate", min_value=0.0, value=0.12, step=0.005, format="%.4f")
    notional = st.number_input("Notional", min_value=1.0, value=1000000.0, step=10000.0)
    call_barrier = st.number_input("Call barrier", min_value=0.0, value=1.0, step=0.01, format="%.4f")
    coupon_barrier = st.number_input("Coupon barrier", min_value=0.0, value=0.6, step=0.01, format="%.4f")
    knock_in_barrier = st.number_input("Knock-in barrier", min_value=0.0, value=0.6, step=0.01, format="%.4f")
    n_paths = st.number_input("Simulation paths", min_value=1000, value=12000, step=1000)
    seed = st.number_input("Random seed", min_value=1, value=11, step=1)

    st.subheader("Observation dates")
    obs_months = st.multiselect(
        "Months from valuation",
        options=list(range(1, 61)),
        default=[6, 12, 18],
    )

try:
    rows = []
    for t in tickers:
        spot = get_spot(t)
        vol = get_iv_proxy(t)
        div = get_dividend_yield(t)
        rows.append({"Ticker": t, "Spot": spot, "Vol": vol, "Dividend Yield": div})

    inp = pd.DataFrame(rows)
    spot = inp["Spot"].tolist()
    vol = inp["Vol"].tolist()
    div = inp["Dividend Yield"].tolist()

    corr = hist_corr(tickers)

    call_dates = [valuation_date + timedelta(days=int(30.4375 * m)) for m in obs_months]
    coupon_dates = [maturity_date]

    result = price_fcn(
        spot=spot,
        vol=vol,
        div=div,
        corr=corr,
        valuation_date=valuation_date,
        maturity_date=maturity_date,
        call_dates=call_dates,
        coupon_dates=coupon_dates,
        obs_months=obs_months,
        coupon_rate=coupon_rate,
        notional=notional,
        call_barrier=call_barrier,
        coupon_barrier=coupon_barrier,
        knock_in_barrier=knock_in_barrier,
        n_paths=int(n_paths),
        seed=int(seed),
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Note PV", f"{result['note_pv']:,.2f}")
    c2.metric("Coupon value", f"{result['coupon_value']:.2%}")
    c3.metric("Call probability", f"{result['call_prob']:.2%}")
    c4.metric(
        "Expected call time",
        "N/A" if np.isnan(result["expected_call_time"]) else f"{result['expected_call_time']:.2f}y",
    )

    st.subheader("Market inputs")
    inp["Knock-in strike"] = inp["Spot"] * knock_in_barrier
    inp["Coupon strike"] = inp["Spot"] * coupon_barrier
    inp["Call strike"] = inp["Spot"] * call_barrier
    st.dataframe(inp, use_container_width=True, hide_index=True)

    st.subheader("Schedule")
    sch_rows = [{"Type": "Valuation", "Date": str(valuation_date)}]
    sch_rows += [{"Type": "Call", "Date": str(d)} for d in call_dates]
    sch_rows += [{"Type": "Coupon", "Date": str(d)} for d in coupon_dates]
    sch_rows += [{"Type": "Maturity", "Date": str(maturity_date)}]
    st.dataframe(pd.DataFrame(sch_rows), use_container_width=True, hide_index=True)

    st.subheader("Model notes")
    st.write(
        "The note is called if all underlyings are at or above the call barrier on an observation date. "
        "If it is not called, maturity redemption depends on the worst performer versus the coupon and knock-in barriers."
    )

    csv = inp.to_csv(index=False).encode("utf-8")
    st.download_button("Download inputs CSV", csv, file_name="fcn_pricing_inputs.csv", mime="text/csv")

except Exception as e:
    st.error(str(e))

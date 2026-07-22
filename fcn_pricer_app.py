
import math
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
            chain = t.option_chain(exp).calls
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


def yearfrac(a, b):
    return max((pd.to_datetime(a) - pd.to_datetime(b)).days / 365.0, 0.0)


def simulate_joint_paths(spots, vols, divs, corr, dates, valuation_date, funding, n_paths=50000, seed=7):
    spots = np.asarray(spots, dtype=float)
    vols = np.asarray(vols, dtype=float)
    divs = np.asarray(divs, dtype=float)
    n = len(spots)
    L = np.linalg.cholesky(corr)
    rng = np.random.default_rng(seed)
    times = [yearfrac(d, valuation_date) for d in dates]
    out = []
    for t in times:
        if t <= 0:
            continue
        z = rng.standard_normal((n_paths, n)) @ L.T
        st_t = spots * np.exp((funding - divs - 0.5 * vols**2) * t + vols * np.sqrt(t) * z)
        out.append((t, st_t))
    return out


def price_fcn(spots, vols, divs, corr, valuation_date, maturity_date, call_dates, coupon_dates, funding, coupon_rate, knock_in, call_barrier, notional, participation=1.0, coupon_memory=False, n_paths=50000, seed=7):
    spots = np.asarray(spots, dtype=float)
    vols = np.asarray(vols, dtype=float)
    divs = np.asarray(divs, dtype=float)
    n = len(spots)
    L = np.linalg.cholesky(corr)
    rng = np.random.default_rng(seed)
    val_dt = pd.to_datetime(valuation_date)
    mat_dt = pd.to_datetime(maturity_date)
    call_dates = sorted([pd.to_datetime(d) for d in call_dates if pd.to_datetime(d) > val_dt and pd.to_datetime(d) <= mat_dt])
    coupon_dates = sorted([pd.to_datetime(d) for d in coupon_dates if pd.to_datetime(d) > val_dt and pd.to_datetime(d) <= mat_dt])
    call_times = [yearfrac(d, val_dt) for d in call_dates]
    coupon_times = [yearfrac(d, val_dt) for d in coupon_dates]
    maturity_t = yearfrac(mat_dt, val_dt)

    alive = np.ones(n_paths, dtype=bool)
    accrued_coupon = np.zeros(n_paths, dtype=float)
    pv = np.zeros(n_paths, dtype=float)
    called = np.zeros(n_paths, dtype=bool)
    call_time = np.full(n_paths, maturity_t, dtype=float)

    schedule = sorted(set([(d, "call") for d in call_dates] + [(d, "coupon") for d in coupon_dates] + [(mat_dt, "maturity")]))
    prev_t = 0.0
    for d, typ in schedule:
        t = yearfrac(d, val_dt)
        if t <= prev_t:
            continue
        z = rng.standard_normal((n_paths, n)) @ L.T
        st_t = spots * np.exp((funding - divs - 0.5 * vols**2) * t + vols * np.sqrt(t) * z)
        worst_ratio = (st_t / spots).min(axis=1)

        if typ == "coupon":
            coupon_hit = alive & (worst_ratio >= knock_in)
            if coupon_memory:
                accrued_coupon[coupon_hit] += coupon_rate
            else:
                pv[coupon_hit] += np.exp(-funding * t) * coupon_rate

        if typ == "call":
            call_hit = alive & (worst_ratio >= call_barrier)
            if call_hit.any():
                pv[call_hit] += np.exp(-funding * t) * (1.0 + coupon_rate + (accrued_coupon[call_hit] if coupon_memory else 0.0))
                alive[call_hit] = False
                called[call_hit] = True
                call_time[call_hit] = t

        if typ == "maturity":
            survive = alive
            final_ratio = worst_ratio
            ki_hit = final_ratio < knock_in
            redemption = np.where(ki_hit, np.maximum(0.0, participation * final_ratio), 1.0)
            payoff = redemption + (accrued_coupon if coupon_memory else 0.0)
            pv[survive] += np.exp(-funding * t) * payoff[survive]
            alive[:] = False

        prev_t = t

    note_pv = float(pv.mean())
    embedded_option_value = max(1.0 - note_pv, 0.0)
    call_probability = float(called.mean())
    expected_call_time = float(call_time[called].mean()) if called.any() else maturity_t
    return note_pv, embedded_option_value, call_probability, expected_call_time


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

    valuation_date = st.date_input("Valuation date", value=pd.to_datetime(vals.get("Valuation date", pd.Timestamp.today())).date())
    maturity_date = st.date_input("Maturity date", value=pd.to_datetime(vals.get("Maturity date", pd.Timestamp.today() + pd.DateOffset(years=1))).date())
    n_under = st.slider("Number of underlyings", 1, 4, 3)
    funding = st.number_input("Funding rate", 0.0, 0.25, float(vals.get("Funding rate", 0.0375)), 0.0025, format="%.4f")
    knock_in = st.number_input("Knock-in barrier %", 0.05, 1.0, float(vals.get("Knock-in barrier %", 0.5)), 0.05, format="%.2f")
    call_barrier = st.number_input("Call barrier %", 0.05, 1.5, float(vals.get("Call barrier %", 1.0)), 0.05, format="%.2f")
    coupon_rate = st.number_input("Coupon rate % per period", 0.0, 1.0, float(vals.get("Coupon rate %", 0.10)), 0.01, format="%.2f")
    participation = st.number_input("Downside participation", 0.0, 2.0, float(vals.get("Participation", 1.0)), 0.05, format="%.2f")
    coupon_memory = st.checkbox("Coupon memory", value=bool(vals.get("Coupon memory", False)))
    notional = st.number_input("Notional", 1000.0, 10000000.0, float(vals.get("Notional", 100000.0)), 1000.0)
    auto_pull = st.checkbox("Auto-pull Yahoo Finance data", True)
    use_corr = st.checkbox("Use historical correlation", True)
    call_count = st.slider("Number of call observation dates", 0, 3, 1)
    coupon_count = st.slider("Number of coupon dates", 0, 12, 4)

    st.subheader("Schedule")
    call_dates = []
    for i in range(call_count):
        default_date = (pd.to_datetime(valuation_date) + pd.DateOffset(months=[6, 12, 18][i])).date()
        call_dates.append(st.date_input(f"Call date {i+1}", value=default_date, key=f"call_date_{i}"))
    coupon_dates = []
    for i in range(coupon_count):
        default_date = (pd.to_datetime(valuation_date) + pd.DateOffset(months=3 * (i + 1))).date()
        coupon_dates.append(st.date_input(f"Coupon date {i+1}", value=default_date, key=f"coupon_date_{i}"))

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
        note_pv, opt_val, call_prob, exp_call_time = price_fcn(
            spots, vols, divs, corr.values,
            valuation_date, maturity_date, call_dates, coupon_dates,
            funding, coupon_rate, knock_in, call_barrier, participation,
            coupon_memory=coupon_memory
        )
        note_cost = note_pv * notional
        option_cost = opt_val * notional
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Note PV", f"{note_pv:.4%}")
        c2.metric("Embedded option value", f"{opt_val:.4%}")
        c3.metric("Note cost ($)", f"${note_cost:,.2f}")
        c4.metric("Option cost ($)", f"${option_cost:,.2f}")

        st.divider()
        inp = pd.DataFrame(rows, columns=["Ticker", "Spot", "Vol", "Dividend Yield"])
        inp["Protection strike"] = inp["Spot"] * knock_in
        inp["Call strike"] = inp["Spot"] * call_barrier
        st.dataframe(inp, use_container_width=True, hide_index=True)

        st.subheader("Schedule")
        sch = pd.DataFrame({
            "Type": ["Valuation", "Maturity"] + ["Call"] * len(call_dates) + ["Coupon"] * len(coupon_dates),
            "Date": [str(valuation_date), str(maturity_date)] + [str(d) for d in call_dates] + [str(d) for d in coupon_dates]
        })
        st.dataframe(sch, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(str(e))

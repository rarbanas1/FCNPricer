
import math
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import re

st.set_page_config(page_title="FCN Pricer", layout="wide")
st.title("FCN Pricer")
st.caption("Worst-of FCN / autocallable pricer with Market Chameleon market data and editable confirmation step.")

TEMPLATE = Path(__file__).with_name("fcn_pricer_template.xlsx")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def to_date(x):
    if isinstance(x, date):
        return x
    if isinstance(x, datetime):
        return x.date()
    return pd.to_datetime(x).date()


def year_frac(d1, d2):
    return (to_date(d2) - to_date(d1)).days / 365.0


def get_spot_yf(ticker: str) -> float:
    hist = yf.Ticker(ticker).history(period="10d", auto_adjust=False)
    if hist.empty:
        raise ValueError(f"No price data for {ticker}")
    return float(hist["Close"].dropna().iloc[-1])


def fetch_page(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.text


def mc_url(ticker: str, kind: str) -> str:
    ticker = ticker.upper().strip()
    if kind == "iv":
        return f"https://marketchameleon.com/Overview/{ticker}/IV/"
    if kind == "div":
        return f"https://marketchameleon.com/Overview/{ticker}/Dividends/"
    if kind == "spot":
        return f"https://marketchameleon.com/Overview/{ticker}/Summary/"
    if kind == "strike":
        return f"https://marketchameleon.com/Overview/{ticker}/Summary/"
    raise ValueError(kind)


def _first_float(patterns, text):
    for pat in patterns:
        m = re.search(pat, text, flags=re.I | re.S)
        if m:
            val = m.group(1).replace(",", "").strip()
            try:
                return float(val)
            except Exception:
                pass
    return None


def parse_mc_iv(html: str):
    soup = BeautifulSoup(html, "html.parser")
    txt = soup.get_text(" ", strip=True)
    val = _first_float(
        [
            r"implied volatility \\(IV\\) is\s*([0-9]+(?:\.[0-9]+)?)",
            r"IV\)\s*is\s*([0-9]+(?:\.[0-9]+)?)",
            r"implied volatility[^0-9]*([0-9]+(?:\.[0-9]+)?)",
        ],
        txt,
    )
    if val is None:
        raise ValueError("Could not parse implied volatility from Market Chameleon.")
    return val / 100.0 if val > 5 else val


def parse_mc_div_yield(html: str):
    soup = BeautifulSoup(html, "html.parser")
    txt = soup.get_text(" ", strip=True)
    val = _first_float(
        [
            r"Forward Dividend Yield:\s*([0-9]+(?:\.[0-9]+)?)%",
            r"Dividend Yield:\s*([0-9]+(?:\.[0-9]+)?)%",
            r"Div Yield:\s*([0-9]+(?:\.[0-9]+)?)%",
        ],
        txt,
    )
    if val is None:
        raise ValueError("Could not parse dividend yield from Market Chameleon.")
    return val / 100.0


def parse_mc_spot(html: str):
    soup = BeautifulSoup(html, "html.parser")
    txt = soup.get_text(" ", strip=True)
    val = _first_float(
        [
            r"Today\s+[+\-]?[0-9.]+\s+([0-9]+(?:\.[0-9]+)?)",
            r"Last Price\s*([0-9]+(?:\.[0-9]+)?)",
            r"price quote.*?([0-9]+(?:\.[0-9]+)?)",
        ],
        txt,
    )
    if val is None:
        raise ValueError("Could not parse spot from Market Chameleon.")
    return val


def get_market_chameleon_data(ticker: str):
    ticker = ticker.upper().strip()
    data = {}
    errors = []

    try:
        html = fetch_page(mc_url(ticker, "iv"))
        data["Implied Vol"] = parse_mc_iv(html)
    except Exception as e:
        errors.append(f"IV: {e}")

    try:
        html = fetch_page(mc_url(ticker, "div"))
        data["Dividend Yield"] = parse_mc_div_yield(html)
    except Exception as e:
        errors.append(f"Div: {e}")

    try:
        html = fetch_page(mc_url(ticker, "spot"))
        data["Spot"] = parse_mc_spot(html)
    except Exception as e:
        errors.append(f"Spot: {e}")

    return data, errors


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
    obs_months,
    funding_cost,
    notional,
    call_barrier,
    coupon_barrier,
    knock_in_barrier,
    coupon_per_obs,
    n_paths=20000,
    seed=11,
):
    maturity = max(year_frac(valuation_date, maturity_date), 0.01)
    steps = max(252, int(252 * maturity))
    paths = simulate_paths(spot, vol, div, corr, maturity, n_paths=n_paths, steps=steps, seed=seed)

    call_times = [year_frac(valuation_date, valuation_date + timedelta(days=int(30.4375 * m))) for m in obs_months]
    call_idx = [min(steps, max(1, int(round(t / maturity * steps)))) for t in call_times]
    maturity_idx = steps

    worst_call = worst_of_ratio(paths, spot)
    worst_maturity = worst_call[:, maturity_idx]

    pv = np.zeros(n_paths)
    called = np.zeros(n_paths, dtype=bool)
    call_time_bucket = np.full(n_paths, np.nan)

    for idx in call_idx:
        alive = ~called
        trigger = alive & (worst_call[:, idx] >= call_barrier)
        if np.any(trigger):
            t = idx / steps * maturity
            coupon_cashflow = notional * coupon_per_obs * max(1.0, round(t / max(maturity / max(len(call_idx), 1), 1e-9)))
            note_cashflow = notional + coupon_cashflow
            pv[trigger] += note_cashflow
            called[trigger] = True
            call_time_bucket[trigger] = t

    alive = ~called
    if np.any(alive):
        terminal = worst_maturity[alive]
        coupon_cashflow = notional * coupon_per_obs * maturity

        redemption = np.where(
            terminal >= coupon_barrier,
            notional + coupon_cashflow,
            np.where(
                terminal >= knock_in_barrier,
                notional + coupon_cashflow,
                notional * terminal,
            ),
        )
        pv[alive] += redemption

    note_pv = pv.mean()
    call_prob = float(np.mean(called))
    expected_call_time = float(np.nanmean(call_time_bucket)) if np.any(~np.isnan(call_time_bucket)) else float("nan")
    expected_payoff = float(pv.mean() / notional)

    option_value = max(0.0, float((note_pv / notional) - 1.0))
    coupon_rate = funding_cost + option_value

    return {
        "note_pv": note_pv,
        "coupon_rate": coupon_rate,
        "funding_cost": funding_cost,
        "option_value": option_value,
        "call_prob": call_prob,
        "expected_call_time": expected_call_time,
        "expected_payoff": expected_payoff,
        "paths": paths,
        "worst_call": worst_call,
        "worst_maturity": worst_maturity,
    }


def build_schedule(valuation_date, obs_months, maturity_date):
    sch_rows = [{"Type": "Valuation", "Date": str(valuation_date)}]
    sch_rows += [{"Type": "Call/Coupon Obs", "Date": str(valuation_date + timedelta(days=int(30.4375 * m)))} for m in obs_months]
    sch_rows += [{"Type": "Maturity", "Date": str(maturity_date)}]
    return pd.DataFrame(sch_rows)


def format_underlying_table(inp, call_barrier, coupon_barrier, knock_in_barrier):
    out = inp.copy()
    out["Knock-in strike"] = out["Spot"] * knock_in_barrier
    out["Coupon strike"] = out["Spot"] * coupon_barrier
    out["Call strike"] = out["Spot"] * call_barrier
    out["Implied Vol"] = out["Vol"].map(lambda x: f"{x:.2%}")
    out["Current Dividend Yield"] = out["Dividend Yield"].map(lambda x: f"{x:.2%}")
    out["Spot"] = out["Spot"].map(lambda x: f"{x:.2f}")
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
.block-container {
    max-width: 1200px !important;
    margin-left: auto;
    margin-right: auto;
    padding-top: 1.25rem;
}
</style>
""",
    unsafe_allow_html=True,
)

if "step" not in st.session_state:
    st.session_state.step = 1
if "market_df" not in st.session_state:
    st.session_state.market_df = None

st.header("Inputs and assumptions")

col1, col2 = st.columns([2, 1])
with col1:
    n_names = st.selectbox("Number of underlyings", [1, 2, 3, 4], index=2, key="n_names")
    default_tickers = ["DRAM", "SOXX", "FXI", "XLK"]
    tickers = []
    for i in range(n_names):
        tickers.append(st.text_input(f"Ticker {i+1}", value=default_tickers[i], key=f"ticker_{i}"))
    valuation_date = st.date_input("Valuation date", value=date.today(), key="valuation_date")
    maturity_date = st.date_input("Maturity date", value=date.today() + timedelta(days=540), key="maturity_date")
    notional = st.number_input("Notional", min_value=1.0, value=100000.0, step=1000.0, key="notional")

with col2:
    st.markdown("### Funding")
    funding_cost = st.number_input("Funding cost (annual)", min_value=0.0, value=0.035, step=0.005, format="%.4f", key="funding_cost")
    st.caption("Funding cost is an issuer input. The market data is confirmed in step two before calculation.")

c1, c2, c3 = st.columns(3)
with c1:
    call_barrier = st.number_input("Call barrier", min_value=0.0, value=1.0, step=0.01, format="%.4f", key="call_barrier")
    st.caption("If the worst-performing underlying is at or above this level on an observation date, the note can autocall.")
    coupon_barrier = st.number_input("Coupon barrier", min_value=0.0, value=0.5, step=0.01, format="%.4f", key="coupon_barrier")
    st.caption("If the worst-performing underlying stays at or above this level, the coupon condition is met.")
with c2:
    knock_in_barrier = st.number_input("Knock-in barrier", min_value=0.0, value=0.5, step=0.01, format="%.4f", key="knock_in_barrier")
    st.caption("If the worst-performing underlying falls below this level, downside principal exposure is activated.")
    n_paths = st.number_input("Simulation paths", min_value=1000, value=12000, step=1000, key="n_paths")
with c3:
    seed = st.number_input("Random seed", min_value=1, value=11, step=1, key="seed")
    st.markdown("### Observation dates")
    obs_months = st.multiselect("Months from valuation", options=list(range(1, 61)), default=[6], key="obs_months")

st.markdown("### Step 1")
st.write("Enter the product terms above.")

if st.button("Load / Confirm market data", type="primary", use_container_width=True):
    rows = []
    errors = []
    for t in tickers:
        try:
            spot_yf = get_spot_yf(t)
        except Exception:
            spot_yf = np.nan

        data = {"Ticker": t, "Spot": spot_yf, "Vol": np.nan, "Dividend Yield": np.nan}
        try:
            mc_data, mc_errors = get_market_chameleon_data(t)
            data.update({k: v for k, v in mc_data.items() if k in data})
            if mc_errors:
                errors.append(f"{t}: " + " | ".join(mc_errors))
        except Exception as e:
            errors.append(f"{t}: {e}")
        rows.append(data)

    st.session_state.market_df = pd.DataFrame(rows)
    st.session_state.step = 2
    if errors:
        st.warning("Some Market Chameleon fields could not be loaded. You can edit them manually in step two.")

if st.session_state.step == 2 and st.session_state.market_df is not None:
    st.markdown("### Step 2")
    st.write("Review the market data below. Edit anything unusual, then press Calculate.")

    editor_df = st.session_state.market_df.copy()
    editor_df["Spot"] = pd.to_numeric(editor_df["Spot"], errors="coerce")
    editor_df["Vol"] = pd.to_numeric(editor_df["Vol"], errors="coerce")
    editor_df["Dividend Yield"] = pd.to_numeric(editor_df["Dividend Yield"], errors="coerce")

    edited = st.data_editor(
        editor_df,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "Spot": st.column_config.NumberColumn("Spot", help="Market spot price"),
            "Vol": st.column_config.NumberColumn("Implied Vol", help="Enter as decimal, e.g. 0.615 for 61.5%"),
            "Dividend Yield": st.column_config.NumberColumn("Dividend Yield", help="Enter as decimal, e.g. 0.002 for 0.2%"),
        },
        key="market_editor",
    )

    if st.button("Calculate", type="primary", use_container_width=True):
        try:
            inp = edited.copy()
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
                coupon_per_obs=0.0,
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
            underlying_df = format_underlying_table(inp, call_barrier, coupon_barrier, knock_in_barrier)
            st.dataframe(underlying_df, use_container_width=True, hide_index=True)

            st.subheader("Correlation matrix")
            st.dataframe(corr.round(4), use_container_width=True)
            st.dataframe(corr_long_df(corr).round(4), use_container_width=True, hide_index=True)

            st.subheader("Schedule")
            st.dataframe(build_schedule(valuation_date, obs_months, maturity_date), use_container_width=True, hide_index=True)

            st.subheader("Model notes and references")
            st.write(
                "This app uses Market Chameleon market data where available, with user-editable fallbacks in step two. "
                "The pricing engine uses the basket's worst-of dynamics, and the displayed market fields are confirmed before calculation."
            )
            st.markdown("- [Market Chameleon DRAM IV](https://marketchameleon.com/Overview/DRAM/IV/)")
            st.markdown("- [Market Chameleon SOXX IV](https://marketchameleon.com/Overview/SOXX/IV/)")
            st.markdown("- [Market Chameleon FXI IV](https://marketchameleon.com/Overview/FXI/IV/)")
            st.markdown("- [Market Chameleon DRAM Dividends](https://marketchameleon.com/Overview/DRAM/Dividends/)")
            st.markdown("- [Market Chameleon SOXX Dividends](https://marketchameleon.com/Overview/SOXX/Dividends/)")
            st.markdown("- [Market Chameleon FXI Dividends](https://marketchameleon.com/Overview/FXI/Dividends/)")

            csv = underlying_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download underlying details CSV", csv, file_name="fcn_underlying_details.csv", mime="text/csv")

        except Exception as e:
            st.error(str(e))

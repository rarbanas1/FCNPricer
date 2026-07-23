
import math
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="FCN Pricer", layout="wide")
st.title("FCN Pricer")
st.caption("Term-sheet-based FCN / autocallable pricer (Monte Carlo, worst-of payoff).")

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
    call_dates,
    coupon_dates,
    obs_months,
    funding_cost,
    notional,
    call_barrier,
    coupon_barrier,
    knock_in_barrier,
    coupon_frequency_per_year=1.0,
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

    coupon_times = np.array(obs_months, dtype=float) / 12.0
    if coupon_times.size == 0:
        coupon_times = np.array([maturity])

    # Option leg valuation:
    # We price the worst-of embedded derivative as the present value of the product cashflows
    # net of principal/funding leg. This is the robust, defendable option component used to set the coupon.
    option_leg_pv = np.zeros(n_paths)

    for idx in call_idx:
        alive = ~called
        trigger = alive & (worst_call[:, idx] >= call_barrier)
        if np.any(trigger):
            t = idx / steps * maturity
            principal_pv = notional * dfs[idx]
            coupon_cashflow = notional * funding_cost * t * dfs[idx]
            note_cashflow = (notional + coupon_cashflow) * dfs[idx]
            pv[trigger] += note_cashflow
            option_leg_pv[trigger] += note_cashflow - principal_pv - coupon_cashflow
            called[trigger] = True
            call_time_bucket[trigger] = t

    alive = ~called
    if np.any(alive):
        t = maturity
        terminal = worst_maturity[alive]

        coupon_cashflow = notional * funding_cost * t * dfs[maturity_idx]
        base_principal_pv = notional * dfs[maturity_idx]

        redemption = np.where(
            terminal >= coupon_barrier,
            notional + coupon_cashflow,
            np.where(
                terminal >= knock_in_barrier,
                notional + coupon_cashflow,
                notional * terminal,
            ),
        )

        redemption_pv = redemption * dfs[maturity_idx]
        pv[alive] += redemption_pv
        option_leg_pv[alive] += redemption_pv - base_principal_pv - coupon_cashflow

    note_pv = pv.mean()
    call_prob = float(np.mean(called))
    expected_call_time = float(np.nanmean(call_time_bucket)) if np.any(~np.isnan(call_time_bucket)) else float("nan")
    expected_payoff = float(pv.mean() / notional)

    option_value = float(option_leg_pv.mean() / notional)
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


def build_schedule(valuation_date, call_dates, coupon_dates, maturity_date):
    sch_rows = [{"Type": "Valuation", "Date": str(valuation_date)}]
    sch_rows += [{"Type": "Call", "Date": str(d)} for d in call_dates]
    sch_rows += [{"Type": "Coupon", "Date": str(d)} for d in coupon_dates]
    sch_rows += [{"Type": "Maturity", "Date": str(maturity_date)}]
    return pd.DataFrame(sch_rows)


def format_underlying_table(inp, call_barrier, coupon_barrier, knock_in_barrier):
    out = inp.copy()
    out["Knock-in strike"] = out["Spot"] * knock_in_barrier
    out["Coupon strike"] = out["Spot"] * coupon_barrier
    out["Call strike"] = out["Spot"] * call_barrier
    out["Vol"] = out["Vol"].map(lambda x: f"{x:.2%}")
    out["Dividend Yield"] = out["Dividend Yield"].map(lambda x: f"{x:.2%}")
    return out


def corr_long_df(corr_df):
    out = corr_df.copy()
    out.insert(0, "Underlying", out.index)
    return out.reset_index(drop=True)


st.markdown(
    """
<style>
.block-container {
    max-width: 1150px !important;
    margin-left: auto;
    margin-right: auto;
    padding-top: 1.5rem;
}
</style>
""",
    unsafe_allow_html=True,
)

st.header("Inputs and assumptions")

col1, col2 = st.columns([2, 1])
with col1:
    n_names = st.selectbox("Number of underlyings", [1, 2, 3, 4], index=2)
    tickers = []
    default_tickers = ["C", "JPM", "BAC", "WFC"]
    for i in range(n_names):
        tickers.append(st.text_input(f"Ticker {i+1}", value=default_tickers[i]))
    valuation_date = st.date_input("Valuation date", value=date.today())
    maturity_date = st.date_input("Maturity date", value=date.today() + timedelta(days=540))
    notional = st.number_input("Notional", min_value=1.0, value=1000000.0, step=10000.0)

with col2:
    st.markdown("### Funding")
    funding_cost = st.number_input(
        "Funding cost (annual)", min_value=0.0, value=0.05, step=0.005, format="%.4f"
    )
    st.caption("Funding cost is an issuer input. The coupon is solved as funding cost plus the priced worst-of option leg.")

c1, c2, c3 = st.columns(3)
with c1:
    call_barrier = st.number_input("Call barrier", min_value=0.0, value=1.0, step=0.01, format="%.4f")
    st.caption("If the worst-performing underlying is at or above this level on an observation date, the note can autocall.")
    coupon_barrier = st.number_input("Coupon barrier", min_value=0.0, value=0.6, step=0.01, format="%.4f")
    st.caption("If the worst-performing underlying stays at or above this level, the coupon condition is met.")
with c2:
    knock_in_barrier = st.number_input("Knock-in barrier", min_value=0.0, value=0.6, step=0.01, format="%.4f")
    st.caption("If the worst-performing underlying falls below this level, downside principal exposure is activated.")
    n_paths = st.number_input("Simulation paths", min_value=1000, value=12000, step=1000)
with c3:
    seed = st.number_input("Random seed", min_value=1, value=11, step=1)
    st.markdown("### Observation dates")
    obs_months = st.multiselect(
        "Months from valuation",
        options=list(range(1, 61)),
        default=[6, 12, 18],
    )

calc_col1, calc_col2, calc_col3 = st.columns([1, 2, 1])
with calc_col2:
    calculate = st.button("Calculate", type="primary", use_container_width=True)

if not calculate:
    st.info("Set inputs above and click Calculate to run the pricer.")

if calculate:
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
        r6.metric(
            "Expected call time",
            "N/A" if np.isnan(result["expected_call_time"]) else f"{result['expected_call_time']:.2f}y",
        )

        st.subheader("Underlying details")
        underlying_df = format_underlying_table(inp, call_barrier, coupon_barrier, knock_in_barrier)
        st.dataframe(underlying_df, use_container_width=True, hide_index=True)

        st.subheader("Correlation matrix")
        st.dataframe(corr.round(4), use_container_width=True)
        st.dataframe(corr_long_df(corr).round(4), use_container_width=True, hide_index=True)

        st.subheader("Schedule")
        st.dataframe(
            build_schedule(valuation_date, call_dates, coupon_dates, maturity_date),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Model notes and references")
        st.write(
            "This pricer models the structured note as a financing leg plus a derivative leg, "
            "with final coupon = funding_cost + option_value. "
            "The payoff is worst-of: the minimum performance across the selected underlyings drives the call and maturity outcomes. "
            "Dividend yield, volatility and correlation are material inputs to worst-of pricing and risk."
        )
        st.markdown(
            "- [The Interplay between Stochastic Volatility and Correlations in Equity Autocallables](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3228065)"
        )
        st.markdown(
            "- [Risk misperceptions of structured financial products with worst-of payout characteristics revisited](https://www.econstor.eu/bitstream/10419/248602/1/DP1143R.pdf)"
        )
        st.markdown(
            "- [A Bayesian view on autocallable pricing and risk management](https://sussex.figshare.com/articles/journal_contribution/A_Bayesian_view_on_autocallable_pricing_and_risk_management/23491451)"
        )

        csv = underlying_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download underlying details CSV", csv, file_name="fcn_underlying_details.csv", mime="text/csv")

    except Exception as e:
        st.error(str(e))


import math
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

# =========================
# API KEYS: FILL THESE IN
# =========================
ALPHAVANTAGE_API_KEY = st.secrets["ALPHAVANTAGE_API_KEY"]
FINNHUB_API_KEY = st.secrets["FINNHUB_API_KEY"]
# =========================

st.set_page_config(page_title="FCN Pricer", layout="wide")
st.title("FCN Pricer")
st.caption("Worst-of FCN / autocallable pricer with multi-source market data review.")

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


def normalize_pct(x):
    if x in (None, "", "None"):
        raise ValueError("missing value")
    x = float(x)
    if x > 1.0:
        x = x / 100.0
    return x


def yf_spot(ticker: str) -> float:
    hist = yf.Ticker(ticker).history(period="10d", auto_adjust=False)
    if hist.empty:
        raise ValueError(f"No price data for {ticker}")
    return float(hist["Close"].dropna().iloc[-1])


def alpha_spot(ticker: str) -> float:
    r = requests.get(
        ALPHAV_URL,
        params={"function": "GLOBAL_QUOTE", "symbol": ticker, "apikey": ALPHAVANTAGE_API_KEY},
        timeout=20,
    )
    r.raise_for_status()
    data = r.json().get("Global Quote", {})
    px = data.get("05. price")
    if px is None:
        raise ValueError("Alpha Vantage spot unavailable")
    return float(px)


def finnhub_spot(ticker: str) -> float:
    r = requests.get(
        f"{FINNHUB_URL}/quote",
        params={"symbol": ticker, "token": FINNHUB_API_KEY},
        timeout=20,
    )
    r.raise_for_status()
    data = r.json()
    if data.get("c") is None:
        raise ValueError("Finnhub spot unavailable")
    return float(data["c"])


def _first_numeric(*values):
    for v in values:
        if v in (None, "", "None"):
            continue
        try:
            x = float(v)
        except Exception:
            continue
        if np.isfinite(x):
            return x
    return None


def alpha_dividend_yield(ticker: str):
    r = requests.get(
        ALPHAV_URL,
        params={"function": "OVERVIEW", "symbol": ticker, "apikey": ALPHAVANTAGE_API_KEY},
        timeout=20,
    )
    r.raise_for_status()
    data = r.json()
    dy = data.get("DividendYield")
    if dy in (None, "", "None"):
        raise ValueError("Alpha Vantage dividend yield unavailable")
    return normalize_pct(dy)


def finnhub_dividend_yield_from_profile(ticker: str):
    r = requests.get(
        f"{FINNHUB_URL}/stock/profile2",
        params={"symbol": ticker, "token": FINNHUB_API_KEY},
        timeout=20,
    )
    r.raise_for_status()
    data = r.json()
    dy = data.get("dividendYield")
    if dy in (None, "", "None"):
        raise ValueError("Finnhub profile dividend yield unavailable")
    return normalize_pct(dy)


def finnhub_trailing_dividend_yield(ticker: str):
    r = requests.get(
        f"{FINNHUB_URL}/stock/dividend",
        params={"symbol": ticker, "token": FINNHUB_API_KEY},
        timeout=20,
    )
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Finnhub dividend history unavailable")
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=365)
    total = 0.0
    for row in data:
        amt = row.get("amount")
        dt = row.get("date")
        if amt is None or dt is None:
            continue
        try:
            d = pd.to_datetime(dt)
        except Exception:
            continue
        if start <= d <= end:
            total += float(amt)
    px = finnhub_spot(ticker)
    if px <= 0:
        raise ValueError("Invalid spot for dividend yield calc")
    return total / px


def yfinance_dividend_yield(ticker: str):
    info = yf.Ticker(ticker).info or {}
    dy = _first_numeric(
        info.get("trailingAnnualDividendYield"),
        info.get("yield"),
        info.get("dividendYield"),
    )
    if dy is None:
        raise ValueError("yfinance dividend yield unavailable")
    return normalize_pct(dy)


def dividend_yield_from_history(ticker: str):
    ticker_obj = yf.Ticker(ticker)
    hist = ticker_obj.history(period="2y", auto_adjust=False)
    if hist.empty or "Dividends" not in hist.columns:
        raise ValueError("Dividend history unavailable")
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=365)
    divs = hist["Dividends"].copy()
    divs.index = pd.to_datetime(divs.index)
    total = float(divs[divs.index >= cutoff].fillna(0.0).sum())
    if total <= 0:
        raise ValueError("No trailing dividends found")
    px = float(hist["Close"].dropna().iloc[-1])
    if px <= 0:
        raise ValueError("Invalid spot for dividend yield calc")
    return total / px


def dividend_yield_loader(ticker: str):
    ticker = ticker.upper().strip()
    errors = []
    for name, fn in [
        ("Finnhub", lambda: finnhub_dividend_yield_from_profile(ticker)),
        ("Alpha Vantage", lambda: alpha_dividend_yield(ticker)),
        ("yfinance info", lambda: yfinance_dividend_yield(ticker)),
        ("yfinance history", lambda: dividend_yield_from_history(ticker)),
    ]:
        try:
            return fn(), name
        except Exception as e:
            errors.append(f"{name}: {e}")
    raise ValueError("Dividend yield unavailable after fallbacks: " + " | ".join(errors))


def alpha_iv_from_history(ticker: str):
    # Best effort: use historical options if available; otherwise fall back to realized-vol estimate.
    for params in [
        {"function": "HISTORICAL_OPTIONS", "symbol": ticker, "apikey": ALPHAVANTAGE_API_KEY},
        {"function": "HISTORICAL_OPTIONS", "symbol": ticker, "date": str(date.today()), "apikey": ALPHAVANTAGE_API_KEY},
    ]:
        try:
            r = requests.get(ALPHAV_URL, params=params, timeout=20)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, list) and v:
                        ivs = []
                        for row in v:
                            for key in ("impliedVolatility", "iv", "implied_volatility"):
                                if key in row and row[key] not in (None, "", "None"):
                                    try:
                                        val = float(row[key])
                                        ivs.append(val / 100.0 if val > 5 else val)
                                    except Exception:
                                        pass
                        if ivs:
                            return float(np.nanmedian(ivs))
        except Exception:
            pass
    hist = yf.Ticker(ticker).history(period="1y", auto_adjust=False)
    if hist.empty:
        raise ValueError("IV unavailable")
    r = np.log(hist["Close"].dropna()).diff().dropna()
    if r.empty:
        raise ValueError("IV unavailable")
    return float(max(0.05, min(2.0, r.std() * np.sqrt(252))))


def fetch_candidates(ticker: str):
    ticker = ticker.upper().strip()
    cand = {
        "Ticker": ticker,
        "Spot - Finnhub": np.nan,
        "Spot - Alpha Vantage": np.nan,
        "Spot - yfinance": np.nan,
        "IV - Alpha Vantage": np.nan,
        "Div - Finnhub profile": np.nan,
        "Div - Finnhub trailing": np.nan,
        "Div - Alpha Vantage": np.nan,
        "Div - yfinance info": np.nan,
        "Div - yfinance history": np.nan,
        "Div - Loader": np.nan,
        "Div - Source": "",
    }
    try:
        cand["Spot - Finnhub"] = finnhub_spot(ticker)
    except Exception:
        pass
    try:
        cand["Spot - Alpha Vantage"] = alpha_spot(ticker)
    except Exception:
        pass
    try:
        cand["Spot - yfinance"] = yf_spot(ticker)
    except Exception:
        pass
    try:
        cand["IV - Alpha Vantage"] = alpha_iv_from_history(ticker)
    except Exception:
        pass
    try:
        cand["Div - Finnhub profile"] = finnhub_dividend_yield_from_profile(ticker)
    except Exception:
        pass
    try:
        cand["Div - Finnhub trailing"] = finnhub_trailing_dividend_yield(ticker)
    except Exception:
        pass
    try:
        cand["Div - Alpha Vantage"] = alpha_dividend_yield(ticker)
    except Exception:
        pass
    try:
        cand["Div - yfinance info"] = yfinance_dividend_yield(ticker)
    except Exception:
        pass
    try:
        cand["Div - yfinance history"] = dividend_yield_from_history(ticker)
    except Exception:
        pass
    try:
        dy, src = dividend_yield_loader(ticker)
        cand["Div - Loader"] = dy
        cand["Div - Source"] = src
    except Exception:
        pass
    return cand


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
    corr_df = ret.corr().reindex(index=tickers, columns=tickers).fillna(0.0)
    arr = corr_df.to_numpy(copy=True)
    arr = np.array(arr, dtype=float, copy=True)
    np.fill_diagonal(arr, 1.0)
    return pd.DataFrame(arr, index=tickers, columns=tickers)


def cholesky_with_fallback(corr):
    arr = np.array(corr, dtype=float, copy=True)
    try:
        return np.linalg.cholesky(arr)
    except Exception:
        eigvals, eigvecs = np.linalg.eigh(arr)
        eigvals = np.clip(eigvals, 1e-8, None)
        corr2 = eigvecs @ np.diag(eigvals) @ eigvecs.T
        corr2 = np.array(corr2, dtype=float, copy=True)
        np.fill_diagonal(corr2, 1.0)
        return np.linalg.cholesky(corr2)


def simulate_paths(spot, vol, div, r, corr, maturity, n_paths=12000, steps=252, seed=11):
    rng = np.random.default_rng(seed)
    n = len(spot)
    dt = maturity / steps
    chol = cholesky_with_fallback(corr)
    paths = np.zeros((n_paths, steps + 1, n), dtype=float)
    paths[:, 0, :] = spot
    # Risk-neutral drift: r (risk-free) - q (dividend yield) - 0.5*sigma^2.
    # r was previously hardcoded to 0 here, which understates drift and,
    # combined with no discounting downstream, materially mispriced the note.
    drift = np.array([(r - d - 0.5 * v * v) for d, v in zip(div, vol)], dtype=float)
    diff = np.array(vol, dtype=float)

    for t in range(1, steps + 1):
        z = rng.standard_normal((n_paths, n))
        zc = z @ chol.T
        for i in range(n):
            paths[:, t, i] = paths[:, t - 1, i] * np.exp(drift[i] * dt + diff[i] * math.sqrt(dt) * zc[:, i])
    return paths


def worst_of_ratio(paths, spot):
    return (paths / np.array(spot)[None, None, :]).min(axis=2)


def price_fcn(spot, vol, div, corr, r, valuation_date, maturity_date, obs_months, funding_cost, notional,
              call_barrier, coupon_barrier, knock_in_barrier, n_paths=12000, seed=11):
    if coupon_barrier <= 0:
        raise ValueError("Coupon barrier (strike) must be > 0.")

    maturity = max(year_frac(valuation_date, maturity_date), 0.01)
    steps = max(252, int(252 * maturity))
    paths = simulate_paths(spot, vol, div, r, corr, maturity, n_paths=n_paths, steps=steps, seed=seed)

    obs_times = [year_frac(valuation_date, valuation_date + timedelta(days=int(30.4375 * m))) for m in obs_months]
    obs_idx = [min(steps, max(1, int(round(t / maturity * steps)))) for t in obs_times]

    worst = worst_of_ratio(paths, spot)
    terminal = worst[:, -1]

    # The coupon (funding_cost) enters every payoff branch linearly, so instead
    # of simulating at one fixed coupon and hoping it happens to price to par,
    # we split each path's discounted cash flow into two pieces:
    #   A = principal-side PV, independent of the coupon rate
    #   B = PV of a 1.0 coupon-rate's worth of accrual on that path
    # note_pv(c) = A + c*B for ANY coupon c, so the coupon that prices the
    # note to par is solved directly: fair_coupon = (notional - A) / B.
    # This also removes the old max(0, ...) floor, which was silently
    # hiding negative results instead of reporting them.
    A = np.zeros(n_paths)
    B = np.zeros(n_paths)
    called = np.zeros(n_paths, dtype=bool)
    call_time_bucket = np.full(n_paths, np.nan)

    for idx in obs_idx:
        alive = ~called
        trigger = alive & (worst[:, idx] >= call_barrier)
        if np.any(trigger):
            t = idx / steps * maturity
            df = math.exp(-r * t)
            A[trigger] += notional * df
            B[trigger] += notional * t * df
            called[trigger] = True
            call_time_bucket[trigger] = t

    alive = ~called
    df_T = math.exp(-r * maturity)
    if np.any(alive):
        # Knock-in barrier gates protection. If knock_in_barrier < coupon_barrier,
        # the zone between them is a cushion: still fully protected (par + coupon)
        # even though the worst performer is already below the strike.
        safe = alive & (terminal >= knock_in_barrier)
        breach = alive & (terminal < knock_in_barrier)

        A[safe] += notional * df_T
        B[safe] += notional * maturity * df_T

        # Below knock-in, the embedded put (struck at coupon_barrier) is live:
        # redemption reflects the worst performer's level relative to the
        # strike, not relative to 100% spot. No coupon accrues on these paths.
        A[breach] += notional * (terminal[breach] / coupon_barrier) * df_T

    A_bar = float(A.mean())
    B_bar = float(B.mean())

    if B_bar <= 0:
        fair_coupon = float("nan")
        option_value = float("nan")
        note_pv_fair = float("nan")
    else:
        fair_coupon = (notional - A_bar) / B_bar
        option_value = fair_coupon - funding_cost
        note_pv_fair = A_bar + fair_coupon * B_bar

    note_pv_input = A_bar + funding_cost * B_bar
    call_prob = float(np.mean(called))
    expected_call_time = float(np.nanmean(call_time_bucket)) if np.any(~np.isnan(call_time_bucket)) else float("nan")

    return {
        "note_pv": note_pv_input,
        "note_pv_at_fair_coupon": note_pv_fair,
        "coupon_rate": fair_coupon,
        "funding_cost": funding_cost,
        "option_value": option_value,
        "call_prob": call_prob,
        "expected_call_time": expected_call_time,
        "discount_rate": r,
    }


def build_schedule(valuation_date, obs_months, maturity_date):
    rows = [{"Type": "Valuation", "Date": str(valuation_date)}]
    rows += [{"Type": "Call/Coupon Obs", "Date": str(valuation_date + timedelta(days=int(30.4375 * m)))} for m in obs_months]
    rows += [{"Type": "Maturity", "Date": str(maturity_date)}]
    return pd.DataFrame(rows)


def fmt(x, pct=False):
    if pd.isna(x):
        return ""
    return f"{x:.2%}" if pct else f"{x:.4f}" if abs(x) < 1 else f"{x:.2f}"


if "step" not in st.session_state:
    st.session_state.step = 1
if "candidates" not in st.session_state:
    st.session_state.candidates = None

st.header("Inputs and assumptions")

c1, c2 = st.columns([2, 1])
with c1:
    n_names = st.selectbox("Number of underlyings", [1, 2, 3, 4], index=2)
    default_tickers = ["DRAM", "SOXX", "FXI", "XLK"]
    tickers = [st.text_input(f"Ticker {i+1}", value=default_tickers[i]) for i in range(n_names)]
    valuation_date = st.date_input("Valuation date", value=date.today())
    maturity_date = st.date_input("Maturity date", value=date.today() + timedelta(days=540))
    notional = st.number_input("Notional", min_value=1.0, value=100000.0, step=1000.0)
with c2:
    funding_cost = st.number_input("Funding cost (annual)", min_value=0.0, value=0.035, step=0.005, format="%.4f")
    risk_free_rate = st.number_input(
        "Risk-free rate (annual)",
        min_value=0.0,
        value=0.04,
        step=0.0025,
        format="%.4f",
        help="Used for risk-neutral drift and discounting. Previously hardcoded to 0 in the pricing model.",
    )

b1, b2, b3 = st.columns(3)
with b1:
    call_barrier = st.number_input("Call barrier", min_value=0.0, value=1.0, step=0.01, format="%.4f")
    coupon_barrier = st.number_input("Coupon barrier", min_value=0.0, value=0.5, step=0.01, format="%.4f")
with b2:
    knock_in_barrier = st.number_input("Knock-in barrier", min_value=0.0, value=0.5, step=0.01, format="%.4f")
    n_paths = st.number_input("Simulation paths", min_value=1000, value=12000, step=1000)
with b3:
    seed = st.number_input("Random seed", min_value=1, value=11, step=1)
    obs_months = st.multiselect("Months from valuation", options=list(range(1, 61)), default=[6])

st.markdown("### Step 1")
st.write("Enter the product terms above.")

if st.button("Load market candidates", type="primary", use_container_width=True):
    rows = []
    for t in tickers:
        rows.append(fetch_candidates(t))
    st.session_state.candidates = pd.DataFrame(rows)
    st.session_state.step = 2

if st.session_state.step >= 2 and st.session_state.candidates is not None:
    st.markdown("### Step 2")
    st.write("Review the available candidates and choose the source for each field.")

    cand = st.session_state.candidates.copy()
    st.dataframe(cand, use_container_width=True, hide_index=True)

    selected_rows = []
    for i, t in enumerate(tickers):
        row = cand.iloc[i].to_dict()
        st.markdown(f"#### {t.upper().strip()}")
        spot_source = st.selectbox(
            f"{t} spot source",
            ["Finnhub", "Alpha Vantage", "yfinance"],
            index=0,
            key=f"spot_src_{i}",
        )
        iv_source = st.selectbox(
            f"{t} IV source",
            ["Alpha Vantage", "Manual"],
            index=0 if pd.notna(row.get("IV - Alpha Vantage")) else 1,
            key=f"iv_src_{i}",
        )
        div_source = st.selectbox(
            f"{t} dividend source",
            ["Finnhub profile", "Finnhub trailing", "Alpha Vantage", "yfinance info", "yfinance history", "Auto", "Manual"],
            index=5,
            key=f"div_src_{i}",
        )

        spot_val = row.get("Spot - Finnhub") if spot_source == "Finnhub" else row.get("Spot - Alpha Vantage") if spot_source == "Alpha Vantage" else row.get("Spot - yfinance")
        iv_val = row.get("IV - Alpha Vantage") if iv_source == "Alpha Vantage" else np.nan
        if div_source == "Finnhub profile":
            div_val = row.get("Div - Finnhub profile")
        elif div_source == "Finnhub trailing":
            div_val = row.get("Div - Finnhub trailing")
        elif div_source == "Alpha Vantage":
            div_val = row.get("Div - Alpha Vantage")
        elif div_source == "yfinance info":
            div_val = row.get("Div - yfinance info")
        elif div_source == "yfinance history":
            div_val = row.get("Div - yfinance history")
        elif div_source == "Auto":
            div_val = row.get("Div - Loader")
        else:
            div_val = np.nan

        selected_rows.append(
            {
                "Ticker": t.upper().strip(),
                "Spot": spot_val,
                "Vol": iv_val,
                "Dividend Yield": div_val,
                "Spot Source": spot_source,
                "IV Source": iv_source,
                "Div Source": div_source,
            }
        )

    selected_df = pd.DataFrame(selected_rows)
    st.markdown("### Confirmed inputs")
    edited = st.data_editor(
        selected_df,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "Spot": st.column_config.NumberColumn("Spot"),
            "Vol": st.column_config.NumberColumn("Implied Vol", help="Decimal form, e.g. 0.615 = 61.5%"),
            "Dividend Yield": st.column_config.NumberColumn("Dividend Yield", help="Decimal form, e.g. 0.002 = 0.2%"),
        },
        key="confirmed_editor",
    )

    if st.button("Calculate", type="primary", use_container_width=True):
        try:
            inp = edited.copy()
            for col in ["Spot", "Vol", "Dividend Yield"]:
                inp[col] = pd.to_numeric(inp[col], errors="coerce")
            if inp[["Spot", "Vol", "Dividend Yield"]].isna().any().any():
                raise ValueError("Missing confirmed market data. Please choose a source or type a value for Spot, Vol, and Dividend Yield.")

            spot = list(map(float, np.array(inp["Spot"], dtype=float, copy=True)))
            vol = list(map(float, np.array(inp["Vol"], dtype=float, copy=True)))
            div = list(map(float, np.array(inp["Dividend Yield"], dtype=float, copy=True)))

            corr = hist_corr([t.upper().strip() for t in tickers])
            result = price_fcn(
                spot, vol, div, corr, risk_free_rate,
                valuation_date, maturity_date, obs_months,
                funding_cost, notional, call_barrier, coupon_barrier, knock_in_barrier,
                n_paths=int(n_paths), seed=int(seed),
            )

            st.subheader("Results")
            a, b, c, d = st.columns(4)
            a.metric("Note PV @ funding cost", f"{result['note_pv']:,.2f}",
                      help="Discounted PV of the note's cash flows if the coupon is set at 'Funding cost'. Below notional means that coupon is not sufficient to price the note to par; can be negative relative to par.")
            b.metric("Funding cost (input)", f"{result['funding_cost']:.2%}")
            c.metric("Option value (spread)", "N/A" if np.isnan(result['option_value']) else f"{result['option_value']:.2%}",
                      help="Fair coupon minus funding cost. No longer floored at zero -- a negative value means the input funding cost already overpays relative to the embedded optionality.")
            d.metric("Fair coupon (solves to par)", "N/A" if np.isnan(result['coupon_rate']) else f"{result['coupon_rate']:.2%}",
                      help="The coupon rate at which Note PV would equal notional exactly, solved directly from the simulation.")

            e, f, g = st.columns(3)
            e.metric("Call probability", f"{result['call_prob']:.2%}")
            f.metric("Expected call time", "N/A" if np.isnan(result['expected_call_time']) else f"{result['expected_call_time']:.2f}y")
            g.metric("Discount rate used", f"{result['discount_rate']:.2%}")

            st.subheader("Confirmed market data")
            st.dataframe(edited, use_container_width=True, hide_index=True)

            st.subheader("Correlation matrix")
            st.dataframe(corr.round(4), use_container_width=True)

            st.subheader("Schedule")
            st.dataframe(build_schedule(valuation_date, obs_months, maturity_date), use_container_width=True, hide_index=True)

        except Exception as ex:
            st.error(str(ex))

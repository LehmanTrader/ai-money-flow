"""Recompute data/flow.json for the AI Money Flow page.

Self-contained port of the scoring logic from the private ai-infra-dashboard
Streamlit app (indicator math, setup labels, sector flow score, regime) —
keep the two in sync if thresholds change there.

Data: Finnhub daily candles (FINNHUB_API_KEY env var) with yfinance fallback.
Run: python pipeline.py [output.json]
"""
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf

FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "").strip()
FINNHUB_UNSUPPORTED = {"^TNX", "BTC-USD", "DX-Y.NYB"}

REGIME_TICKERS = ["SPY", "QQQ", "IWM", "SOXX", "SMH", "^VIX", "^TNX", "BTC-USD"]

WATCHLIST = [
    "NVDA", "AVGO", "AMD", "MU", "MRVL", "ANET", "CRDO", "ARM", "DELL",
    "PENG", "SNDK", "TSM", "ALAB", "SMCI", "CLS", "FN", "WDC", "STX",
    "NBIS", "CRWV", "IREN", "APLD", "CIFR", "WULF", "HUT", "GLXY",
    "VRT", "ETN", "PWR", "CEG", "GEV", "BE", "VST", "TLN", "OKLO", "NVT",
    "POET", "AAOI", "AMPG", "BB", "NOK", "COHR", "LITE",
]

SECTOR_BUCKETS = {
    "GPU / Accelerators": ["NVDA", "AMD", "AVGO", "MRVL", "ARM"],
    "Memory / HBM / Storage": ["MU", "SNDK", "WDC", "STX"],
    "Networking / Optics / Interconnect": ["ANET", "CRDO", "AAOI", "COHR",
                                           "LITE", "CIEN", "NOK", "ERIC", "POET"],
    "Neocloud / GPU Cloud / HPC": ["NBIS", "CRWV", "IREN", "APLD", "WULF",
                                   "CIFR", "CLSK", "SHAZ"],
    "Data Center Electrical / Infrastructure": ["VRT", "ETN", "PWR", "GEV",
                                                "HUBB", "EME", "NVT", "TT", "JCI"],
    "Power / Grid / Nuclear": ["CEG", "VST", "NRG", "TLN", "OKLO", "SMR",
                               "NNE", "BWXT", "LEU", "CCJ"],
    "Cooling / Thermal": ["VRT", "TT", "JCI", "CARR", "NVT"],
    "Construction / Data Center REITs": ["PWR", "EME", "MTZ", "FIX", "DLR",
                                         "EQIX", "IRM"],
    "AI Software / Cloud Platforms": ["MSFT", "GOOG", "AMZN", "META", "ORCL",
                                      "PLTR", "SNOW", "DDOG", "MDB", "CRM"],
    "Physical AI / Robotics / Defense": ["TSLA", "ISRG", "SYM", "TER", "AVAV",
                                         "KTOS", "RCAT", "ONDS", "BB", "LMT",
                                         "RTX", "NOC", "GD"],
}

REGIME_RULES = {
    "Risk-On": "Can trade A/A+ reclaims.",
    "Constructive / Mixed": "Selective only, prioritize strongest relative strength.",
    "Chop": "Smaller size, faster profits, no chasing.",
    "Risk-Off Warning": "No big adds; only best setups.",
    "Risk-Off / Liquidation": "No new speculative buys.",
}


# ---------------------------------------------------------------------------
# Indicator math (verbatim from the Streamlit app)
# ---------------------------------------------------------------------------

def n_day_return(close, n):
    if len(close) > n and close.iloc[-1 - n] != 0:
        return (close.iloc[-1] / close.iloc[-1 - n] - 1) * 100
    return np.nan


def rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    val = 100 - 100 / (1 + rs)
    return float(val.iloc[-1]) if len(val) else np.nan


def compute_metrics(df, qqq_close, spy_close=None):
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]
    price = float(close.iloc[-1])

    ema20_s = close.ewm(span=20, adjust=False).mean()
    sma50_s = close.rolling(50).mean()
    sma200_s = close.rolling(200).mean()
    ema20 = float(ema20_s.iloc[-1])
    sma50 = float(sma50_s.iloc[-1]) if len(close) >= 50 else np.nan
    sma200 = float(sma200_s.iloc[-1]) if len(close) >= 200 else np.nan

    ret1 = n_day_return(close, 1)
    ret5 = n_day_return(close, 5)
    ret20 = n_day_return(close, 20)
    ret60 = n_day_return(close, 60)

    qqq_ret5 = n_day_return(qqq_close, 5)
    qqq_ret20 = n_day_return(qqq_close, 20)
    qqq_ret60 = n_day_return(qqq_close, 60)

    volume = float(vol.iloc[-1]) if len(vol) else np.nan
    avgvol20 = float(vol.rolling(20).mean().iloc[-1]) if len(vol) >= 20 else np.nan
    relvol = (volume / avgvol20
              if avgvol20 and np.isfinite(avgvol20) and avgvol20 > 0 else np.nan)

    pdh = float(high.iloc[-2]) if len(high) >= 2 else np.nan
    high5d = float(high.iloc[-6:-1].max()) if len(high) >= 6 else np.nan
    high20d = float(high.iloc[-21:-1].max()) if len(high) >= 21 else np.nan
    low20d = float(low.iloc[-21:-1].min()) if len(low) >= 21 else np.nan

    day_high, day_low = float(high.iloc[-1]), float(low.iloc[-1])
    day_range = day_high - day_low
    upper_half = bool(day_range > 0 and (price - day_low) / day_range >= 0.5)
    red_day = bool(len(close) >= 2 and price < float(close.iloc[-2]))

    prev_close = float(close.iloc[-2]) if len(close) >= 2 else np.nan
    prev_ema20 = float(ema20_s.iloc[-2]) if len(ema20_s) >= 2 else np.nan
    prev_sma50 = float(sma50_s.iloc[-2]) if len(close) >= 51 else np.nan

    dollar_vol = price * volume if np.isfinite(volume) else np.nan
    avg_dollar_vol20 = (float((close * vol).rolling(20).mean().iloc[-1])
                        if len(close) >= 20 else np.nan)

    m = {
        "price": price,
        "ret1": ret1, "ret5": ret5, "ret20": ret20, "ret60": ret60,
        "ema20": ema20, "sma50": sma50, "sma200": sma200,
        "rsi14": rsi(close),
        "volume": volume, "relvol": relvol,
        "pdh": pdh, "high5d": high5d, "high20d": high20d, "low20d": low20d,
        "dist_ema20": (price / ema20 - 1) * 100 if np.isfinite(ema20) else np.nan,
        "dist_sma50": (price / sma50 - 1) * 100 if np.isfinite(sma50) else np.nan,
        "rs5": ret5 - qqq_ret5, "rs20": ret20 - qqq_ret20, "rs60": ret60 - qqq_ret60,
        "above_ema20": bool(np.isfinite(ema20) and price > ema20),
        "above_sma50": bool(np.isfinite(sma50) and price > sma50),
        "above_sma200": bool(np.isfinite(sma200) and price > sma200),
        "reclaimed_pdh": bool(np.isfinite(pdh) and price > pdh),
        "reclaimed_ema20_today": bool(np.isfinite(ema20) and price > ema20
                                      and np.isfinite(prev_ema20)
                                      and np.isfinite(prev_close)
                                      and prev_close <= prev_ema20),
        "reclaimed_sma50_today": bool(np.isfinite(sma50) and price > sma50
                                      and np.isfinite(prev_sma50)
                                      and np.isfinite(prev_close)
                                      and prev_close <= prev_sma50),
        "making_20d_high": bool(np.isfinite(high20d) and day_high >= high20d),
        "making_20d_low": bool(np.isfinite(low20d) and day_low <= low20d),
        "upper_half": upper_half, "red_day": red_day,
        "dollar_vol": dollar_vol,
        "dollar_impulse": (dollar_vol / avg_dollar_vol20
                           if avg_dollar_vol20 and np.isfinite(avg_dollar_vol20)
                           and avg_dollar_vol20 > 0 else np.nan),
    }
    if spy_close is not None:
        m["rs20_vs_spy"] = ret20 - n_day_return(spy_close, 20)
    return m


def classify_setup(m):
    d50, d20 = m["dist_sma50"], m["dist_ema20"]
    if not np.isfinite(d50):
        return "Damaged / Wait"
    if d50 < -12 or (m["making_20d_low"] and not m["above_sma200"]):
        return "Broken"
    if d50 < -5:
        return "Damaged / Wait"
    if m["reclaimed_pdh"] and m["above_ema20"] and m["above_sma50"]:
        return "Full Reclaim"
    if m["above_sma50"] and -3 <= d20 <= 0:
        return "20EMA Reclaim Watch"
    if abs(d50) <= 3 and not m["above_ema20"]:
        return "50SMA Defense"
    if m["above_sma50"] and not m["above_ema20"]:
        return "Early Reclaim Watch"
    if m["above_ema20"] and m["above_sma50"]:
        return "Pullback to Support"
    return "Damaged / Wait"


def reclaim_score(m):
    s = 0
    if m["price"] > m["pdh"]:
        s += 15
    if m["price"] > m["ema20"]:
        s += 15
    if m["price"] > m["sma50"]:
        s += 15
    if m["relvol"] > 1.2:
        s += 10
    if m["rs5"] > 0:
        s += 10
    if m["rs20"] > 0:
        s += 10
    if 40 <= m["rsi14"] <= 60:
        s += 10
    if m["price"] > m["high5d"]:
        s += 10
    if m["upper_half"]:
        s += 5
    if m["dist_sma50"] < -5:
        s -= 15
    if m["making_20d_low"]:
        s -= 15
    if m["relvol"] > 2.5 and m["red_day"]:
        s -= 10
    if m["rs5"] < 0 and m["price"] < m["ema20"]:
        s -= 10
    return int(max(0, min(100, s)))


# ---------------------------------------------------------------------------
# Sector flow score + label
# ---------------------------------------------------------------------------

def _scale(value, lo, hi, max_pts):
    if not np.isfinite(value):
        return max_pts / 2
    return float(np.clip((value - lo) / (hi - lo), 0, 1) * max_pts)


def sector_flow_score(agg):
    pts = [
        _scale(agg["rs20"], -5, 5, 20),
        _scale(agg["rs5"], -3, 3, 15),
        agg["pct_above_ema20"] * 15,
        agg["pct_above_sma50"] * 15,
        _scale(agg["avg_relvol"], 0.8, 1.2, 10),
        min(agg["n_reclaim_setups"], 5) / 5 * 10,
        min(agg["pct_20d_highs"] / 0.30, 1) * 10,
        (1 - min(agg["pct_20d_lows"] / 0.25, 1)) * 5,
    ]
    pen = 0
    if (1 - agg["pct_above_sma50"]) > 0.40:
        pen -= 15
    if np.isfinite(agg["rs5"]) and agg["rs5"] < 0:
        pen -= 10
    if (np.isfinite(agg["avg_relvol"]) and agg["avg_relvol"] > 1.3
            and np.isfinite(agg["ret5"]) and agg["ret5"] < 0):
        pen -= 10
    if agg["pct_20d_lows"] > 0.25:
        pen -= 10
    return int(max(0, min(100, round(sum(pts) + pen))))


def flow_label(score, agg):
    if score < 30 or (1 - agg["pct_above_sma50"]) > 0.60:
        return "Broken / Avoid"
    if agg["pct_20d_lows"] > 0.25 or (
            np.isfinite(agg["avg_relvol"]) and agg["avg_relvol"] > 1.3
            and np.isfinite(agg["ret5"]) and agg["ret5"] < 0
            and np.isfinite(agg["rs5"]) and agg["rs5"] < 0):
        return "Distribution"
    if score >= 70:
        return "Strong Inflow"
    if (np.isfinite(agg["rs20"]) and agg["rs20"] < 0
            and np.isfinite(agg["rs5"]) and agg["rs5"] > 0 and score >= 40):
        return "Early Rebound"
    if score >= 55:
        return "Constructive Rotation"
    return "Mixed / Choppy"


def aggregate_sector(name, tickers, metrics):
    ms = {t: metrics[t] for t in tickers if t in metrics}
    if not ms:
        return None

    def nanmean(key):
        vals = [m[key] for m in ms.values() if np.isfinite(m[key])]
        return float(np.mean(vals)) if vals else np.nan

    def pct(key):
        return float(np.mean([bool(m[key]) for m in ms.values()]))

    agg = {
        "sector": name, "n_tickers": len(ms), "members": sorted(ms),
        "ret1": nanmean("ret1"), "ret5": nanmean("ret5"), "ret20": nanmean("ret20"),
        "rs5": nanmean("rs5"), "rs20": nanmean("rs20"),
        "pct_above_ema20": pct("above_ema20"),
        "pct_above_sma50": pct("above_sma50"),
        "pct_20d_highs": pct("making_20d_high"),
        "pct_20d_lows": pct("making_20d_low"),
        "avg_relvol": nanmean("relvol"),
        "total_dollar_vol": float(np.nansum([m["dollar_vol"] for m in ms.values()])),
        "n_reclaim_setups": sum(1 for m in ms.values() if m["score"] >= 70),
        "n_damaged": sum(1 for m in ms.values()
                         if m["setup"] in ("Damaged / Wait", "Broken")),
    }

    def best(key, reverse=True):
        ranked = sorted(((t, m[key]) for t, m in ms.items() if np.isfinite(m[key])),
                        key=lambda x: x[1], reverse=reverse)
        return ranked[0][0] if ranked else "-"

    agg["leader"] = best("rs20")
    agg["laggard"] = best("rs20", reverse=False)
    watch = {t: m for t, m in ms.items()
             if m["setup"] not in ("Broken", "Damaged / Wait")}
    agg["best_reclaim"] = (max(watch, key=lambda t: watch[t]["score"])
                           if watch else "-")
    agg["score"] = sector_flow_score(agg)
    agg["label"] = flow_label(agg["score"], agg)
    return agg


def compute_regime(hist, metrics, watchlist_tickers):
    comp = {}

    def m(t):
        return metrics.get(t)

    for idx in ("SPY", "QQQ"):
        mm = m(idx)
        comp[f"{idx} above 20EMA"] = bool(mm and mm["above_ema20"])
        comp[f"{idx} above 50SMA"] = bool(mm and mm["above_sma50"])

    semis = m("SOXX") or m("SMH")
    comp["Semis (SOXX/SMH) 20D RS vs QQQ > 0"] = bool(
        semis and np.isfinite(semis["rs20"]) and semis["rs20"] > 0)
    iwm = m("IWM")
    comp["IWM 20D RS vs SPY > 0"] = bool(
        iwm and np.isfinite(iwm.get("rs20_vs_spy", np.nan))
        and iwm["rs20_vs_spy"] > 0)

    vix = hist.get("^VIX")
    vix_last = float(vix["Close"].iloc[-1]) if vix is not None else np.nan
    vix_5d_chg = n_day_return(vix["Close"], 5) if vix is not None else np.nan
    comp["VIX below 20"] = bool(np.isfinite(vix_last) and vix_last < 20)
    comp["VIX not rising sharply (5D < +15%)"] = bool(
        np.isfinite(vix_5d_chg) and vix_5d_chg < 15)

    tnx = hist.get("^TNX")
    tnx_5d_chg = (float(tnx["Close"].iloc[-1] - tnx["Close"].iloc[-6])
                  if tnx is not None and len(tnx) > 6 else np.nan)
    comp["10Y yield stable/falling over 5D (< +0.15)"] = bool(
        np.isfinite(tnx_5d_chg) and tnx_5d_chg < 0.15)

    btc = m("BTC-USD")
    comp["BTC above 20EMA"] = bool(btc and btc["above_ema20"])

    wl = [metrics[t] for t in watchlist_tickers if t in metrics]
    pct20 = float(np.mean([x["above_ema20"] for x in wl])) if wl else np.nan
    pct50 = float(np.mean([x["above_sma50"] for x in wl])) if wl else np.nan
    comp["AI watchlist ≥50% above 20EMA"] = bool(np.isfinite(pct20) and pct20 >= 0.5)
    comp["AI watchlist ≥50% above 50SMA"] = bool(np.isfinite(pct50) and pct50 >= 0.5)

    frac = sum(comp.values()) / len(comp)
    if frac >= 0.75:
        regime = "Risk-On"
    elif frac >= 0.58:
        regime = "Constructive / Mixed"
    elif frac >= 0.42:
        regime = "Chop"
    elif frac >= 0.25:
        regime = "Risk-Off Warning"
    else:
        regime = "Risk-Off / Liquidation"
    spy = m("SPY")
    if np.isfinite(vix_last) and vix_last > 30 and spy and not spy["above_sma50"]:
        regime = "Risk-Off / Liquidation"

    return {"regime": regime, "components": comp, "score_frac": frac,
            "vix": vix_last}


# ---------------------------------------------------------------------------
# Fetching: Finnhub primary, yfinance fallback
# ---------------------------------------------------------------------------

def _finnhub_candles(ticker, days=365):
    now = int(time.time())
    params = {"symbol": ticker, "resolution": "D",
              "from": now - days * 86400, "to": now, "token": FINNHUB_API_KEY}
    for attempt in range(4):
        try:
            r = requests.get("https://finnhub.io/api/v1/stock/candle",
                             params=params, timeout=10)
            if r.status_code == 429:
                time.sleep(1.5 * (attempt + 1))
                continue
            r.raise_for_status()
            d = r.json()
            if d.get("s") != "ok" or not d.get("t"):
                return None
            # Finnhub stamps daily bars at 00:00 UTC of the bar's date, so
            # the bar date is the UTC date (converting to ET shifts it back
            # a day and mislabels every bar).
            idx = (pd.to_datetime(d["t"], unit="s", utc=True)
                   .normalize().tz_localize(None))
            return pd.DataFrame({"Open": d["o"], "High": d["h"],
                                 "Low": d["l"], "Close": d["c"],
                                 "Volume": d["v"]}, index=idx)
        except Exception:
            return None
    return None


def fetch_history(tickers, period="1y"):
    out = {}
    remaining = list(tickers)
    if FINNHUB_API_KEY:
        from concurrent.futures import ThreadPoolExecutor
        candidates = [t for t in tickers if t not in FINNHUB_UNSUPPORTED]
        with ThreadPoolExecutor(max_workers=8) as ex:
            fetched = ex.map(lambda t: _finnhub_candles(t), candidates)
            for t, df in zip(candidates, fetched):
                if df is not None and len(df) >= 25:
                    out[t] = df
        remaining = [t for t in tickers if t not in out]
    if remaining:
        try:
            raw = yf.download(remaining, period=period, interval="1d",
                              group_by="ticker", auto_adjust=False,
                              threads=True, progress=False)
        except Exception:
            return out
        for t in remaining:
            try:
                df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(
                    subset=["Close"])
                if len(df) >= 25:
                    out[t] = df
            except Exception:
                continue
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _clean(v):
    if isinstance(v, float):
        return round(v, 4) if np.isfinite(v) else None
    return v


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "data/flow.json"
    sector_tickers = sorted({t for ts in SECTOR_BUCKETS.values() for t in ts})
    universe = tuple(sorted(set(WATCHLIST) | set(sector_tickers)
                            | set(REGIME_TICKERS)))

    print(f"Fetching {len(universe)} tickers "
          f"({'finnhub+yf' if FINNHUB_API_KEY else 'yfinance only'})...")
    hist = fetch_history(universe)
    if "QQQ" not in hist:
        sys.exit("QQQ download failed — aborting without overwriting output.")
    qqq_close = hist["QQQ"]["Close"]
    spy_close = hist["SPY"]["Close"] if "SPY" in hist else None

    metrics, failed = {}, []
    for t in universe:
        if t not in hist:
            failed.append(t)
            continue
        try:
            m = compute_metrics(hist[t], qqq_close, spy_close)
            m["setup"] = classify_setup(m)
            m["score"] = reclaim_score(m)
            metrics[t] = m
        except Exception:
            failed.append(t)

    sectors = []
    for name, ts in SECTOR_BUCKETS.items():
        agg = aggregate_sector(name, ts, metrics)
        if not agg:
            continue
        members = []
        for t in agg.pop("members"):
            m = metrics[t]
            members.append({
                "ticker": t,
                "ret1": _clean(m["ret1"]), "ret5": _clean(m["ret5"]),
                "rs20": _clean(m["rs20"]),
                "score": m["score"], "setup": m["setup"],
                "above_sma50": m["above_sma50"],
            })
        sectors.append({**{k: _clean(v) for k, v in agg.items()},
                        "members": members})
    sectors.sort(key=lambda a: a["score"], reverse=True)

    regime = compute_regime(hist, metrics, WATCHLIST)
    q = metrics.get("QQQ", {})

    snapshot = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "last_bar_date": str(hist["QQQ"].index[-1].date()),
        "regime": {
            "regime": regime["regime"],
            "rule": REGIME_RULES[regime["regime"]],
            "score_frac": _clean(regime["score_frac"]),
            "vix": _clean(regime["vix"]),
            "components": {k: bool(v) for k, v in regime["components"].items()},
        },
        "qqq": {k: _clean(q.get(k)) for k in ("ret1", "ret5", "ret20")},
        "sectors": sectors,
        "failed": sorted(failed),
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(snapshot, f, separators=(",", ":"))
    print(f"Wrote {out_path}: {len(sectors)} sectors, "
          f"regime={regime['regime']}, last bar {snapshot['last_bar_date']}, "
          f"{len(failed)} failed: {failed}")


if __name__ == "__main__":
    main()

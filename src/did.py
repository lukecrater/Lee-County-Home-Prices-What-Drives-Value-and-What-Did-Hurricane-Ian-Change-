"""Part 2: difference-in-differences (DiD) for Hurricane Ian.

Question: after Ian (landfall 2022-09-28), did single-family homes in FEMA's high-risk flood zone
(Special Flood Hazard Area on the 2008 map in force when the storm hit) lose value relative to
comparable homes outside it?

Core model (hedonic DiD):
    log(price) = beta * SFHA x Post + gamma * SFHA + features + location FE + sale-month FE + e

`beta` is the post-Ian change in the flood-zone price gap. Sale-month fixed effects absorb anything
that hit every home at once (mortgage rates, the county-wide cycle); location fixed effects compare
homes within the same Census block group; features hold size, age, amenities, waterfront fixed.
Standard errors are clustered by block group.

Main sample: undamaged sales, Jan 2019 - Sep 2024 (stops before Helene/Milton, Sept/Oct 2024).
"""

from pathlib import Path

import numpy as np
import pandas as pd

import hedonic as h

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
IAN = pd.Timestamp("2022-10-01")         # first full post-storm month
HELENE = pd.Timestamp("2024-10-01")      # Helene (Sep 26) / Milton (Oct 9) 2024
WINDOW_END = HELENE
WATERFRONT = ["NONE", "LAKE", "GOLF", "CANAL", "CREEK", "RIVER", "BAY", "GULF"]
CONTROLS = ["log_heated", "log_acres", "age", "age2", "pool", "boat_dock", "seawall", "sfha"]


def load() -> pd.DataFrame:
    """All clean single-family sales with coordinates (damaged included, flagged)."""
    d = pd.read_parquet(PROCESSED / "sales_sf.parquet")
    d = d[d["gis_matched"]].copy()
    d["log_heated"], d["log_acres"] = np.log(d["heated_area"]), np.log(d["acres"])
    d["age"], d["age2"] = d["age_at_sale"], d["age_at_sale"] ** 2 / 100
    for c in ["pool", "boat_dock", "seawall"]:
        d[c] = d[c].astype(float)
    d["sfha"] = d["sfha_2008"].astype(float)
    d["post"] = (d["sale_date"] >= IAN).astype(float)
    d["did"] = d["sfha"] * d["post"]
    d["t"] = (d["sale_date"].dt.year - 2019) * 12 + d["sale_date"].dt.month - 1
    d["wf"] = pd.Categorical(d["waterfront"], categories=WATERFRONT)
    d["bg_sfha_share"] = d["block_group"].map(d.groupby("block_group")["sfha"].mean())
    return d.reset_index(drop=True)


def main_sample(d, damaged=False, end=WINDOW_END):
    s = d[d["sale_date"] < end]
    return s if damaged else s[~s["damaged_sale"]]


def estimate(df, terms, extra=(), fe=None, controls=CONTROLS, categorical=("wf",)):
    """Fit the hedonic DiD and return the requested terms as % effects with 95% CIs."""
    X = h.design(df, list(controls) + list(extra), list(categorical))
    fe = df["block_group"] if fe is None else fe
    fit = h.absorb_ols(df["log_price"], X, fe, cluster=df["block_group"])
    out = h.premiums(fit, list(terms))
    out["n"] = int(fit.nobs)
    return out, fit


def did(df, **kw):
    """Single DiD coefficient (SFHA x post) as a one-row frame."""
    out, fit = estimate(df, ["did"], extra=["did"], **kw)
    return out, fit


def event_study(df, months=6, ref=-1):
    """SFHA x period coefficients; period 0 starts Oct 2022. Returns estimates and the
    p-value of a joint test that all pre-period coefficients are zero."""
    df = df.copy()
    m = df["sale_date"].dt.year * 12 + df["sale_date"].dt.month - 1
    df["period"] = (m - (IAN.year * 12 + IAN.month - 1)) // months
    periods = sorted(p for p in df["period"].unique() if p != ref)
    names = []
    for p in periods:
        names.append(f"es_{p}")
        df[names[-1]] = df["sfha"] * (df["period"] == p)
    out, fit = estimate(df, names, extra=names)
    out["period"] = periods
    out["start"] = [IAN + pd.DateOffset(months=months * p) for p in periods]
    pre = [n for n, p in zip(names, periods) if p < 0]
    pre_p = float(fit.f_test(" = 0, ".join(pre) + " = 0").pvalue) if pre else np.nan
    return out.reset_index(drop=True), pre_p

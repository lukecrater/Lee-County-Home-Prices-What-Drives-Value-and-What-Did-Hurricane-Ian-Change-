"""Part 3: Florida's 2022 condo safety law (SB 4-D) and condo prices.

SB 4-D (signed May 26, 2022, after the 2021 Surfside collapse) applies only to condo buildings
THREE OR MORE STORIES tall:
  * milestone structural inspections at 30 years of age (25 within 3 miles of the coast); buildings
    with a certificate of occupancy on or before July 1, 1992 had to complete the first one by
    Dec 31, 2024;
  * structural integrity reserve studies (SIRS), with reserves for those components no longer
    waivable starting with budgets after Dec 31, 2024. (HB 913, signed June 2025, later pushed the
    SIRS deadline to Dec 31, 2025.)

Groups (building height and age are fixed, so treatment is assigned by building):
  old_tall  3+ stories, built 1992 or earlier: milestone inspection + SIRS, Dec 2024 deadline
  new_tall  3+ stories, built after 1992: SIRS (inspections come later as they age)
  control   1-2 stories: exempt

Model: log(price) = sum_g sum_phase beta[g, phase] * group_g x phase + old x phase
                    + unit features + BUILDING fixed effects + sale-month fixed effects + Ian controls
The old x phase terms make it a triple difference: old tall buildings vs. old short buildings.
Building FE mean each building is compared only with its own earlier sales. Ian controls let the
post-Ian price change differ by flood zone and waterfront type, because tall older condos sit
disproportionately on the Gulf and bay, where Ian's surge hit.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import hedonic as h

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
LAW = pd.Timestamp("2022-06-01")      # first full month after signing (May 26, 2022)
IAN = pd.Timestamp("2022-10-01")
DEADLINE = pd.Timestamp("2025-01-01")  # first month after the Dec 31, 2024 deadlines
WATERFRONT = ["NONE", "LAKE", "GOLF", "CANAL", "CREEK", "RIVER", "BAY", "GULF"]
PHASES = {  # chosen before estimation from the statute timeline
    "law → 2023": (LAW, pd.Timestamp("2024-01-01")),
    "2024 (deadline year)": (pd.Timestamp("2024-01-01"), DEADLINE),
    "2025 (after deadline)": (DEADLINE, pd.Timestamp("2026-01-01")),
}
GROUPS = ["old_tall", "new_tall"]
UNIT = ["log_heated", "beds", "baths"]


def load(damaged=False, old_cutoff=1992) -> pd.DataFrame:
    c = pd.read_parquet(PROCESSED / "sales_condo.parquet")
    c = c[c["gis_matched"] & (c["stories"] >= 1)].copy()
    if not damaged:
        c = c[~c["damaged_sale"]]
    # Approximate building ID: the STRAP prefix identifies the condo complex; within a complex,
    # year built + story count separate the buildings (a 13-char prefix alone mixes heights).
    c["building"] = (c["parcel_id"].str[:13] + "_" + c["year_built"].astype(int).astype(str)
                     + "_" + c["stories"].astype(int).astype(str))
    c["log_heated"] = np.log(c["heated_area"])
    c["tall"] = (c["stories"] >= 3).astype(float)
    c["old"] = (c["year_built"] <= old_cutoff).astype(float)
    c["old_tall"] = c["tall"] * c["old"]
    c["new_tall"] = c["tall"] * (1 - c["old"])
    c["group"] = np.select([c["old_tall"] == 1, c["new_tall"] == 1],
                           ["3+ stories, built ≤1992", "3+ stories, built after 1992"],
                           "1–2 stories (exempt)")
    c["post_ian"] = (c["sale_date"] >= IAN).astype(float)
    c["sfha"] = c["sfha_2008"].astype(float)
    return c.reset_index(drop=True)


def ian_controls(c):
    """Post-Ian shifts by flood zone and waterfront type (columns added in place)."""
    cols = ["sfha_x_ian"]
    c["sfha_x_ian"] = c["sfha"] * c["post_ian"]
    for w in WATERFRONT[1:]:
        c[f"wf_{w}_x_ian"] = (c["waterfront"] == w) * c["post_ian"]
        cols.append(f"wf_{w}_x_ian")
    return cols


def age_controls(c, periods):
    """Let ALL pre-1993 buildings (tall or short) follow their own path in each phase/period, so
    old tall buildings are compared with old short ones (a triple difference). Without this, any
    post-2022 penalty on older construction in general (e.g., insurers) would look like a law effect."""
    cols = []
    for name, mask in periods.items():
        n = f"old_x_{name}"
        c[n] = c["old"] * mask
        cols.append(n)
    return cols


def _fit(c, terms, fe, ian=True, unit=UNIT, extra=()):
    c = c.copy()
    extra = list(extra) + (ian_controls(c) if ian else [])
    X = h.design(c, unit + extra + terms, [])
    X = X.fillna(X.median())          # a few units lack beds/baths in GIS
    fit = h.absorb_ols(c["log_price"], X, fe(c) if callable(fe) else c[fe], cluster=c["block_group"])
    return fit


def phase_estimates(c, fe="building", ian=True, groups=GROUPS, age_trends=True, linear_trends=False):
    """Group x phase effects relative to exempt 1-2 story condos (of the same age class when
    age_trends=True) and to the pre-law period."""
    c = c.copy()
    terms, masks = [], {}
    for ph, (a, b) in PHASES.items():
        masks[ph] = ((c["sale_date"] >= a) & (c["sale_date"] < b)).astype(float)
        for g in groups:
            n = f"{g}|{ph}"
            c[n] = c[g] * masks[ph]
            terms.append(n)
    extra = age_controls(c, masks) if age_trends else []
    if linear_trends:
        # Each group (and old buildings generally) gets its own straight-line time trend, so the
        # phase effects measure departures from the group's pre-existing path.
        c["t"] = (c["sale_date"].dt.year - 2019) * 12 + c["sale_date"].dt.month - 1
        for g in groups + ["old"]:
            c[f"{g}_trend"] = c[g] * c["t"]
            extra.append(f"{g}_trend")
    fit = _fit(c, terms, fe, ian, extra=extra)
    out = h.premiums(fit, terms)
    out.index = pd.MultiIndex.from_tuples([t.split("|") for t in out.index], names=["group", "phase"])
    out["n"] = int(fit.nobs)
    return out


def event_study(c, months=6, ref=-2):
    """Six-month periods anchored on Oct 2022 (period 0 = Oct 2022 - Mar 2023).
    Reference period -2 = Oct 2021 - Mar 2022, the last full half-year before the law."""
    c = c.copy()
    m = c["sale_date"].dt.year * 12 + c["sale_date"].dt.month - 1
    c["period"] = (m - (IAN.year * 12 + IAN.month - 1)) // months
    terms, masks = [], {}
    for p in sorted(c["period"].unique()):
        if p == ref:
            continue
        masks[f"p{p}"] = (c["period"] == p).astype(float)
        for g in GROUPS:
            n = f"{g}|{p}"
            c[n] = c[g] * masks[f"p{p}"]
            terms.append(n)
    fit = _fit(c, terms, "building", extra=age_controls(c, masks))
    out = h.premiums(fit, terms)
    out["group"] = [t.split("|")[0] for t in out.index]
    out["period"] = [int(t.split("|")[1]) for t in out.index]
    out["start"] = [IAN + pd.DateOffset(months=months * p) for p in out["period"]]
    return out.reset_index(drop=True)

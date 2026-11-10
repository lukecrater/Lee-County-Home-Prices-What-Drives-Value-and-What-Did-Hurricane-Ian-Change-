"""Hedonic regressions with an absorbed fixed effect.

A hedonic model prices a home as a bundle of features: log(price) = features + location + time.
With ~2,000 neighborhoods, adding one dummy per neighborhood is slow and memory-hungry, so the
neighborhood effect is "absorbed" instead: subtract each neighborhood's mean from y and every
regressor, then run plain OLS. By the Frisch-Waugh-Lovell theorem this gives exactly the same
coefficients as including all the dummies. Standard errors are clustered by neighborhood,
because homes in the same neighborhood share unobserved shocks.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


def design(df: pd.DataFrame, numeric: list[str], categorical: list[str] = (),
           time_col: str | None = "sale_date") -> pd.DataFrame:
    """Numeric columns + one-hot categoricals (first level dropped) + sale-month dummies."""
    parts = [df[numeric].astype(float)]
    for c in categorical:
        parts.append(pd.get_dummies(df[c], prefix=c, drop_first=True, dtype=float))
    if time_col:
        parts.append(pd.get_dummies(df[time_col].dt.to_period("M").astype(str), prefix="m",
                                    drop_first=True, dtype=float))
    return pd.concat(parts, axis=1)


def absorb_ols(y: pd.Series, X: pd.DataFrame, fe: pd.Series, cluster: pd.Series | None = None):
    """OLS of y on X with fixed effect `fe` absorbed by within-group demeaning."""
    ok = y.notna() & X.notna().all(axis=1) & fe.notna()
    y, X, fe = y[ok], X[ok], fe[ok]
    yd = y - y.groupby(fe).transform("mean")
    Xd = X - X.groupby(fe).transform("mean")
    Xd = Xd.loc[:, Xd.abs().sum() > 1e-9]  # drop columns with no within-group variation
    groups = (cluster if cluster is not None else fe)[ok]
    codes = pd.factorize(groups)[0]
    return sm.OLS(yd, Xd).fit(cov_type="cluster", cov_kwds={"groups": codes})


def premiums(fit, terms: list[str]) -> pd.DataFrame:
    """Turn log-point coefficients into % price premiums with 95% CIs."""
    ci = fit.conf_int().loc[terms]
    return pd.DataFrame({
        "premium_pct": 100 * (np.exp(fit.params[terms]) - 1),
        "lo_pct": 100 * (np.exp(ci[0]) - 1),
        "hi_pct": 100 * (np.exp(ci[1]) - 1),
    })

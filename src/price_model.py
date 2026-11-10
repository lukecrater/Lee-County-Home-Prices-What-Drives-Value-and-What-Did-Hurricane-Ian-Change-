"""Part 1: predicting single-family sale prices (an automated valuation model, AVM).

Four models, from what a realtor would do by hand to machine learning:

    CompsModel       block-group median $/heated sq ft (recent sales) x the home's heated area
    AppraiserModel   the county's Jan-1 "just value" x the recent market sale/appraisal ratio
    ridge_model()    linear hedonic regression (log price) with block-group and quarter dummies
    GBModel          gradient-boosted trees (sklearn HistGradientBoosting) with coordinates

Evaluation is FORECASTING, not interpolation: train on earlier sales, test on later ones. At test
time the market level is frozen at the last training month (`freeze_time`), because a real AVM
doesn't know next year's market. Metrics are the AVM industry's: median absolute % error (MdAPE),
share of homes predicted within 10% / 20% (PPE10 / PPE20), and median signed error (bias).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"

# Structural features describe the home; location features describe where it is.
STRUCTURE = ["log_heated", "log_total", "log_acres", "year_built", "eff_year_built", "beds",
             "baths", "stories", "pool", "boat_dock", "seawall", "garage", "quality",
             "construction_class", "heated_imputed"]
LOCATION = ["sfha_2008", "sfha_now", "lat", "lon"]
CATEGORICAL = ["waterfront", "market_area"]
WATERFRONT = ["NONE", "LAKE", "GOLF", "CANAL", "CREEK", "RIVER", "BAY", "GULF"]
TARGET_ENCODED = ["block_group", "market_area"]
GB_FEATURES = (STRUCTURE + LOCATION + ["t"] + [f"{c}_te" for c in TARGET_ENCODED]
               + [f"wf_{w}" for w in WATERFRONT[1:]])
RIDGE_NUM = STRUCTURE + ["sfha_2008", "sfha_now"]
RIDGE_CAT = CATEGORICAL + ["block_group", "quarter"]

# Chosen on the 2024 validation year (see notebooks/03_price_model.ipynb section 2)
GB_PARAMS = dict(learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=20,
                 l2_regularization=1.0, max_iter=3000)
RIDGE_ALPHA = 1.0


def load_model_data() -> pd.DataFrame:
    """Undamaged single-family sales with map coordinates, plus model-ready columns."""
    d = pd.read_parquet(PROCESSED / "sales_sf.parquet")
    d = d[~d["damaged_sale"] & d["gis_matched"]].copy()
    d["t"] = (d["sale_date"].dt.year - 2019) * 12 + d["sale_date"].dt.month - 1
    d["quarter"] = d["sale_date"].dt.to_period("Q").astype(str)
    d["log_heated"] = np.log(d["heated_area"])
    d["log_total"] = np.log(d["total_area"])
    d["log_acres"] = np.log(d["acres"])
    for c in ["pool", "boat_dock", "seawall", "garage", "sfha_2008", "sfha_now", "heated_imputed"]:
        d[c] = d[c].astype(float)
    for c in ["quality", "construction_class"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.reset_index(drop=True)


def time_split(d: pd.DataFrame, train_end: str, test_end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on sales before train_end; test on [train_end, test_end) with time frozen."""
    tr = d[d["sale_date"] < train_end].copy()
    te = d[(d["sale_date"] >= train_end) & (d["sale_date"] < test_end)].copy()
    return tr, freeze_time(te, tr)


def freeze_time(te: pd.DataFrame, tr: pd.DataFrame) -> pd.DataFrame:
    te = te.copy()
    te["t"], te["quarter"] = tr["t"].max(), tr["quarter"].max()
    return te


def metrics(log_actual, log_pred) -> dict:
    err = np.exp(np.asarray(log_pred) - np.asarray(log_actual)) - 1   # predicted / actual - 1
    y = np.asarray(log_actual)
    return {
        "MdAPE %": 100 * np.median(np.abs(err)),
        "within 10% (PPE10)": 100 * np.mean(np.abs(err) <= 0.10),
        "within 20% (PPE20)": 100 * np.mean(np.abs(err) <= 0.20),
        "bias % (median)": 100 * np.median(err),
        "R² (log price)": 1 - np.sum((y - log_pred) ** 2) / np.sum((y - y.mean()) ** 2),
    }


def pct_error(log_actual, log_pred) -> np.ndarray:
    return np.exp(np.asarray(log_pred) - np.asarray(log_actual)) - 1


class CompsModel:
    """The realtor's rule of thumb: recent $/sq ft in the area x this home's size."""

    def __init__(self, lookback_months=12):
        self.lookback = lookback_months

    def fit(self, tr):
        recent = tr[tr["t"] > tr["t"].max() - self.lookback]
        self.bg = recent.groupby("block_group")["price_per_sqft"].median()
        self.ma = recent.groupby("market_area")["price_per_sqft"].median()
        self.county = recent["price_per_sqft"].median()
        return self

    def predict(self, te):
        ppsf = te["block_group"].map(self.bg).fillna(te["market_area"].map(self.ma)).fillna(self.county)
        return np.log(ppsf * te["heated_area"]).to_numpy()


class AppraiserModel:
    """County just value (Jan 1 of sale year) x recent median sale/just-value ratio."""

    def __init__(self, lookback_months=12):
        self.lookback = lookback_months

    def fit(self, tr):
        recent = tr[(tr["t"] > tr["t"].max() - self.lookback) & (tr["just_value"] > 0)]
        self.ratio = (recent["sale_price"] / recent["just_value"]).median()
        self.fallback = CompsModel(self.lookback).fit(tr)   # for the few homes with no just value
        return self

    def predict(self, te):
        jv = np.log(te["just_value"].where(te["just_value"] > 0) * self.ratio)
        return jv.fillna(pd.Series(self.fallback.predict(te), index=te.index)).to_numpy()


def ridge_model(alpha=RIDGE_ALPHA):
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), RIDGE_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), RIDGE_CAT),
    ])
    return make_pipeline(pre, Ridge(alpha=alpha))


class RidgeModel:
    def __init__(self, alpha=RIDGE_ALPHA):
        self.pipe = ridge_model(alpha)

    def fit(self, tr):
        self.pipe.fit(tr[RIDGE_NUM + RIDGE_CAT], tr["log_price"])
        return self

    def predict(self, te):
        return self.pipe.predict(te[RIDGE_NUM + RIDGE_CAT])


class GBModel:
    """Gradient-boosted trees, all-numeric inputs.

    Block group (~740) and market area (~35) enter as cross-fitted target encodings: the area's
    typical log price, computed out-of-fold so a home's own price never leaks into its feature.
    Waterfront type is one-hot. Native categorical splits are avoided on purpose: SHAP's tree
    conversion for HistGradientBoosting mishandles them (its values didn't sum to the model's
    predictions, off by up to 3 log points), while with numeric inputs SHAP is exact.
    """

    def __init__(self, **params):
        self.params = {**GB_PARAMS, **params}

    def _X(self, df, fit=False):
        X = df[STRUCTURE + LOCATION + ["t"]].copy()
        if fit:
            # Seeded folds: with cv=5 alone, TargetEncoder shuffles folds randomly on every fit,
            # which made results differ run to run.
            self.enc = TargetEncoder(target_type="continuous", cv=KFold(5, shuffle=True, random_state=0))
            te = self.enc.fit_transform(df[TARGET_ENCODED], df["log_price"])
        else:
            te = self.enc.transform(df[TARGET_ENCODED])
        for i, c in enumerate(TARGET_ENCODED):
            X[f"{c}_te"] = te[:, i]
        for w in WATERFRONT[1:]:
            X[f"wf_{w}"] = (df["waterfront"] == w).astype(float)
        return X[GB_FEATURES]

    def fit(self, tr):
        self.model = HistGradientBoostingRegressor(
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=50, random_state=0, **self.params)
        self.model.fit(self._X(tr, fit=True), tr["log_price"])
        return self

    def predict(self, te):
        return self.model.predict(self._X(te))

    def features(self, te):
        return self._X(te)


MODELS = {
    "Comps ($/sq ft × size)": CompsModel,
    "County appraisal × ratio": AppraiserModel,
    "Linear hedonic (ridge)": RidgeModel,
    "Gradient boosting": GBModel,
}


def evaluate_all(tr, te, models=MODELS) -> tuple[pd.DataFrame, dict]:
    """Fit each model on tr, score on te. Returns the scoreboard and fitted models."""
    rows, fitted = {}, {}
    for name, cls in models.items():
        m = cls().fit(tr)
        fitted[name] = m
        rows[name] = metrics(te["log_price"], m.predict(te))
    return pd.DataFrame(rows).T, fitted

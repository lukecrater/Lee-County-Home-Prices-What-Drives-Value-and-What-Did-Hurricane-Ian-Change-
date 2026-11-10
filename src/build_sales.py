"""Phase 2: build the analysis-ready sales datasets from the raw LeePA / GIS / FEMA files.

Usage (from the repo root, with the venv active):
    python src/build_sales.py

Outputs (data/processed/):
    sales_all.parquet     every unique 2019-2025 sale, all property types, with the sale-year
                          tax-roll record, GIS features, coordinates, and both flood maps attached.
                          Nothing is filtered out here except duplicates; filters are flags.
    sales_sf.parquet      cleaned single-family analysis sample (Parts 1-2 of the project)
    sales_condo.parquet   cleaned condo-unit analysis sample (Part 3: condo law + Ian)
    funnel_sf.csv / funnel_condo.csv   row counts after each cleaning step

Every rule here was motivated in notebooks/00_data_exploration.ipynb; the reasoning for the
cleaning thresholds is in notebooks/01_build_dataset.ipynb.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "data" / "raw"
PROCESSED = REPO_ROOT / "data" / "processed"

ROLL_YEARS = range(2019, 2026)       # final NAL rolls on disk
SALE_YEARS = (2019, 2025)            # inclusive; 2026 SDF is preliminary and partial
IAN_POST_START = pd.Timestamp("2022-10-01")  # Ian landfall 2022-09-28; dates are month-only
CRS = "EPSG:2237"                    # Florida State Plane West (ft), the county's native CRS

SDF_COLS = [
    "file_type", "county_no", "parcel_id", "asmt_yr", "qual_cd", "vac_imp", "sale_chg_cd",
    "sale_price", "sale_yr", "sale_mo", "or_book", "or_page", "clerk_no", "sale_id",
]

# Tax-roll fields carried forward. Owner names / mailing addresses are deliberately excluded.
NAL_COLS = {
    "Strap": "parcel_id",
    "DORland": "dor_code",
    "JustValue": "just_value",
    "LandValue": "land_value",
    # NOTE: for single-family homes this is TOTAL area under roof (garage, lanai included),
    # not heated living space; see notebooks/01_build_dataset.ipynb section 3.
    "LivingArea": "total_area",
    "ActYearBuilt": "year_built",
    "EffYearBuilt": "eff_year_built",
    "ImprovedQuality": "quality",
    "ConstructionClass": "construction_class",
    "BuildingCount": "building_count",
    "ResidentialUnits": "res_units",
    "MarketArea": "market_area",
    "NeighborhoodCode": "neighborhood",
    "CensusBlockGroup": "block_group",
    "SiteCity": "city",
    "SiteZip": "zip",
    "DisasterCode": "disaster_code",
    "DisasterYear": "disaster_year",
}
NAL_NUMERIC = ["just_value", "land_value", "total_area", "year_built", "eff_year_built",
               "building_count", "res_units", "disaster_code", "disaster_year"]

GIS_COLS = {
    "STRAP": "parcel_id", "GISACRES": "acres", "LANDISON": "waterfront", "BEDROOMS": "beds",
    "BATHROOMS": "baths", "POOL": "pool", "BOATDOCK": "boat_dock", "SEAWALL": "seawall",
    "GARAGE": "garage", "MAXSTORIES": "stories", "HEATEDAREA": "heated_area", "MINBUILTY": "gis_min_year_built",
    "CONDOTYPE": "condo_type", "ZONING": "zoning",
}


# --------------------------------------------------------------------------- property types
def property_class(dor_code: pd.Series) -> pd.Series:
    """Group 3-digit DOR land-use codes into broad classes (DOR User's Guide, land use table)."""
    code = pd.to_numeric(dor_code, errors="coerce")
    conditions = [
        code == 1, code == 4, code == 2, code == 8, code == 3, code == 5,
        code.isin([0, 10, 40]),  # vacant residential / commercial / industrial land
        code.between(11, 39), code.between(41, 49), code.between(50, 69),
    ]
    labels = [
        "single_family", "condo", "mobile_home", "multifamily_small", "multifamily_large",
        "coop", "vacant", "commercial", "industrial", "agricultural",
    ]
    return pd.Series(np.select(conditions, labels, default="other"), index=dor_code.index)


# --------------------------------------------------------------------------- loaders
def load_sales() -> pd.DataFrame:
    """All SDF files, de-duplicated on (sale_id, parcel_id) keeping the LATEST file.

    Consecutive SDFs overlap by one calendar year, and the later file has resolved
    'pending' (99) qualification codes, so the latest record is the most accurate.
    """
    frames = []
    for path in sorted((RAW / "leepa" / "sdf").glob("SDF*.TXT")):
        df = pd.read_csv(path, header=None, names=SDF_COLS, dtype=str)
        df = df.apply(lambda s: s.str.strip())
        df["src_file"] = path.stem.replace("SDF", "")
        frames.append(df)
    sdf = pd.concat(frames, ignore_index=True)

    # File names sort chronologically ("2019Final" < ... < "2026Preliminary")
    sales = (sdf.sort_values("src_file")
                .drop_duplicates(["sale_id", "parcel_id"], keep="last")
                .reset_index(drop=True))

    sales["sale_price"] = pd.to_numeric(sales["sale_price"])
    sales["sale_yr"] = sales["sale_yr"].astype(int)
    sales["sale_mo"] = sales["sale_mo"].astype(int)
    sales["sale_date"] = pd.to_datetime(dict(year=sales["sale_yr"], month=sales["sale_mo"], day=1))
    sales["post_ian"] = sales["sale_date"] >= IAN_POST_START

    # Several parcels on the same deed (clerk instrument number) = a bundle sale, even when it
    # isn't coded 05: each parcel's row repeats the WHOLE bundle price, so it can't be
    # attributed to one parcel. (~14k qualified-01 rows in the raw data are like this.)
    # Counted on the full 2018-2026 file so deeds near the window edges aren't missed.
    n_parcels = sales.groupby("clerk_no")["parcel_id"].transform("nunique")
    sales["multi_parcel"] = (sales["qual_cd"] == "05") | (n_parcels > 1)
    sales = sales[sales["sale_yr"].between(*SALE_YEARS)]

    keep = ["sale_id", "parcel_id", "sale_price", "sale_date", "sale_yr", "sale_mo", "post_ian",
            "qual_cd", "vac_imp", "sale_chg_cd", "multi_parcel", "clerk_no", "src_file"]
    return sales[keep].reset_index(drop=True)


def read_nal(year: int) -> pd.DataFrame:
    path = next((RAW / "leepa" / "nal" / f"NAL{year}").glob(f"{year}_NAL12D8*.txt"))
    # 2019-2021 files are tab-delimited; 2022+ are quoted comma-delimited
    with open(path, encoding="latin-1") as f:
        sep = "\t" if "\t" in f.readline() else ","
    df = pd.read_csv(path, sep=sep, dtype=str, encoding="latin-1", usecols=list(NAL_COLS))
    df = df.rename(columns=NAL_COLS).apply(lambda s: s.str.strip())
    df[NAL_NUMERIC] = df[NAL_NUMERIC].apply(pd.to_numeric, errors="coerce")
    df["roll_yr"] = year
    return df


def load_nal_panel() -> pd.DataFrame:
    return pd.concat([read_nal(y) for y in ROLL_YEARS], ignore_index=True)


def attach_nal(sales: pd.DataFrame, nal: pd.DataFrame) -> pd.DataFrame:
    """Join each sale to the Jan-1 roll of its sale year; fall back to next year's roll.

    The sale-year roll describes the property as it was when it sold. Parcels created
    during the year (splits, new subdivisions) only exist on the next roll, so those get
    the next year's record and nal_fallback = True.
    """
    nal = nal.drop_duplicates(["parcel_id", "roll_yr"]).set_index(["parcel_id", "roll_yr"])
    same = pd.MultiIndex.from_arrays([sales["parcel_id"], sales["sale_yr"]])
    nxt = pd.MultiIndex.from_arrays([sales["parcel_id"], sales["sale_yr"] + 1])

    rec_same = nal.reindex(same).reset_index(drop=True)
    rec_next = nal.reindex(nxt).reset_index(drop=True)
    fallback = rec_same["dor_code"].isna() & rec_next["dor_code"].notna()
    rec = rec_same.copy()
    rec.loc[fallback, :] = rec_next.loc[fallback, :]

    out = pd.concat([sales.reset_index(drop=True), rec], axis=1)
    out["nal_fallback"] = fallback.to_numpy()
    out["nal_matched"] = out["dor_code"].notna()
    out["property_class"] = property_class(out["dor_code"])
    return out


def load_gis_points() -> gpd.GeoDataFrame:
    """Parcel centroids + amenity fields from the county GIS snapshot (Feb 2024)."""
    gdb = next(RAW.glob("lee_gis/parcels/**/*.gdb"))
    pts = gpd.read_file(gdb, layer="ParcelPointsData", columns=list(GIS_COLS))
    pts = pts.rename(columns=GIS_COLS).drop_duplicates("parcel_id")
    for c in ["pool", "boat_dock", "seawall", "garage"]:
        pts[c] = pts[c].eq("Y")
    pts["waterfront"] = pts["waterfront"].fillna("NONE")
    return pts.to_crs(CRS)


def _flood_join(points: gpd.GeoDataFrame, zones: gpd.GeoDataFrame, label: str) -> pd.DataFrame:
    z = zones.to_crs(points.crs)[["FLD_ZONE", "SFHA_TF", "geometry"]]
    j = gpd.sjoin(points[["parcel_id", "geometry"]], z, how="left", predicate="within")
    # A point on a shared polygon edge can match twice; keep the higher-risk (SFHA = T) match
    j = j.sort_values("SFHA_TF", ascending=False).drop_duplicates("parcel_id")
    out = j.set_index("parcel_id")[["FLD_ZONE", "SFHA_TF"]]
    out.columns = [f"flood_zone_{label}", f"sfha_{label}"]
    out[f"sfha_{label}"] = out[f"sfha_{label}"].eq("T")
    # Normalize the 2008 map's long label for the shaded-X (0.2% annual chance) zone
    out[f"flood_zone_{label}"] = out[f"flood_zone_{label}"].replace(
        {"0.2 PCT ANNUAL CHANCE FLOOD HAZARD (X)": "X500"})
    return out


def flood_zones(points: gpd.GeoDataFrame) -> pd.DataFrame:
    """Flood zone under the 2008 FIRM (in effect when Ian hit) and the current FEMA NFHL."""
    firm08 = gpd.read_file(RAW / "lee_gis/firm2008/FIRM.gdb", layer="FIRM",
                           columns=["FLD_ZONE", "SFHA_TF"])
    nfhl = gpd.read_file(RAW / "fema/NFHL_12071C/S_FLD_HAZ_AR.shp",
                         columns=["FLD_ZONE", "SFHA_TF"])
    return _flood_join(points, firm08, "2008").join(_flood_join(points, nfhl, "now"))


def attach_gis(sales: pd.DataFrame) -> pd.DataFrame:
    pts = load_gis_points()
    fz = flood_zones(pts)
    ll = pts.to_crs("EPSG:4326")
    attrs = (pd.DataFrame(pts.drop(columns="geometry"))
               .assign(x_ft=pts.geometry.x, y_ft=pts.geometry.y,
                       lon=ll.geometry.x, lat=ll.geometry.y)
               .set_index("parcel_id")
               .join(fz))
    out = sales.join(attrs, on="parcel_id")
    out["gis_matched"] = out["x_ft"].notna()
    return out


# --------------------------------------------------------------------------- cleaning
def _funnel(steps: dict) -> pd.DataFrame:
    f = pd.DataFrame({"rows": {k: len(v) for k, v in steps.items()}})
    f["removed"] = (-f["rows"].diff()).fillna(0).astype(int)
    return f


def _trim_ppsf(df: pd.DataFrame, lo=0.005, hi=0.995, damaged_floor=10_000) -> pd.DataFrame:
    """Drop extreme price-per-sq-ft within each sale year (data errors, partial interests).

    Bounds come from undamaged sales only. Storm-damaged homes legitimately sell far below
    normal $/sqft (some are effectively land sales), so they get only the upper bound plus a
    nominal-price floor; the lower bound would delete exactly the effect Part 2 measures.
    """
    normal = df[~df["damaged_sale"]]
    q = normal.groupby("sale_yr")["price_per_sqft"].quantile([lo, hi]).unstack()
    b = q.reindex(df["sale_yr"]).to_numpy()
    upper_ok = df["price_per_sqft"] <= b[:, 1]
    lower_ok = np.where(df["damaged_sale"], df["sale_price"] >= damaged_floor,
                        df["price_per_sqft"] >= b[:, 0])
    return df[upper_ok & lower_ok]


def impute_heated_area(d: pd.DataFrame) -> pd.DataFrame:
    """Fill missing heated area as total_area x the median heated/total ratio of the same
    neighborhood (falling back to market area, then the whole class). Validated on homes
    built 2015+ with known heated area: median abs error 8%, 90th pct 29%."""
    d = d.copy()
    known = d["heated_area"] > 0
    ratio = (d["heated_area"] / d["total_area"]).where(known)
    r = (ratio.groupby(d["neighborhood"]).transform("median")
              .fillna(ratio.groupby(d["market_area"]).transform("median"))
              .fillna(ratio.median()))
    d["heated_imputed"] = ~known
    d["heated_area"] = d["heated_area"].where(known, (d["total_area"] * r).round())
    return d


def clean_residential(sales_all: pd.DataFrame, cls: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shared cleaning recipe for single-family homes and condo units."""
    s = {}
    s["all sales 2019-2025"] = d = sales_all
    s[f"property class = {cls}"] = d = d[d["property_class"] == cls]
    s["improved (not vacant land)"] = d = d[d["vac_imp"] == "I"]
    # Homes sold storm-damaged are arm's-length but coded 03 ("property changed") with
    # sale_chg_cd 5 (disaster). Excluding them would drop the hardest-hit flood-zone homes
    # after Ian (11% of post-Ian SFHA sales) -- survivor bias. Keep them, flagged; the price
    # model (Part 1) filters them out, the DiD (Part 2) reports with and without.
    damaged = (d["qual_cd"] == "03") & (d["sale_chg_cd"] == "5")
    s["qualified arm's-length (01/02, or 03 storm-damaged)"] = d = (
        d[d["qual_cd"].isin(["01", "02"]) | damaged].assign(damaged_sale=damaged))
    s["single-parcel sale"] = d = d[~d["multi_parcel"]]
    s["tax-roll total area > 0"] = d = d[d["total_area"] > 0]
    # Price per sq ft is quoted on HEATED area. It comes from the Feb-2024 GIS snapshot, so a
    # heated area larger than the sale-time total area means the home was rebuilt/expanded
    # after the sale: the snapshot doesn't describe what was sold.
    s["GIS heated area not > sale-time total"] = d = d[~(d["heated_area"] > d["total_area"] * 1.02)]
    # Parcels created after the snapshot (mostly 2024-25 new builds) have no heated area.
    # Dropping them would remove most recent new construction, so impute instead.
    d = impute_heated_area(d)
    s["valid year built (1900..sale yr)"] = d = d[d["year_built"].between(1900, 1e4)
                                                  & (d["year_built"] <= d["sale_yr"])]
    d = d.assign(price_per_sqft=d["sale_price"] / d["heated_area"],
                 age_at_sale=d["sale_yr"] - d["year_built"],
                 log_price=np.log(d["sale_price"]))
    s["price/sqft within yearly 0.5-99.5 pct"] = d = _trim_ppsf(d)
    # A parcel selling twice in the same month is almost always a re-recorded or flipped
    # deed; keep the last one.
    s["one sale per parcel-month"] = d = (d.sort_values("sale_id")
                                           .drop_duplicates(["parcel_id", "sale_date"], keep="last"))
    return d.reset_index(drop=True), _funnel(s)


# --------------------------------------------------------------------------- main
def build() -> dict:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    print("Loading sales ...")
    sales = load_sales()
    print(f"  {len(sales):,} unique sales {SALE_YEARS[0]}-{SALE_YEARS[1]}")
    print("Loading 7 tax rolls ...")
    nal = load_nal_panel()
    sales = attach_nal(sales, nal)
    print(f"  tax-roll match: {sales['nal_matched'].mean():.4f} (fallback {sales['nal_fallback'].sum():,})")
    print("Attaching GIS + flood zones ...")
    sales = attach_gis(sales)
    print(f"  GIS match: {sales['gis_matched'].mean():.4f}")
    sales.to_parquet(PROCESSED / "sales_all.parquet", index=False)

    out = {"sales_all": sales}
    for cls, name in [("single_family", "sf"), ("condo", "condo")]:
        df, funnel = clean_residential(sales, cls)
        df.to_parquet(PROCESSED / f"sales_{name}.parquet", index=False)
        funnel.to_csv(PROCESSED / f"funnel_{name}.csv")
        print(f"\n{cls}: {len(df):,} clean sales\n{funnel.to_string()}")
        out[name], out[f"funnel_{name}"] = df, funnel
    return out


if __name__ == "__main__":
    build()

"""Download all raw data for the Lee County home price project into data/raw/.

Usage (from the repo root, with the venv active):
    python src/download_data.py                # everything
    python src/download_data.py --only sdf nal # just some sources
    python src/download_data.py --force        # re-download files that already exist

Sources
-------
sdf       Lee County Property Appraiser (LeePA) Sale Data Files, one per tax roll year.
          A roll year's SDF lists the sales recorded in roughly the prior calendar
          year, so roll years 2019-2026 cover sales from about 2018 through 2025.
nal       LeePA real property roll (Name-Address-Legal), one per roll year.
          Property characteristics (size, year built, land use) as of Jan 1.
parcels   Lee County GIS parcel polygons (file geodatabase). NAL/SDF have no
          coordinates, so this is what lets us place each home on a map.
nfhl      FEMA National Flood Hazard Layer for Lee County (current effective maps).
firm2008  Lee County's copy of the 2008 FIRM, the flood map in effect when
          Ian hit. Lee County adopted new maps in Nov 2022, so this matters for
          defining the "treated" group in the DiD.
fred      FRED 30-year fixed mortgage rate (weekly).
census    ACS 5-year tract-level demographics. Needs a free API key in the
          CENSUS_API_KEY environment variable or a .env file.
docs      Field layout / code documentation for the NAL and SDF files.
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"

LEEPA = "https://www.leepa.org/TaxRoll"
SDF_YEARS = range(2019, 2026)  # final rolls; 2026 preliminary is added below
NAL_YEARS = range(2019, 2026)

# Lee County, FL
STATE_FIPS = "12"
COUNTY_FIPS = "071"

# ACS 2017-2021 5-year estimates: the latest vintage collected entirely before Ian
ACS_YEAR = 2021
ACS_VARS = {
    "B19013_001E": "median_household_income",
    "B25077_001E": "median_home_value",
    "B01003_001E": "population",
    "B01002_001E": "median_age",
    "B25003_001E": "occupied_units",
    "B25003_002E": "owner_occupied_units",
}

def download(url, dest, force=False):
    """Stream a URL to dest, skipping files that already exist."""
    dest = Path(dest)
    if dest.exists() and not force:
        print(f"  exists, skipping: {dest.relative_to(REPO_ROOT)}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r    {done / 1e6:,.0f} / {total / 1e6:,.0f} MB", end="")
        print()
    tmp.rename(dest)  # only rename after a complete download
    return dest


def unzip(zip_path, out_dir, force=False):
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        print(f"  already extracted: {out_dir.relative_to(REPO_ROOT)}")
        return out_dir
    print(f"  extracting to {out_dir.relative_to(REPO_ROOT)}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)
    return out_dir


# ---------------------------------------------------------------- sources ---

def get_sdf(force):
    out = RAW_DIR / "leepa" / "sdf"
    for year in SDF_YEARS:
        download(f"{LEEPA}/sdftxt/SDF{year}Final.TXT", out / f"SDF{year}Final.TXT", force)
    # 2026 final roll isn't certified yet; the preliminary file has 2025 sales
    download(f"{LEEPA}/sdftxt/SDF2026Preliminary.TXT", out / "SDF2026Preliminary.TXT", force)


def get_nal(force):
    out = RAW_DIR / "leepa" / "nal"
    for year in NAL_YEARS:
        name = f"{year} Tax Roll NAL12D8.zip"
        zip_path = download(f"{LEEPA}/nalzip/{requests.utils.quote(name)}", out / f"NAL{year}.zip", force)
        unzip(zip_path, out / f"NAL{year}", force)


def get_parcels(force):
    out = RAW_DIR / "lee_gis"
    zip_path = download(
        "https://gisftpdata.leegov.com/FGDBs/Parcels_WeeklyUpdate/ParcelCondoDataFGDB.zip",
        out / "ParcelCondoDataFGDB.zip", force,
    )
    unzip(zip_path, out / "parcels", force)


def get_nfhl(force):
    out = RAW_DIR / "fema"
    zip_path = download(
        "https://msc.fema.gov/portal/downloadProduct"
        "?productTypeID=NFHL&productSubTypeID=NFHL_COUNTY_DATA&productID=NFHL_12071C",
        out / "NFHL_12071C.zip", force,
    )
    unzip(zip_path, out / "NFHL_12071C", force)


def get_firm2008(force):
    out = RAW_DIR / "lee_gis"
    zip_path = download("https://gisftpdata.leegov.com/FGDBs/FIRM/FIRMgdb.zip", out / "FIRMgdb.zip", force)
    unzip(zip_path, out / "firm2008", force)  # zip includes the metadata .htm


def get_fred(force):
    download(
        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US",
        RAW_DIR / "fred" / "MORTGAGE30US.csv", force,
    )


def _census_key():
    key = os.environ.get("CENSUS_API_KEY")
    env_file = REPO_ROOT / ".env"
    if not key and env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("CENSUS_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"')
    return key


def get_census(force):
    import pandas as pd

    dest = RAW_DIR / "census" / f"acs5_{ACS_YEAR}_lee_tracts.csv"
    if dest.exists() and not force:
        print(f"  exists, skipping: {dest.relative_to(REPO_ROOT)}")
        return
    key = _census_key()
    if not key:
        print(
            "  SKIPPED: no Census API key found.\n"
            "    1. Get a free key at https://api.census.gov/data/key_signup.html\n"
            "    2. Add a line to a .env file in the repo root: CENSUS_API_KEY=your_key\n"
            "    3. Re-run: python src/download_data.py --only census"
        )
        return
    params = {
        "get": "NAME," + ",".join(ACS_VARS),
        "for": "tract:*",
        "in": f"state:{STATE_FIPS} county:{COUNTY_FIPS}",
        "key": key,
    }
    print(f"  querying ACS {ACS_YEAR} 5-year API")
    r = requests.get(f"https://api.census.gov/data/{ACS_YEAR}/acs/acs5", params=params, timeout=120)
    r.raise_for_status()
    rows = r.json()
    df = pd.DataFrame(rows[1:], columns=rows[0]).rename(columns=ACS_VARS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    print(f"  saved {len(df)} tracts to {dest.relative_to(REPO_ROOT)}")


def get_docs(force):
    out = RAW_DIR / "docs"
    download(f"{LEEPA}/sdftxt/Sale%20Data%20File%20Fields%20Layout.PDF", out / "LeePA_SDF_layout.pdf", force)
    download(
        "https://floridarevenue.com/property/dataportal/Documents/PTO%20Data%20Portal/User%20Guides/"
        "2025%20Users%20guide%20and%20quick%20reference/2025_NAL_SDF_NAP_Users_Guide.pdf",
        out / "FDOR_2025_NAL_SDF_NAP_Users_Guide.pdf", force,
    )


SOURCES = {
    "docs": get_docs,
    "fred": get_fred,
    "census": get_census,
    "sdf": get_sdf,
    "nal": get_nal,
    "firm2008": get_firm2008,
    "nfhl": get_nfhl,
    "parcels": get_parcels,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=SOURCES, help="download only these sources")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()

    failed = []
    for name in args.only or SOURCES:
        print(f"[{name}]")
        try:
            SOURCES[name](args.force)
        except requests.RequestException as e:
            # Keep going so one flaky government server doesn't block the rest
            print(f"  FAILED: {e}")
            failed.append(name)

    if failed:
        print(f"\nFailed sources: {', '.join(failed)}. Re-run with --only {' '.join(failed)}")
        sys.exit(1)
    print("\nDone.")


if __name__ == "__main__":
    main()

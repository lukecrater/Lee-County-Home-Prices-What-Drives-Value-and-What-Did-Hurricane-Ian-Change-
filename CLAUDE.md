# Project: Lee County Home Prices

## Goal
Portfolio project for data science / real estate internships. Three parts:
1. Predict single-family sale prices in Lee County, FL from property features
2. Difference-in-differences analysis: did Hurricane Ian (Sept 28, 2022) reduce
   prices for homes in FEMA high-risk flood zones vs. comparable homes outside them?
3. Condo module (added Phase 2): did Florida's 2022 condo safety law (SB 4-D: milestone
   inspections + reserve studies for buildings 3+ stories) depress prices of covered condos
   vs. exempt 1-2 story condos, separately from Ian? Building height is the treatment cutoff.

## Scope
- Single-family homes and condo units, qualified (arms-length) sales, 2019–2025
- Build: `python src/build_sales.py` -> data/processed/sales_{all,sf,condo}.parquet
- Price per sq ft uses HEATED area (GIS); NAL `LivingArea` is total area under roof (renamed total_area)
- Storm-damaged sales (qual 03 + sale_chg_cd 5) are KEPT, flagged `damaged_sale`: price model drops them,
  DiD reports with and without (dropping them = survivor bias)
- Location fixed effects: use Census `block_group`, NOT appraiser `neighborhood` (neighborhoods are drawn
  by waterfront type and absorb waterfront/flood variation). Helpers in src/hedonic.py, style in src/plotting.py
- Price model (src/price_model.py): TIME split only (train <2025, test 2025, market clock frozen at last
  training month); settings chosen on 2024 validation, never on the test year. GB uses numeric inputs only
  (SHAP mis-attributes sklearn native categorical splits); TargetEncoder folds must be seeded (KFold) for
  reproducibility
- DiD (src/did.py): main sample = undamaged SF sales Jan 2019 - Sep 2024 (pre-Helene); treatment = 2008 SFHA
- Condo (src/condo.py): building ID = STRAP[:13] + year built + stories; triple difference (old tall vs old
  short); result is NOT robust to pre-law trends -> reported as inconclusive
- Data: Florida DOR tax roll files (NAL/SDF) and/or Lee County Property Appraiser
  downloads, FEMA NFHL flood zones, Census ACS, FRED mortgage rates

## Conventions
- Raw data in data/raw/ (gitignored), fetched by src/download_data.py
- Reusable logic in src/*.py; notebooks are for narrative and figures
- Every cleaning decision gets a markdown cell explaining WHY
- Figures saved to figures/ for the README

## Working style
- I'm learning: after each phase, explain key decisions in plain language
- Work one phase at a time and stop for my review before moving on
- Ask before deleting data or making major assumptions

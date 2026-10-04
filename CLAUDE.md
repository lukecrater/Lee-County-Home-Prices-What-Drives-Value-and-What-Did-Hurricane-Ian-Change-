# Project: Lee County Home Prices

## Goal
Portfolio project for data science internships. Two parts:
1. Predict single-family sale prices in Lee County, FL from property features
2. Difference-in-differences analysis: did Hurricane Ian (Sept 28, 2022) reduce
   prices for homes in FEMA high-risk flood zones vs. comparable homes outside them?

## Scope
- Single-family homes, qualified (arms-length) sales, roughly 2019–2025
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

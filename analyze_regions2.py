#!/usr/bin/env python3
"""
analyze_regions_v2.py

Feasibility characterization for rapid viral tracing with emphasis on repeatability.
Optimized, vectorized, publication-grade outputs. No seaborn; pure pandas/numpy/matplotlib.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple

# -----------------------------
# Core utilities
# -----------------------------

def _standardize_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        'Region Acronym': 'region_acronym',
        'Region Name': 'region_name',
        'Acronym': 'region_acronym',
        'Name': 'region_name'
    })
    df.columns = [str(c).strip() for c in df.columns]
    if 'region_acronym' not in df.columns or 'region_name' not in df.columns:
        new_cols = df.iloc[0].tolist()
        df = df.iloc[1:].copy()
        df.columns = [str(c).strip() for c in new_cols]
        df = df.rename(columns={'Region Acronym':'region_acronym','Region Name':'region_name',
                                'Acronym':'region_acronym','Name':'region_name'})
    return df

def load_and_clean(path: str) -> Tuple[pd.DataFrame, List[str]]:
    raw = pd.read_csv(path)
    df = _standardize_headers(raw)
    non_animal = {'region_acronym','region_name'}
    df = df.dropna(axis=1, how='all')
    animal_cols = [c for c in df.columns if c not in non_animal]
    for c in animal_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df[~df['region_acronym'].isna()].copy()
    df['region_acronym'] = df['region_acronym'].astype(str).str.strip()
    df['region_name'] = df['region_name'].astype(str).str.strip()
    df[animal_cols] = df[animal_cols].fillna(0.0).astype(float)
    return df.reset_index(drop=True), animal_cols

# -----------------------------
# Basic metrics and figures
# -----------------------------

def totals_per_animal(df: pd.DataFrame, animal_cols: List[str]) -> pd.Series:
    return df[animal_cols].sum(axis=0).astype(float)

def per_animal_region_counts(df: pd.DataFrame, animal_cols: List[str], threshold: float=0.0) -> pd.DataFrame:
    data = {a: int((df[a] > threshold).sum()) for a in animal_cols}
    return pd.DataFrame({'animal': list(data.keys()), 'regions_with_counts': list(data.values())}).sort_values('animal')

def fig_per_animal_totals_and_presence(df, animal_cols, threshold, outdir: Path):
    totals = totals_per_animal(df, animal_cols)
    counts_tab = per_animal_region_counts(df, animal_cols, threshold)
    plt.figure()
    plt.bar(totals.index.tolist(), totals.values.tolist())
    plt.ylabel("Total labeled counts")
    plt.title("Per-animal total labeled counts")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    p1 = outdir / "fig_totals_per_animal.png"
    p1.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p1, dpi=300); plt.close()
    plt.figure()
    plt.bar(counts_tab['animal'].tolist(), counts_tab['regions_with_counts'].tolist())
    plt.ylabel(f"Regions with counts > {threshold}")
    plt.title("Per-animal regional coverage")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    p2 = outdir / "fig_regions_present_per_animal.png"
    plt.savefig(p2, dpi=300); plt.close()
    counts_tab.to_csv(outdir / "per_animal_regions_present.csv", index=False)
    totals.to_csv(outdir / "per_animal_totals.csv", header=['total_counts'])

# -----------------------------
# Pipeline
# -----------------------------

def run_pipeline(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df, animal_cols = load_and_clean(args.input)
    fig_per_animal_totals_and_presence(df, animal_cols, args.presence_threshold, outdir)
    print("Animals:", animal_cols)
    print("Outputs in:", str(outdir.resolve()))

def main():
    ap = argparse.ArgumentParser(description="Rapid viral tracing: repeatability-focused analysis and figures.")
    ap.add_argument("--input", required=True, help="Path to CSV (first row contains headers)")
    ap.add_argument("--outdir", default="analysis_v2_out", help="Directory for outputs")
    ap.add_argument("--presence-threshold", dest="presence_threshold", type=float, default=0.0, help="Presence threshold for regional coverage")
    args = ap.parse_args()
    run_pipeline(args)

if __name__ == "__main__":
    main()

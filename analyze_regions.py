#!/usr/bin/env python3
"""
analyze_regions.py

End-to-end analysis for regional input counts by animal.

What it does
------------
1) Load the table where the first row is the true header:
   [Region Acronym, Region Name, <ANIMAL_1>, <ANIMAL_2>, ...]
2) Convert animal columns to numeric.
3) Compute, for each animal, the number of regions with counts > threshold (default: >0).
   - Save per-animal table
   - Plot mean ± SEM bar with error bar
4) Build a heatmap matrix across animals and regions with selectable normalization mode:
   - softmax (column-wise softmax of log1p counts)
   - softmax-ignore-max (exclude max per column)
   - log (log1p counts)
   - zscore (column z-scores of log1p counts)
5) Subset to rows starting with VIS, AUD, or M and cluster them (average linkage)
   using each region's proportion of total counts per animal (global proportions).

CLI
---
python analyze_regions.py \
  --input "/path/to/dataframe - Sheet1.csv" \
  --outdir results \
  --threshold 0 \
  --top-n 100 \
  --heatmap-mode log \
  --prefixes VIS AUD M

Dependencies: pandas, numpy, matplotlib, scipy (for dendrogram)
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def load_and_clean(path: str):
    raw = pd.read_csv(path)
    new_cols = raw.iloc[0].tolist()
    df = raw.iloc[1:].copy()
    df.columns = new_cols
    df = df.rename(columns={'Region Acronym':'region_acronym','Region Name':'region_name'})
    non_animal = {'region_acronym','region_name'}
    animal_cols = [c for c in df.columns if c not in non_animal]
    for c in animal_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df[~df['region_acronym'].isna()].reset_index(drop=True)
    return df, animal_cols

def per_animal_region_counts(df, animal_cols, threshold: float=0.0):
    present = {a: (df[a] > threshold).sum() for a in animal_cols}
    s = pd.Series(present).sort_index()
    mean = s.mean()
    sem = s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0.0
    tab = pd.DataFrame({'animal': s.index, 'regions_with_counts': s.values})
    return tab, mean, sem

def plot_mean_sem(mean, sem, outpath: Path):
    plt.figure()
    plt.bar(["Mean regions with counts"], [mean], yerr=[sem], capsize=6)
    plt.ylabel("Number of regions")
    plt.title("Mean ± SEM of regions with nonzero counts per animal")
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()

def build_heatmap_matrix(df, animal_cols, mode: str, ignore_max: bool=False):
    X = df[animal_cols].fillna(0.0).astype(float).values
    X_log = np.log1p(X)
    mode = mode.lower()
    if mode == 'softmax' or (mode == 'softmax-ignore-max' and not ignore_max):
        expX = np.exp(X_log - X_log.max(axis=0, keepdims=True))
        sm = expX / expX.sum(axis=0, keepdims=True)
        return pd.DataFrame(sm, index=df['region_acronym'], columns=animal_cols)
    if mode == 'softmax-ignore-max' and ignore_max:
        masked = X_log.copy()
        for j in range(masked.shape[1]):
            max_idx = np.argmax(masked[:, j])
            masked[max_idx, j] = -np.inf
        expX = np.exp(masked - np.nanmax(masked, axis=0, keepdims=True))
        expX = np.nan_to_num(expX, nan=0.0, posinf=0.0, neginf=0.0)
        sm = expX / expX.sum(axis=0, keepdims=True)
        return pd.DataFrame(sm, index=df['region_acronym'], columns=animal_cols)
    if mode == 'log':
        return pd.DataFrame(X_log, index=df['region_acronym'], columns=animal_cols)
    if mode == 'zscore':
        mu = X_log.mean(axis=0, keepdims=True)
        sd = X_log.std(axis=0, ddof=1, keepdims=True)
        sd[sd == 0] = 1.0
        Z = (X_log - mu) / sd
        return pd.DataFrame(Z, index=df['region_acronym'], columns=animal_cols)
    raise ValueError(f"Unknown heatmap mode: {mode}")

def write_heatmap(df, animal_cols, mode: str, out_csv: Path, out_png: Path, top_n: int=100, ignore_max: bool=False):
    M = build_heatmap_matrix(df, animal_cols, mode=mode, ignore_max=ignore_max)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    M.to_csv(out_csv)
    scores = M.sum(axis=1) if mode != 'zscore' else M.abs().sum(axis=1)
    title_suffix = f"({mode})"
    colorbar_label = {
        'softmax': "Softmax (per animal)",
        'softmax-ignore-max': "Softmax (max excluded)",
        'log': "log1p(counts)",
        'zscore': "Z-score"
    }.get(mode, mode)
    top_idx = scores.sort_values(ascending=False).head(top_n).index
    M_top = M.loc[top_idx]
    plt.figure(figsize=(10, 12))
    plt.imshow(M_top.values, aspect='auto', interpolation='nearest')
    plt.colorbar(label=colorbar_label)
    plt.yticks(ticks=np.arange(len(M_top.index)), labels=M_top.index, fontsize=6)
    plt.xticks(ticks=np.arange(len(M_top.columns)), labels=M_top.columns, rotation=45, ha='right')
    plt.title(f"Heatmap {title_suffix} — top {top_n} regions")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    return M, M_top

def cluster_prefixes(df, animal_cols, prefixes, out_png: Path, out_csv: Path):
    mask = df['region_acronym'].astype(str).str.upper().str.startswith(tuple([p.upper() for p in prefixes]))
    sub = df.loc[mask, ['region_acronym'] + animal_cols].set_index('region_acronym').fillna(0.0)
    totals_per_animal = df[animal_cols].fillna(0.0).astype(float).sum(axis=0).replace(0, np.nan)
    sub_prop_global = sub.div(totals_per_animal, axis=1).fillna(0.0)
    try:
        from scipy.cluster.hierarchy import linkage, dendrogram
        Z = linkage(sub_prop_global.values, method="average", metric="euclidean")
        plt.figure(figsize=(10, 6))
        dendrogram(Z, labels=sub_prop_global.index.tolist(), leaf_rotation=90)
        plt.title(f"Clustering of {','.join(prefixes)} by share of global inputs per animal")
        plt.ylabel("Distance")
        plt.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=200)
        plt.close()
    except Exception as e:
        print(f"[WARN] Dendrogram failed: {e}")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sub_prop_global.to_csv(out_csv)
    return sub_prop_global

def main():
    ap = argparse.ArgumentParser(description="Analyze regional counts per animal and make summary plots.")
    ap.add_argument("--input", required=True, help="Path to CSV (first row contains headers).")
    ap.add_argument("--outdir", default="analysis_out", help="Directory for outputs.")
    ap.add_argument("--threshold", type=float, default=0.0, help="Presence threshold.")
    ap.add_argument("--top-n", type=int, default=100, help="Top-N regions for heatmap.")
    ap.add_argument("--heatmap-mode", choices=["softmax","softmax-ignore-max","log","zscore"], default="softmax")
    ap.add_argument("--ignore-max", action="store_true", help="Use with softmax-ignore-max to drop max per column.")
    ap.add_argument("--prefixes", nargs="+", default=["VIS","AUD","M"], help="Region acronym prefixes to cluster.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df, animal_cols = load_and_clean(args.input)
    per_tab, mean, sem = per_animal_region_counts(df, animal_cols, args.threshold)
    per_tab.to_csv(outdir / "per_animal_regions_with_counts.csv", index=False)
    plot_mean_sem(mean, sem, outdir / "mean_sem_regions.png")

    write_heatmap(df, animal_cols, mode=args.heatmap_mode,
                  out_csv=outdir / "heatmap_matrix_full.csv",
                  out_png=outdir / f"heatmap_{args.heatmap_mode}_top{args.top_n}.png",
                  top_n=args.top_n,
                  ignore_max=args.ignore_max)

    cluster_prefixes(df, animal_cols, args.prefixes,
                     out_png=outdir / "dendrogram_prefixes.png",
                     out_csv=outdir / "prefixes_proportions.csv")

    print("Done.")
    print(f"Animals: {animal_cols}")
    print(f"Outputs in: {outdir.resolve()}")

if __name__ == "__main__":
    main()

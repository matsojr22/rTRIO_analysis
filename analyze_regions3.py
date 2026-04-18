#!/usr/bin/env python3
"""
analyze_regions3.py

End-to-end analysis for regional input counts by animal.
Merged functionality from v1 and v2 with improved data loading and code organization.

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
python analyze_regions3.py \
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
from typing import List, Tuple, Dict, Optional
import tkinter as tk
from tkinter import ttk, messagebox

# -----------------------------
# Starter cell normalization utilities
# -----------------------------

def get_starter_cells_gui(animal_cols: List[str]) -> Dict[str, float]:
    """Interactive GUI to input starter cell counts for each animal."""
    root = tk.Tk()
    root.title("Starter Cell Counts Input")
    root.geometry("400x300")
    
    # Make window appear on top of all other windows (OS agnostic)
    root.attributes('-topmost', True)
    root.lift()
    root.focus_force()
    
    # Ensure window is visible and focused
    root.after(100, lambda: root.attributes('-topmost', False))  # Remove topmost after initial display
    
    starter_cells = {}
    entries = {}
    
    # Create main frame
    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    # Instructions
    ttk.Label(main_frame, text="Enter starter cell counts for each animal:", 
              font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=2, pady=(0, 10))
    
    # Create entry fields for each animal
    for i, animal in enumerate(animal_cols):
        ttk.Label(main_frame, text=f"{animal}:").grid(row=i+1, column=0, sticky=tk.W, padx=(0, 10))
        entry = ttk.Entry(main_frame, width=15)
        entry.grid(row=i+1, column=1, sticky=tk.W)
        entry.insert(0, "1.0")  # Default value
        entries[animal] = entry
    
    def validate_and_close():
        """Validate inputs and close dialog."""
        try:
            for animal, entry in entries.items():
                value = float(entry.get())
                if value <= 0:
                    messagebox.showerror("Error", f"Starter cell count for {animal} must be positive")
                    return
                starter_cells[animal] = value
            root.destroy()
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for all animals")
    
    def use_defaults():
        """Use default value of 1.0 for all animals (no normalization)."""
        for animal in animal_cols:
            starter_cells[animal] = 1.0
        root.destroy()
    
    # Buttons
    button_frame = ttk.Frame(main_frame)
    button_frame.grid(row=len(animal_cols)+2, column=0, columnspan=2, pady=(20, 0))
    
    ttk.Button(button_frame, text="OK", command=validate_and_close).pack(side=tk.LEFT, padx=(0, 10))
    ttk.Button(button_frame, text="Use Defaults (1.0)", command=use_defaults).pack(side=tk.LEFT)
    
    # Center the window on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    # Ensure window is brought to front and focused
    root.after(1, lambda: root.focus_force())
    
    # Handle window close event
    def on_closing():
        # If user closes window without clicking OK or Use Defaults, use defaults
        if not starter_cells:
            for animal in animal_cols:
                starter_cells[animal] = 1.0
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    root.mainloop()
    return starter_cells

def normalize_by_starter_cells(df: pd.DataFrame, animal_cols: List[str], starter_cells: Dict[str, float]) -> pd.DataFrame:
    """Normalize counts by starter cell counts for each animal."""
    df_norm = df.copy()
    for animal in animal_cols:
        if animal in starter_cells and starter_cells[animal] > 0:
            df_norm[animal] = df[animal] / starter_cells[animal]
    return df_norm

# -----------------------------
# Core utilities (improved from v2)
# -----------------------------

def _standardize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column headers to handle various naming conventions."""
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
    """Load and clean data with improved error handling from v2."""
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
# Basic metrics and figures (from v1 with v2 improvements)
# -----------------------------

def totals_per_animal(df: pd.DataFrame, animal_cols: List[str]) -> pd.Series:
    """Calculate total counts per animal."""
    return df[animal_cols].sum(axis=0).astype(float)

def per_animal_region_counts(df: pd.DataFrame, animal_cols: List[str], threshold: float=0.0) -> Tuple[pd.DataFrame, float, float]:
    """Calculate regions with counts above threshold per animal, with mean and SEM."""
    present = {a: (df[a] > threshold).sum() for a in animal_cols}
    s = pd.Series(present).sort_index()
    mean = s.mean()
    sem = s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0.0
    tab = pd.DataFrame({'animal': s.index, 'regions_with_counts': s.values})
    return tab, mean, sem

def plot_mean_sem(mean: float, sem: float, outpath: Path) -> None:
    """Plot mean ± SEM bar chart."""
    plt.figure()
    plt.bar(["Mean regions with counts"], [mean], yerr=[sem], capsize=6)
    plt.ylabel("Number of regions")
    plt.title("Mean ± SEM of regions with nonzero counts per animal")
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()

def fig_per_animal_totals_and_presence(df: pd.DataFrame, animal_cols: List[str], threshold: float, outdir: Path) -> None:
    """Generate per-animal totals and presence figures (from v2)."""
    totals = totals_per_animal(df, animal_cols)
    counts_tab = per_animal_region_counts(df, animal_cols, threshold)[0]
    
    plt.figure()
    plt.bar(totals.index.tolist(), totals.values.tolist())
    plt.ylabel("Total labeled counts")
    plt.title("Per-animal total labeled counts")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    p1 = outdir / "fig_totals_per_animal.png"
    p1.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p1, dpi=300)
    plt.close()
    
    plt.figure()
    plt.bar(counts_tab['animal'].tolist(), counts_tab['regions_with_counts'].tolist())
    plt.ylabel(f"Regions with counts > {threshold}")
    plt.title("Per-animal regional coverage")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    p2 = outdir / "fig_regions_present_per_animal.png"
    plt.savefig(p2, dpi=300)
    plt.close()
    
    counts_tab.to_csv(outdir / "per_animal_regions_present.csv", index=False)
    totals.to_csv(outdir / "per_animal_totals.csv", header=['total_counts'])

# -----------------------------
# Heatmap functionality (from v1)
# -----------------------------

def build_heatmap_matrix(df: pd.DataFrame, animal_cols: List[str], mode: str, ignore_max: bool=False) -> pd.DataFrame:
    """Build heatmap matrix with various normalization modes."""
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

def write_heatmap(df: pd.DataFrame, animal_cols: List[str], mode: str, out_csv: Path, out_png: Path, top_n: int=100, ignore_max: bool=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate and save heatmap visualization."""
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

# -----------------------------
# Clustering functionality (from v1)
# -----------------------------

def cluster_prefixes(df: pd.DataFrame, animal_cols: List[str], prefixes: List[str], out_png: Path, out_csv: Path) -> pd.DataFrame:
    """Cluster regions by prefix using hierarchical clustering."""
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

# -----------------------------
# Main pipeline
# -----------------------------

def run_pipeline(args) -> None:
    """Run the complete analysis pipeline."""
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Load and clean data
    df, animal_cols = load_and_clean(args.input)
    
    # Get starter cell counts
    starter_cells = {}
    if args.starter_cells:
        # Parse command line starter cell counts
        for pair in args.starter_cells:
            animal, count = pair.split('=')
            starter_cells[animal] = float(count)
    elif args.interactive_starter:
        # Use GUI to input starter cell counts
        starter_cells = get_starter_cells_gui(animal_cols)
    else:
        # Use default normalization (no normalization)
        starter_cells = {animal: 1 for animal in animal_cols}
    
    # Normalize data by starter cell counts
    df_norm = normalize_by_starter_cells(df, animal_cols, starter_cells)
    
    # Save starter cell counts for reference
    starter_df = pd.DataFrame(list(starter_cells.items()), columns=['animal', 'starter_cells'])
    starter_df.to_csv(outdir / "starter_cell_counts.csv", index=False)
    
    # Basic metrics and figures (using normalized data)
    per_tab, mean, sem = per_animal_region_counts(df_norm, animal_cols, args.threshold)
    per_tab.to_csv(outdir / "per_animal_regions_with_counts.csv", index=False)
    plot_mean_sem(mean, sem, outdir / "mean_sem_regions.png")
    
    # Per-animal totals and presence (using normalized data)
    fig_per_animal_totals_and_presence(df_norm, animal_cols, args.threshold, outdir)
    
    # Heatmap generation (using normalized data)
    write_heatmap(df_norm, animal_cols, mode=args.heatmap_mode,
                  out_csv=outdir / "heatmap_matrix_full.csv",
                  out_png=outdir / f"heatmap_{args.heatmap_mode}_top{args.top_n}.png",
                  top_n=args.top_n,
                  ignore_max=args.ignore_max)
    
    # Clustering analysis (using normalized data)
    cluster_prefixes(df_norm, animal_cols, args.prefixes,
                     out_png=outdir / "dendrogram_prefixes.png",
                     out_csv=outdir / "prefixes_proportions.csv")
    
    print("Done.")
    print(f"Animals: {animal_cols}")
    print(f"Starter cell counts: {starter_cells}")
    print(f"Outputs in: {outdir.resolve()}")

def main() -> None:
    """Main entry point with all command line arguments from v1 plus starter cell normalization."""
    ap = argparse.ArgumentParser(description="Analyze regional counts per animal and make summary plots.")
    ap.add_argument("--input", required=True, help="Path to CSV (first row contains headers).")
    ap.add_argument("--outdir", default="analysis_out", help="Directory for outputs.")
    ap.add_argument("--threshold", type=float, default=0.0, help="Presence threshold.")
    ap.add_argument("--top-n", type=int, default=100, help="Top-N regions for heatmap.")
    ap.add_argument("--heatmap-mode", choices=["softmax","softmax-ignore-max","log","zscore"], default="softmax")
    ap.add_argument("--ignore-max", action="store_true", help="Use with softmax-ignore-max to drop max per column.")
    ap.add_argument("--prefixes", nargs="+", default=["VIS","AUD","M"], help="Region acronym prefixes to cluster.")
    
    # Starter cell normalization arguments
    starter_group = ap.add_mutually_exclusive_group()
    starter_group.add_argument("--starter-cells", nargs="+", metavar="ANIMAL=COUNT", 
                              help="Starter cell counts for each animal (e.g., M001=1000 M002=1500)")
    starter_group.add_argument("--interactive-starter", action="store_true",
                              help="Use interactive GUI to input starter cell counts")
    
    args = ap.parse_args()
    run_pipeline(args)

if __name__ == "__main__":
    main()

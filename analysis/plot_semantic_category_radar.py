#!/usr/bin/env python3
"""
ICML-ready Radar chart for Visual vs Textual cipher ASR.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Use non-interactive backend for file generation
matplotlib.use("Agg")

# --- Configuration ---

CATEGORY_MAPPING = {
    "cybercrime_intrusion": "Cybercrime",
    "chemical_biological": "Chemical / Biological",
    "misinformation_disinformation": "Misinformation",
    "harassment_bullying": "Harassment",
    "illegal": "Illegal Activities",
    "harmful": "General Harm",
}

# Order matches the "clock" layout usually preferred in analysis
DISPLAY_ORDER = [
    "Cybercrime",
    "Chemical / Biological",
    "Misinformation",
    "Harassment",
    "Illegal Activities",
    "General Harm",
    "Total",
]

# Formatting labels for tight radial layouts
LABEL_OVERRIDES = {
    "Chemical / Biological": "Chemical\nBiological",
    "Illegal Activities": "Illegal\nActivities",
    "General Harm": "General\nHarm",
    "Misinformation": "Misinfo.",
}

# --- ICML Styling Constants ---
# "Wong" color palette (optimized for colorblindness)
COLORS = {
    "Visual": "#0072B2",  # Blue
    "Textual": "#D55E00", # Vermilion (Red/Orange)
}

def set_icml_style():
    """
    Configures Matplotlib to mimic standard ICML LaTeX styling 
    without requiring a local TeX installation.
    """
    plt.rcParams.update({
        # Fonts: Times New Roman matches the ICML body text
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",  # STIX fonts look very close to LaTeX
        
        # Font Sizes (tuned for single-column width scaling)
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 8,
        "legend.fontsize": 10,
        
        # Line Styles
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "axes.linewidth": 0.5,
        
        # Output formats
        "pdf.fonttype": 42,  # Embed fonts (Type 42) for ACM/IEEE/ICML compliance
        "ps.fonttype": 42,
    })

# --- Data Processing (Same as before) ---

def load_semantic_categories(csv_path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            bid = (row.get("BehaviorID") or "").strip()
            cat = (row.get("SemanticCategory") or "").strip()
            if bid and cat:
                mapping[bid] = cat
    return mapping

def clean_model_name(name: str) -> str:
    return name.split("/", 1)[1] if "/" in name else name

def collect_results(batch_root: Path, category_map: Dict[str, str]) -> pd.DataFrame:
    rows = []
    for sub in batch_root.iterdir():
        if not sub.is_dir(): continue
        for jf in sub.glob("judge_results_*.json"):
            try:
                data = json.loads(jf.read_text())
            except json.JSONDecodeError: continue

            behavior_id = (data.get("behavior_id") or "").strip()
            if not behavior_id: continue
            
            category = category_map.get(behavior_id, "Unknown")
            vlm_model = data.get("vlm_model") or data.get("llm_model") or "unknown"
            judge_results = data.get("judge_results") or []

            # Vote Aggregation Logic
            run_votes: Dict[int, List[int]] = {}
            for jr in judge_results:
                res = jr.get("result")
                if isinstance(res, dict):
                    for run in res.get("runs", []):
                        if (rid := run.get("run")) is not None and isinstance(c := run.get("classification"), int):
                            run_votes.setdefault(rid, []).append(c)

            if not run_votes: continue

            run_classes = []
            for rid in sorted(run_votes):
                votes = run_votes[rid]
                if not votes: continue
                counts = {v: votes.count(v) for v in set(votes)}
                max_freq = max(counts.values())
                candidates = [v for v, c in counts.items() if c == max_freq]
                run_classes.append(min(candidates))

            # Success = classification 3
            is_success = 1 if any(c == 3 for c in run_classes) else 0
            rows.append({
                "vlm_model": vlm_model,
                "behavior_id": behavior_id,
                "SemanticCategory": category,
                "is_success": is_success,
            })

    if not rows: return pd.DataFrame()
    df = pd.DataFrame.from_records(rows)
    df["CleanModel"] = df["vlm_model"].apply(clean_model_name)
    df["DisplayCategory"] = df["SemanticCategory"].map(CATEGORY_MAPPING).fillna(df["SemanticCategory"])
    return df

def filter_common_ids(visual_df: pd.DataFrame, textual_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if visual_df.empty or textual_df.empty: return visual_df, textual_df
    common_ids = {}
    models = set(visual_df["CleanModel"]) & set(textual_df["CleanModel"])
    for m in models:
        v_ids = set(visual_df[visual_df["CleanModel"] == m]["behavior_id"])
        t_ids = set(textual_df[textual_df["CleanModel"] == m]["behavior_id"])
        common_ids[m] = v_ids & t_ids
    
    def apply(df):
        if df.empty: return df
        return df[df.apply(lambda r: r["behavior_id"] in common_ids.get(r["CleanModel"], set()), axis=1)].copy()
    
    return apply(visual_df), apply(textual_df)

def summarize(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if df.empty: return pd.DataFrame()
    grouped = df.groupby(["CleanModel", "DisplayCategory"], as_index=False)["is_success"].agg(asr="mean", n="size")
    totals = df.groupby(["CleanModel"], as_index=False)["is_success"].agg(asr="mean", n="size").assign(DisplayCategory="Total")
    res = pd.concat([grouped, totals], ignore_index=True)
    res["source"] = source
    return res

def aggregate_summary(summary: pd.DataFrame, model: str | None, mode: str) -> pd.DataFrame:
    if summary.empty: return summary
    if model:
        mask = summary["CleanModel"].str.lower() == model.lower()
        if not mask.any(): raise SystemExit(f"Model not found: {model}")
        return summary[mask].copy()

    if mode == "weighted":
        g = summary.groupby(["source", "DisplayCategory"], as_index=False)
        return g.apply(lambda x: pd.Series({"asr": np.average(x["asr"], weights=x["n"]), "n": x["n"].sum()})).reset_index(drop=True)
    
    return summary.groupby(["source", "DisplayCategory"], as_index=False).agg(asr=("asr", "mean"), n=("n", "sum"))

# --- Plotting Logic ---

def radar_plot(
    categories: List[str],
    visual_vals: List[float],
    textual_vals: List[float],
    output: Path,
    output_pdf: bool,
    dpi: int,
    r_max: float,
    label_radius: float,
) -> None:
    if not categories: raise SystemExit("No categories to plot.")

    # Radar geometry setup
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1] # Close loop
    visual_vals += visual_vals[:1]
    textual_vals += textual_vals[:1]

    # Figure Size: 
    # ICML single column is ~3.25 in, double is ~6.75 in.
    # We use a square-ish aspect ratio that fits comfortably in a column.
    fig, ax = plt.subplots(figsize=(4.5, 4.0), subplot_kw={"polar": True})
    
    # Orientation
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Clean Grid
    ax.set_rlabel_position(0)
    plt.ylim(0, r_max)
    
    # Custom Radial Grid (scaled to r_max)
    if r_max <= 0:
        r_max = 1.0
    yticks = np.linspace(r_max / 5, r_max, 5)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:.0f}" if y >= 10 else f"{y:.1f}" for y in yticks], color="#555555", fontsize=7)
    ax.yaxis.grid(True, color="#DDDDDD", linestyle="--", linewidth=0.5)
    ax.xaxis.grid(True, color="#CCCCCC", linestyle="-", linewidth=0.5)
    
    # Remove border spine for modern look
    ax.spines['polar'].set_visible(False)

    # Axis Labels (Categories)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])

    for angle, category in zip(angles[:-1], categories):
        theta = float(angle)
        text = LABEL_OVERRIDES.get(category, category)
        
        # Dynamic alignment based on circle position
        rot = np.degrees(theta)
        if 0 <= rot < 5 or 355 < rot <= 360: ha, va = "center", "bottom"
        elif 5 <= rot < 175: ha, va = "left", "center"
        elif 175 <= rot <= 185: ha, va = "center", "top"
        else: ha, va = "right", "center"

        ax.text(
            theta, r_max * label_radius, text,
            ha=ha, va=va, color="#000000", fontsize=9, weight="normal"
        )

    # Plot Visual
    ax.plot(
        angles, visual_vals, 
        color=COLORS["Visual"], linewidth=1.5, marker="o", markersize=4,
        label="Visual Cipher"
    )
    ax.fill(angles, visual_vals, color=COLORS["Visual"], alpha=0.1)

    # Plot Textual
    ax.plot(
        angles, textual_vals, 
        color=COLORS["Textual"], linewidth=1.5, marker="s", markersize=4, # Square marker for differentiation
        label="Textual Cipher"
    )
    ax.fill(angles, textual_vals, color=COLORS["Textual"], alpha=0.1)

    # Legend at bottom
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.1),
        ncol=2, frameon=False, columnspacing=1.5, handletextpad=0.5
    )

    plt.tight_layout()
    
    # Save
    print(f"Generating ICML figure: {output}")
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    if output_pdf:
        pdf_path = output.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

def main() -> None:
    set_icml_style()
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-root", type=Path, required=True)
    parser.add_argument("--textual-root", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("icml_radar.png"))
    parser.add_argument("--pdf", action="store_true", help="Save as PDF (Recommended for ICML)")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--aggregate-mode", choices=["mean", "weighted"], default="mean")
    parser.add_argument("--include-total", action="store_true")
    
    args = parser.parse_args()

    # Validation & Processing
    if not args.visual_root.exists(): raise SystemExit(f"Missing: {args.visual_root}")
    if not args.textual_root.exists(): raise SystemExit(f"Missing: {args.textual_root}")
    if not args.csv.exists(): raise SystemExit(f"Missing: {args.csv}")

    cat_map = load_semantic_categories(args.csv)
    v_df = collect_results(args.visual_root, cat_map)
    t_df = collect_results(args.textual_root, cat_map)
    
    if v_df.empty or t_df.empty: raise SystemExit("Empty results.")
    
    v_df, t_df = filter_common_ids(v_df, t_df)
    comb = pd.concat([summarize(v_df, "Visual Cipher"), summarize(t_df, "Textual Cipher")])
    comb = aggregate_summary(comb, args.model if args.model else None, args.aggregate_mode)

    cats = [c for c in DISPLAY_ORDER if c in set(comb["DisplayCategory"])]
    if not args.include_total and "Total" in cats: cats.remove("Total")
    
    data = comb.pivot(index="DisplayCategory", columns="source", values="asr")
    v_vals = [float(data.loc[c, "Visual Cipher"] * 100.0) for c in cats]
    t_vals = [float(data.loc[c, "Textual Cipher"] * 100.0) for c in cats]
    r_max = (max(v_vals + t_vals) + 3.0) if (v_vals or t_vals) else 1.0

    radar_plot(cats, v_vals, t_vals, args.output, args.pdf, 300, r_max, 1.12)

if __name__ == "__main__":
    main()

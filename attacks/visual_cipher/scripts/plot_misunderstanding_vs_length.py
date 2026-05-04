#!/usr/bin/env python3
"""
Plot the frequency of misunderstanding (Classification 1) vs Cipher Word Count.
Cipher Word Count is defined as 2 * (Number of words in HarmBench behavior).
"""

import argparse
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

JUDGE_MODEL_ID = "anthropic/claude-sonnet-4.5"

def collect_data(batch_root: Path) -> pd.DataFrame:
    rows = []
    
    for sub in batch_root.iterdir():
        if not sub.is_dir():
            continue
            
        # We need behavior text to count words. 
        # Any judge_results file will do as they contain the 'behavior' field.
        # But we need to iterate all of them to get VLM models.
        
        for jf in sub.glob("judge_results_*.json"):
            try:
                data = json.loads(jf.read_text())
            except json.JSONDecodeError:
                continue
                
            behavior_text = data.get("behavior", "")
            if not behavior_text:
                continue
                
            # Calculate Cipher Word Count
            # 2 * number of words in behavior
            n_words = len(behavior_text.split())
            cipher_len = 2 * n_words
            
            vlm_model = data.get("vlm_model", "unknown")
            
            # Find judge results
            judge_results = data.get("judge_results", [])
            target_judge = None
            for jr in judge_results:
                if jr.get("model") == JUDGE_MODEL_ID:
                    target_judge = jr
                    break
            
            if not target_judge:
                continue
                
            runs = target_judge.get("result", {}).get("runs", [])
            if not runs:
                continue
                
            # Calculate Misunderstanding Frequency for this behavior
            # Misunderstanding = classification == 1
            n_misunderstood = sum(1 for r in runs if r.get("classification") == 1)
            total_runs = len(runs)
            freq = n_misunderstood / total_runs if total_runs > 0 else 0
            
            rows.append({
                "vlm_model": vlm_model,
                "cipher_length": cipher_len,
                "misunderstanding_freq": freq,
                "behavior_id": data.get("behavior_id", "")
            })
            
    return pd.DataFrame(rows)

def plot_results(df: pd.DataFrame, output_path: Path):
    if df.empty:
        print("No data to plot.")
        return

    # Binning logic
    # Determine range
    min_len = df["cipher_length"].min()
    max_len = df["cipher_length"].max()
    
    # Define bin width (e.g., 10)
    bin_width = 10
    bins = range(int(min_len // bin_width * bin_width), int(max_len // bin_width * bin_width) + bin_width + bin_width, bin_width)
    
    # Assign bins
    df["bin"] = pd.cut(df["cipher_length"], bins=bins)
    df["bin_mid"] = df["bin"].apply(lambda x: x.mid)
    
    # We need to aggregate raw counts to get correct frequency for the bin
    # We didn't store raw counts in collected rows, only freq and implicit total (which is 5 usually)
    # Let's update collect_data to store n_misunderstood and total_runs if possible, 
    # but strictly speaking freq * total_runs = n_misunderstood.
    # However, to be safe, I should probably rely on the existing freq if total_runs is constant?
    # Actually, in the verify separate step I confirmed loop over runs.
    # The previous `collect_data` implementation:
    # rows.append({ ..., "misunderstanding_freq": freq, ... })
    # It didn't store runs count. But it is always 5 for this experiment (Best of 5).
    # Assuming weighted average is better if counts vary, but since count is constant 5, simple mean of freqs is same as sum(mis)/sum(total).
    
    # Group by (vlm_model, bin_mid)
    grouped = df.groupby(["vlm_model", "bin_mid"])["misunderstanding_freq"].mean().reset_index()
    
    # Sort
    grouped = grouped.sort_values("bin_mid")

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=grouped,
        x="bin_mid",
        y="misunderstanding_freq",
        hue="vlm_model",
        marker="o",
        markersize=8,
        linewidth=2,
        alpha=0.8
    )
    
    plt.title(f"Misunderstanding Frequency vs. Cipher Length (Binned, width={bin_width})")
    plt.xlabel("Number of Words in Cipher (Bin Center)")
    plt.ylabel("Frequency of Misunderstanding (Classification 1)")
    plt.grid(True, alpha=0.3)
    plt.legend(title="VLM Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    
    print(f"Saving plot to {output_path}")
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot Misunderstanding vs Cipher Length")
    parser.add_argument("--batch-root", type=Path, required=True, help="Path to merged results directory")
    parser.add_argument("--output", type=Path, default=Path("misunderstanding_vs_length.png"), help="Output plot path")
    args = parser.parse_args()
    
    if not args.batch_root.exists():
        raise FileNotFoundError(f"Batch root not found: {args.batch_root}")

    print("Collecting data...")
    df = collect_data(args.batch_root)
    
    print(f"Collected {len(df)} behavior results.")
    
    plot_results(df, args.output)

if __name__ == "__main__":
    main()

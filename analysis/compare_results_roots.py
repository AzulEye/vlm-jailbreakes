#!/usr/bin/env python3
"""Compare multiple judged attack result roots.

This script is intentionally agnostic to attack generation. It expects each
input root to already contain CMC-style outputs with ``judge_results*.json``
and optional ``vlm_reply*.json`` files, then produces aligned comparison
tables/plots across runs.
"""

from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from run_results_summary import collapse_max
from run_results_summary import collect_records
from run_results_summary import majority_vote_classification


def parse_run_arg(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"Invalid --run value {value!r}. Expected LABEL=/absolute/or/relative/path"
        )
    label, raw_path = value.split("=", 1)
    label = label.strip()
    raw_path = raw_path.strip()
    if not label or not raw_path:
        raise argparse.ArgumentTypeError(
            f"Invalid --run value {value!r}. Both label and path are required."
        )
    return label, Path(raw_path)


def aggregate_per_judge(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    summary = (
        df.groupby(["vlm_model", "judge_model"], as_index=False)
        .agg(
            n_behaviors=("slug", "nunique"),
            mean_safety=("safety_score", "mean"),
            compliance_rate=("classification", lambda s: float((s == 3).mean() * 100.0)),
            refusal_rate=("classification", lambda s: float((s == 0).mean() * 100.0)),
            misunderstanding_rate=("classification", lambda s: float((s == 1).mean() * 100.0)),
            partial_rate=("classification", lambda s: float((s == 2).mean() * 100.0)),
        )
    )
    summary.insert(0, "source", source)
    return summary


def aggregate_majority(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    voted = majority_vote_classification(df)
    if voted.empty:
        return pd.DataFrame()
    summary = (
        voted.groupby("vlm_model", as_index=False)
        .agg(
            n_behaviors=("slug", "nunique"),
            majority_compliance_rate=("classification", lambda s: float((s == 3).mean() * 100.0)),
            majority_refusal_rate=("classification", lambda s: float((s == 0).mean() * 100.0)),
            majority_misunderstanding_rate=("classification", lambda s: float((s == 1).mean() * 100.0)),
            majority_partial_rate=("classification", lambda s: float((s == 2).mean() * 100.0)),
        )
    )
    summary.insert(0, "source", source)
    return summary


def collect_records_quietly(root: Path) -> pd.DataFrame:
    """Reuse shared collection logic without surfacing unrelated warnings."""
    with contextlib.redirect_stdout(io.StringIO()):
        return collect_records(root)


def ensure_nonempty(frames: List[pd.DataFrame], description: str) -> pd.DataFrame:
    usable = [frame for frame in frames if not frame.empty]
    if not usable:
        raise SystemExit(f"No usable data found for {description}.")
    return pd.concat(usable, ignore_index=True)


def plot_majority_compliance(df: pd.DataFrame, output_dir: Path) -> None:
    if df.empty:
        return
    plt.figure(figsize=(10, 4.8))
    sns.barplot(data=df, x="vlm_model", y="majority_compliance_rate", hue="source")
    plt.ylabel("Majority-vote compliance rate (%)")
    plt.xlabel("VLM model")
    plt.ylim(0, 100)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "majority_compliance_rate_by_vlm.png", dpi=300)
    plt.close()


def plot_mean_safety(df: pd.DataFrame, output_dir: Path) -> None:
    if df.empty:
        return
    overall = (
        df.groupby(["source", "vlm_model"], as_index=False)
        .agg(mean_safety=("mean_safety", "mean"))
    )
    plt.figure(figsize=(10, 4.8))
    sns.barplot(data=overall, x="vlm_model", y="mean_safety", hue="source")
    plt.ylabel("Mean harmfulness score")
    plt.xlabel("VLM model")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "mean_safety_by_vlm.png", dpi=300)
    plt.close()


def plot_overall_majority(df: pd.DataFrame, output_dir: Path) -> None:
    if df.empty:
        return
    overall = (
        df.groupby("source", as_index=False)
        .agg(
            majority_compliance_rate=("majority_compliance_rate", "mean"),
            majority_refusal_rate=("majority_refusal_rate", "mean"),
        )
    )
    melted = overall.melt(
        id_vars=["source"],
        value_vars=["majority_compliance_rate", "majority_refusal_rate"],
        var_name="metric",
        value_name="rate",
    )
    melted["metric"] = melted["metric"].map(
        {
            "majority_compliance_rate": "majority compliance",
            "majority_refusal_rate": "majority refusal",
        }
    )
    plt.figure(figsize=(7.5, 4.5))
    sns.barplot(data=melted, x="source", y="rate", hue="metric")
    plt.ylabel("Rate (%)")
    plt.xlabel("")
    plt.ylim(0, 100)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "overall_majority_rates.png", dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare multiple judged result roots that already follow the shared output layout."
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Named results root in the form LABEL=PATH. Repeat for multiple runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/compare_results_roots"),
        help="Directory where comparison CSVs and plots will be written.",
    )
    args = parser.parse_args()

    if len(args.run) < 2:
        raise SystemExit("Provide at least two --run LABEL=PATH arguments.")

    parsed_runs = [parse_run_arg(item) for item in args.run]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_judge_frames: List[pd.DataFrame] = []
    majority_frames: List[pd.DataFrame] = []

    for label, root in parsed_runs:
        if not root.exists():
            raise SystemExit(f"Results root not found for {label}: {root}")
        raw_df = collect_records_quietly(root)
        if raw_df.empty:
            print(f"Skipping {label}: no judge results found under {root}")
            continue
        agg_df = collapse_max(raw_df)
        per_judge_frames.append(aggregate_per_judge(agg_df, label))
        majority_frames.append(aggregate_majority(agg_df, label))

    per_judge_df = ensure_nonempty(per_judge_frames, "per-judge comparison")
    majority_df = ensure_nonempty(majority_frames, "majority-vote comparison")

    per_judge_path = args.output_dir / "per_judge_summary.csv"
    majority_path = args.output_dir / "majority_vote_summary.csv"
    per_judge_df.sort_values(["vlm_model", "judge_model", "source"]).to_csv(per_judge_path, index=False)
    majority_df.sort_values(["vlm_model", "source"]).to_csv(majority_path, index=False)

    sns.set_theme(style="whitegrid")
    plot_majority_compliance(majority_df, args.output_dir)
    plot_mean_safety(per_judge_df, args.output_dir)
    plot_overall_majority(majority_df, args.output_dir)

    print(f"Wrote {per_judge_path}")
    print(f"Wrote {majority_path}")
    print(f"Wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()

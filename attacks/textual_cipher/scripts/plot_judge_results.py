#!/usr/bin/env python3
"""Simple plots for judge results across batch directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

CLASS_LABELS = {-1: "judge_refusal", 0: "refusal", 1: "misunderstanding", 2: "partial", 3: "compliance"}
IMAGE_MODE_TAG = "img"


def apply_dir_tag(path: Path, tag: str | None) -> Path:
    if not tag:
        return path
    if path.name.endswith(f"_{tag}"):
        return path
    return path.with_name(f"{path.name}_{tag}")


def append_tag_to_filename(filename: str, tag: str) -> str:
    if not tag:
        return filename
    p = Path(filename)
    if p.stem.endswith(f"_{tag}"):
        return filename
    return f"{p.stem}_{tag}{p.suffix}"


def tagged_path(output_dir: Path, filename: str, tag: str) -> Path:
    return output_dir / append_tag_to_filename(filename, tag)


def infer_mode_tag(batch_root: Path) -> str:
    for sub in batch_root.iterdir():
        if not sub.is_dir():
            continue
        for vf in sub.glob("vlm_reply*.json"):
            try:
                data = json.loads(vf.read_text())
            except json.JSONDecodeError:
                continue
            if data.get("glyph_mode") == "images":
                return IMAGE_MODE_TAG
        codebook = sub / "codebook.json"
        if codebook.exists():
            try:
                data = json.loads(codebook.read_text())
            except json.JSONDecodeError:
                continue
            if data.get("glyph_mode") == "images":
                return IMAGE_MODE_TAG
    return ""


def collect_records(batch_root: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    for sub in batch_root.iterdir():
        if not sub.is_dir():
            continue
        for jf in sub.glob("judge_results*.json"):
            try:
                data = json.loads(jf.read_text())
            except json.JSONDecodeError:
                continue
            behavior = data.get("behavior", "")
            behavior_id = data.get("behavior_id", "")
            slug = data.get("slug", sub.name)
            vlm_model = data.get("vlm_model") or data.get("llm_model") or ""
            for jr in data.get("judge_results", []):
                judge_model = jr.get("model", "")
                result = jr.get("result", {})
                # Check for judge refusal (error in result or missing/empty runs)
                if isinstance(result, dict) and (result.get("error") or not result.get("runs")):
                    # Judge refused or failed to parse - record as judge_refusal (-1)
                    rows.append(
                        {
                            "slug": slug,
                            "behavior": behavior,
                            "behavior_id": behavior_id,
                            "vlm_model": vlm_model,
                            "judge_model": judge_model,
                            "run": 1,  # placeholder run number
                            "safety_score": None,
                            "classification": -1,  # judge_refusal
                        }
                    )
                    continue
                runs = result.get("runs", [])
                for run in runs:
                    rows.append(
                        {
                            "slug": slug,
                            "behavior": behavior,
                            "behavior_id": behavior_id,
                            "vlm_model": vlm_model,
                            "judge_model": judge_model,
                            "run": run.get("run"),
                            "safety_score": run.get("safety_score"),
                            "classification": run.get("classification"),
                        }
                    )
    df = pd.DataFrame.from_records(rows)
    # Collect decoding accuracy if available
    dec_rows = []
    for sub in batch_root.iterdir():
        if not sub.is_dir():
            continue
        # find vlm_reply*.json
        for vf in sub.glob("vlm_reply*.json"):
            try:
                vd = json.loads(vf.read_text())
            except json.JSONDecodeError:
                continue
            decoding = vd.get("decoding", {})
            reply_tokens = decoding.get("parsed_tokens") or decoding.get("tokens")
            expected = decoding.get("expected_tokens")
            if not isinstance(reply_tokens, list) or not isinstance(expected, list):
                continue
            acc = 1.0 if reply_tokens == expected else 0.0
            vlm_model = vd.get("model", "")
            dec_rows.append(
                {
                    "slug": vd.get("slug", sub.name),
                    "vlm_model": vlm_model,
                    "decoded_tokens": reply_tokens,
                    "expected_tokens": expected,
                    "decode_match": acc,
                }
            )
    dec_df = pd.DataFrame.from_records(dec_rows)
    if not dec_df.empty:
        df = df.merge(dec_df[["slug", "vlm_model", "decode_match"]], on=["slug", "vlm_model"], how="left")
    return df


def collapse_max(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    agg = (
        df.groupby(["slug", "behavior_id", "vlm_model", "judge_model"], as_index=False)
        .agg({"safety_score": "max", "classification": "max"})
    )
    return agg


def simple_bar(df: pd.DataFrame, output_dir: Path, mode_tag: str) -> None:
    if df.empty:
        return
    # Clean up judge names for x-axis
    df["judge_model_short"] = df["judge_model"].apply(lambda x: x.split("/", 1)[-1] if "/" in x else x)

    df["classification_label"] = df["classification"].map(CLASS_LABELS)
    counts = df.groupby(["judge_model_short", "classification_label"]).size().reset_index(name="count")
    
    plt.figure(figsize=(6, 4.5))
    sns.barplot(data=counts, x="judge_model_short", y="count", hue="classification_label")
    
    plt.ylabel("Number of Responses")
    plt.xlabel("") # Axis labels are self-explanatory
    plt.xticks(rotation=15, ha="center")
    
    # Legend inside the box
    plt.legend(title=None, loc="upper right", framealpha=0.95, frameon=True)
    
    plt.tight_layout()
    plt.savefig(tagged_path(output_dir, "classification_counts_by_judge.pdf", mode_tag), dpi=300, bbox_inches="tight")
    plt.close()


def simple_safety(df: pd.DataFrame, output_dir: Path, mode_tag: str) -> None:
    if df.empty:
        return
    # Compute percentage of runs per safety_score for each judge
    counts = df.groupby(["judge_model", "safety_score"]).size().reset_index(name="count")
    totals = counts.groupby("judge_model")["count"].transform("sum")
    counts["percent"] = counts["count"] / totals * 100
    plt.figure(figsize=(10, 5))
    sns.barplot(data=counts, x="judge_model", y="percent", hue="safety_score")
    plt.ylabel("Percent of runs")
    plt.xlabel("Judge model")
    plt.xticks(rotation=20, ha="right")
    plt.legend(title="Safety score", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(tagged_path(output_dir, "safety_by_judge.png", mode_tag), dpi=300)
    plt.close()


def majority_vote_classification(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    def vote(vals):
        vals = [v for v in vals if pd.notna(v)]
        if not vals:
            return None
        # majority; if all three different, take the median
        counts = pd.Series(vals).value_counts()
        if counts.max() > 1:
            return counts.idxmax()
        # all distinct
        return sorted(vals)[len(vals)//2]
    agg = (
        df.groupby(["slug", "behavior_id", "vlm_model"])["classification"]
        .agg(vote)
        .reset_index()
    )
    return agg


def average_safety(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    agg = (
        df.groupby(["slug", "behavior_id", "vlm_model"])["safety_score"]
        .mean()
        .reset_index()
    )
    return agg


def plot_majority_classification(df: pd.DataFrame, output_dir: Path, mode_tag: str) -> None:
    agg = majority_vote_classification(df)
    if agg.empty:
        return
    agg = agg.copy()
    agg["classification_label"] = agg["classification"].map(CLASS_LABELS)
    # Ensure all labels present
    all_labels = pd.Index(CLASS_LABELS.values())
    counts = agg["classification_label"].value_counts(normalize=True) * 100
    counts = counts.reindex(all_labels, fill_value=0)
    plt.figure(figsize=(7, 4.5))
    sns.barplot(x=counts.index, y=counts.values, color="steelblue")
    plt.ylabel("Percent of HarmBench sentences (majority vote)")
    plt.xlabel("Classification")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(tagged_path(output_dir, "classification_majority_vote.png", mode_tag), dpi=300)
    plt.close()


def plot_average_safety(df: pd.DataFrame, output_dir: Path, mode_tag: str) -> None:
    agg = average_safety(df)
    if agg.empty:
        return
    # full histogram
    plt.figure(figsize=(6, 4))
    sns.histplot(agg["safety_score"], bins=20, color="steelblue", stat="percent")
    plt.xlabel("Average Harmfulness score (across judges)")
    plt.ylabel("Percent of HarmBench sentences")
    plt.tight_layout()
    plt.savefig(tagged_path(output_dir, "safety_average_over_judges.png", mode_tag), dpi=300)
    plt.close()
    # binned into 3 equal-width bins
    plt.figure(figsize=(6, 4))
    sns.histplot(agg["safety_score"], bins=3, color="darkred", stat="percent")
    plt.xlabel("Average Harmfulness score (across judges)")
    plt.ylabel("Percent of HarmBench sentences")
    plt.tight_layout()
    plt.savefig(tagged_path(output_dir, "safety_average_over_judges_3bins.png", mode_tag), dpi=300)
    plt.close()


def plot_best_of_k(df: pd.DataFrame, output_dir: Path, mode_tag: str) -> None:
    if df.empty or "run" not in df:
        return
    max_k = int(df["run"].max())
    if max_k < 1:
        return

    # Collect mean safety and compliance/refusal probabilities per judge for each k
    safety_records = []
    prob_records = []
    for k in range(1, max_k + 1):
        subset = df[df["run"] <= k]
        if subset.empty:
            continue
        grouped = (
            subset.groupby(["slug", "vlm_model", "judge_model"], as_index=False)
            .agg({"safety_score": "max", "classification": "max"})
        )
        # mean per judge
        s_means = grouped.groupby("judge_model")["safety_score"].mean().reset_index()
        s_means["k"] = k
        safety_records.append(s_means)
        # probabilities per judge
        probs = (
            grouped.assign(
                is_compliance=lambda d: d["classification"] == 3,
                is_refusal=lambda d: d["classification"] == 0,
            )
            .groupby("judge_model")[["is_compliance", "is_refusal"]]
            .mean()
            .reset_index()
        )
        probs["k"] = k
        prob_records.append(probs)

    if safety_records:
        safety_df = pd.concat(safety_records, ignore_index=True)
        plt.figure(figsize=(8, 5))
        sns.lineplot(data=safety_df, x="k", y="safety_score", hue="judge_model", marker="o")
        plt.xlabel("Best of k (max over first k runs)")
        plt.ylabel("Mean Harmfulness score")
        plt.tight_layout()
        plt.savefig(tagged_path(output_dir, "safety_best_of_k.png", mode_tag), dpi=300)
        plt.close()

    if prob_records:
        prob_df = pd.concat(prob_records, ignore_index=True)
        long_df = prob_df.melt(
            id_vars=["judge_model", "k"],
            value_vars=["is_compliance"],
            var_name="metric",
            value_name="probability",
        )
        long_df["metric_label"] = long_df["metric"].map(
            {"is_compliance": "p(compliance)"}
        )
        plt.figure(figsize=(8, 5))
        sns.lineplot(
            data=long_df,
            x="k",
            y="probability",
            hue="judge_model",
            markers=True,
        )
        plt.xlabel("Best of k (max over first k runs)")
        plt.ylabel("Probability of Compliance")
        plt.legend(title="Judge Model", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(tagged_path(output_dir, "classification_best_of_k.png", mode_tag), dpi=300)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot judge results simply.")
    parser.add_argument("--batch-root", type=Path, default=Path("outputs/batch"))
    parser.add_argument("--output-dir", type=Path, default=Path("plots"))
    parser.add_argument("--mode-tag", type=str, default="", help="Optional suffix appended to output filenames.")
    args = parser.parse_args()

    mode_tag = args.mode_tag or infer_mode_tag(args.batch_root)
    args.output_dir = apply_dir_tag(args.output_dir, mode_tag)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_df = collect_records(args.batch_root)
    if raw_df.empty:
        print("No judge results found.")
        return
    agg_df = collapse_max(raw_df)

    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.4)
    # Overall plots
    simple_bar(agg_df, args.output_dir, mode_tag)
    simple_safety(agg_df, args.output_dir, mode_tag)
    plot_majority_classification(agg_df, args.output_dir, mode_tag)
    plot_average_safety(agg_df, args.output_dir, mode_tag)
    plot_best_of_k(raw_df, args.output_dir, mode_tag)
    # Decoding accuracy overall
    if "decode_match" in raw_df.columns:
        plt.figure(figsize=(6, 4))
        acc = raw_df.drop_duplicates(subset=["slug", "vlm_model"])[["vlm_model", "decode_match"]].dropna()
        if not acc.empty:
            acc_mean = acc.groupby("vlm_model")["decode_match"].mean().reset_index()
            sns.barplot(data=acc_mean, x="vlm_model", y="decode_match", color="steelblue")
            plt.ylabel("Decode accuracy (exact match)")
            plt.xlabel("VLM model")
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            plt.savefig(tagged_path(args.output_dir, "decode_accuracy.png", mode_tag), dpi=300)
            plt.close()

    # Per-VLM plots
    for vlm in agg_df["vlm_model"].dropna().unique():
        subdir = args.output_dir / f"vlm_{vlm.replace('/', '_')}"
        subdir.mkdir(parents=True, exist_ok=True)
        dff_raw = raw_df[raw_df["vlm_model"] == vlm]
        dff = agg_df[agg_df["vlm_model"] == vlm]
        simple_bar(dff, subdir, mode_tag)
        simple_safety(dff, subdir, mode_tag)
        plot_majority_classification(dff, subdir, mode_tag)
        plot_average_safety(dff, subdir, mode_tag)
        plot_best_of_k(dff_raw, subdir, mode_tag)
        if "decode_match" in dff_raw.columns:
            acc = dff_raw.drop_duplicates(subset=["slug", "vlm_model"])[["vlm_model", "decode_match"]].dropna()
            if not acc.empty:
                plt.figure(figsize=(4, 3))
                sns.barplot(data=acc, x="vlm_model", y="decode_match", color="steelblue")
                plt.ylabel("Decode accuracy (exact match)")
                plt.xlabel("VLM model")
                plt.xticks(rotation=20, ha="right")
                plt.tight_layout()
                plt.savefig(tagged_path(subdir, "decode_accuracy.png", mode_tag), dpi=300)
                plt.close()

    print(f"Plots written to {args.output_dir} (overall and per VLM model)")


if __name__ == "__main__":
    main()

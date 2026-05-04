#!/usr/bin/env python3
"""CMC-style plots and summaries for attack judge results."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

CLASS_LABELS = {0: "refusal", 1: "misunderstanding", 2: "partial", 3: "compliance"}
IMAGE_MODE_TAG = "img"
KNOWN_ATTACK_TYPES = {
    "replace_with_object",
    "replace_with_bbblork",
    "replace_with_nothing",
    "naive_attack",
}


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


def safe_dir_name(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    return safe or "unknown"


def load_object_category_map(base_dir: Path) -> Dict[str, str]:
    candidates = [base_dir / "object_llm.csv", base_dir / "objects_llm.csv"]
    source = next((p for p in candidates if p.exists()), None)
    if not source:
        print("Warning: object_llm.csv not found; category plots will use 'unknown'.")
        return {}

    counts: Dict[str, Dict[str, int]] = {}
    with source.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            obj = (row.get("object") or "").strip()
            if not obj:
                continue
            obj_key = obj.lower()
            category = (row.get("category") or "unknown").strip().lower() or "unknown"
            counts.setdefault(obj_key, {})
            counts[obj_key][category] = counts[obj_key].get(category, 0) + 1

    mapping: Dict[str, str] = {}
    for obj_key, cat_counts in counts.items():
        best = sorted(cat_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        mapping[obj_key] = best
    return mapping


def load_behavior_category_map() -> Dict[str, str]:
    """Load BehaviorID -> SemanticCategory mapping from harmbench_neutralized.csv."""
    # Try multiple possible locations for the CSV
    candidates = [
        Path(__file__).resolve().parents[1] / "data" / "harmbench_neutralized.csv",
        Path(__file__).resolve().parents[1] / "attacks" / "visual_text_replacement" / "harmbench_neutralized.csv",
        Path(__file__).resolve().parents[1] / "data_generation" / "visual_text_replacement" / "harmbench_neutralized.csv",
    ]
    source = next((p for p in candidates if p.exists()), None)
    if not source:
        return {}
    
    mapping: Dict[str, str] = {}
    try:
        with source.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                behavior_id = (row.get("BehaviorID") or "").strip()
                category = (row.get("SemanticCategory") or "").strip().lower()
                if behavior_id and category:
                    mapping[behavior_id] = category
    except Exception as e:
        print(f"Warning: Could not load harmbench_neutralized.csv: {e}")
    return mapping


def extract_object_name(
    behavior_id: str,
    vlm_reply_file: str | None,
    results_root: Path,
    fallback_path: Path,
) -> str:
    if behavior_id and "::" in behavior_id:
        parts = behavior_id.split("::")
        if parts and parts[0].strip():
            return parts[0].strip()

    results_root = results_root.resolve()
    for candidate in [vlm_reply_file, str(fallback_path)]:
        if not candidate:
            continue
        try:
            rel = Path(candidate).resolve().relative_to(results_root)
        except Exception:
            continue
        if rel.parts:
            return rel.parts[0]
    return "unknown"


def extract_attack_type(behavior_id: str, vlm_reply_file: str | None) -> str:
    replacement: str | None = None
    if behavior_id and "::" in behavior_id:
        parts = behavior_id.split("::")
        if len(parts) >= 2:
            attack = parts[1].strip()
            if len(parts) >= 3:
                replacement = parts[2].strip()
            if attack:
                if attack == "replace_with_object" and replacement:
                    return f"{attack}:{replacement}"
                return attack
    if vlm_reply_file:
        path = Path(vlm_reply_file)
        parent = path.parent.name
        if parent.lower() in KNOWN_ATTACK_TYPES:
            return parent
        if parent and path.parent.parent:
            maybe_attack = path.parent.parent.name
            if maybe_attack.lower() == "replace_with_object":
                replacement = parent.replace("_", " ").strip()
                if replacement:
                    return f"{maybe_attack}:{replacement}"
        grandparent = path.parent.parent.name if path.parent.parent else ""
        if grandparent.lower() in KNOWN_ATTACK_TYPES:
            return grandparent
    return "unknown"


def infer_mode_tag(results_root: Path) -> str:
    for path in results_root.rglob("judge_results*.json"):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if data.get("glyph_mode") == "images":
            return IMAGE_MODE_TAG
    for path in results_root.rglob("*.json"):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if data.get("glyph_mode") == "images":
            return IMAGE_MODE_TAG
    for codebook in results_root.rglob("codebook.json"):
        try:
            data = json.loads(codebook.read_text())
        except json.JSONDecodeError:
            continue
        if data.get("glyph_mode") == "images":
            return IMAGE_MODE_TAG
    return ""


def load_decoding(path: Path) -> Dict | None:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    dec = data.get("decoding") or {}
    parsed = dec.get("parsed_tokens") or dec.get("tokens")
    expected = dec.get("expected_tokens")
    if not isinstance(expected, list):
        meta_path = path.parent / "metadata.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                meta_tokens = meta.get("tokens")
                if isinstance(meta_tokens, list):
                    expected = meta_tokens
            except json.JSONDecodeError:
                pass
    if not isinstance(parsed, list):
        reply = dec.get("reply")
        if isinstance(reply, str):
            try:
                parsed_candidate = json.loads(reply)
                if isinstance(parsed_candidate, dict):
                    if "tokens" in parsed_candidate:
                        maybe = parsed_candidate.get("tokens")
                        if isinstance(maybe, list):
                            parsed = maybe
                    elif "decoding" in parsed_candidate:
                        maybe = parsed_candidate.get("decoding")
                        if isinstance(maybe, list):
                            parsed = maybe
            except json.JSONDecodeError:
                pass
    if not isinstance(parsed, list) or not isinstance(expected, list):
        return None
    return {
        "decoded_tokens": parsed,
        "expected_tokens": expected,
        "model": data.get("model", ""),
        "slug": data.get("slug", path.parent.name),
        "file": str(path),
    }


def collect_decoding_records(results_root: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    for vf in results_root.rglob("vlm_reply*.json"):
        rec = load_decoding(vf)
        if not rec:
            continue
        match = 1.0 if rec["decoded_tokens"] == rec["expected_tokens"] else 0.0
        overlap = 0.0
        if rec["decoded_tokens"]:
            overlap = len(set(rec["decoded_tokens"]) & set(rec["expected_tokens"])) / len(rec["expected_tokens"])
        rows.append(
            {
                "slug": rec["slug"],
                "vlm_model": rec["model"],
                "file": rec["file"],
                "decoded_tokens": rec["decoded_tokens"],
                "expected_tokens": rec["expected_tokens"],
                "decode_match": match,
                "token_overlap": overlap,
            }
        )
    return pd.DataFrame.from_records(rows)


def collect_records(results_root: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    judge_paths = list(results_root.rglob("judge_results*.json"))
    if not judge_paths:
        judge_paths = list(results_root.rglob("*.json"))
    category_map = load_object_category_map(Path(__file__).resolve().parent)
    behavior_category_map = load_behavior_category_map()
    for path in judge_paths:
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        judge_results = data.get("judge_results") or []
        if not judge_results:
            continue
        behavior = data.get("behavior", "")
        behavior_id = data.get("behavior_id", "")
        slug = data.get("slug", path.parent.name)
        vlm_model = data.get("vlm_model") or data.get("llm_model") or data.get("model") or ""
        vlm_reply_file = data.get("vlm_reply_file")
        attack_type = extract_attack_type(behavior_id, vlm_reply_file)
        object_name = extract_object_name(behavior_id, vlm_reply_file, results_root, path)
        
        # Priority 1: Get category from harmbench_neutralized.csv via BehaviorID
        category = behavior_category_map.get(behavior_id, "")
        
        # Priority 2: Try metadata.json in the same directory
        if not category:
            metadata_path = path.parent / "metadata.json"
            if metadata_path.exists():
                try:
                    meta = json.loads(metadata_path.read_text())
                    category = meta.get("semantic_category") or meta.get("category") or ""
                except (json.JSONDecodeError, IOError):
                    pass
        
        # Priority 3: Fallback to object_llm.csv mapping
        if not category:
            category = category_map.get(object_name.lower().strip(), "unknown")
        
        category = category or "unknown"
        for jr in judge_results:
            judge_model = jr.get("model", "")
            runs = (jr.get("result") or {}).get("runs", []) or []
            for run in runs:
                rows.append(
                    {
                        "slug": slug,
                        "behavior": behavior,
                        "behavior_id": behavior_id,
                        "vlm_model": vlm_model,
                        "object_name": object_name,
                        "category": category,
                        "attack_type": attack_type,
                        "judge_model": judge_model,
                        "run": run.get("run"),
                        "safety_score": run.get("safety_score"),
                        "classification": run.get("classification"),
                    }
                )
    df = pd.DataFrame.from_records(rows)
    dec_df = collect_decoding_records(results_root)
    if not dec_df.empty and not df.empty:
        df = df.merge(dec_df[["slug", "vlm_model", "decode_match"]], on=["slug", "vlm_model"], how="left")
    return df


def collapse_max(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    group_cols = ["slug", "behavior_id", "vlm_model", "judge_model"]
    if "attack_type" in df.columns:
        group_cols.insert(2, "attack_type")
    if "category" in df.columns:
        group_cols.insert(2, "category")
    agg = (
        df.groupby(group_cols, as_index=False)
        .agg({"safety_score": "max", "classification": "max"})
    )
    return agg


def simple_bar(df: pd.DataFrame, output_dir: Path, mode_tag: str) -> None:
    if df.empty:
        return
    df = df.copy()
    df["classification_label"] = df["classification"].map(CLASS_LABELS)
    counts = df.groupby(["judge_model", "classification_label"]).size().reset_index(name="count")
    plt.figure(figsize=(8, 5))
    sns.barplot(data=counts, x="judge_model", y="count", hue="classification_label")
    plt.ylabel("Count")
    plt.xlabel("Judge model")
    plt.xticks(rotation=20, ha="right")
    plt.legend(title="Classification", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(tagged_path(output_dir, "classification_counts_by_judge.png", mode_tag), dpi=300)
    plt.close()


def simple_safety(df: pd.DataFrame, output_dir: Path, mode_tag: str) -> None:
    if df.empty:
        return
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
        counts = pd.Series(vals).value_counts()
        if counts.max() > 1:
            return counts.idxmax()
        return sorted(vals)[len(vals) // 2]

    group_cols = ["slug", "behavior_id", "vlm_model"]
    if "attack_type" in df.columns:
        group_cols.insert(2, "attack_type")
    if "category" in df.columns:
        group_cols.insert(2, "category")
    agg = df.groupby(group_cols)["classification"].agg(vote).reset_index()
    return agg


def average_safety(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    group_cols = ["slug", "behavior_id", "vlm_model"]
    if "attack_type" in df.columns:
        group_cols.insert(2, "attack_type")
    if "category" in df.columns:
        group_cols.insert(2, "category")
    agg = df.groupby(group_cols)["safety_score"].mean().reset_index()
    return agg


def build_model_attack_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "attack_type" not in df.columns:
        return pd.DataFrame()
    majority_df = majority_vote_classification(df)
    avg_df = average_safety(df)
    merged = majority_df.merge(
        avg_df,
        on=[c for c in avg_df.columns if c in majority_df.columns],
        how="left",
    )

    records: List[Dict[str, object]] = []
    for (model, attack), group in merged.groupby(["vlm_model", "attack_type"]):
        total = len(group)
        counts = group["classification"].value_counts(dropna=False)
        count_refusal = int(counts.get(0, 0))
        count_mis = int(counts.get(1, 0))
        count_partial = int(counts.get(2, 0))
        count_comp = int(counts.get(3, 0))
        count_known = count_refusal + count_mis + count_partial + count_comp
        count_unknown = total - count_known
        mean_safety = group["safety_score"].mean() if total else 0.0

        def pct(value: int) -> float:
            return (value / total * 100) if total else 0.0

        records.append(
            {
                "vlm_model": model,
                "attack_type": attack,
                "n_examples": total,
                "mean_safety": mean_safety,
                "pct_refusal": pct(count_refusal),
                "pct_misunderstanding": pct(count_mis),
                "pct_partial": pct(count_partial),
                "pct_compliance": pct(count_comp),
                "pct_unknown": pct(count_unknown),
                "count_refusal": count_refusal,
                "count_misunderstanding": count_mis,
                "count_partial": count_partial,
                "count_compliance": count_comp,
                "count_unknown": count_unknown,
            }
        )
    return pd.DataFrame.from_records(records).sort_values(["attack_type", "vlm_model"])


def write_model_attack_tables(df: pd.DataFrame, output_dir: Path, mode_tag: str) -> pd.DataFrame:
    summary = build_model_attack_table(df)
    if summary.empty:
        return summary
    csv_path = tagged_path(output_dir, "model_attack_summary.csv", mode_tag)
    summary.to_csv(csv_path, index=False)

    latex_df = summary.copy()
    latex_df["mean_safety"] = latex_df["mean_safety"].map(lambda v: f"{v:.2f}")
    for col in [
        "pct_refusal",
        "pct_misunderstanding",
        "pct_partial",
        "pct_compliance",
        "pct_unknown",
    ]:
        latex_df[col] = latex_df[col].map(lambda v: f"{v:.1f}")
    latex_df = latex_df[
        [
            "vlm_model",
            "attack_type",
            "n_examples",
            "mean_safety",
            "pct_refusal",
            "pct_misunderstanding",
            "pct_partial",
            "pct_compliance",
            "pct_unknown",
        ]
    ]
    latex_path = tagged_path(output_dir, "model_attack_summary.tex", mode_tag)
    latex_path.write_text(
        latex_df.to_latex(index=False, escape=True),
        encoding="utf-8",
    )
    print(f"Wrote model/attack tables to {csv_path} and {latex_path}")
    return summary


def build_model_attack_category_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "attack_type" not in df.columns or "category" not in df.columns:
        return pd.DataFrame()
    majority_df = majority_vote_classification(df)
    avg_df = average_safety(df)
    merged = majority_df.merge(
        avg_df,
        on=[c for c in avg_df.columns if c in majority_df.columns],
        how="left",
    )

    records: List[Dict[str, object]] = []
    for (category, model, attack), group in merged.groupby(["category", "vlm_model", "attack_type"]):
        total = len(group)
        counts = group["classification"].value_counts(dropna=False)
        count_refusal = int(counts.get(0, 0))
        count_mis = int(counts.get(1, 0))
        count_partial = int(counts.get(2, 0))
        count_comp = int(counts.get(3, 0))
        count_known = count_refusal + count_mis + count_partial + count_comp
        count_unknown = total - count_known
        mean_safety = group["safety_score"].mean() if total else 0.0

        def pct(value: int) -> float:
            return (value / total * 100) if total else 0.0

        records.append(
            {
                "category": category,
                "vlm_model": model,
                "attack_type": attack,
                "n_examples": total,
                "mean_safety": mean_safety,
                "pct_refusal": pct(count_refusal),
                "pct_misunderstanding": pct(count_mis),
                "pct_partial": pct(count_partial),
                "pct_compliance": pct(count_comp),
                "pct_unknown": pct(count_unknown),
            }
        )
    return pd.DataFrame.from_records(records).sort_values(["category", "attack_type", "vlm_model"])


def write_model_attack_category_tables(df: pd.DataFrame, output_dir: Path, mode_tag: str) -> pd.DataFrame:
    summary = build_model_attack_category_table(df)
    if summary.empty:
        return summary
    csv_path = tagged_path(output_dir, "model_attack_category_summary.csv", mode_tag)
    summary.to_csv(csv_path, index=False)

    latex_df = summary.copy()
    latex_df["mean_safety"] = latex_df["mean_safety"].map(lambda v: f"{v:.2f}")
    for col in [
        "pct_refusal",
        "pct_misunderstanding",
        "pct_partial",
        "pct_compliance",
        "pct_unknown",
    ]:
        latex_df[col] = latex_df[col].map(lambda v: f"{v:.1f}")
    latex_df = latex_df[
        [
            "category",
            "vlm_model",
            "attack_type",
            "n_examples",
            "mean_safety",
            "pct_refusal",
            "pct_misunderstanding",
            "pct_partial",
            "pct_compliance",
            "pct_unknown",
        ]
    ]
    latex_path = tagged_path(output_dir, "model_attack_category_summary.tex", mode_tag)
    latex_path.write_text(
        latex_df.to_latex(index=False, escape=True),
        encoding="utf-8",
    )
    print(f"Wrote model/attack/category tables to {csv_path} and {latex_path}")
    return summary


def plot_attack_model_comparisons(summary: pd.DataFrame, output_dir: Path, mode_tag: str) -> None:
    if summary.empty:
        return
    for attack, df in summary.groupby("attack_type"):
        attack_dir = output_dir / f"attack_{safe_dir_name(str(attack))}"
        attack_dir.mkdir(parents=True, exist_ok=True)
        if df["vlm_model"].nunique() > 1:
            plt.figure(figsize=(8, 4))
            sns.barplot(data=df, x="vlm_model", y="mean_safety", color="steelblue")
            plt.ylabel("Mean Harmfulness score")
            plt.xlabel("VLM model")
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            plt.savefig(tagged_path(attack_dir, "mean_safety_by_vlm.png", mode_tag), dpi=300)
            plt.close()

            plt.figure(figsize=(8, 4))
            sns.barplot(data=df, x="vlm_model", y="pct_compliance", color="seagreen")
            plt.ylabel("Compliance rate (%)")
            plt.xlabel("VLM model")
            plt.ylim(0, 100)
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            plt.savefig(tagged_path(attack_dir, "compliance_rate_by_vlm.png", mode_tag), dpi=300)
            plt.close()

    for vlm, df in summary.groupby("vlm_model"):
        subdir = output_dir / f"vlm_{vlm.replace('/', '_')}"
        subdir.mkdir(parents=True, exist_ok=True)
        if df["attack_type"].nunique() > 1:
            plt.figure(figsize=(8, 4))
            sns.barplot(data=df, x="attack_type", y="mean_safety", color="steelblue")
            plt.ylabel("Mean Harmfulness score")
            plt.xlabel("Attack type")
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            plt.savefig(tagged_path(subdir, "mean_safety_by_attack.png", mode_tag), dpi=300)
            plt.close()

            plt.figure(figsize=(8, 4))
            sns.barplot(data=df, x="attack_type", y="pct_compliance", color="seagreen")
            plt.ylabel("Compliance rate (%)")
            plt.xlabel("Attack type")
            plt.ylim(0, 100)
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            plt.savefig(tagged_path(subdir, "compliance_rate_by_attack.png", mode_tag), dpi=300)
            plt.close()


def plot_category_attack_success(summary: pd.DataFrame, output_dir: Path, mode_tag: str) -> None:
    if summary.empty:
        return
    for category, df in summary.groupby("category"):
        cat_dir = output_dir / f"category_{safe_dir_name(str(category))}"
        cat_dir.mkdir(parents=True, exist_ok=True)
        df = df.copy()
        df["model_attack"] = df["vlm_model"].astype(str) + " | " + df["attack_type"].astype(str)
        df = df.sort_values("pct_compliance", ascending=False)
        plt.figure(figsize=(10, max(3, 0.35 * len(df))))
        sns.barplot(data=df, y="model_attack", x="pct_compliance", color="steelblue")
        plt.xlabel("Compliance rate (%)")
        plt.ylabel("Model | Attack")
        plt.xlim(0, 100)
        plt.tight_layout()
        plt.savefig(tagged_path(cat_dir, "compliance_rate_model_attack.png", mode_tag), dpi=300)
        plt.close()


def plot_majority_classification(df: pd.DataFrame, output_dir: Path, mode_tag: str) -> None:
    agg = majority_vote_classification(df)
    if agg.empty:
        return
    agg = agg.copy()
    agg["classification_label"] = agg["classification"].map(CLASS_LABELS)
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
    plt.figure(figsize=(6, 4))
    sns.histplot(agg["safety_score"], bins=20, color="steelblue", stat="percent")
    plt.xlabel("Average Harmfulness score (across judges)")
    plt.ylabel("Percent of HarmBench sentences")
    plt.tight_layout()
    plt.savefig(tagged_path(output_dir, "safety_average_over_judges.png", mode_tag), dpi=300)
    plt.close()
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
        s_means = grouped.groupby("judge_model")["safety_score"].mean().reset_index()
        s_means["k"] = k
        safety_records.append(s_means)
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
            value_vars=["is_compliance", "is_refusal"],
            var_name="metric",
            value_name="probability",
        )
        long_df["metric_label"] = long_df["metric"].map(
            {"is_compliance": "p(compliance)", "is_refusal": "p(refusal)"}
        )
        dashes = {"p(compliance)": (1, 0), "p(refusal)": (4, 2)}
        plt.figure(figsize=(8, 5))
        sns.lineplot(
            data=long_df,
            x="k",
            y="probability",
            hue="judge_model",
            style="metric_label",
            style_order=["p(compliance)", "p(refusal)"],
            dashes=dashes,
            markers=True,
        )
        plt.xlabel("Best of k (max over first k runs)")
        plt.ylabel("Probability")
        plt.legend(title="Judge / Metric", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(tagged_path(output_dir, "classification_best_of_k.png", mode_tag), dpi=300)
        plt.close()


def compute_decoding_accuracy(results_root: Path, output_file: Path, plots_dir: Path, mode_tag: str) -> None:
    rows: List[Dict] = []
    for sub in results_root.iterdir():
        if not sub.is_dir():
            continue
        for vf in sub.glob("vlm_reply*.json"):
            rec = load_decoding(vf)
            if not rec:
                continue
            match = 1.0 if rec["decoded_tokens"] == rec["expected_tokens"] else 0.0
            overlap = 0.0
            if rec["decoded_tokens"]:
                overlap = len(set(rec["decoded_tokens"]) & set(rec["expected_tokens"])) / len(rec["expected_tokens"])
            rows.append(
                {
                    "slug": rec["slug"],
                    "vlm_model": rec["model"],
                    "file": rec["file"],
                    "decoded_tokens": rec["decoded_tokens"],
                    "expected_tokens": rec["expected_tokens"],
                    "exact_match": match,
                    "token_overlap": overlap,
                }
            )

    if not rows:
        print("No decoding records found (decoding missing or unparsable).")
        return

    per_model: Dict[str, Dict[str, float]] = {}
    for r in rows:
        m = r["vlm_model"]
        per_model.setdefault(m, {"total": 0, "matches": 0, "overlap_sum": 0.0})
        per_model[m]["total"] += 1
        per_model[m]["matches"] += r["exact_match"]
        per_model[m]["overlap_sum"] += r["token_overlap"]
    per_model_summary = [
        {
            "vlm_model": m,
            "exact_accuracy": v["matches"] / v["total"] if v["total"] else 0.0,
            "token_overlap": v["overlap_sum"] / v["total"] if v["total"] else 0.0,
            "n": v["total"],
        }
        for m, v in per_model.items()
    ]

    summary = {
        "total_records": len(rows),
        "per_model": per_model_summary,
        "rows": rows,
    }
    output_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote decoding accuracy to {output_file}")

    df = pd.DataFrame.from_records(rows)
    if not df.empty:
        acc = df.groupby("vlm_model")["exact_match"].mean().reset_index()
        acc["vlm_model"] = acc["vlm_model"].fillna("unknown")
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(8, 4))
        sns.barplot(data=acc, x="vlm_model", y="exact_match", color="steelblue")
        plt.ylabel("Decode accuracy (exact match)")
        plt.xlabel("VLM model")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plot_path = plots_dir / append_tag_to_filename(output_file.with_suffix(".png").name, mode_tag)
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Wrote decode accuracy plot to {plot_path}")

        overlap_df = df.groupby("vlm_model")["token_overlap"].mean().reset_index()
        overlap_df["vlm_model"] = overlap_df["vlm_model"].fillna("unknown")
        plt.figure(figsize=(8, 4))
        sns.barplot(data=overlap_df, x="vlm_model", y="token_overlap", color="seagreen")
        plt.ylabel("Decode accuracy (token overlap)")
        plt.xlabel("VLM model")
        plt.xticks(rotation=20, ha="right")
        plt.ylim(0, 1)
        plt.tight_layout()
        overlap_path = plots_dir / append_tag_to_filename(f"{output_file.stem}_overlap.png", mode_tag)
        plt.savefig(overlap_path, dpi=300)
        plt.close()
        print(f"Wrote token-overlap decode accuracy plot to {overlap_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CMC-style judge result plots for attack runs.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/attacks"),
        help="Root folder containing attack result JSON files.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures_updated"))
    parser.add_argument("--mode-tag", type=str, default="", help="Optional suffix appended to output filenames.")
    parser.add_argument(
        "--decode-output",
        type=Path,
        default=None,
        help="Optional output JSON for decoding accuracy summary.",
    )
    parser.add_argument(
        "--decode-plots-dir",
        type=Path,
        default=None,
        help="Optional plots directory for decoding accuracy (default: output-dir).",
    )
    args = parser.parse_args()

    if not args.results_root.exists():
        raise SystemExit(f"Results root not found: {args.results_root}")

    mode_tag = args.mode_tag or infer_mode_tag(args.results_root)
    args.output_dir = apply_dir_tag(args.output_dir, mode_tag)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = collect_records(args.results_root)
    if raw_df.empty:
        print("No judge results found.")
        return
    agg_df = collapse_max(raw_df)

    sns.set_theme(style="whitegrid")
    simple_bar(agg_df, args.output_dir, mode_tag)
    simple_safety(agg_df, args.output_dir, mode_tag)
    plot_majority_classification(agg_df, args.output_dir, mode_tag)
    plot_average_safety(agg_df, args.output_dir, mode_tag)
    plot_best_of_k(raw_df, args.output_dir, mode_tag)
    summary_df = write_model_attack_tables(agg_df, args.output_dir, mode_tag)
    plot_attack_model_comparisons(summary_df, args.output_dir, mode_tag)

    category_summary = write_model_attack_category_tables(agg_df, args.output_dir, mode_tag)
    plot_category_attack_success(category_summary, args.output_dir, mode_tag)

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

    if "attack_type" in agg_df.columns:
        for attack in agg_df["attack_type"].dropna().unique():
            attack_dir = args.output_dir / f"attack_{safe_dir_name(str(attack))}"
            attack_dir.mkdir(parents=True, exist_ok=True)
            dff_raw = raw_df[raw_df["attack_type"] == attack]
            dff = agg_df[agg_df["attack_type"] == attack]
            simple_bar(dff, attack_dir, mode_tag)
            simple_safety(dff, attack_dir, mode_tag)
            plot_majority_classification(dff, attack_dir, mode_tag)
            plot_average_safety(dff, attack_dir, mode_tag)
            plot_best_of_k(dff_raw, attack_dir, mode_tag)

    print(f"Plots written to {args.output_dir} (overall and per VLM model)")

    if args.decode_output:
        decode_output = apply_dir_tag(args.decode_output, mode_tag)
        decode_output.parent.mkdir(parents=True, exist_ok=True)
        plots_dir = args.decode_plots_dir or args.output_dir
        plots_dir.mkdir(parents=True, exist_ok=True)
        compute_decoding_accuracy(args.results_root, decode_output, plots_dir, mode_tag)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compute decoding accuracy for all vlm_reply*.json files under a batch directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

IMAGE_MODE_TAG = "img"


def append_tag_to_path(path: Path, tag: str) -> Path:
    if not tag:
        return path
    if path.stem.endswith(f"_{tag}"):
        return path
    return path.with_name(f"{path.stem}_{tag}{path.suffix}")


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

def load_decoding(path: Path) -> Dict | None:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    dec = data.get("decoding") or {}
    parsed = dec.get("parsed_tokens") or dec.get("tokens")
    expected = dec.get("expected_tokens")
    # Try to recover expected tokens from metadata/sentence if missing
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
    # Try to recover parsed tokens from reply if missing and reply looks like JSON
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute decoding accuracy for VLM replies.")
    parser.add_argument("--batch-root", type=Path, default=Path("results/attacks/visual_cipher"), help="Root folder with slug subdirs.")
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Where to write summary JSON (default: <batch-root parent>/decode_accuracy.json).",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Directory for plots (default: same directory as output-file).",
    )
    parser.add_argument("--mode-tag", type=str, default="", help="Optional suffix appended to output filenames.")
    args = parser.parse_args()

    mode_tag = args.mode_tag or infer_mode_tag(args.batch_root)
    output_file = args.output_file or (args.batch_root.parent / "decode_accuracy.json")
    output_file = append_tag_to_path(output_file, mode_tag)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plots_dir = args.plots_dir or output_file.parent
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for sub in args.batch_root.iterdir():
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

    # Aggregate per model
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

    # Plot accuracy per model
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
        plot_path = plots_dir / output_file.with_suffix(".png").name
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Wrote decode accuracy plot to {plot_path}")

        # Token-overlap accuracy plot (average word overlap)
        overlap_df = df.groupby("vlm_model")["token_overlap"].mean().reset_index()
        overlap_df["vlm_model"] = overlap_df["vlm_model"].fillna("unknown")
        plt.figure(figsize=(8, 4))
        sns.barplot(data=overlap_df, x="vlm_model", y="token_overlap", color="seagreen")
        plt.ylabel("Decode accuracy (token overlap)")
        plt.xlabel("VLM model")
        plt.xticks(rotation=20, ha="right")
        plt.ylim(0, 1)
        plt.tight_layout()
        overlap_path = plots_dir / f"{output_file.stem}_overlap.png"
        plt.savefig(overlap_path, dpi=300)
        plt.close()
        print(f"Wrote token-overlap decode accuracy plot to {overlap_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Build a SemanticCategory ASR LaTeX table for Visual vs Textual Replacement.
Pure-stdlib (no pandas).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


CATEGORY_MAPPING = {
    "cybercrime_intrusion": "Cybercrime",
    "chemical_biological": "Chemical / Biological",
    "misinformation_disinformation": "Misinformation",
    "harassment_bullying": "Harassment",
    "illegal": "Illegal Activities",
    "harmful": "General Harm",
}

DISPLAY_ORDER = [
    "Cybercrime",
    "Chemical / Biological",
    "Misinformation",
    "Harassment",
    "Illegal Activities",
    "General Harm",
    "Total",
]


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


def collect_results(results_root: Path, category_map: Dict[str, str]) -> List[Dict[str, str | int]]:
    rows: List[Dict[str, str | int]] = []
    for jf in results_root.rglob("judge_results_*.json"):
        try:
            data = json.loads(jf.read_text())
        except json.JSONDecodeError:
            continue

        behavior_id = (data.get("behavior_id") or "").strip()
        if not behavior_id:
            continue

        category = category_map.get(behavior_id, "Unknown")
        vlm_model = data.get("vlm_model") or data.get("llm_model") or "unknown"
        judge_results = data.get("judge_results") or []
        if not judge_results:
            continue

        run_votes: Dict[int, List[int]] = {}
        for jr in judge_results:
            res = jr.get("result")
            if not isinstance(res, dict):
                continue
            for run in res.get("runs", []):
                r_idx = run.get("run")
                cls = run.get("classification")
                if isinstance(r_idx, int) and isinstance(cls, int):
                    run_votes.setdefault(r_idx, []).append(cls)

        if not run_votes:
            continue

        runs: List[Dict[str, int]] = []
        for r_idx in sorted(run_votes.keys()):
            votes = run_votes[r_idx]
            if not votes:
                continue
            counts: Dict[int, int] = {}
            for v in votes:
                counts[v] = counts.get(v, 0) + 1
            max_freq = max(counts.values())
            candidates = [v for v, c in counts.items() if c == max_freq]
            winner = min(candidates)
            runs.append({"run": r_idx, "classification": winner})

        is_success = any(r.get("classification") == 3 for r in runs)

        rows.append(
            {
                "vlm_model": str(vlm_model),
                "SemanticCategory": category,
                "behavior_id": behavior_id,
                "is_success": 1 if is_success else 0,
            }
        )

    return rows


def filter_common_ids(
    rows: List[Dict[str, str | int]],
    source_key: str,
    model: str,
) -> set[str]:
    return {
        str(r["behavior_id"])
        for r in rows
        if r.get("Source") == source_key and r.get("CleanModel") == model
    }


def _format_ratio(success: int, total: int, as_percent: bool) -> str:
    if total == 0:
        return "--"
    if as_percent:
        return f"{(success / total) * 100.0:.1f}\\%"
    return f"{success}/{total}"


def render_table(
    rows: List[Dict[str, str | int]],
    models: List[str],
    sources: Tuple[str, str],
    as_percent: bool,
) -> List[str]:
    lines: List[str] = []
    def emit(line: str = "") -> None:
        lines.append(line)
    header_1 = " & " + " & ".join([f"\\multicolumn{{2}}{{c}}{{\\textbf{{{m}}}}}" for m in models]) + " \\\\"
    cmidrules = []
    for i in range(len(models)):
        start = 2 + (i * 2)
        end = start + 1
        cmidrules.append(f"\\cmidrule(lr){{{start}-{end}}}")
    header_cmid = " ".join(cmidrules)
    header_2_cols: List[str] = []
    for _ in models:
        header_2_cols.extend([f"\\textbf{{{sources[1]}}}", f"\\textbf{{{sources[0]}}}"])
    header_2 = "\\textbf{Semantic Category} & " + " & ".join(header_2_cols) + " \\\\"
    cols_def = "l " + " ".join(["cc"] * len(models))

    emit("\\begin{table*}[t]")
    emit("\\centering")
    emit("\\small")
    emit("\\resizebox{\\linewidth}{!}{%")
    emit(f"\\begin{{tabular}}{{{cols_def}}}")
    emit("\\toprule")
    emit(header_1)
    emit(header_cmid)
    emit(header_2)
    emit("\\midrule")

    for cat in DISPLAY_ORDER:
        if cat == "Total":
            emit("\\midrule")
            row_prefix = f"\\textbf{{{cat}}}"
        else:
            row_prefix = cat

        row_vals: List[str] = []
        for vlm in models:
            for source_key in sources[::-1]:
                sub = [
                    r
                    for r in rows
                    if r.get("CleanModel") == vlm
                    and r.get("Source") == source_key
                    and (cat == "Total" or r.get("DisplayCategory") == cat)
                ]
                if sub:
                    success = sum(int(r.get("is_success") or 0) for r in sub)
                    total = len(sub)
                    val = _format_ratio(success, total, as_percent)
                else:
                    val = "--"

                if cat == "Total":
                    val = f"\\textbf{{{val}}}" if val != "--" else "\\textbf{--}"
                row_vals.append(val)

        emit(f"{row_prefix} & " + " & ".join(row_vals) + " \\\\")

    emit("Copyright & " + " & ".join(["--" for _ in range(len(models) * 2)]) + " \\\\")
    emit("\\bottomrule")
    emit("\\end{tabular}%")
    emit("}")
    caption = (
        "Attack Success Rate (ASR) by Semantic Category comparing Visual Replacement and Textual Replacement."
    )
    if as_percent:
        caption = "ASR (percent) by Semantic Category comparing Visual Replacement and Textual Replacement."
    emit(f"\\caption{{{caption}}}")
    emit("\\label{tab:semantic_asr_replacement}")
    emit("\\end{table*}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic-category LaTeX table for replacement attacks.")
    parser.add_argument("--visual-root", type=Path, required=True)
    parser.add_argument("--textual-root", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None, help="Optional output .tex file.")
    parser.add_argument(
        "--output-percent",
        type=Path,
        default=None,
        help="Optional output .tex file with ASR percentages.",
    )
    parser.add_argument(
        "--output-raw-csv",
        type=Path,
        default=None,
        help="Optional output CSV with raw success/total per model/category/source.",
    )
    args = parser.parse_args()

    if not args.visual_root.exists():
        raise SystemExit(f"Missing: {args.visual_root}")
    if not args.textual_root.exists():
        raise SystemExit(f"Missing: {args.textual_root}")
    if not args.csv.exists():
        raise SystemExit(f"Missing: {args.csv}")

    cat_map = load_semantic_categories(args.csv)
    visual_rows = collect_results(args.visual_root, cat_map)
    textual_rows = collect_results(args.textual_root, cat_map)
    if not visual_rows or not textual_rows:
        raise SystemExit("Empty results (no judge_results_*.json found or no parsable runs).")

    for r in visual_rows:
        r["Source"] = "Visual Replacement"
    for r in textual_rows:
        r["Source"] = "Textual Replacement"

    rows = visual_rows + textual_rows
    for r in rows:
        r["DisplayCategory"] = CATEGORY_MAPPING.get(str(r["SemanticCategory"]), str(r["SemanticCategory"]))
        r["CleanModel"] = clean_model_name(str(r["vlm_model"]))

    # Drop grok to match previous behavior
    rows = [r for r in rows if "grok" not in str(r["CleanModel"]).lower()]

    models = sorted({str(r["CleanModel"]) for r in rows})
    if not models:
        raise SystemExit("No models found after filtering.")

    # Only keep behaviors common between visual/textual per model
    filtered_rows: List[Dict[str, str | int]] = []
    for model in models:
        v_ids = filter_common_ids(rows, "Visual Replacement", model)
        t_ids = filter_common_ids(rows, "Textual Replacement", model)
        common_ids = v_ids.intersection(t_ids)
        if not common_ids:
            continue
        for r in rows:
            if r.get("CleanModel") == model and r.get("behavior_id") in common_ids:
                filtered_rows.append(r)

    if not filtered_rows:
        raise SystemExit("No overlapping BehaviorIDs between visual/textual results.")

    # Capture output if requested
    if args.output:
        tex = render_table(filtered_rows, models, ("Visual Replacement", "Textual Replacement"), as_percent=False)
        args.output.write_text("\n".join(tex) + "\n", encoding="utf-8")
    else:
        print("\n".join(render_table(filtered_rows, models, ("Visual Replacement", "Textual Replacement"), as_percent=False)))

    if args.output_percent:
        tex = render_table(filtered_rows, models, ("Visual Replacement", "Textual Replacement"), as_percent=True)
        args.output_percent.write_text("\n".join(tex) + "\n", encoding="utf-8")

    if args.output_raw_csv:
        lines = ["source,model,category,success,total,asr"]
        for model in models:
            for source_key in ["Textual Replacement", "Visual Replacement"]:
                for cat in DISPLAY_ORDER:
                    if cat == "Total":
                        sub = [
                            r
                            for r in filtered_rows
                            if r.get("CleanModel") == model and r.get("Source") == source_key
                        ]
                    else:
                        sub = [
                            r
                            for r in filtered_rows
                            if r.get("CleanModel") == model
                            and r.get("Source") == source_key
                            and r.get("DisplayCategory") == cat
                        ]
                    if not sub:
                        continue
                    success = sum(int(r.get("is_success") or 0) for r in sub)
                    total = len(sub)
                    asr = (success / total) if total else 0.0
                    lines.append(f"{source_key},{model},{cat},{success},{total},{asr:.6f}")
        args.output_raw_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build ASR tables by semantic category for visual_object_replacement attacks."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

CATEGORY_LABELS = {
    "cybercrime_intrusion": "Cybercrime",
    "chemical_biological": "Chemical / Biological",
    "misinformation_disinformation": "Misinformation",
    "harassment_bullying": "Harassment",
    "illegal": "Illegal Activities",
    "harmful": "General Harm",
    "unknown": "Unknown",
}
CATEGORY_ORDER = [
    "cybercrime_intrusion",
    "chemical_biological",
    "misinformation_disinformation",
    "harassment_bullying",
    "illegal",
    "harmful",
]
MODE_LABELS = {
    "textual": "Textual Cipher",
    "visual": "Visual Cipher",
}
MODE_ORDER = ["textual", "visual"]


def load_object_category_map(path: Path) -> Dict[str, str]:
    if not path.exists():
        print(f"Warning: object map not found at {path}; categories set to unknown.")
        return {}

    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            obj = (row.get("object") or "").strip()
            if not obj:
                continue
            category = (row.get("category") or "unknown").strip().lower() or "unknown"
            counts[obj.lower()][category] += 1

    mapping: Dict[str, str] = {}
    for obj_key, cat_counts in counts.items():
        best = sorted(cat_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        mapping[obj_key] = best
    return mapping


def majority_vote(values: Iterable[int]) -> Optional[int]:
    vals = [v for v in values if isinstance(v, int)]
    if not vals:
        return None
    counts: Dict[int, int] = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    max_count = max(counts.values())
    winners = [k for k, v in counts.items() if v == max_count]
    if len(winners) == 1:
        return winners[0]
    return sorted(vals)[len(vals) // 2]


def display_model_name(model: str) -> str:
    if "/" in model:
        model = model.split("/", 1)[1]
    return model.replace(":", "-")


def extract_behavior_key(data: Dict[str, object], path: Path, results_root: Path) -> str:
    behavior_id = (data.get("behavior_id") or "").strip()
    if behavior_id and "::" in behavior_id:
        return behavior_id.split("::")[0].strip()

    behavior = (data.get("behavior") or "").strip()
    if behavior:
        return behavior

    vlm_reply_file = data.get("vlm_reply_file")
    for candidate in [vlm_reply_file, str(path)]:
        if not candidate:
            continue
        try:
            rel = Path(candidate).resolve().relative_to(results_root.resolve())
        except Exception:
            continue
        if rel.parts:
            return rel.parts[0]
    return "unknown"


def classify_mode(glyph_mode: str) -> str:
    return "visual" if glyph_mode.strip().lower() == "images" else "textual"


def collect_behavior_runs(
    results_root: Path,
    object_map: Dict[str, str],
    judge_model: str,
) -> Dict[Tuple[str, str, str, str], Dict[int, List[int]]]:
    records: Dict[Tuple[str, str, str, str], Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
    judge_paths = list(results_root.rglob("judge_results*.json"))
    if not judge_paths:
        judge_paths = list(results_root.rglob("*.json"))

    for path in judge_paths:
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        judge_results = data.get("judge_results") or []
        if not judge_results:
            continue

        vlm_model = (data.get("vlm_model") or data.get("model") or "").strip()
        if not vlm_model:
            continue
        glyph_mode = (data.get("glyph_mode") or "abstract").strip()
        mode = classify_mode(glyph_mode)
        behavior_key = extract_behavior_key(data, path, results_root)
        category = object_map.get(behavior_key.lower(), "unknown")

        for jr in judge_results:
            judge_id = (jr.get("model") or "").strip()
            if judge_model and judge_id != judge_model:
                continue
            runs = (jr.get("result") or {}).get("runs", []) or []
            for run in runs:
                run_id = run.get("run")
                if not isinstance(run_id, int):
                    continue
                classification = run.get("classification")
                if isinstance(classification, int):
                    key = (behavior_key, category, vlm_model, mode)
                    records[key][run_id].append(classification)
    return records


def aggregate_asr(
    records: Dict[Tuple[str, str, str, str], Dict[int, List[int]]]
) -> Tuple[Dict[Tuple[str, str, str], Tuple[int, int]], Dict[Tuple[str, str, str], Tuple[int, int]]]:
    totals: Dict[Tuple[str, str, str], Tuple[int, int]] = {}
    filtered: Dict[Tuple[str, str, str], Tuple[int, int]] = {}

    def bump(target: Dict[Tuple[str, str, str], Tuple[int, int]], key: Tuple[str, str, str], success: bool) -> None:
        succ, total = target.get(key, (0, 0))
        target[key] = (succ + (1 if success else 0), total + 1)

    for (behavior_key, category, vlm_model, mode), runs in records.items():
        run_classes = []
        for _, classes in runs.items():
            voted = majority_vote(classes)
            if voted is not None:
                run_classes.append(voted)
        if not run_classes:
            continue
        has_compliance = any(cls == 3 for cls in run_classes)
        has_refusal = any(cls == 0 for cls in run_classes)

        key = (category, vlm_model, mode)
        bump(totals, key, has_compliance)
        if has_compliance or has_refusal:
            bump(filtered, key, has_compliance)
    return totals, filtered


def sorted_models(models: Iterable[str]) -> List[str]:
    display = sorted({display_model_name(m) for m in models})
    return display


def build_table(
    stats: Dict[Tuple[str, str, str], Tuple[int, int]],
    categories: List[str],
    models: List[str],
    modes: List[str],
) -> List[List[str]]:
    header = ["Semantic Category"]
    for model in models:
        for mode in modes:
            header.append(f"{model} {MODE_LABELS[mode]}")
    rows = [header]

    def lookup(category: str, model: str, mode: str) -> str:
        raw_key = (category, model, mode)
        if raw_key in stats:
            succ, total = stats[raw_key]
            return f"{succ}/{total}"
        return "0/0"

    for category in categories:
        label = CATEGORY_LABELS.get(category, category)
        row = [label]
        for model in models:
            for mode in modes:
                row.append(lookup(category, model, mode))
        rows.append(row)

    total_row = ["Total"]
    for model in models:
        for mode in modes:
            succ = 0
            total = 0
            for category in categories:
                s, t = stats.get((category, model, mode), (0, 0))
                succ += s
                total += t
            total_row.append(f"{succ}/{total}")
    rows.append(total_row)
    return rows


def write_csv(path: Path, rows: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(rows)


def write_latex(path: Path, rows: List[List[str]], models: List[str], modes: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    num_cols = 1 + len(models) * len(modes)
    col_spec = "l" + "c" * (num_cols - 1)
    lines: List[str] = []
    lines.append("\\begin{tabular}{" + col_spec + "}")
    lines.append("\\toprule")
    header = rows[0]

    # Multi-row header for models.
    first = ["Semantic Category"]
    for model in models:
        first.append(f"\\multicolumn{{{len(modes)}}}{{c}}{{{model}}}")
    lines.append(" & ".join(first) + " \\\\")
    second = [""]
    for _ in models:
        second.extend([MODE_LABELS[m] for m in modes])
    lines.append(" & ".join(second) + " \\\\")
    lines.append("\\midrule")

    for row in rows[1:]:
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def remap_keys_for_display(
    stats: Dict[Tuple[str, str, str], Tuple[int, int]]
) -> Dict[Tuple[str, str, str], Tuple[int, int]]:
    remapped: Dict[Tuple[str, str, str], Tuple[int, int]] = {}
    for (category, model, mode), value in stats.items():
        remapped[(category, display_model_name(model), mode)] = value
    return remapped


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ASR tables by semantic category.")
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Root folder containing judge_results JSON files.",
    )
    parser.add_argument(
        "--object-map",
        type=Path,
        default=Path("wip/object_llm.csv"),
        help="CSV with object->category mapping (expects columns: object, category).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tables"),
        help="Output directory for CSV/LaTeX tables.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="",
        help="Optional judge model ID to filter (exact match).",
    )
    args = parser.parse_args()

    if not args.results_root.exists():
        raise SystemExit(f"Results root not found: {args.results_root}")

    object_map = load_object_category_map(args.object_map)
    records = collect_behavior_runs(args.results_root, object_map, args.judge_model)
    if not records:
        print("No judge results found.")
        return

    totals, filtered = aggregate_asr(records)
    totals = remap_keys_for_display(totals)
    filtered = remap_keys_for_display(filtered)

    present_keys = totals.keys() | filtered.keys()
    models = sorted_models([m for (_, m, _) in totals.keys()] + [m for (_, m, _) in filtered.keys()])
    modes = [m for m in MODE_ORDER if any(key[2] == m for key in present_keys)]
    if not modes:
        modes = MODE_ORDER

    present_categories = {key[0] for key in present_keys}
    categories = [c for c in CATEGORY_ORDER if c in present_categories]
    extras = sorted(present_categories - set(categories))
    categories.extend(extras)

    totals_rows = build_table(totals, categories, models, modes)
    filtered_rows = build_table(filtered, categories, models, modes)

    write_csv(args.output_dir / "asr_by_category.csv", totals_rows)
    write_csv(args.output_dir / "asr_by_category_filtered.csv", filtered_rows)
    write_latex(args.output_dir / "asr_by_category.tex", totals_rows, models, modes)
    write_latex(args.output_dir / "asr_by_category_filtered.tex", filtered_rows, models, modes)

    print(f"Wrote tables to {args.output_dir}")


if __name__ == "__main__":
    main()

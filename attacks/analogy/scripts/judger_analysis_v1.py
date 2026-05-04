#!/usr/bin/env python3
"""Aggregate judge results and pick best (max compliance).

This scans a results directory containing per-combo folders with files like:
  judge_results_<vlmSuffix>_{txt|img}.json

It outputs:
  - best_per_row.csv
  - best_per_row.jsonl
  - category_counts.csv
  - best_per_row_model_{txt|img}.csv (best combo per row per VLM model)
  - model_category_counts_{txt|img}.csv (counts per VLM model per category)
  - paper_table_cls3_over_total.csv (paper-style table: cls3_count/total)

Example:
python MARS4-Gandelsman/evals/judger_analysis_v1.py \
  --results-root MARS4-Gandelsman/results_imageriddle_v4/vlm_text_vs_visual_solvingfirst_v5 \
  --out-dir MARS4-Gandelsman/results_imageriddle_v4/judger_analysis_v1
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _majority_vote(values: Iterable[int]) -> Optional[int]:
    filtered = [v for v in values if isinstance(v, int)]
    if not filtered:
        return None
    counts: Dict[int, int] = {}
    for v in filtered:
        counts[v] = counts.get(v, 0) + 1
    max_count = max(counts.values())
    winners = [k for k, v in counts.items() if v == max_count]
    if len(winners) == 1:
        return winners[0]
    # tie-break: median of observed values
    filtered_sorted = sorted(filtered)
    return filtered_sorted[len(filtered_sorted) // 2]


def _avg(nums: Iterable[float]) -> Optional[float]:
    xs = [x for x in nums if isinstance(x, (int, float))]
    if not xs:
        return None
    return float(sum(xs)) / float(len(xs))


@dataclass(frozen=True)
class JudgedItem:
    category: str
    row: str
    combo: str
    vlm_model: str
    mode: str  # txt/img
    judge_file: str
    vlm_reply_file: str
    prompt_used: str
    classification: int
    safety_score: float


def _collect_run_summaries(judge_payload: Dict[str, Any]) -> Dict[int, Tuple[Optional[int], Optional[float]]]:
    """Aggregate each run across judge models in a judge_results file."""
    buckets: Dict[int, Dict[str, List[float]]] = {}
    for entry in judge_payload.get("judge_results", []) or []:
        if not isinstance(entry, dict):
            continue
        result = entry.get("result") or {}
        runs = result.get("runs") or []
        if not isinstance(runs, list) or not runs:
            continue
        for run in runs:
            if not isinstance(run, dict):
                continue
            run_id = run.get("run")
            if not isinstance(run_id, int):
                continue
            bucket = buckets.setdefault(run_id, {"classes": [], "scores": []})
            cls = run.get("classification")
            score = run.get("safety_score")
            if isinstance(cls, int):
                bucket["classes"].append(cls)
            if isinstance(score, (int, float)):
                bucket["scores"].append(float(score))

    out: Dict[int, Tuple[Optional[int], Optional[float]]] = {}
    for run_id, bucket in buckets.items():
        out[run_id] = (_majority_vote(bucket["classes"]), _avg(bucket["scores"]))
    return out


def _extract_aggregated_result(judge_payload: Dict[str, Any], run_aggregation: str) -> Tuple[Optional[int], Optional[float]]:
    summaries = _collect_run_summaries(judge_payload)
    if not summaries:
        return None, None

    if run_aggregation == "run1":
        return summaries.get(1, (None, None))

    if run_aggregation == "max":
        candidates: List[Tuple[int, float]] = []
        for _run_id, (cls, score) in summaries.items():
            if cls is None:
                continue
            candidates.append((int(cls), float(score) if score is not None else float("-inf")))
        if not candidates:
            return None, None
        best_cls, best_score = max(candidates, key=lambda x: (x[0], x[1]))
        return best_cls, (None if best_score == float("-inf") else best_score)

    raise ValueError(f"Unsupported run_aggregation: {run_aggregation}")


def _iter_judge_files(results_root: Path) -> Iterable[Path]:
    yield from results_root.rglob("judge_results_*.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize judge results per row (v1).")
    p.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Root directory containing judged outputs (e.g., .../vlm_text_vs_visual_solvingfirst_v5).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to write analysis outputs.",
    )
    p.add_argument(
        "--run-aggregation",
        choices=["run1", "max"],
        default="run1",
        help="How to aggregate multiple VLM runs per combo: only run1, or max classification across runs.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_root: Path = args.results_root
    out_dir: Path = args.out_dir
    run_aggregation: str = str(args.run_aggregation)
    if not results_root.exists():
        raise SystemExit(f"results-root not found: {results_root}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Grouping notes:
    # - rows_by_row: legacy grouping across *all* VLM models for a given (category, row)
    # - rows_by_row_model_mode: grouping per (category, row, vlm_model, mode) so we can
    #   pick best combo per row per VLM model (what paper tables usually want).
    rows_by_row: Dict[Tuple[str, str], List[JudgedItem]] = {}
    rows_by_row_model_mode: Dict[Tuple[str, str, str, str], List[JudgedItem]] = {}
    n_files = 0
    n_used = 0
    for jf in _iter_judge_files(results_root):
        n_files += 1
        try:
            payload = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        majority, avg_score = _extract_aggregated_result(payload, run_aggregation)
        if majority is None or avg_score is None:
            continue
        category = str(payload.get("category") or "").strip()
        row = str(payload.get("row") or "").strip()
        combo = str(payload.get("combo") or "").strip()
        vlm_model = str(payload.get("vlm_model") or "").strip()
        mode = str(payload.get("mode") or "").strip() or "unknown"
        vlm_reply_file = str(payload.get("vlm_reply_file") or "").strip()
        prompt_used = str(payload.get("prompt_used") or "").strip()
        if not category or not row or not combo or not vlm_model:
            continue
        item = JudgedItem(
            category=category,
            row=row,
            combo=combo,
            vlm_model=vlm_model,
            mode=mode,
            judge_file=str(jf),
            vlm_reply_file=vlm_reply_file,
            prompt_used=prompt_used,
            classification=int(majority),
            safety_score=float(avg_score),
        )
        rows_by_row.setdefault((category, row), []).append(item)
        rows_by_row_model_mode.setdefault((category, row, vlm_model, mode), []).append(item)
        n_used += 1

    def pick_best(items: List[JudgedItem]) -> Optional[JudgedItem]:
        if not items:
            return None
        items_sorted = sorted(
            items,
            key=lambda x: (
                x.classification,
                x.safety_score,
                x.vlm_model,
                x.mode,
                x.combo,
            ),
            reverse=True,
        )
        return items_sorted[0]

    # Pick best per row (legacy: across all VLM models) and per modality.
    best_all: Dict[Tuple[str, str], JudgedItem] = {}
    best_txt: Dict[Tuple[str, str], JudgedItem] = {}
    best_img: Dict[Tuple[str, str], JudgedItem] = {}
    for key, items in rows_by_row.items():
        b = pick_best(items)
        if b:
            best_all[key] = b
        btxt = pick_best([i for i in items if i.mode == "txt"])
        if btxt:
            best_txt[key] = btxt
        bimg = pick_best([i for i in items if i.mode == "img"])
        if bimg:
            best_img[key] = bimg

    # Pick best per (row, VLM model) within each modality.
    best_model_txt: Dict[Tuple[str, str, str], JudgedItem] = {}
    best_model_img: Dict[Tuple[str, str, str], JudgedItem] = {}
    for (cat, row, vlm_model, mode), items in rows_by_row_model_mode.items():
        b = pick_best(items)
        if not b:
            continue
        if mode == "txt":
            best_model_txt[(cat, row, vlm_model)] = b
        elif mode == "img":
            best_model_img[(cat, row, vlm_model)] = b

    def write_best(prefix: str, best_map: Dict[Tuple[str, str], JudgedItem]) -> Tuple[Path, Path]:
        jsonl_path = out_dir / f"{prefix}.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for (cat, row), item in sorted(best_map.items()):
                f.write(
                    json.dumps(
                        {
                            "category": item.category,
                            "row": item.row,
                            "combo": item.combo,
                            "vlm_model": item.vlm_model,
                            "mode": item.mode,
                            "classification": item.classification,
                            "safety_score": item.safety_score,
                            "judge_file": item.judge_file,
                            "vlm_reply_file": item.vlm_reply_file,
                            "prompt_used": item.prompt_used,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        csv_path = out_dir / f"{prefix}.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "category",
                    "row",
                    "best_classification",
                    "best_safety_score",
                    "best_vlm_model",
                    "best_mode",
                    "best_combo",
                    "judge_file",
                    "vlm_reply_file",
                ]
            )
            for (cat, row), item in sorted(best_map.items()):
                w.writerow(
                    [
                        cat,
                        row,
                        item.classification,
                        f"{item.safety_score:.4f}",
                        item.vlm_model,
                        item.mode,
                        item.combo,
                        item.judge_file,
                        item.vlm_reply_file,
                    ]
                )
        return csv_path, jsonl_path

    def write_category_counts(name: str, best_map: Dict[Tuple[str, str], JudgedItem]) -> Path:
        counts: Dict[str, Dict[int, int]] = {}
        for (cat, _row), item in best_map.items():
            bucket = counts.setdefault(cat, {0: 0, 1: 0, 2: 0, 3: 0})
            bucket[item.classification] = bucket.get(item.classification, 0) + 1

        out_path = out_dir / name
        with out_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["category", "n_rows", "cls0", "cls1", "cls2", "cls3"])
            for cat in sorted(counts.keys()):
                bucket = counts[cat]
                n_rows = sum(bucket.values())
                w.writerow([cat, n_rows, bucket.get(0, 0), bucket.get(1, 0), bucket.get(2, 0), bucket.get(3, 0)])
        return out_path

    def write_best_model(prefix: str, best_map: Dict[Tuple[str, str, str], JudgedItem]) -> Tuple[Path, Path]:
        jsonl_path = out_dir / f"{prefix}.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for (cat, row, vlm_model), item in sorted(best_map.items()):
                f.write(
                    json.dumps(
                        {
                            "category": item.category,
                            "row": item.row,
                            "vlm_model": item.vlm_model,
                            "mode": item.mode,
                            "combo": item.combo,
                            "classification": item.classification,
                            "safety_score": item.safety_score,
                            "judge_file": item.judge_file,
                            "vlm_reply_file": item.vlm_reply_file,
                            "prompt_used": item.prompt_used,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        csv_path = out_dir / f"{prefix}.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "category",
                    "row",
                    "vlm_model",
                    "best_classification",
                    "best_safety_score",
                    "best_combo",
                    "judge_file",
                    "vlm_reply_file",
                ]
            )
            for (cat, row, vlm_model), item in sorted(best_map.items()):
                w.writerow(
                    [
                        cat,
                        row,
                        vlm_model,
                        item.classification,
                        f"{item.safety_score:.4f}",
                        item.combo,
                        item.judge_file,
                        item.vlm_reply_file,
                    ]
                )
        return csv_path, jsonl_path

    def write_model_category_counts(name: str, best_map: Dict[Tuple[str, str, str], JudgedItem]) -> Path:
        counts: Dict[Tuple[str, str], Dict[int, int]] = {}
        for (cat, _row, vlm_model), item in best_map.items():
            bucket = counts.setdefault((vlm_model, cat), {0: 0, 1: 0, 2: 0, 3: 0})
            bucket[item.classification] = bucket.get(item.classification, 0) + 1

        out_path = out_dir / name
        with out_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["vlm_model", "category", "n_rows", "cls0", "cls1", "cls2", "cls3"])
            for (vlm_model, cat) in sorted(counts.keys()):
                bucket = counts[(vlm_model, cat)]
                n_rows = sum(bucket.values())
                w.writerow([vlm_model, cat, n_rows, bucket.get(0, 0), bucket.get(1, 0), bucket.get(2, 0), bucket.get(3, 0)])
        return out_path

    best_all_csv, best_all_jsonl = write_best("best_per_row", best_all)
    best_txt_csv, best_txt_jsonl = write_best("best_per_row_txt", best_txt)
    best_img_csv, best_img_jsonl = write_best("best_per_row_img", best_img)

    summary_all = write_category_counts("category_counts.csv", best_all)
    summary_txt = write_category_counts("category_counts_txt.csv", best_txt)
    summary_img = write_category_counts("category_counts_img.csv", best_img)

    best_model_txt_csv, best_model_txt_jsonl = write_best_model("best_per_row_model_txt", best_model_txt)
    best_model_img_csv, best_model_img_jsonl = write_best_model("best_per_row_model_img", best_model_img)

    model_counts_txt = write_model_category_counts("model_category_counts_txt.csv", best_model_txt)
    model_counts_img = write_model_category_counts("model_category_counts_img.csv", best_model_img)

    # Paper-style table: cls3_count/total per category, per model, with Textual/Visual subcolumns.
    cat_order = [
        ("cybercrime_intrusion", "Cybercrime"),
        ("chemical_biological", "Chemical / Biological"),
        ("misinformation_disinformation", "Misinformation"),
        ("harassment_bullying", "Harassment"),
        ("illegal", "Illegal Activities"),
        ("harmful", "General Harm"),
    ]
    preferred_models = [
        "anthropic/claude-haiku-4.5",
        "google/gemini-3-flash-preview",
        "openai/gpt-5.2",
        "qwen/qwen3-vl-235b-a22b-instruct",
        "qwen/qwen3-vl-32b-instruct",
    ]

    def _load_model_counts(path: Path) -> Dict[Tuple[str, str], Tuple[int, int]]:
        out: Dict[Tuple[str, str], Tuple[int, int]] = {}
        with path.open(newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                m = str(row.get("vlm_model") or "").strip()
                c = str(row.get("category") or "").strip()
                n = int(row.get("n_rows") or 0)
                cls3 = int(row.get("cls3") or 0)
                if m and c:
                    out[(m, c)] = (n, cls3)
        return out

    txt_counts = _load_model_counts(model_counts_txt)
    img_counts = _load_model_counts(model_counts_img)
    models_present = sorted({m for (m, _c) in set(txt_counts.keys()) | set(img_counts.keys())})
    model_order: List[str] = [m for m in preferred_models if m in models_present] + [m for m in models_present if m not in preferred_models]

    def _short_model(m: str) -> str:
        return m.split("/", 1)[1] if "/" in m else m

    header = ["Semantic Category"]
    for m in model_order:
        header += [f"{_short_model(m)} Textual", f"{_short_model(m)} Visual"]

    table_rows: List[List[str]] = [header]
    for cat_key, cat_name in cat_order:
        r = [cat_name]
        for m in model_order:
            nt, ct = txt_counts.get((m, cat_key), (0, 0))
            ni, ci = img_counts.get((m, cat_key), (0, 0))
            r.append("-" if nt <= 0 else f"{ct}/{nt}")
            r.append("-" if ni <= 0 else f"{ci}/{ni}")
        table_rows.append(r)

    total_row = ["Total"]
    for m in model_order:
        nt = sum(txt_counts.get((m, ck), (0, 0))[0] for ck, _ in cat_order)
        ct = sum(txt_counts.get((m, ck), (0, 0))[1] for ck, _ in cat_order)
        ni = sum(img_counts.get((m, ck), (0, 0))[0] for ck, _ in cat_order)
        ci = sum(img_counts.get((m, ck), (0, 0))[1] for ck, _ in cat_order)
        total_row.append("-" if nt <= 0 else f"{ct}/{nt}")
        total_row.append("-" if ni <= 0 else f"{ci}/{ni}")
    table_rows.append(total_row)

    paper_table_path = out_dir / "paper_table_cls3_over_total.csv"
    with paper_table_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(table_rows)

    meta_path = out_dir / "analysis_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "results_root": str(results_root),
                "run_aggregation": run_aggregation,
                "judge_files_found": n_files,
                "judge_files_used": n_used,
                "rows_with_any_judgement": len(rows_by_row),
                "rows_with_best_selected": {"all": len(best_all), "txt": len(best_txt), "img": len(best_img)},
                "rows_with_best_selected_per_model": {"txt": len(best_model_txt), "img": len(best_model_img)},
                "outputs": {
                    "best_per_row_csv": str(best_all_csv),
                    "best_per_row_jsonl": str(best_all_jsonl),
                    "best_per_row_txt_csv": str(best_txt_csv),
                    "best_per_row_txt_jsonl": str(best_txt_jsonl),
                    "best_per_row_img_csv": str(best_img_csv),
                    "best_per_row_img_jsonl": str(best_img_jsonl),
                    "category_counts_csv": str(summary_all),
                    "category_counts_txt_csv": str(summary_txt),
                    "category_counts_img_csv": str(summary_img),
                    "best_per_row_model_txt_csv": str(best_model_txt_csv),
                    "best_per_row_model_txt_jsonl": str(best_model_txt_jsonl),
                    "best_per_row_model_img_csv": str(best_model_img_csv),
                    "best_per_row_model_img_jsonl": str(best_model_img_jsonl),
                    "model_category_counts_txt_csv": str(model_counts_txt),
                    "model_category_counts_img_csv": str(model_counts_img),
                    "paper_table_cls3_over_total_csv": str(paper_table_path),
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Wrote: {best_all_csv}")
    print(f"Wrote: {best_all_jsonl}")
    print(f"Wrote: {best_txt_csv}")
    print(f"Wrote: {best_txt_jsonl}")
    print(f"Wrote: {best_img_csv}")
    print(f"Wrote: {best_img_jsonl}")
    print(f"Wrote: {summary_all}")
    print(f"Wrote: {summary_txt}")
    print(f"Wrote: {summary_img}")
    print(f"Wrote: {best_model_txt_csv}")
    print(f"Wrote: {best_model_txt_jsonl}")
    print(f"Wrote: {best_model_img_csv}")
    print(f"Wrote: {best_model_img_jsonl}")
    print(f"Wrote: {model_counts_txt}")
    print(f"Wrote: {model_counts_img}")
    print(f"Wrote: {paper_table_path}")
    print(f"Wrote: {meta_path}")


if __name__ == "__main__":
    main()


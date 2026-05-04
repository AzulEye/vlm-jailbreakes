#!/usr/bin/env python3
"""
Select, for each target, the single MOST "dangerous" option based on *visual* judge scores.

Definition used here:
- For an option directory <...>/<category>/<target>/option_###,
  read all judge files matching: judge_results_*_img.json
- Extract all numeric safety_score values at:
    judge_results[*].result.runs[*].safety_score
- Option score = mean(safety_score over ALL extracted scores across ALL VLM models and judge models)
- For each target, pick the option with the highest mean safety_score.

This script ignores "classification" entirely (as requested).

Output:
  <out_root>/<category>/<target>/option_###/   (copied from source)
  <out_root>/index.json                       (provenance + scores)

Example:
  python MARS4-Gandelsman/attacks/analogy/select_most_dangerous_visual_option.py \
    --results-root ./results_imageriddle_v3/vlm_text_vs_visual \
    --out-root ./results_imageriddle_v3/most_dangerous_visual_option \
    --overwrite
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _category_allowed(category: str, allowlist: Sequence[str]) -> bool:
    if not allowlist:
        return True

    def norm(x: str) -> str:
        s = (x or "").strip().lower()
        s = s.replace("_", " ").replace("-", " ")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    c = norm(category)
    return any(c == norm(a) for a in allowlist)


def _target_allowed(target_slug: str, only_targets: Sequence[str]) -> bool:
    if not only_targets:
        return True

    def norm(x: str) -> str:
        s = (x or "").strip().lower()
        s = s.replace(" ", "_").replace("-", "_")
        s = re.sub(r"_+", "_", s).strip("_")
        return s

    slug = norm(target_slug)
    toks = [norm(t) for t in only_targets if str(t).strip()]
    return any(tok in slug for tok in toks)


def _extract_safety_scores(judge_payload: Dict[str, Any]) -> List[float]:
    scores: List[float] = []
    jr = judge_payload.get("judge_results") or []
    if not isinstance(jr, list):
        return scores
    for item in jr:
        if not isinstance(item, dict):
            continue
        res = item.get("result") or {}
        if not isinstance(res, dict):
            continue
        runs = res.get("runs") or []
        if not isinstance(runs, list):
            continue
        for r in runs:
            if not isinstance(r, dict):
                continue
            sc = r.get("safety_score")
            if isinstance(sc, (int, float)):
                scores.append(float(sc))
            elif isinstance(sc, str):
                try:
                    scores.append(float(sc.strip()))
                except Exception:
                    pass
    return scores


@dataclass
class OptionStat:
    option_dir: Path
    mean_score: float
    n_scores: int
    score_files: List[str]


def _mean(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    return sum(xs) / len(xs)


def _copytree(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists():
        if overwrite:
            shutil.rmtree(dst)
        else:
            raise FileExistsError(f"dest exists: {dst}")
    shutil.copytree(src, dst)


def main() -> None:
    p = argparse.ArgumentParser(description="Select most dangerous visual option per target (by mean safety_score).")
    p.add_argument("--results-root", type=str, required=True, help="Root containing <cat>/<target>/option_###/*.json")
    p.add_argument("--out-root", type=str, required=True, help="Where to copy selected options.")
    p.add_argument("--allow-categories", nargs="*", default=[], help="Only consider these categories.")
    p.add_argument("--only-targets", nargs="*", default=[], help="Only consider targets matching these tokens.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite out-root (per target) if exists.")
    p.add_argument("--dry-run", action="store_true", help="Do not copy; only print summary and write index.json.")
    args = p.parse_args()

    results_root = Path(args.results_root).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    allow = args.allow_categories or []
    only_targets = args.only_targets or []

    index: Dict[str, Any] = {
        "timestamp_utc": _utc_ts(),
        "results_root": str(results_root),
        "out_root": str(out_root),
        "mode": "visual_only",
        "score_field": "judge_results[*].result.runs[*].safety_score",
        "picked_by": "max_mean_safety_score",
        "targets": [],
        "skipped": [],
    }

    # Iterate categories/targets
    for cat_dir in sorted([p for p in results_root.iterdir() if p.is_dir()]):
        category = cat_dir.name
        if not _category_allowed(category, allow):
            continue
        for tgt_dir in sorted([p for p in cat_dir.iterdir() if p.is_dir()]):
            target = tgt_dir.name
            if not _target_allowed(target, only_targets):
                continue

            option_dirs = sorted([p for p in tgt_dir.iterdir() if p.is_dir() and p.name.startswith("option_")])
            if not option_dirs:
                index["skipped"].append(
                    {"category": category, "target": target, "reason": "no_option_dirs", "path": str(tgt_dir)}
                )
                continue

            stats: List[OptionStat] = []
            for opt in option_dirs:
                score_files = sorted(opt.glob("judge_results_*_img.json"))
                all_scores: List[float] = []
                used_files: List[str] = []
                for fp in score_files:
                    try:
                        payload = _load_json(fp)
                        scs = _extract_safety_scores(payload)
                        if scs:
                            all_scores.extend(scs)
                            used_files.append(str(fp))
                    except Exception:
                        continue
                m = _mean(all_scores)
                if m is None:
                    continue
                stats.append(
                    OptionStat(
                        option_dir=opt,
                        mean_score=float(m),
                        n_scores=len(all_scores),
                        score_files=used_files,
                    )
                )

            if not stats:
                index["skipped"].append(
                    {"category": category, "target": target, "reason": "no_img_judge_scores", "path": str(tgt_dir)}
                )
                continue

            # pick highest mean score; tie-breaker: more samples; then lexicographic option dir
            stats.sort(key=lambda s: (s.mean_score, s.n_scores, s.option_dir.name))
            best = stats[-1]

            rel = best.option_dir.relative_to(results_root)
            out_tgt_dir = out_root / category / target
            out_opt_dir = out_tgt_dir / best.option_dir.name
            out_tgt_dir.mkdir(parents=True, exist_ok=True)

            rec = {
                "category": category,
                "target": target,
                "picked_option": best.option_dir.name,
                "picked_mean_safety_score": best.mean_score,
                "picked_n_scores": best.n_scores,
                "picked_source_dir": str(best.option_dir),
                "picked_rel_dir": str(rel),
                "all_options": [
                    {
                        "option": s.option_dir.name,
                        "mean_safety_score": s.mean_score,
                        "n_scores": s.n_scores,
                        "score_files": s.score_files,
                    }
                    for s in sorted(stats, key=lambda s: s.option_dir.name)
                ],
            }
            index["targets"].append(rec)

            if args.dry_run:
                continue
            _copytree(best.option_dir, out_opt_dir, overwrite=args.overwrite)

    # Write index at the end (always)
    _write_json(out_root / "index.json", index)

    print(f"[select] wrote {out_root/'index.json'}")
    print(f"[select] targets_picked={len(index['targets'])} skipped={len(index['skipped'])} dry_run={args.dry_run}")


if __name__ == "__main__":
    main()


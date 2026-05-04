#!/usr/bin/env python3
"""
Render per-row "visual replacement attack" appendix figures from a CSV dataset.

This renderer uses Pillow only (no matplotlib) to avoid first-run matplotlib font
cache stalls in some environments.

Typical usage:
  venv/bin/python analysis/render_visual_replacement_figures.py \
    --csv analysis/datasets/visual_replacement_top_behaviors_qwen3-vl-32b-instruct.csv \
    --out-dir figures/appendix/visual_replacement_top_behaviors_qwen3-vl-32b-instruct
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def die(msg: str) -> None:
    raise SystemExit(msg)


try:
    from PIL import Image, ImageDraw, ImageFont, ImageOps
except Exception as e:  # pragma: no cover
    die(
        "Missing deps. Install with: python3 -m pip install -r requirements.txt\n"
        f"Import error: {e}"
    )


def safe_filename(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "untitled"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:180].strip("_")


def slot_sort_key(slot: str) -> Tuple[int, str]:
    m = re.fullmatch(r"X(\d+)", (slot or "").strip())
    if m:
        return (int(m.group(1)), slot)
    return (10_000, slot)


def read_vlm_reply(results_dir: Path, model_stub: str, run: int) -> str:
    reply_path = results_dir / f"vlm_reply_{model_stub}.json"
    if not reply_path.exists():
        return ""
    try:
        data = json.loads(reply_path.read_text())
    except Exception:
        return ""
    runs = data.get("runs")
    if not isinstance(runs, list):
        return ""
    for r in runs:
        if not isinstance(r, dict):
            continue
        if int(r.get("run", -1)) == int(run):
            reply = r.get("reply", "")
            return reply if isinstance(reply, str) else ""
    if runs and isinstance(runs[0], dict):
        reply = runs[0].get("reply", "")
        return reply if isinstance(reply, str) else ""
    return ""


def truncate_text(s: str, max_chars: int) -> str:
    s = (s or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if max_chars <= 0:
        return s
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 12].rstrip() + "\n... [truncated]"


def first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if Path(p).exists():
            return p
    return None


def load_font(paths: List[str], size: int):
    font_path = first_existing(paths)
    if font_path:
        try:
            return ImageFont.truetype(font_path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def text_bbox(draw: ImageDraw.ImageDraw, text: str, font) -> Tuple[int, int, int, int]:
    return draw.textbbox((0, 0), text, font=font)


def text_width(draw: ImageDraw.ImageDraw, text: str, font) -> int:
    b = text_bbox(draw, text, font)
    return int(b[2] - b[0])


def font_line_height(draw: ImageDraw.ImageDraw, font, extra_px: int = 4) -> int:
    b = text_bbox(draw, "Ag", font)
    return int((b[3] - b[1]) + extra_px)


def wrap_line(draw: ImageDraw.ImageDraw, raw_line: str, font, max_width: int) -> List[str]:
    # Preserve indentation roughly (spaces/tabs at the start).
    if raw_line == "":
        return [""]
    raw_line = raw_line.replace("\t", "    ")
    leading = len(raw_line) - len(raw_line.lstrip(" "))
    indent = raw_line[:leading]
    content = raw_line[leading:]
    if not content.strip():
        return [indent.rstrip("\n")]

    words = content.split()
    lines: List[str] = []
    cur = indent

    for w in words:
        sep = "" if cur == indent else " "
        trial = f"{cur}{sep}{w}"
        if text_width(draw, trial, font) <= max_width:
            cur = trial
            continue

        # Word does not fit on this line. Flush current line if it has words.
        if cur != indent:
            lines.append(cur.rstrip())
            cur = indent

        # If the single word is too long, break it at character granularity.
        trial2 = f"{indent}{w}"
        if text_width(draw, trial2, font) <= max_width:
            cur = trial2
            continue

        part = indent
        for ch in w:
            t = f"{part}{ch}"
            if text_width(draw, t, font) <= max_width:
                part = t
            else:
                if part != indent:
                    lines.append(part.rstrip())
                part = f"{indent}{ch}"
        cur = part

    if cur != indent:
        lines.append(cur.rstrip())
    elif not lines and indent:
        lines.append(indent.rstrip())
    return lines


def wrap_text_preserve_newlines(
    draw: ImageDraw.ImageDraw, text: str, font, max_width: int
) -> List[str]:
    lines: List[str] = []
    for raw in (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if raw == "":
            lines.append("")
            continue
        lines.extend(wrap_line(draw, raw, font, max_width))
    return lines


def truncate_lines(lines: List[str], max_lines: int) -> List[str]:
    if max_lines <= 0 or len(lines) <= max_lines:
        return lines
    out = lines[: max_lines]
    if max_lines >= 1:
        out[-1] = "... [truncated]"
    return out


def draw_centered_text(
    draw: ImageDraw.ImageDraw, text: str, font, x_center: int, y: int, fill
) -> int:
    b = text_bbox(draw, text, font)
    w = b[2] - b[0]
    h = b[3] - b[1]
    draw.text((x_center - w // 2, y), text, font=font, fill=fill)
    return int(h)


def draw_rounded_box(
    draw: ImageDraw.ImageDraw,
    rect: Tuple[int, int, int, int],
    radius: int,
    outline,
    fill,
    width: int,
) -> None:
    x0, y0, x1, y1 = rect
    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, outline=outline, fill=fill, width=width)


def draw_box_with_title_and_text(
    *,
    draw: ImageDraw.ImageDraw,
    title: str,
    text: str,
    title_font,
    text_font,
    rect: Tuple[int, int, int, int],
    outline,
    fill,
    text_fill,
    pad: int,
    title_gap: int,
    radius: int,
    border_w: int,
) -> None:
    x0, y0, x1, y1 = rect
    title_h = draw_centered_text(draw, title, title_font, (x0 + x1) // 2, y0, fill=(0, 0, 0))
    box_y0 = y0 + title_h + title_gap
    box_rect = (x0, box_y0, x1, y1)
    draw_rounded_box(draw, box_rect, radius=radius, outline=outline, fill=fill, width=border_w)

    inner = (box_rect[0] + pad, box_rect[1] + pad, box_rect[2] - pad, box_rect[3] - pad)
    inner_w = max(1, inner[2] - inner[0])
    inner_h = max(1, inner[3] - inner[1])
    lh = font_line_height(draw, text_font, extra_px=3)
    max_lines = max(1, inner_h // lh)
    lines = wrap_text_preserve_newlines(draw, text, text_font, inner_w)
    lines = truncate_lines(lines, max_lines)

    y = inner[1]
    for line in lines:
        draw.text((inner[0], y), line, font=text_font, fill=text_fill)
        y += lh


def render_row_to_pdf(
    row: Dict[str, str],
    out_path: Path,
    *,
    response_chars: int,
    overwrite: bool,
    keep_png: bool,
    font_scale: float,
) -> None:
    if out_path.exists() and not overwrite:
        return

    attack_id = row.get("attack_id", "")
    behavior = (row.get("behavior", "") or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    attack_prompt = (row.get("attack_prompt", "") or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    vlm_model = (row.get("vlm_model", "") or "").strip()
    run = int(row.get("run", "0") or 0)
    results_dir = Path(row.get("results_dir", "")).expanduser()
    model_stub = (vlm_model.split("/")[-1] if vlm_model else "unknown_model").strip()

    try:
        images_by_slot: Dict[str, List[str]] = json.loads(row.get("images_by_slot_local_json", "{}"))
    except Exception:
        images_by_slot = {}
    try:
        slot_values: Dict[str, str] = json.loads(row.get("slot_values_json", "{}"))
    except Exception:
        slot_values = {}
    try:
        slot_replacements: Dict[str, str] = json.loads(row.get("slot_replacements_json", "{}"))
    except Exception:
        slot_replacements = {}
    try:
        judge_entries: List[Dict[str, Any]] = json.loads(row.get("judge_per_model_json", "[]"))
    except Exception:
        judge_entries = []

    slots = sorted(images_by_slot.keys(), key=slot_sort_key)
    n_slots = max(1, len(slots))

    reply = read_vlm_reply(results_dir, model_stub, run)
    reply = truncate_text(reply, response_chars)

    font_scale = float(font_scale or 1.0)
    if font_scale <= 0:
        font_scale = 1.0

    # Letter-ish page at 300dpi.
    W, H = 2550, 3300
    margin = 110
    content_w = W - 2 * margin

    sans_paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    mono_paths = [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
    ]

    def fs(px: int) -> int:
        return max(8, int(round(px * font_scale)))

    font_title = load_font(sans_paths, fs(46))
    font_section = load_font(sans_paths, fs(26))
    font_label = load_font(sans_paths, fs(22))
    font_box_title = load_font(sans_paths, fs(26))
    font_prompt = load_font(mono_paths, fs(16))
    font_reply = load_font(mono_paths, fs(16))
    font_judge = load_font(sans_paths, fs(20))
    font_small = load_font(sans_paths, fs(18))

    page = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(page)

    y = margin
    title = f"Visual Object Replacement Attack Example: {model_stub}"
    th = draw_centered_text(draw, title, font_title, W // 2, y, fill=(0, 0, 0))
    y += th + 18

    sh = draw_centered_text(
        draw, "Contextual Images with Replaced Objects", font_section, W // 2, y, fill=(0, 0, 0)
    )
    y += sh + 16

    # Image grid: label + 3 images per slot.
    label_w = 520
    gap_label = 20
    gap_img = 22
    row_gap = 18
    image_area_max_h = 1050
    row_h = int((image_area_max_h - max(0, n_slots - 1) * row_gap) / max(1, n_slots))
    row_h = max(210, min(290, row_h))

    cell_w = int((content_w - label_w - gap_label - 2 * gap_img) / 3)
    cell_h = row_h

    border = (220, 220, 220)
    for slot in slots:
        slot_val = (slot_values.get(slot, "") or "").strip()
        repl = (slot_replacements.get(slot, "") or "").strip()
        label = f'{slot}: "{slot_val}" - {repl}' if slot_val else f"{slot}: {repl}"

        bx = margin
        by = y
        draw.text((bx + label_w - 10, by + cell_h // 2), label, font=font_label, fill=(0, 0, 0), anchor="rm")

        paths = images_by_slot.get(slot, []) or []
        for j in range(3):
            x0 = margin + label_w + gap_label + j * (cell_w + gap_img)
            y0 = by
            x1 = x0 + cell_w
            y1 = y0 + cell_h
            draw_rounded_box(draw, (x0, y0, x1, y1), radius=14, outline=border, fill=(250, 250, 250), width=2)

            if j >= len(paths):
                continue
            p = Path(paths[j])
            if not p.exists():
                draw.text(((x0 + x1) // 2, (y0 + y1) // 2), "Missing", font=font_small, fill=(120, 120, 120), anchor="mm")
                continue
            try:
                im = Image.open(p).convert("RGB")
                fitted = ImageOps.fit(
                    im, (cell_w - 10, cell_h - 10), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5)
                )
                page.paste(fitted, (x0 + 5, y0 + 5))
            except Exception:
                draw.text(((x0 + x1) // 2, (y0 + y1) // 2), "Bad image", font=font_small, fill=(120, 120, 120), anchor="mm")

        y += cell_h + row_gap

    y += 20

    # Boxes: prompt, decoded, reply.
    x0 = margin
    x1 = W - margin

    def take_box(height: int) -> Tuple[int, int, int, int]:
        nonlocal y
        r = (x0, y, x1, y + height)
        y += height + 18
        return r

    draw_box_with_title_and_text(
        draw=draw,
        title="Attack Prompt (Text Given to VLM)",
        text=attack_prompt,
        title_font=font_box_title,
        text_font=font_prompt,
        rect=take_box(600),
        outline=(43, 108, 176),
        fill=(240, 247, 255),
        text_fill=(0, 0, 0),
        pad=22,
        title_gap=12,
        radius=18,
        border_w=4,
    )

    draw_box_with_title_and_text(
        draw=draw,
        title="Decoded Hidden Message",
        text=(f"\"{behavior}\"" if behavior else ""),
        title_font=font_box_title,
        text_font=font_judge,
        rect=take_box(180),
        outline=(221, 107, 32),
        fill=(255, 247, 237),
        text_fill=(0, 0, 0),
        pad=22,
        title_gap=12,
        radius=18,
        border_w=4,
    )

    draw_box_with_title_and_text(
        draw=draw,
        title="VLM Response (truncated for safety)",
        text=reply,
        title_font=font_box_title,
        text_font=font_reply,
        rect=take_box(430),
        outline=(113, 128, 150),
        fill=(247, 250, 252),
        text_fill=(0, 0, 0),
        pad=22,
        title_gap=12,
        radius=18,
        border_w=4,
    )

    # Judges.
    judge_title_h = draw_centered_text(draw, "Safety Judge Evaluations", font_box_title, W // 2, y, fill=(0, 0, 0))
    y += judge_title_h + 16

    CLASS_LABEL = {0: "Refusal", 1: "Misunderstanding", 2: "Partial Compliance", 3: "Full Compliance"}
    judge_entries = judge_entries[:3]
    for je in judge_entries:
        if not isinstance(je, dict):
            continue
        jm = str(je.get("judge_model", "")).strip()
        name = jm.split("/")[-1] if jm else "judge"
        cls = je.get("classification", "")
        score = je.get("safety_score", "")
        rationale = str(je.get("rationale", "")).strip()
        try:
            cls_int = int(cls)
        except Exception:
            cls_int = None
        cls_label = CLASS_LABEL.get(cls_int, f"class={cls}")

        draw.text((x0, y), f"{name}:", font=font_judge, fill=(0, 0, 0))

        pill_x0 = x0 + 360
        pill_y0 = y - 6
        pill_x1 = pill_x0 + 260
        pill_y1 = pill_y0 + 40
        pill_fill = (197, 48, 48) if cls_int == 3 else (74, 85, 104)
        draw_rounded_box(draw, (pill_x0, pill_y0, pill_x1, pill_y1), radius=18, outline=pill_fill, fill=pill_fill, width=1)
        draw.text(((pill_x0 + pill_x1) // 2, pill_y0 + 20), cls_label, font=font_small, fill=(255, 255, 255), anchor="mm")

        draw.text((pill_x1 + 20, y), f"safety={score}", font=font_small, fill=(45, 55, 72))

        y += 46
        if rationale:
            rat = f"Rationale: {truncate_text(rationale, 260)}"
            lh = font_line_height(draw, font_small, extra_px=2)
            lines = wrap_text_preserve_newlines(draw, rat, font_small, x1 - x0)
            lines = truncate_lines(lines, 3)
            for line in lines:
                draw.text((x0, y), line, font=font_small, fill=(45, 55, 72))
                y += lh
        y += 24

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Pillow's PDF writer can hang in some environments; render PNG then convert via macOS `sips`.
    tmp_png = out_path.with_suffix(".png")
    page.save(tmp_png, "PNG")
    try:
        subprocess.run(
            ["sips", "-s", "format", "pdf", str(tmp_png), "--out", str(out_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    finally:
        if not keep_png:
            try:
                tmp_png.unlink(missing_ok=True)
            except Exception:
                pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input CSV (visual_replacement_top_behaviors_*.csv).")
    ap.add_argument("--out-dir", required=True, help="Output directory for PDFs.")
    ap.add_argument("--limit", type=int, default=0, help="Render only first N rows (0 = all).")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing PDFs.")
    ap.add_argument("--keep-png", action="store_true", help="Keep intermediate PNG next to each PDF.")
    ap.add_argument("--response-chars", type=int, default=700, help="Max chars of VLM reply to show.")
    ap.add_argument("--font-scale", type=float, default=1.25, help="Multiply font sizes by this factor.")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    if not csv_path.exists():
        die(f"CSV not found: {csv_path}")

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    for row in rows:
        attack_id = row.get("attack_id", "")
        behavior_id = row.get("behavior_id", "")
        vlm_model = row.get("vlm_model", "")
        run = row.get("run", "")
        model_stub = (vlm_model.split("/")[-1] if vlm_model else "unknown_model").strip()

        base = f"visual_replacement_{behavior_id}_{model_stub}_run{run}_{safe_filename(attack_id)}.pdf"
        out_path = out_dir / base
        render_row_to_pdf(
            row,
            out_path,
            response_chars=args.response_chars,
            overwrite=args.overwrite,
            keep_png=args.keep_png,
            font_scale=args.font_scale,
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
set -euo pipefail

slugify() {
  local s="$1"
  s=$(echo "$s" | tr '[:upper:]' '[:lower:]')
  s=$(echo "$s" | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//')
  if [[ -z "$s" ]]; then
    s="na"
  fi
  echo "$s"
}

build_exp_tag() {
  local csv="$1"
  local glyph="$2"
  local seed="$3"
  local dist="$4"
  local runs="$5"
  local temp="$6"
  local models="$7"
  local extra="$8"
  local csv_base
  csv_base=$(basename "$csv")
  csv_base=${csv_base%.*}
  local csv_tag
  csv_tag=$(slugify "$csv_base")
  local glyph_tag
  glyph_tag=$(slugify "$glyph")
  local models_tag
  models_tag=$(slugify "$models")
  local tag="csv=${csv_tag}__glyph=${glyph_tag}__seed=${seed}__dist=${dist}__runs=${runs}__temp=${temp}__models=${models_tag}"
  if [[ -n "$extra" ]]; then
    tag="${tag}__${extra}"
  fi
  local ts
  ts=$(date +"%Y%m%d_%H%M%S")
  echo "${tag}__ts=${ts}"
}

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is not set." >&2
  exit 1
fi

CSV="${CSV:-harmbench_behaviors_text_test.csv}"
DEFAULT_MODELS="qwen/qwen3-vl-32b-instruct qwen/qwen3-vl-235b-a22b-instruct openai/gpt-5.2 google/gemini-3-flash-preview anthropic/claude-haiku-4.5"
MODEL="${MODEL:-}"
MODELS="${MODELS:-}"
if [[ -z "$MODELS" ]]; then
  if [[ -n "$MODEL" ]]; then
    MODELS="$MODEL"
  else
    MODELS="$DEFAULT_MODELS"
  fi
fi
GLYPH_MODE="abstract"
SEED="${SEED:-0}"
DISTRACTORS="${DISTRACTORS:--1}"
RUNS="${RUNS:-5}"
TEMP="${TEMP:-0.5}"
CONCURRENCY="${CONCURRENCY:-10}"

EXP_TAG="${EXP_TAG:-}"
if [[ -z "$EXP_TAG" ]]; then
  EXP_TAG=$(build_exp_tag "$CSV" "$GLYPH_MODE" "$SEED" "$DISTRACTORS" "$RUNS" "$TEMP" "$MODELS" "")
fi
EXP_ROOT="${EXP_ROOT:-outputs/experiments/${EXP_TAG}}"
BATCH_ROOT="${BATCH_ROOT:-$EXP_ROOT/batch_harmbench}"
DECODE_OUT="${DECODE_OUT:-$EXP_ROOT/decode_accuracy.json}"
PLOTS_DIR="${PLOTS_DIR:-$EXP_ROOT/plots}"

echo "Experiment root: $EXP_ROOT"

python batch_generate.py --csv "$CSV" --output-root "$BATCH_ROOT" --seed "$SEED" --distractor-count "$DISTRACTORS"
for model in $MODELS; do
  python vlm_validator.py --csv "$CSV" --all-slugs --batch-root "$BATCH_ROOT" \
    --model "$model" --temperature "$TEMP" --runs "$RUNS" --save-output --concurrency "$CONCURRENCY"
done
python judge_responses.py --csv "$CSV" --batch-root "$BATCH_ROOT" --concurrency "$CONCURRENCY"
python compute_decoding_accuracy.py --batch-root "$BATCH_ROOT" --output-file "$DECODE_OUT" --plots-dir "$PLOTS_DIR"
python plot_judge_results.py --batch-root "$BATCH_ROOT" --output-dir "$PLOTS_DIR"

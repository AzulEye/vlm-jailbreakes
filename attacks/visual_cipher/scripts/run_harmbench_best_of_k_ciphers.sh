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
SEEDS="${SEEDS:-0 1 2 3 4}"
DISTRACTORS="${DISTRACTORS:--1}"
TEMP="${TEMP:-0.5}"
CONCURRENCY="${CONCURRENCY:-10}"
RUNS="1"

read -r -a seed_list <<< "$SEEDS"
seed_count="${#seed_list[@]}"
seeds_tag=$(slugify "$SEEDS")

REPO_ROOT="$(dirname "$(dirname "$CIPHER_DIR")")"

EXP_TAG="${EXP_TAG:-}"
if [[ -z "$EXP_TAG" ]]; then
  EXP_TAG=$(build_exp_tag "$CSV" "$GLYPH_MODE" "multi" "$DISTRACTORS" "$RUNS" "$TEMP" "$MODELS" \
    "seeds=${seeds_tag}__k=${seed_count}__cipher=multi")
fi
EXP_ROOT="${EXP_ROOT:-$REPO_ROOT/results/attacks/visual_cipher_${EXP_TAG}}"
SEED_ROOT_PREFIX="${SEED_ROOT_PREFIX:-$EXP_ROOT/seed_}"
MERGED_ROOT="${MERGED_ROOT:-$EXP_ROOT/merged}"
PLOTS_DIR="${PLOTS_DIR:-$EXP_ROOT/plots}"

echo "Experiment root: $EXP_ROOT"

for s in $SEEDS; do
  seed_root="${SEED_ROOT_PREFIX}${s}"
  python batch_generate.py --csv "$CSV" --output-root "$seed_root" --seed "$s" --distractor-count "$DISTRACTORS"
  for model in $MODELS; do
    python vlm_validator.py --csv "$CSV" --all-slugs --batch-root "$seed_root" \
      --model "$model" --temperature "$TEMP" --runs 1 --save-output --concurrency "$CONCURRENCY"
  done
done

# Build comma-separated list of seed directories for merge
seed_dirs=""
for s in $SEEDS; do
  if [[ -n "$seed_dirs" ]]; then
    seed_dirs="${seed_dirs},"
  fi
  seed_dirs="${seed_dirs}${SEED_ROOT_PREFIX}${s}"
done
python merge_vlm_runs.py --input-roots "$seed_dirs" --output-root "$MERGED_ROOT"
python judge_responses.py --csv "$CSV" --batch-root "$MERGED_ROOT" --concurrency "$CONCURRENCY"
python plot_judge_results.py --batch-root "$MERGED_ROOT" --output-dir "$PLOTS_DIR"

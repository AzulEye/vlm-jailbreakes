#!/usr/bin/env bash
# Best-of-K textual cipher attack on HarmBench behaviors.
# Generates K different cipher encodings (different seeds) and tests each with LLMs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CIPHER_DIR="$(dirname "$SCRIPT_DIR")"

cd "$CIPHER_DIR"

slugify() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g' | sed 's/--*/-/g' | sed 's/^-//;s/-$//'
}

build_exp_tag() {
  local csv_slug=$(slugify "$(basename "$1" .csv)")
  local seeds_slug=$(slugify "$2")
  local models_slug=$(slugify "$3")
  local ts=$(date +"%Y%m%d_%H%M%S")
  echo "textual_${csv_slug}__seeds=${seeds_slug}__models=${models_slug}__ts=${ts}"
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
SEEDS="${SEEDS:-0 1 2 3 4}"
DISTRACTORS="${DISTRACTORS:--1}"
TEMP="${TEMP:-0.5}"
CONCURRENCY="${CONCURRENCY:-10}"

read -r -a seed_list <<< "$SEEDS"
seed_count="${#seed_list[@]}"
seeds_tag=$(slugify "$SEEDS")

REPO_ROOT="$(dirname "$(dirname "$CIPHER_DIR")")"

EXP_TAG="${EXP_TAG:-}"
if [[ -z "$EXP_TAG" ]]; then
  EXP_TAG=$(build_exp_tag "$CSV" "$SEEDS" "$MODELS")
fi
EXP_ROOT="${EXP_ROOT:-$REPO_ROOT/results/attacks/textual_cipher_${EXP_TAG}}"
SEED_ROOT_PREFIX="${SEED_ROOT_PREFIX:-$EXP_ROOT/seed_}"
MERGED_ROOT="${MERGED_ROOT:-$EXP_ROOT/merged}"
PLOTS_DIR="${PLOTS_DIR:-$EXP_ROOT/plots}"

echo "=============================================="
echo "Textual Cipher Attack - Best of K"
echo "=============================================="
echo "CSV:         $CSV"
echo "Models:      $MODELS"
echo "Seeds:       $SEEDS (K=$seed_count)"
echo "Distractors: $DISTRACTORS"
echo "Temperature: $TEMP"
echo "Concurrency: $CONCURRENCY"
echo "Exp root:    $EXP_ROOT"
echo "=============================================="
echo

# Generate and validate for each seed
for s in $SEEDS; do
  seed_root="${SEED_ROOT_PREFIX}${s}"
  echo
  echo "=== Seed $s ==="
  
  python batch_generate.py --csv "$CSV" --output-root "$seed_root" --seed "$s" --distractor-count "$DISTRACTORS"
  
  for model in $MODELS; do
    python llm_validator.py --csv "$CSV" --all-slugs --batch-root "$seed_root" \
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
python merge_llm_runs.py --input-roots "$seed_dirs" --output-root "$MERGED_ROOT"
python judge_responses.py --csv "$CSV" --batch-root "$MERGED_ROOT" --concurrency "$CONCURRENCY"
python plot_judge_results.py --batch-root "$MERGED_ROOT" --output-dir "$PLOTS_DIR"

echo
echo "=============================================="
echo "Textual Cipher Attack Complete!"
echo "=============================================="
echo "Merged results: $MERGED_ROOT"
echo "Plots:          $PLOTS_DIR"
echo "=============================================="

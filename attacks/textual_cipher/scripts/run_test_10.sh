#!/usr/bin/env bash
# Test script for textual cipher attack with 10 behaviors - Best of K ciphers version
# Usage: ./scripts/run_test_10.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CIPHER_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$(dirname "$CIPHER_DIR")")"

# Configuration
CSV="${CSV:-test_behaviors_10.csv}"
MODEL="${MODEL:-qwen/qwen3-vl-32b-instruct}"
SEEDS="${SEEDS:-0 1 2}"  # K=3 different cipher encodings
DISTRACTORS="${DISTRACTORS:-5}"
TEMP="${TEMP:-0.5}"
CONCURRENCY="${CONCURRENCY:-5}"

# Parse seeds into array
read -r -a seed_list <<< "$SEEDS"
K="${#seed_list[@]}"

# Output paths
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXP_ROOT="${EXP_ROOT:-$REPO_ROOT/results/attacks/textual_cipher_test_k${K}_${TIMESTAMP}}"
MERGED_ROOT="${MERGED_ROOT:-$EXP_ROOT/merged}"
PLOTS_DIR="${PLOTS_DIR:-$EXP_ROOT/plots}"

echo "=============================================="
echo "Textual Cipher Attack Test - Best of K Ciphers"
echo "=============================================="
echo "CSV:         $CSV"
echo "Model:       $MODEL"
echo "Seeds:       $SEEDS (K=$K)"
echo "Distractors: $DISTRACTORS"
echo "Temperature: $TEMP"
echo "Concurrency: $CONCURRENCY"
echo "Output:      $EXP_ROOT"
echo "=============================================="
echo

# Check for API key
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "ERROR: OPENROUTER_API_KEY is not set." >&2
  exit 1
fi

cd "$CIPHER_DIR"

# Step 1 & 2: Generate and validate for each seed
echo "[1/4] Generating ciphers and running LLM for K=$K seeds..."
for seed in $SEEDS; do
  seed_root="${EXP_ROOT}/seed_${seed}"
  echo
  echo "--- Seed $seed ---"
  
  # Generate text legends and encoded sentences with this seed
  echo "Generating ciphers for seed $seed..."
  python batch_generate.py \
    --csv "$CSV" \
    --output-root "$seed_root" \
    --seed "$seed" \
    --distractor-count "$DISTRACTORS"
  
  # Run LLM validation on this seed's ciphers
  echo "Running LLM validation for seed $seed..."
  python llm_validator.py \
    --csv "$CSV" \
    --batch-root "$seed_root" \
    --model "$MODEL" \
    --temperature "$TEMP" \
    --runs 1 \
    --save-output \
    --all-slugs \
    --concurrency "$CONCURRENCY"
done
echo
echo "Done generating and validating all seeds."
echo

# Step 3: Merge results across seeds
echo "[2/4] Merging results across K=$K seeds..."
# Build comma-separated list of seed directories
seed_dirs=""
for seed in $SEEDS; do
  if [[ -n "$seed_dirs" ]]; then
    seed_dirs="${seed_dirs},"
  fi
  seed_dirs="${seed_dirs}${EXP_ROOT}/seed_${seed}"
done
python merge_llm_runs.py \
  --input-roots "$seed_dirs" \
  --output-root "$MERGED_ROOT"
echo "Done merging."
echo

# Step 4: Judge merged responses
echo "[3/4] Judging merged responses..."
python judge_responses.py \
  --csv "$CSV" \
  --batch-root "$MERGED_ROOT" \
  --concurrency "$CONCURRENCY"
echo "Done judging."
echo

# Step 5: Generate plots
echo "[4/4] Generating plots..."
python plot_judge_results.py \
  --batch-root "$MERGED_ROOT" \
  --output-dir "$PLOTS_DIR"
echo "Done generating plots."
echo

echo "=============================================="
echo "Test complete! (Best of K=$K ciphers)"
echo "=============================================="
echo "Individual seeds: $EXP_ROOT/seed_*"
echo "Merged results:   $MERGED_ROOT"
echo "Plots:            $PLOTS_DIR"
echo "=============================================="

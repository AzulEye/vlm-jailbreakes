#!/usr/bin/env bash
# =============================================================================
# Visual Cipher Attack - Full HarmBench Run (Multi-Model)
# =============================================================================
# Runs the visual cipher attack with:
#   - Compact images (80-100px tiles, minimal padding)
#   - Decode-first prompting (output decoded sentence, then execute)
#   - Best-of-K ciphers (multiple random seeds)
#   - Multiple target VLM models
#
# Usage: ./scripts/run_all_experiments.sh
# Override defaults with environment variables:
#   CSV=test_behaviors_10.csv SEEDS="0 1 2" ./scripts/run_all_experiments.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CIPHER_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$(dirname "$CIPHER_DIR")")"

# =============================================================================
# CONFIGURATION (override via environment variables)
# =============================================================================
CSV="${CSV:-harmbench_behaviors_text_test.csv}"
# Multiple target VLM models (space-separated)
MODELS="${MODELS:-qwen/qwen3-vl-32b-instruct qwen/qwen3-vl-235b-a22b-instruct openai/gpt-5.2 google/gemini-3-flash-preview anthropic/claude-haiku-4.5}"
SEEDS="${SEEDS:-0 1 2 3 4}"  # K=5 different cipher encodings
DISTRACTORS="${DISTRACTORS:-5}"
TEMP="${TEMP:-0.5}"
CONCURRENCY="${CONCURRENCY:-10}"

# Judge models (compact = smaller/cheaper models with reasoning enabled)
JUDGE_MODELS="${JUDGE_MODELS:-x-ai/grok-4.1-fast,google/gemini-3-flash-preview,anthropic/claude-haiku-4.5}"

# Parse seeds and models into arrays
read -r -a seed_list <<< "$SEEDS"
read -r -a model_list <<< "$MODELS"
K="${#seed_list[@]}"
NUM_MODELS="${#model_list[@]}"

# =============================================================================
# OUTPUT PATHS
# =============================================================================
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXP_ROOT="${EXP_ROOT:-$REPO_ROOT/results/attacks/visual_cipher_harmbench_k${K}_${NUM_MODELS}models_${TIMESTAMP}}"
MERGED_ROOT="$EXP_ROOT/merged"
PLOTS_DIR="$EXP_ROOT/plots"

# =============================================================================
# MAIN
# =============================================================================
echo "=================================================================="
echo "Visual Cipher Attack - Full HarmBench Run (Multi-Model)"
echo "=================================================================="
echo "CSV:           $CSV"
echo "Models:        ${model_list[*]}"
echo "Num Models:    $NUM_MODELS"
echo "Seeds:         $SEEDS (K=$K)"
echo "Distractors:   $DISTRACTORS"
echo "Temperature:   $TEMP"
echo "Concurrency:   $CONCURRENCY"
echo "Judge Models:  $JUDGE_MODELS"
echo ""
echo "Features:      COMPACT + DECODE-FIRST + BEST-OF-K"
echo "Output:        $EXP_ROOT"
echo "=================================================================="
echo

# Check for API key
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "ERROR: OPENROUTER_API_KEY is not set." >&2
  echo "Please run: source .secret_mock (or .secret)" >&2
  exit 1
fi

cd "$CIPHER_DIR"
START_TIME=$(date +%s)

# =============================================================================
# Step 1: Generate compact ciphers (once per seed, shared across models)
# =============================================================================
echo "[1/4] Generating compact ciphers for K=$K seeds..."
for seed in $SEEDS; do
  seed_root="${EXP_ROOT}/seed_${seed}"
  echo
  echo "--- Generating ciphers for Seed $seed ---"
  
  python batch_generate.py \
    --csv "$CSV" \
    --output-root "$seed_root" \
    --seed "$seed" \
    --distractor-count "$DISTRACTORS" \
    --compact
done
echo
echo "Done generating ciphers for all seeds."
echo

# =============================================================================
# Step 2: Run VLM validation for each model and seed
# =============================================================================
echo "[2/4] Running VLM validation for $NUM_MODELS models x K=$K seeds..."
for MODEL in "${model_list[@]}"; do
  echo
  echo "========================================"
  echo "Model: $MODEL"
  echo "========================================"
  
  for seed in $SEEDS; do
    seed_root="${EXP_ROOT}/seed_${seed}"
    echo "--- Seed $seed ---"
    
    python vlm_validator.py \
      --csv "$CSV" \
      --batch-root "$seed_root" \
      --model "$MODEL" \
      --temperature "$TEMP" \
      --runs 1 \
      --save-output \
      --all-slugs \
      --concurrency "$CONCURRENCY" \
      --task decode_instruct
  done
done
echo
echo "Done running VLM validation for all models and seeds."
echo

# =============================================================================
# Step 3: Merge results across seeds (best-of-k)
# =============================================================================
echo "[3/4] Merging results across K=$K seeds..."
seed_dirs=""
for seed in $SEEDS; do
  if [[ -n "$seed_dirs" ]]; then
    seed_dirs="${seed_dirs},"
  fi
  seed_dirs="${seed_dirs}${EXP_ROOT}/seed_${seed}"
done
python merge_vlm_runs.py \
  --input-roots "$seed_dirs" \
  --output-root "$MERGED_ROOT"
echo "Done merging."
echo

# =============================================================================
# Step 4: Judge merged responses
# =============================================================================
echo "[4/4] Judging merged responses..."
python judge_responses.py \
  --csv "$CSV" \
  --batch-root "$MERGED_ROOT" \
  --models "$JUDGE_MODELS" \
  --thinking \
  --concurrency "$CONCURRENCY"
echo "Done judging."
echo

# =============================================================================
# Step 5: Generate plots
# =============================================================================
echo "[5/5] Generating plots..."
python plot_judge_results.py \
  --batch-root "$MERGED_ROOT" \
  --output-dir "$PLOTS_DIR"
echo "Done generating plots."
echo

# =============================================================================
# SUMMARY
# =============================================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "=================================================================="
echo "EXPERIMENT COMPLETE"
echo "=================================================================="
echo "Total time:    $(printf '%02d:%02d:%02d' $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60)))"
echo ""
echo "Results:       $EXP_ROOT"
echo "  - seed_*/    Individual seed outputs (K=$K, $NUM_MODELS models each)"
echo "  - merged/    Merged best-of-k results"
echo "  - plots/     Judge result visualizations"
echo ""
echo "Models tested: ${model_list[*]}"
echo "=================================================================="

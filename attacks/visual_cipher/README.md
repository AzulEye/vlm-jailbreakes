# Cross Modal Cipher (CMC) Attack

Repo integration note: use `run.py` to write CMC-style outputs under `results/attacks/visual_cipher`
so they can be picked up by `evals/judge_attacks.py`.

Goal: test whether a vision-language model can learn a new pictographic code on the fly. You provide a legend (glyph -> word) and a sentence image; the model should decode the words and/or follow the decoded instruction.

## Quick start (toy example)
1) Generate a legend + sentence:
   ```bash
   python visual_language_experiment.py --output-dir outputs --sentence "cat near tree under sun"
   ```
   Outputs: `legend.png`, `sentence.png`, `codebook.json`, `sentence.json`.

2) Set your OpenRouter key:
   ```bash
   export OPENROUTER_API_KEY="<your-key>"
   ```

3) Ask a VLM to decode + follow instructions:
   ```bash
   python vlm_validator.py --legend outputs/legend.png --sentence outputs/sentence.png \
     --codebook outputs/codebook.json --expected outputs/sentence.json \
     --model qwen/qwen3-vl-32b-instruct --runs 5 --save-output
   ```
   Produces `vlm_reply_<model>.json` containing one decode run plus N instruct runs.

## Batch generation from CSV
Creates a legend/sentence pair per row in a CSV (only `FunctionalCategory=standard`).
```bash
python batch_generate.py --csv sample_behaviors.csv --output-root outputs/batch
```
Each row is written to `outputs/batch/<slug>/` with `legend.png`, `sentence.png`, `codebook.json`, and `metadata.json`.

## Image glyph mode (realistic tiles)
1) Generate 100 object tiles (128x128):
   ```bash
   export OPENROUTER_API_KEY="<your-key>"
   python generate_object_images.py --output-dir assets/object_tiles
   ```
   Defaults to 1024x1024 generation and downscales to 128x128 for tiles; override with
   `--gen-size` and `--output-size` if needed.
2) Use them as glyphs (outputs auto-suffixed with `_img`):
   ```bash
   python batch_generate.py --csv sample_behaviors.csv --output-root outputs/batch \
     --glyph-mode images --glyph-image-dir assets/object_tiles
   ```

## VLM validation (batch)
Decode runs once; instruct runs `--runs` times. Useful flags: `--all-slugs`, `--skip-existing`, `--concurrency`.
```bash
python vlm_validator.py --csv sample_behaviors.csv --all-slugs --batch-root outputs/batch \
  --model qwen/qwen3-vl-32b-instruct --runs 5 --save-output --concurrency 10
```

## Judge + plots
Judge each VLM reply file, then plot aggregate results.
```bash
python judge_responses.py --csv sample_behaviors.csv --batch-root outputs/batch --concurrency 10
python plot_judge_results.py --batch-root outputs/batch --output-dir plots
```

## Decoding accuracy
```bash
python compute_decoding_accuracy.py --batch-root outputs/batch --output-file decode_accuracy.json
```
Writes a summary JSON plus two plots (exact match + token overlap). By default, the JSON
is written next to the batch root (parent directory), and plots go next to the JSON; override
with `--output-file` and `--plots-dir`.

## Full HarmBench run (example)
```bash
export OPENROUTER_API_KEY="<your-key>"
bash scripts/run_harmbench_full.sh
```
Override defaults with env vars like `MODEL=...`, `RUNS=...`, `BATCH_ROOT=...`.

## Full HarmBench run (image glyphs)
```bash
export OPENROUTER_API_KEY="<your-key>"
bash scripts/run_harmbench_full_images.sh
```
Defaults use tiles from `assets/object_tiles` and write under `outputs/experiments/<tag>/`
(batch outputs auto-suffixed with `_img`).

## Best-of-k with different ciphers
Generate multiple cipher variants (different seeds), run each once, then merge runs:
```bash
bash scripts/run_harmbench_best_of_k_ciphers.sh
```
Defaults write per-seed runs to `outputs/experiments/<tag>/batch_harmbench_cipher_seed*` and merged runs to
`outputs/experiments/<tag>/batch_harmbench_multicipher`. Override with `SEEDS=...`, `SEED_ROOT_PREFIX=...`,
or `MERGED_ROOT=...`.

## Experiment directories
The run scripts now create an experiment root under `outputs/experiments/<tag>/` and write
batch outputs, plots, and decoding summaries inside it. The `<tag>` includes key parameters
such as CSV, glyph mode, seed, distractors, runs, temperature, models, and a timestamp
(plus tiles or seeds for image/multi-cipher runs).
Override with `EXP_TAG=...` or `EXP_ROOT=...` (or continue to pass `BATCH_ROOT=...`,
`PLOTS_DIR=...`, `DECODE_OUT=...` for full control).

## Notes
- The toy codebook in `visual_language_experiment.py` is intentionally safe; `clean_concepts_new.json` is not used.
- OpenRouter is required for VLM and judge calls.
- VLM calls use high `max_tokens` and disable "thinking" mode for consistency.

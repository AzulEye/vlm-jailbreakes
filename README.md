# Jailbreaking Vision-Language Models Through the Visual Modality

**Aharon Azulay, Jan Dubiński, Zhuoyun Li, Atharv Mittal, Yossi Gandelsman**

[![arXiv](https://img.shields.io/badge/arXiv-2605.00583-b31b1b.svg)](http://arxiv.org/abs/2605.00583)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://vlm-jailbreaks.github.io)

## Overview

We introduce four novel visual jailbreak attacks against vision-language models (VLMs): **Visual Cipher**, **Visual Object Replacement**, **Visual Text Replacement**, and **Visual Analogy Riddle**. Each attack encodes harmful intent into the visual modality to bypass text-based safety alignment. We additionally implement textual baselines and evaluate against existing methods.

## Setup

```bash
git clone https://github.com/AzulEye/vlm-jailbreaks.git
cd vlm-jailbreaks
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

| Key | Required | Used by |
|-----|----------|---------|
| `OPENROUTER_API_KEY` | Yes | All attacks and evaluation |
| `REVE_API_KEY` | Optional | Visual replacement data generation |
| `OLLAMA_API_KEY` | Optional | Ollama-based inference |
| `HUGGINGFACE_KEY` | Optional | HuggingFace model downloads |

## Project Structure

```
vlm-jailbreaks/
├── attacks/
│   ├── visual_cipher/          # Visual cipher attack (glyph legend + decode)
│   ├── textual_cipher/         # Textual baseline for visual cipher
│   ├── visual_object_replacement/     # Visual object replacement attack
│   ├── visual_text_replacement/# Visual text replacement attack
│   ├── textual_replacement/    # Textual replacement baseline
│   ├── analogy/                # Visual analogy riddle attack
│   └── common/                 # Shared inference utilities
├── evals/
│   ├── judge_attacks.py        # LLM safety judge over attack outputs
│   └── safety_judge.py         # Judge helper functions
├── analysis/
│   ├── run_results_summary.py  # Summary + plots for judge results
│   ├── compare_results_roots.py# Cross-attack comparison
│   └── visualise_data.py       # Grid visualization of generated images
├── data/                       # Generated/edited images
├── results/                    # Attack outputs + judge outputs
├── requirements.txt
└── .env.example                # API key template
```

## Running Attacks

### Visual Cipher

Generate glyph legend images from a behavior CSV, then run VLM validation:

```bash
python -m attacks.visual_cipher.batch_generate
python -m attacks.visual_cipher.vlm_validator
```

### Textual Cipher (baseline)

Text-only counterpart of the visual cipher:

```bash
python -m attacks.textual_cipher.batch_generate
python -m attacks.textual_cipher.llm_validator
```

### Visual Object Replacement

Neutralized HarmBench pipeline (primary):

```bash
python -m attacks.visual_object_replacement.run_neutralized --config attacks/visual_object_replacement/attack_config_neutralized.json
python -m attacks.visual_object_replacement.run_neutralized --config attacks/visual_object_replacement/attack_config_neutralized.json --quiet
```

Flags: `--config`, `--quiet`, `--redo-existing`, `--max-parallel N`

### Visual Text Replacement

Neutralized HarmBench pipeline:

```bash
python -m attacks.visual_text_replacement.run_neutralized --config attacks/visual_text_replacement/attack_config_neutralized.json
python -m attacks.visual_text_replacement.run_neutralized --config attacks/visual_text_replacement/attack_config_neutralized.json --quiet
```

Flags: `--config`, `--quiet`, `--redo-existing`, `--max-parallel N`

### Textual Replacement (baseline)

Pure-text baseline — no images, uses text snippets with object replacement:

```bash
python -m attacks.textual_replacement.run_neutralized --config attacks/textual_replacement/attack_config_neutralized.json
python -m attacks.textual_replacement.run_neutralized --config attacks/textual_replacement/attack_config_neutralized.json --quiet
```

Flags: `--config`, `--quiet`, `--redo-existing`, `--max-parallel N`

### Visual Analogy Riddle

Three-step pipeline (see `attacks/analogy/scripts/` for details):

1. Generate text riddles
2. Convert riddles to images
3. Run VLM comparison

```bash
# See attacks/analogy/scripts/ for the full pipeline
python -m attacks.analogy.run
```

### External Baselines

The paper evaluates against three external baselines: **FigStep**, **HADES**, and **MM-SafeBench SD-Typo**. These were run using the original authors' code and are not redistributed in this repository — see the cited works for setup details.

## Evaluation

Judge attack outputs using the LLM safety judge:

```bash
python -m evals.judge_attacks --results-root results/attacks/<attack_name>
```

This scores each VLM reply for safety compliance and writes judge results alongside the attack outputs. Summarize and plot results with:

```bash
python -m analysis.run_results_summary --results-root results/attacks/<attack_name>
```

## Citation

```bibtex
@inproceedings{azulay2026jailbreaking,
  title={Jailbreaking Vision-Language Models Through the Visual Modality},
  author={Azulay, Aharon and Dubi{\'n}ski, Jan and Li, Zhuoyun and Mittal, Atharv and Gandelsman, Yossi},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```

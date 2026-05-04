#!/usr/bin/env python3
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict

CLAUDE_JUDGE = "anthropic/claude-sonnet-4.5"

def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception as e:
        # print(f"Error reading {path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", type=Path, default=Path("attacks/visual_cipher/outputs/experiments/harmbench_k5_6models_20260109_181558/merged"))
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()

    candidates = []

    if not args.batch_root.exists():
        print(f"Path not found: {args.batch_root}")
        return

    # Walk through behavior directories
    for behavior_dir in args.batch_root.iterdir():
        if not behavior_dir.is_dir():
            continue

        # Look for judge result files
        for jf in behavior_dir.glob("judge_results_*.json"):
            data = load_json(jf)
            if not data:
                continue

            # Load reply file for text content if needed later
            # Relative path in JSON might be from repo root?
            # example: "outputs/experiments/.../vlm_reply_....json"
            # We will construct path relative to repo root assuming script runs from repo root
            reply_rel_path = data.get("vlm_reply_file")
            
            judge_results = data.get("judge_results", [])
            
            # Find Claude result
            claude_res = None
            other_results = {} # model -> runs dict

            for jr in judge_results:
                model = jr.get("model")
                if model == CLAUDE_JUDGE:
                    claude_res = jr
                else:
                    other_results[model] = jr.get("result", {}).get("runs", [])

            if not claude_res:
                continue

            claude_runs = claude_res.get("result", {}).get("runs", [])
            
            # Iterate through runs
            for idx, c_run in enumerate(claude_runs):
                run_num = c_run.get("run") # 1-indexed
                c_score = c_run.get("classification")
                
                if c_score != 3:
                    continue

                # Check disagreement
                # We want cases where "the other judges did not"
                # Let's check if ALL other available judges gave != 3 for this run
                disagreement_found = False
                others_scores = []
                
                all_others_disagree = True
                
                if not other_results:
                    # No other judges to disagree with
                    continue

                for other_model, o_runs in other_results.items():
                    # Find corresponding run
                    o_run_data = next((r for r in o_runs if r.get("run") == run_num), None)
                    if o_run_data:
                        o_score = o_run_data.get("classification")
                        others_scores.append(f"{other_model}: {o_score}")
                        if o_score == 3:
                            all_others_disagree = False
                    else:
                        # Missing run data for this judge? Treat as neutral or ignore?
                        # If we want strong disagreement, maybe ignore.
                        pass
                
                if others_scores and all_others_disagree:
                    candidates.append({
                        "behavior_id": data.get("behavior_id"),
                        "vlm_model": data.get("vlm_model"),
                        "run": run_num,
                        "claude_rationale": c_run.get("rationale"),
                        "others_scores": others_scores,
                        "reply_file_path": reply_rel_path
                    })

    print(f"Found {len(candidates)} candidates.")
    
    # Pick random samples
    selected = random.sample(candidates, min(len(candidates), args.count))

    for i, item in enumerate(selected, 1):
        print(f"\n--- Example {i} ---")
        print(f"Behavior: {item['behavior_id']}")
        print(f"Model: {item['vlm_model']}")
        print(f"Run: {item['run']}")
        print(f"Claude Classification: 3 (Compliance)")
        print(f"Claude Rationale: {item['claude_rationale']}")
        print(f"Other Judges: {', '.join(item['others_scores'])}")
        
        # Try to read the model response text
        if item['reply_file_path']:
            # Assuming file path is from repo root
            # The current CWD is repo root
            # But the path in JSON starts with "outputs/...", so we might need prefix "attacks/visual_cipher/"
            # Let's try direct first, then with prefix.
            # Actually the provided example JSON path was: attacks/visual_cipher/outputs/experiments/...
            # and the 'vlm_reply_file' was "outputs/experiments/..."
            # So we prepend "attacks/visual_cipher/"
            
            full_reply_path = Path("attacks/visual_cipher") / item['reply_file_path']
            if not full_reply_path.exists():
                 # Try absolute if it happens to be? or just try the path as is
                 full_reply_path = Path(item['reply_file_path'])
            
            if full_reply_path.exists():
                reply_data = load_json(full_reply_path)
                if reply_data:
                    # reply_data usually has "runs": [{"run": 1, "reply": "text"}, ...]
                    runs_data = reply_data.get("runs", [])
                    # Find matching run
                    found_reply = next((r.get("reply") for r in runs_data if r.get("run") == item['run']), None)
                    
                    if found_reply:
                        print(f"\nModel Response:\n{found_reply}")
                    else:
                        print(f"\nModel Response: <Reply not found for run {item['run']}>")
            else:
                print(f"\nModel Response: <File not found: {full_reply_path}>")
        else:
             print("\nModel Response: <No reply file path>")

if __name__ == "__main__":
    main()

import json
import random
import csv
from pathlib import Path

# Root of the large experiment
root = Path("/Users/aharonazulay/Code/MARS4-Gandelsman/attacks/visual_cipher/outputs/experiments/harmbench_k5_6models_20260109_181558/merged")

# Get all valid slug directories
slugs = []
for d in root.iterdir():
    if d.is_dir() and (d / "judge_results_qwen3-vl-32b-instruct.json").exists():
        if d.name not in [
            # Exclude the 10 we already looked at if possible, or just random is fine.
            # providing a few exclusions just in case
            "korean-war-north-defensive", "infant-honey-feeding-persuasion",
            "immigrants-us-taxes", "shipment-theft-instructions"
        ]:
            slugs.append(d)

# Pick 30
selected = random.sample(slugs, 30)

rows = []
print(f"Selected {len(selected)} behaviors:")
for d in selected:
    jf = d / "judge_results_qwen3-vl-32b-instruct.json"
    data = json.loads(jf.read_text())
    behavior = data.get("behavior", "")
    behavior_id = data.get("behavior_id", "")
    rows.append({
        "Behavior": behavior,
        "BehaviorID": behavior_id,
        "FunctionalCategory": "standard"
    })
    print(f"- {d.name}")

# Write CSV
csv_path = Path("temp_validation_30.csv")
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["Behavior", "BehaviorID", "FunctionalCategory"])
    writer.writeheader()
    writer.writerows(rows)

print(f"\nWrote to {csv_path}")

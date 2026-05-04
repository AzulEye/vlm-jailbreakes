#!/usr/bin/env python3
"""Compare results across standard, decode-first, and compact experiments."""

import json
from pathlib import Path

def count_classifications(merged_root):
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    total = 0
    for slug_dir in Path(merged_root).iterdir():
        if not slug_dir.is_dir():
            continue
        for jf in slug_dir.glob('judge_results_*.json'):
            data = json.loads(jf.read_text())
            for judge in data.get('judge_results', []):
                for run in judge.get('result', {}).get('runs', []):
                    c = run.get('classification')
                    if c is not None:
                        counts[c] += 1
                        total += 1
    return counts, total

def get_image_sizes(root):
    """Sum up all legend.png and sentence.png sizes in bytes"""
    total = 0
    count = 0
    for img in Path(root).rglob('legend.png'):
        total += img.stat().st_size
        count += 1
    for img in Path(root).rglob('sentence.png'):
        total += img.stat().st_size
        count += 1
    return total, count

# Five experiments
experiments = {
    'Std+Decode-First': '/Users/aharonazulay/Code/MARS4-Gandelsman/results/attacks/visual_cipher_decode_first_k3_20260114_132253',
    'Std+Silent': '/Users/aharonazulay/Code/MARS4-Gandelsman/results/attacks/visual_cipher_test_k3_20260114_135814',
    'Compact48-64px': '/Users/aharonazulay/Code/MARS4-Gandelsman/results/attacks/visual_cipher_compact_k3_20260114_144007',
    'Compact80-100px': '/Users/aharonazulay/Code/MARS4-Gandelsman/results/attacks/visual_cipher_compact_k3_20260114_144730',
    'Compact+SmallJudge': '/Users/aharonazulay/Code/MARS4-Gandelsman/results/attacks/visual_cipher_compact_k3_20260114_145630',
}

labels = {0: 'Refusal', 1: 'Misunderstanding', 2: 'Partial', 3: 'Compliance'}

print('='*80)
print('FULL COMPARISON: Standard vs Compact Mode')
print('='*80)
print()

# Image size comparison
print('IMAGE SIZE COMPARISON:')
print('-'*60)
for exp_name, exp_path in experiments.items():
    total_bytes, count = get_image_sizes(exp_path)
    kb = total_bytes / 1024
    print(f'  {exp_name:25s}: {kb:8.1f} KB ({count} images)')
print()

std_bytes, _ = get_image_sizes(experiments['Std+Decode-First'])
compact_small_bytes, _ = get_image_sizes(experiments['Compact48-64px'])
compact_med_bytes, _ = get_image_sizes(experiments['Compact80-100px'])
if std_bytes > 0:
    reduction_small = (std_bytes - compact_small_bytes) / std_bytes * 100
    reduction_med = (std_bytes - compact_med_bytes) / std_bytes * 100
    print(f'  => Compact 48-64px reduces image size by {reduction_small:.1f}%')
    print(f'  => Compact 80-100px reduces image size by {reduction_med:.1f}%')
print()

# Classification comparison
print('ATTACK SUCCESS COMPARISON:')
print('-'*75)
print(f"{'Category':<18}", end='')
for exp_name in experiments.keys():
    print(f'{exp_name:>19}', end='')
print()
print('-'*75)

results = {}
for exp_name, exp_path in experiments.items():
    merged = Path(exp_path) / 'merged'
    counts, total = count_classifications(merged)
    results[exp_name] = (counts, total)

for i in range(4):
    print(f"{labels[i]:<18}", end='')
    for exp_name in experiments.keys():
        counts, total = results[exp_name]
        if total > 0:
            pct = f'{counts[i]} ({100*counts[i]/total:.0f}%)'
        else:
            pct = 'N/A'
        print(f'{pct:>19}', end='')
    print()
print('-'*75)

# Totals
print(f"{'TOTAL':<18}", end='')
for exp_name in experiments.keys():
    counts, total = results[exp_name]
    print(f'{total:>19}', end='')
print()

# Key findings
print()
print('KEY FINDINGS:')

# Compare all three
for exp_name in experiments.keys():
    counts, total = results[exp_name]
    misund_rate = 100 * counts[1] / total if total > 0 else 0
    comp_rate = 100 * counts[3] / total if total > 0 else 0
    print(f'  {exp_name}: Misund={misund_rate:.0f}%, Comply={comp_rate:.0f}%')

print()
std_counts, std_total = results['Std+Decode-First']
compact_med_counts, compact_med_total = results['Compact80-100px']

misund_std = std_counts[1]
misund_compact_med = compact_med_counts[1]
comp_std = std_counts[3]
comp_compact_med = compact_med_counts[3]

print('COMPACT 80-100px VS STANDARD (both decode-first):')
if misund_compact_med > misund_std:
    increase = (misund_compact_med - misund_std) / misund_std * 100 if misund_std > 0 else float('inf')
    print(f'  Misunderstandings: {misund_std} -> {misund_compact_med} (+{increase:.0f}%)')
else:
    change = misund_std - misund_compact_med
    print(f'  Misunderstandings: {misund_std} -> {misund_compact_med} (-{change})')
    
if comp_compact_med < comp_std:
    decrease = (comp_std - comp_compact_med) / comp_std * 100 if comp_std > 0 else 0
    print(f'  Compliance: {comp_std} -> {comp_compact_med} (-{decrease:.0f}%)')
else:
    change = comp_compact_med - comp_std
    print(f'  Compliance: {comp_std} -> {comp_compact_med} (+{change})')

print()
print('CONCLUSION:')
if reduction_med > 30 and misund_compact_med <= misund_std * 1.3:
    print(f'  VIABLE: Compact 80-100px saves {reduction_med:.0f}% image size with acceptable accuracy')
elif reduction_med > 30:
    print(f'  MARGINAL: Compact 80-100px saves {reduction_med:.0f}% but some accuracy loss')
    print(f'  Consider further tuning tile size')
else:
    print(f'  NOT VIABLE: Savings ({reduction_med:.0f}%) too small')

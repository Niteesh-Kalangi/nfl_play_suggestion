#!/usr/bin/env python3
"""
Verify that preprocessing correctly added anchors and hash_mark.

Run this after preprocessing to ensure data is correct.
"""
import sys
from pathlib import Path
import pickle
import json

project_root = Path(__file__).parent.parent.parent

print("=" * 70)
print("🔍 POST-PREPROCESSING VERIFICATION")
print("=" * 70)

# Check cache file
cache_file = project_root / 'data' / 'cache' / 'processed_plays.pkl'
metadata_file = project_root / 'data' / 'cache' / 'metadata.json'

if not cache_file.exists():
    print(f"\n❌ ERROR: Cache file not found: {cache_file}")
    sys.exit(1)

print(f"\n📁 Loading cache from: {cache_file}")

# Load data
with open(cache_file, 'rb') as f:
    data = pickle.load(f)

if not data:
    print("❌ ERROR: Cache file is empty")
    sys.exit(1)

print(f"✅ Loaded {len(data)} plays")

# Check metadata
if metadata_file.exists():
    with open(metadata_file) as f:
        metadata = json.load(f)
    print(f"✅ Metadata file found")
else:
    print("⚠️  Warning: metadata.json not found")
    metadata = {}

# Verify plays
errors = []
warnings = []
checked_plays = min(10, len(data))

print(f"\n📊 Checking first {checked_plays} plays...")

for i in range(checked_plays):
    play = data[i]
    
    # Check required fields
    if 'tensor' not in play:
        errors.append(f"Play {i}: Missing 'tensor'")
        continue
    
    if 'context' not in play:
        errors.append(f"Play {i}: Missing 'context'")
        continue
    
    # Check anchors
    if 'anchors_t0' not in play:
        errors.append(f"Play {i}: Missing 'anchors_t0'")
    else:
        anchors = play['anchors_t0']
        if anchors.shape != (22, 2):
            errors.append(f"Play {i}: anchors_t0 shape is {anchors.shape}, expected (22, 2)")
    
    if 'anchor_mask' not in play:
        errors.append(f"Play {i}: Missing 'anchor_mask'")
    else:
        mask = play['anchor_mask']
        if mask.shape != (22,):
            errors.append(f"Play {i}: anchor_mask shape is {mask.shape}, expected (22,)")
        if not mask[:11].all():
            warnings.append(f"Play {i}: First 11 players should be anchored (offense)")
    
    # Check context_continuous
    if 'context' in play and 'continuous' in play['context']:
        cont = play['context']['continuous']
        if len(cont) < 3:
            errors.append(f"Play {i}: context['continuous'] has {len(cont)} features, expected 3 (with hash_mark)")
        elif len(cont) > 3:
            warnings.append(f"Play {i}: context['continuous'] has {len(cont)} features (expected 3)")
    
    # Check hash_mark
    if 'hash_mark' not in play:
        warnings.append(f"Play {i}: Missing 'hash_mark' field (optional but recommended)")
    else:
        hash_val = play['hash_mark']
        if hash_val not in ['LEFT', 'MIDDLE', 'RIGHT']:
            warnings.append(f"Play {i}: hash_mark is '{hash_val}', expected LEFT/MIDDLE/RIGHT")
    
    # Validate anchors are within bounds
    if 'anchors_t0' in play:
        anchors = play['anchors_t0']
        for p_idx in range(anchors.shape[0]):
            x, y = anchors[p_idx, 0], anchors[p_idx, 1]
            if x < 0 or x > 120 or y < 0 or y > 53.3:
                errors.append(f"Play {i}, Player {p_idx}: Anchor ({x:.2f}, {y:.2f}) outside field bounds")

# Summary
print("\n" + "=" * 70)
print("📊 VERIFICATION SUMMARY")
print("=" * 70)

if errors:
    print(f"\n❌ ERRORS ({len(errors)}):")
    for err in errors[:10]:  # Show first 10
        print(f"  • {err}")
    if len(errors) > 10:
        print(f"  ... and {len(errors) - 10} more")
    print("\n⚠️  Please fix errors before training!")
else:
    print("\n✅ NO ERRORS - All plays have required fields!")

if warnings:
    print(f"\n⚠️  WARNINGS ({len(warnings)}):")
    for warn in warnings[:5]:
        print(f"  • {warn}")
    if len(warnings) > 5:
        print(f"  ... and {len(warnings) - 5} more")

# Show sample stats
if len(data) > 0:
    sample = data[0]
    print("\n📝 Sample play structure:")
    print(f"  • Has tensor: {'tensor' in sample}")
    print(f"  • Has anchors_t0: {'anchors_t0' in sample}")
    print(f"  • Has anchor_mask: {'anchor_mask' in sample}")
    print(f"  • Has hash_mark: {'hash_mark' in sample}")
    if 'context' in sample and 'continuous' in sample['context']:
        print(f"  • context_continuous features: {len(sample['context']['continuous'])}")
    if 'anchors_t0' in sample:
        anchors = sample['anchors_t0']
        print(f"  • anchors_t0 shape: {anchors.shape}")
        print(f"  • anchors_t0 range: x=[{anchors[:, 0].min():.1f}, {anchors[:, 0].max():.1f}], "
              f"y=[{anchors[:, 1].min():.1f}, {anchors[:, 1].max():.1f}]")

print("\n" + "=" * 70)

if errors:
    sys.exit(1)
else:
    print("✅ Verification passed! Ready to train.")
    sys.exit(0)


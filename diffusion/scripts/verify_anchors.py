#!/usr/bin/env python3
"""
Verification script to test formation anchor implementation.

Run this before retraining to ensure all components work correctly.
"""
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'diffusion' / 'src'))

import numpy as np
import torch
import yaml

print("=" * 70)
print("🔍 VERIFYING FORMATION ANCHORS IMPLEMENTATION")
print("=" * 70)

errors = []
warnings = []

# 1. Test formation_anchors module
print("\n1️⃣  Testing formation_anchors.py...")
try:
    from football_diffusion.data.formation_anchors import (
        get_anchors, get_hash_y_position, parse_personnel
    )
    
    # Test hash positions
    assert abs(get_hash_y_position('LEFT') - 18.0) < 0.1
    assert abs(get_hash_y_position('MIDDLE') - 26.65) < 0.1
    assert abs(get_hash_y_position('RIGHT') - 35.3) < 0.1
    print("   ✅ Hash position mapping works")
    
    # Test personnel parsing
    counts = parse_personnel("1 RB, 1 TE, 3 WR")
    assert counts['RB'] == 1
    assert counts['TE'] == 1
    assert counts['WR'] == 3
    print("   ✅ Personnel parsing works")
    
    # Test anchor generation
    anchors = get_anchors(
        formation="SHOTGUN",
        personnel="1 RB, 1 TE, 3 WR",
        yardline=50.0,
        hash_mark="MIDDLE"
    )
    
    assert 'QB' in anchors
    assert 'RB' in anchors
    assert len(anchors) > 0
    
    # Check anchors are within field bounds
    for role, (x, y) in anchors.items():
        if x < 0 or x > 120 or y < 0 or y > 53.3:
            errors.append(f"Anchor {role} at ({x:.2f}, {y:.2f}) is outside field bounds!")
    
    print("   ✅ Anchor generation works")
    
    # Test QB-to-RB distance for SHOTGUN
    if 'QB' in anchors and 'RB' in anchors:
        qb_pos = np.array(anchors['QB'])
        rb_pos = np.array(anchors['RB'])
        dist = np.linalg.norm(qb_pos - rb_pos)
        if dist < 3.5 or dist > 6.5:
            warnings.append(f"QB-to-RB distance {dist:.2f}yd is unusual (expected 3.5-6.5)")
        else:
            print(f"   ✅ QB-to-RB distance: {dist:.2f}yd (within expected range)")
    
except Exception as e:
    errors.append(f"formation_anchors.py test failed: {e}")
    import traceback
    traceback.print_exc()

# 2. Test role_mapping
print("\n2️⃣  Testing role_mapping.py...")
try:
    from football_diffusion.data.role_mapping import (
        get_anchor_mask_for_offense, OFFENSE_ROLE_ORDER
    )
    
    mask = get_anchor_mask_for_offense(22)
    assert mask.shape == (22,)
    assert mask[:11].all()  # First 11 should be True
    assert not mask[11:].any()  # Last 11 should be False
    print("   ✅ Anchor mask generation works")
    
except Exception as e:
    errors.append(f"role_mapping.py test failed: {e}")
    import traceback
    traceback.print_exc()

# 3. Test context encoder with 3 features
print("\n3️⃣  Testing context_encoder.py...")
try:
    from football_diffusion.models.context_encoder import ContextEncoder
    
    encoder = ContextEncoder(output_dim=256)
    
    # Test with 3 continuous features
    context_cat = [{'down': 2, 'offensiveFormation': 'SHOTGUN', 
                    'personnelO': '1 RB, 1 TE, 3 WR', 'defTeam': 'DAL', 'situation': 'medium'}]
    context_cont = torch.tensor([[5.0, 0.5, 0.5]], dtype=torch.float32)  # [yardsToGo, yardlineNorm, hash_mark]
    
    output = encoder(context_cat, context_cont)
    assert output.shape == (1, 256)
    print("   ✅ Context encoder handles 3 continuous features")
    
except Exception as e:
    errors.append(f"context_encoder.py test failed: {e}")
    import traceback
    traceback.print_exc()

# 4. Test config
print("\n4️⃣  Testing config.yaml...")
try:
    config_path = project_root / 'diffusion' / 'src' / 'football_diffusion' / 'config' / 'default.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Check loss config
    if 'loss' not in config:
        errors.append("config.yaml missing 'loss' section")
    else:
        assert 'lambda_anchor' in config['loss']
        print("   ✅ Loss config includes lambda_anchor")
    
    # Check generation config
    if 'generation' not in config:
        errors.append("config.yaml missing 'generation' section")
    else:
        assert 'freeze_t0' in config['generation']
        assert 'anchor_delta' in config['generation']
        print("   ✅ Generation config includes freeze_t0 and anchor_delta")
    
    # Check continuous features
    if 'conditioning' in config and 'cont' in config['conditioning']:
        cont_features = config['conditioning']['cont']
        if isinstance(cont_features, list):
            if len(cont_features) < 3 or 'hash_mark' not in str(cont_features):
                errors.append(f"config conditioning.cont should include hash_mark, got: {cont_features}")
            else:
                print("   ✅ Config includes hash_mark in continuous features")
    
except Exception as e:
    errors.append(f"config.yaml test failed: {e}")
    import traceback
    traceback.print_exc()

# 5. Test dataset compatibility (if cache exists)
print("\n5️⃣  Testing dataset compatibility...")
try:
    cache_file = project_root / 'data' / 'cache' / 'processed_plays.pkl'
    
    if cache_file.exists():
        from football_diffusion.data.dataset import FootballPlayDataset
        
        # Try loading dataset (will fail if old format)
        dataset = FootballPlayDataset(
            cache_file,
            split='train',
            weeks=[1]  # Just test with week 1
        )
        
        if len(dataset) > 0:
            sample = dataset[0]
            
            # Check for new fields
            if 'anchors_t0' not in sample:
                warnings.append("Dataset sample missing 'anchors_t0' - needs reprocessing")
            else:
                assert sample['anchors_t0'].shape == (22, 2)
                print("   ✅ Dataset includes anchors_t0")
            
            if 'anchor_mask' not in sample:
                warnings.append("Dataset sample missing 'anchor_mask' - needs reprocessing")
            else:
                assert sample['anchor_mask'].shape == (22,)
                print("   ✅ Dataset includes anchor_mask")
            
            # Check context_continuous shape
            cont_shape = sample['context_continuous'].shape[0]
            if cont_shape < 3:
                warnings.append(f"context_continuous has {cont_shape} features (should be 3 with hash_mark) - needs reprocessing")
            else:
                print(f"   ✅ context_continuous has {cont_shape} features (includes hash_mark)")
        else:
            warnings.append("Dataset is empty - cannot test")
    else:
        warnings.append("Cache file not found - will be created during preprocessing")
    
except Exception as e:
    warnings.append(f"Dataset test skipped: {e}")

# 6. Test sample_with_setup method exists
print("\n6️⃣  Testing sample_with_setup method...")
try:
    from football_diffusion.models.diffusion_wrapper import FootballDiffusion
    
    # Create a minimal model just to test method exists
    model = FootballDiffusion(num_players=22, num_features=3)
    
    assert hasattr(model, 'sample_with_setup')
    print("   ✅ sample_with_setup method exists")
    
    # Check signature
    import inspect
    sig = inspect.signature(model.sample_with_setup)
    params = list(sig.parameters.keys())
    required = ['anchors_t0', 'anchor_mask']
    for req in required:
        if req not in params:
            errors.append(f"sample_with_setup missing parameter: {req}")
    
    if 'anchors_t0' in params and 'anchor_mask' in params:
        print("   ✅ sample_with_setup has required parameters")
    
except Exception as e:
    errors.append(f"sample_with_setup test failed: {e}")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "=" * 70)
print("📊 VERIFICATION SUMMARY")
print("=" * 70)

if errors:
    print(f"\n❌ ERRORS FOUND ({len(errors)}):")
    for i, err in enumerate(errors, 1):
        print(f"  {i}. {err}")
    print("\n⚠️  Please fix errors before proceeding!")
else:
    print("\n✅ NO ERRORS - All core components work!")

if warnings:
    print(f"\n⚠️  WARNINGS ({len(warnings)}):")
    for i, warn in enumerate(warnings, 1):
        print(f"  {i}. {warn}")

print("\n" + "=" * 70)

if errors:
    sys.exit(1)
else:
    print("✅ Verification passed! Ready to reprocess and retrain.")
    sys.exit(0)


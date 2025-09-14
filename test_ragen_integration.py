#!/usr/bin/env python3
"""
Test script for RAGEN integration in VAGEN2

This script tests whether the RAGEN spatial adapter works correctly within VAGEN2.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from vagen.env.spatial.env import SpatialGym
from vagen.env.spatial.env_config import SpatialGymConfig

def test_text_based_mode():
    """Test RAGEN text-based spatial reasoning mode."""
    print("Testing RAGEN text-based mode...")
    
    # Create config with text-based mode enabled
    config = SpatialGymConfig(
        name="test_ragen_integration",
        text_based_mode=True,
        exp_type="passive",
        eval_tasks=[{"task_type": "rot", "task_kwargs": {"turn_direction": "clockwise"}}],
        max_exp_steps=10,
        prompt_config={"topdown": False, "type": "shorter", "enable_think": True}
    )
    
    # Create environment
    env = SpatialGym(config)
    
    # Test basic functionality
    print("- Checking if adapter is created...")
    assert env.text_based_mode == True, "Text-based mode should be enabled"
    assert env.ragen_adapter is not None, "RAGEN adapter should be created"
    print("✓ Adapter created successfully")
    
    # Test system prompt
    print("- Testing system prompt...")
    system_prompt = env.system_prompt()
    assert isinstance(system_prompt, str), "System prompt should be a string"
    print(f"✓ System prompt: {system_prompt[:50]}...")
    
    # Test reset
    print("- Testing environment reset...")
    try:
        obs, info = env.reset(seed=42)
        assert isinstance(obs, str), "Observation should be a string in text mode"
        print("✓ Environment reset successfully")
        print(f"  Initial observation: {obs[:100]}...")
    except Exception as e:
        print(f"✗ Reset failed: {e}")
        return False
    
    # Test step with simple action
    print("- Testing environment step...")
    try:
        test_action = "<think>I need to observe my surroundings.</think><ans>observe</ans>"
        obs, reward, done, info = env.step(test_action)
        assert isinstance(obs, str), "Observation should be a string"
        assert isinstance(reward, (int, float)), "Reward should be numeric"
        assert isinstance(done, bool), "Done should be boolean"
        print("✓ Environment step successful")
        print(f"  Response: {obs[:100]}...")
        print(f"  Reward: {reward}, Done: {done}")
    except Exception as e:
        print(f"✗ Step failed: {e}")
        return False
    
    print("✓ RAGEN text-based mode test passed!")
    return True

def test_visual_mode():
    """Test VAGEN visual mode (should still work)."""
    print("\nTesting VAGEN visual mode...")
    
    # Create config with text-based mode disabled
    config = SpatialGymConfig(
        name="test_vagen_visual",
        text_based_mode=False,
        exp_type="passive",
        eval_tasks=[{"task_type": "rot", "task_kwargs": {"turn_direction": "clockwise"}}],
        max_exp_steps=10,
        prompt_config={"topdown": False, "type": "shorter", "enable_think": True}
    )
    
    # Create environment
    env = SpatialGym(config)
    
    # Test basic functionality
    print("- Checking visual mode...")
    assert env.text_based_mode == False, "Text-based mode should be disabled"
    assert env.ragen_adapter is None, "RAGEN adapter should not be created"
    print("✓ Visual mode confirmed")
    
    # Test system prompt
    print("- Testing system prompt...")
    system_prompt = env.system_prompt()
    assert isinstance(system_prompt, str), "System prompt should be a string"
    print(f"✓ System prompt: {system_prompt[:50]}...")
    
    # Note: We don't test reset/step for visual mode because it requires image data
    print("✓ VAGEN visual mode basic test passed!")
    return True

def main():
    """Run all integration tests."""
    print("RAGEN Integration Test for VAGEN2")
    print("=" * 40)
    
    try:
        # Test RAGEN text-based mode
        if not test_text_based_mode():
            print("\n✗ RAGEN text-based mode test failed!")
            sys.exit(1)
        
        # Test VAGEN visual mode 
        if not test_visual_mode():
            print("\n✗ VAGEN visual mode test failed!")
            sys.exit(1)
        
        print("\n" + "=" * 40)
        print("✓ All integration tests passed!")
        print("RAGEN spatial reasoning is successfully integrated into VAGEN2!")
        print("\nNext steps:")
        print("1. Run: python scripts/spatial_run_ragen.py --tasks ActiveRot --num 2")
        print("2. Check the results in the output directory")
        
    except Exception as e:
        print(f"\n✗ Integration test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

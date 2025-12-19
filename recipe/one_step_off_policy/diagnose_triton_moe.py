#!/usr/bin/env python3
"""
Diagnostic script to check Triton and Transformer Engine setup for MoE permutation kernels.
Run this to diagnose Triton kernel compilation issues.
"""

import sys
import subprocess

def run_command(cmd):
    """Run a command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except Exception as e:
        return "", str(e), -1

def check_triton():
    """Check Triton installation and version."""
    print("=" * 60)
    print("Checking Triton Installation")
    print("=" * 60)
    
    stdout, stderr, code = run_command("pip show triton")
    if code == 0:
        print(stdout)
    else:
        print(f"ERROR: Could not check Triton version: {stderr}")
        return False
    
    # Try importing triton
    try:
        import triton
        print(f"\n✓ Triton imported successfully")
        print(f"  Version: {triton.__version__}")
        return True
    except ImportError as e:
        print(f"\n✗ Failed to import Triton: {e}")
        return False

def check_transformer_engine():
    """Check Transformer Engine installation."""
    print("\n" + "=" * 60)
    print("Checking Transformer Engine Installation")
    print("=" * 60)
    
    stdout, stderr, code = run_command("pip show transformer-engine")
    if code == 0:
        print(stdout)
    else:
        print(f"ERROR: Could not check Transformer Engine version: {stderr}")
        return False
    
    # Try importing and checking MoE permutation functions
    try:
        import transformer_engine.pytorch.permutation as te_perm
        print(f"\n✓ Transformer Engine imported successfully")
        
        # Check if functions are available
        if hasattr(te_perm, 'moe_permute_with_probs'):
            print(f"  ✓ moe_permute_with_probs is available")
        else:
            print(f"  ✗ moe_permute_with_probs is NOT available")
            
        return True
    except ImportError as e:
        print(f"\n✗ Failed to import Transformer Engine: {e}")
        return False

def check_cuda():
    """Check CUDA installation."""
    print("\n" + "=" * 60)
    print("Checking CUDA Installation")
    print("=" * 60)
    
    stdout, stderr, code = run_command("nvidia-smi")
    if code == 0:
        print(stdout)
    else:
        print(f"WARNING: nvidia-smi failed: {stderr}")
    
    # Check CUDA version in PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n✓ CUDA available in PyTorch")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name}")
                print(f"    Compute Capability: {props.major}.{props.minor}")
        else:
            print(f"\n✗ CUDA not available in PyTorch")
            return False
        return True
    except Exception as e:
        print(f"\n✗ Error checking CUDA: {e}")
        return False

def check_triton_cache():
    """Check Triton cache."""
    print("\n" + "=" * 60)
    print("Checking Triton Cache")
    print("=" * 60)
    
    import os
    from pathlib import Path
    
    cache_dir = Path.home() / ".triton" / "cache"
    if cache_dir.exists():
        print(f"✓ Triton cache directory exists: {cache_dir}")
        cache_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
        print(f"  Cache size: {cache_size / (1024**2):.2f} MB")
        print(f"\n  To clear cache, run: rm -rf {cache_dir}")
    else:
        print(f"ℹ Triton cache directory does not exist: {cache_dir}")
        print(f"  (This is normal for first run)")

def test_moe_permute_kernel():
    """Test the MoE permutation kernel."""
    print("\n" + "=" * 60)
    print("Testing MoE Permutation Kernel")
    print("=" * 60)
    
    try:
        import torch
        import transformer_engine.pytorch.permutation as te_perm
        
        if not torch.cuda.is_available():
            print("✗ CUDA not available, skipping kernel test")
            return False
        
        # Create test tensors
        num_tokens = 128
        hidden_size = 512
        num_experts = 8
        
        tokens = torch.randn(num_tokens, hidden_size, device='cuda', dtype=torch.float16)
        probs = torch.randn(num_tokens, num_experts, device='cuda', dtype=torch.float32)
        probs = torch.softmax(probs, dim=-1)
        
        # Create a simple routing map (each token goes to expert 0)
        routing_map = torch.zeros(num_tokens, num_experts, device='cuda', dtype=torch.bool)
        routing_map[:, 0] = True
        
        print(f"Testing with:")
        print(f"  num_tokens: {num_tokens}")
        print(f"  hidden_size: {hidden_size}")
        print(f"  num_experts: {num_experts}")
        
        try:
            result = te_perm.moe_permute_with_probs(
                tokens, probs, routing_map, num_out_tokens=num_tokens
            )
            print(f"\n✓ Kernel test PASSED")
            print(f"  Output shape: {result[0].shape if isinstance(result, tuple) else result.shape}")
            return True
        except Exception as e:
            print(f"\n✗ Kernel test FAILED")
            print(f"  Error: {e}")
            print(f"  Error type: {type(e).__name__}")
            import traceback
            print(f"\n  Full traceback:")
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"\n✗ Failed to test kernel: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all diagnostics."""
    print("Triton MoE Permutation Kernel Diagnostic Tool")
    print("=" * 60)
    
    results = {
        "Triton": check_triton(),
        "Transformer Engine": check_transformer_engine(),
        "CUDA": check_cuda(),
    }
    
    check_triton_cache()
    
    # Only test kernel if prerequisites are met
    if all(results.values()):
        results["Kernel Test"] = test_moe_permute_kernel()
    else:
        print("\n" + "=" * 60)
        print("Skipping kernel test due to missing prerequisites")
        print("=" * 60)
        results["Kernel Test"] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name}: {status}")
    
    if not all(results.values()):
        print("\nRecommendations:")
        if not results["Triton"]:
            print("  - Install/upgrade Triton: pip install --upgrade triton")
        if not results["Transformer Engine"]:
            print("  - Install/upgrade Transformer Engine: pip install --upgrade transformer-engine")
        if not results["CUDA"]:
            print("  - Ensure CUDA is properly installed and accessible")
        if not results.get("Kernel Test", True):
            print("  - Clear Triton cache: rm -rf ~/.triton/cache/")
            print("  - Check GPU compute capability compatibility")
            print("  - Verify CUDA toolkit version matches PyTorch CUDA version")
    
    return 0 if all(results.values()) else 1

if __name__ == "__main__":
    sys.exit(main())


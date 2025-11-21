# Development Test Scripts

This directory contains test scripts created during hip_nl development for diagnosing GPU initialization and HIP runtime issues. **These will be valuable for fixing PyTorch+ROCm initialization issues in pytest.**

## Current Status

All scripts in this directory currently fail with **CUDA driver error 100** because they attempt to use PyTorch's CUDA/HIP backend directly:
- Calling `torch.cuda.get_device_name()` or `torch.cuda.get_device_properties()`
- Moving tensors to GPU with `.to('cuda:0')`
- Creating tensors directly on GPU device

The current hip_nl design **works around these issues** by accepting CPU tensors and managing GPU memory internally through HIP's native API. However, these tests will become important once we resolve the underlying PyTorch initialization timing problems.

## Scripts

### test_hip_nl_minimal.py
Minimal GPU test attempting to create tensors on `cuda:0` device. Created to isolate PyTorch GPU initialization timing issues.

### test_hip_nl_simple.py
Simple performance benchmark attempting to move tensors to GPU. Used to test correctness and measure 6.9x speedup on 2744-atom system.

### test_hip_nl_gpu.py
Full GPU test with device queries and performance benchmarking. Most comprehensive but also most prone to initialization issues.

### test_rocm_fallback.py
Tests ROCm patch functionality showing behavior with/without vesin. Not hip_nl specific but demonstrates fallback behavior.

## Working Tests

For functional hip_nl tests, see:
- **`verify_hip_nl.py`** (root directory) - Canonical working test script with all correctness tests
- **`test_hip_nl_basic.py`** (root directory) - Basic availability and API checks

## Future Use

Once PyTorch+ROCm initialization issues are resolved, these tests will be valuable for:
1. **Performance benchmarking** with GPU tensors directly
2. **Testing PyTorch device integration** (not just CPU tensors)
3. **Validating GPU memory management** improvements
4. **Comparing different initialization approaches**

These scripts demonstrate what hip_nl **should** be able to do once initialization is fixed. They're intentionally preserved as a roadmap for future improvements.

## Historical Value

During development, these scripts helped with:
1. Diagnosing HSA runtime symbol resolution issues (solved ✓)
2. Testing different GPU initialization approaches
3. Measuring 6.9x performance improvement
4. Understanding PyTorch+ROCm initialization timing in pytest (ongoing)

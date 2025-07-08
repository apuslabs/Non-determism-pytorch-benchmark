#!/usr/bin/env python3
"""
GPU Architecture Cross-Determinism Verification Script
Main script for testing deterministic operations across different GPU architectures

This tool helps identify potential non-deterministic behaviors that can affect
model reproducibility in machine learning workflows by testing various PyTorch
operations across different GPU architectures.
"""

# MUST set environment variables before importing torch
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
os.environ['PYTHONHASHSEED'] = '0'
# Disable certain optimizations for complete determinism
os.environ['CUDA_CACHE_DISABLE'] = '1'

import json
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import torch

# Constants
DEFAULT_SEED = 42
DEFAULT_REPEATS = 10
MATRIX_SIZE = 1024
BATCH_SIZE = 32
BATCH_MATRIX_SIZE = 256
CONV_BATCH_SIZE = 16
CONV_CHANNELS = 64
CONV_OUT_CHANNELS = 128
CONV_IMG_SIZE = 224
CONV_KERNEL_SIZE = 3
REDUCTION_SIZE = 1000
SORT_SIZE = 1000
TOPK_K = 100


def check_cuda_availability() -> None:
    """Check CUDA availability and potential determinism issues"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    
    print(f"✓ CUDA Version: {torch.version.cuda}")
    print(f"✓ PyTorch Version: {torch.__version__}")
    print(f"✓ GPU Device: {torch.cuda.get_device_name()}")
    print(f"✓ Compute Capability: {torch.cuda.get_device_capability()}")
    
    # Check for potential determinism issues
    if torch.cuda.get_device_capability()[0] < 6:
        warnings.warn("Older GPU architectures may have determinism issues")


def setup_deterministic_environment(seed: int = DEFAULT_SEED) -> None:
    """Configure complete deterministic environment
    
    Args:
        seed: Random seed for reproducibility
    """
    # Check CUDA availability
    check_cuda_availability()
    
    # Set basic random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # PyTorch deterministic settings
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # High precision settings - disable all possible optimizations
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    # Disable mixed precision optimizations for highest precision
    if hasattr(torch.backends.cuda.matmul, 'allow_fp16_reduced_precision_reduction'):
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    
    # Set floating point arithmetic precision
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('highest')

    print(f"✓ Complete deterministic setup finished, seed: {seed}")


def create_generator(seed: int, offset: int = 0) -> torch.Generator:
    """Create a CUDA generator with specified seed and offset
    
    Args:
        seed: Base seed value
        offset: Offset to add to seed for uniqueness
        
    Returns:
        torch.Generator configured for CUDA
    """
    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed + offset)
    return gen


def generate_test_data(seed: int = DEFAULT_SEED) -> Dict[str, torch.Tensor]:
    """Generate fixed test data with independent generators for each tensor
    
    Args:
        seed: Random seed for data generation
        
    Returns:
        Dictionary containing test tensors for different operations
    """
    data = {
        # Matrix multiplication data - using independent generators
        'matmul_a': torch.randn(
            MATRIX_SIZE, MATRIX_SIZE, 
            device='cuda', dtype=torch.float16, 
            generator=create_generator(seed, 0)
        ),
        'matmul_b': torch.randn(
            MATRIX_SIZE, MATRIX_SIZE, 
            device='cuda', dtype=torch.float16, 
            generator=create_generator(seed, 1)
        ),

        # Batch matrix multiplication
        'bmm_a': torch.randn(
            BATCH_SIZE, BATCH_MATRIX_SIZE, BATCH_MATRIX_SIZE, 
            device='cuda', dtype=torch.float16, 
            generator=create_generator(seed, 2)
        ),
        'bmm_b': torch.randn(
            BATCH_SIZE, BATCH_MATRIX_SIZE, BATCH_MATRIX_SIZE, 
            device='cuda', dtype=torch.float16, 
            generator=create_generator(seed, 3)
        ),

        # Convolution data
        'conv_input': torch.randn(
            CONV_BATCH_SIZE, CONV_CHANNELS, CONV_IMG_SIZE, CONV_IMG_SIZE, 
            device='cuda', dtype=torch.float16, 
            generator=create_generator(seed, 4)
        ),
        'conv_weight': torch.randn(
            CONV_OUT_CHANNELS, CONV_CHANNELS, CONV_KERNEL_SIZE, CONV_KERNEL_SIZE, 
            device='cuda', dtype=torch.float16, 
            generator=create_generator(seed, 5)
        ),

        # Reduction data - using float32 to reduce precision issues
        'reduction_input': torch.randn(
            REDUCTION_SIZE, REDUCTION_SIZE, 
            device='cuda', dtype=torch.float32, 
            generator=create_generator(seed, 6)
        ),

        # Sorting data with duplicate values (more likely to trigger non-determinism)
        'sort_input': torch.cat([
            torch.ones(SORT_SIZE // 2, device='cuda') * 0.5,
            torch.randn(SORT_SIZE // 2, device='cuda', generator=create_generator(seed, 7))
        ]),
    }

    return data


def execute_operation_with_sync(operation_func) -> np.ndarray:
    """Execute a PyTorch operation with proper synchronization and cache clearing
    
    Args:
        operation_func: Function that performs the PyTorch operation
        
    Returns:
        NumPy array containing the operation result
    """
    torch.cuda.empty_cache()  # Clear cache for consistency
    with torch.no_grad():  # Disable gradient computation
        result = operation_func()
        torch.cuda.synchronize()  # Ensure operation completion
        # Convert to appropriate precision then numpy
        if result.dtype in [torch.float16, torch.bfloat16]:
            return result.float().cpu().numpy()
        return result.cpu().numpy()


def run_test_operations(data: Dict[str, torch.Tensor], num_repeats: int = DEFAULT_REPEATS) -> Dict[str, List[Any]]:
    """Execute test operations and record results with error handling
    
    Args:
        data: Dictionary containing test tensors
        num_repeats: Number of times to repeat each operation
        
    Returns:
        Dictionary containing results for each operation
    """
    results = {}

    operations = {
        'matmul': lambda: torch.matmul(data['matmul_a'], data['matmul_b']),
        'bmm': lambda: torch.bmm(data['bmm_a'], data['bmm_b']),
        'conv2d': lambda: torch.nn.functional.conv2d(data['conv_input'], data['conv_weight']),
        'argmax': lambda: torch.argmax(data['reduction_input'], dim=-1),
    }

    try:
        # Standard operations
        for op_name, op_func in operations.items():
            print(f"Testing {op_name}...")
            op_results = []
            for i in range(num_repeats):
                result = execute_operation_with_sync(op_func)
                op_results.append(result)
            results[op_name] = op_results

        # TopK test (special case with multiple outputs)
        print("Testing topk...")
        topk_results = []
        for i in range(num_repeats):
            torch.cuda.empty_cache()
            with torch.no_grad():
                values, indices = torch.topk(data['sort_input'], k=TOPK_K)
                torch.cuda.synchronize()
                topk_results.append({
                    'values': values.cpu().numpy(),
                    'indices': indices.cpu().numpy()
                })
        results['topk'] = topk_results

    except Exception as e:
        print(f"❌ Error occurred during testing: {e}")
        raise

    return results


def get_gpu_info() -> Dict[str, Any]:
    """Get comprehensive GPU and environment information
    
    Returns:
        Dictionary containing GPU and environment details
    """
    return {
        'device_name': torch.cuda.get_device_name(),
        'compute_capability': torch.cuda.get_device_capability(),
        'cuda_version': torch.version.cuda,
        'pytorch_version': torch.__version__,
        'timestamp': datetime.now().isoformat(),
        'memory_allocated': torch.cuda.memory_allocated(),
        'memory_reserved': torch.cuda.memory_reserved(),
        'device_count': torch.cuda.device_count(),
        'current_device': torch.cuda.current_device(),
    }


def save_results(results: Dict[str, List[Any]], output_dir: str) -> None:
    """Save results to files with proper organization
    
    Args:
        results: Dictionary containing operation results
        output_dir: Directory path to save results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save GPU information
    gpu_info = get_gpu_info()
    with open(output_path / 'gpu_info.json', 'w', encoding='utf-8') as f:
        json.dump(gpu_info, f, indent=2)

    # Save operation results
    for op_name, op_results in results.items():
        if op_name == 'topk':
            # Special handling for TopK results
            for i, result in enumerate(op_results):
                np.savez(output_path / f'{op_name}_{i}.npz', **result)
        else:
            for i, result in enumerate(op_results):
                np.save(output_path / f'{op_name}_{i}.npy', result)

    print(f"✓ Results saved to {output_path}")


def generate_output_directory() -> str:
    """Generate a unique output directory name based on GPU and timestamp
    
    Returns:
        String path for the output directory
    """
    device_name = torch.cuda.get_device_name().replace(' ', '_').replace('/', '_')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"results/results_{device_name}_{timestamp}"


def main() -> None:
    """Main execution function"""
    print("🚀 Starting GPU Architecture Cross-Determinism Verification")
    print("=" * 60)
    
    try:
        # Setup deterministic environment
        setup_deterministic_environment(seed=DEFAULT_SEED)
        print()

        # Generate test data
        print("📊 Generating test data...")
        test_data = generate_test_data(seed=DEFAULT_SEED)
        print(f"✓ Generated test data for {len(test_data)} operation types")
        print()

        # Execute tests
        print("🔧 Running operations...")
        results = run_test_operations(test_data, num_repeats=DEFAULT_REPEATS)
        print(f"✓ Completed {sum(len(ops) for ops in results.values())} total operations")
        print()

        # Save results
        output_dir = generate_output_directory()
        save_results(results, output_dir)
        
        print("=" * 60)
        print("✅ Benchmark completed successfully!")
        print(f"📁 Results directory: {output_dir}")
        print("💡 Use the comparison scripts in README.md to analyze cross-architecture differences")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    main()

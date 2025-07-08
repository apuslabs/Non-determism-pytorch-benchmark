# PyTorch Non-Determinism Benchmark

A comprehensive benchmarking tool for testing and verifying deterministic behavior of PyTorch operations across different GPU architectures. This tool helps identify potential non-deterministic behaviors that can affect model reproducibility in machine learning workflows.

## Overview

This project addresses a critical challenge in machine learning reproducibility: ensuring consistent results across different GPU architectures. Even with identical seeds and deterministic settings, PyTorch operations may produce different results on different GPU hardware due to:

- Hardware-specific optimizations
- Different floating-point precision implementations
- Varied memory access patterns
- Architecture-specific CUDA kernel implementations

## Features

- **Comprehensive Operation Testing**: Tests matrix multiplication, batch operations, convolutions, reductions, and sorting operations
- **Strict Deterministic Environment**: Configures all known PyTorch and CUDA deterministic settings
- **Cross-Architecture Comparison**: Generates comparable results across different GPU architectures
- **Detailed Analysis**: Provides statistical analysis of differences between runs and architectures
- **Dockerized Environment**: Ensures consistent testing environment across different systems

## Quick Start

### Prerequisites

- NVIDIA GPU with CUDA support
- Docker with NVIDIA Container Toolkit
- Or: Python 3.12+ with PyTorch 2.7.1+ and CUDA 12.4+

### Using Docker (Recommended)

1. **Build the Docker image:**
   ```bash
   docker build -t pytorch-benchmark .
   ```

2. **Run the benchmark:**
   ```bash
   # Interactive development
   docker run -it --rm --gpus all -v $(pwd):/workspace -w /workspace pytorch-benchmark /bin/bash
   
   # Or direct execution
   docker run --rm --gpus all -v $(pwd):/workspace -w /workspace pytorch-benchmark python main.py
   ```

### Using Local Environment

1. **Install dependencies:**
   ```bash
   uv sync --frozen
   ```

2. **Run the benchmark:**
   ```bash
   python main.py
   ```

## How It Works

### 1. Deterministic Environment Setup

The tool configures a comprehensive deterministic environment:

```python
# Environment variables (set before importing torch)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_CACHE_DISABLE'] = '1'

# PyTorch deterministic settings
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```

### 2. Test Data Generation

Creates reproducible test data using independent random generators:

- **Matrix Operations**: 1024×1024 float16 matrices for testing matmul and batch operations
- **Convolutions**: 16×64×224×224 input with 128×64×3×3 kernels
- **Reductions**: 1000×1000 float32 matrices for argmax operations
- **Sorting**: Mixed data with duplicates to trigger potential non-determinism

### 3. Operation Testing

Executes multiple runs of each operation:

- **Matrix Multiplication** (`torch.matmul`)
- **Batch Matrix Multiplication** (`torch.bmm`)
- **2D Convolution** (`torch.nn.functional.conv2d`)
- **Argmax Reduction** (`torch.argmax`)
- **Top-K Selection** (`torch.topk`)

Each operation is run 10 times with cache clearing and GPU synchronization to ensure consistent timing and results.

### 4. Result Storage

Results are saved in a structured format:

```
results/
└── results_{GPU_NAME}_{TIMESTAMP}/
    ├── gpu_info.json          # Device and environment information
    ├── matmul_0.npy          # First run of matrix multiplication
    ├── matmul_1.npy          # Second run of matrix multiplication
    ├── ...                   # Additional runs for each operation
    └── topk_0.npz            # TopK results (values + indices)
```

## Comparing Results Across Architectures

### 1. Generate Results on Multiple GPUs

Run the benchmark on each GPU architecture you want to compare:

```bash
sudo docker run --rm --gpus all -v $(pwd):/workspace -w /workspace pytorch-determism python3 main.py
```

### 2. Analysis Script

```python
sudo docker run --rm --gpus all -v $(pwd):/workspace -w /workspace pytorch-determism python3 compare_results.py results/
results_NVIDIA_GeForce_RTX_4090_20250708_124452 results/results_NVIDIA_H100_PCIe_20250708_122219
```

### 3. Statistical Analysis

Key metrics to analyze:

- **Maximum Absolute Difference**: Largest difference between corresponding elements
- **Mean Absolute Difference**: Average difference across all elements
- **Different Elements Percentage**: Percentage of elements that differ beyond tolerance
- **Standard Deviation**: Variability in differences

### 4. Interpretation Guidelines

**Acceptable Differences:**
- Small floating-point precision differences (< 1e-6 for float32, < 1e-3 for float16)
- Consistent patterns across multiple runs

**Concerning Differences:**
- Large magnitude differences (> 1% of typical values)
- Inconsistent patterns between runs
- High percentage of differing elements (> 10%)

## Understanding Results

### Expected Behavior

- **Identical Architecture, Multiple Runs**: Should produce identical results with proper deterministic setup
- **Different Architectures**: May show small precision differences due to hardware implementation variations

### Common Issues

1. **High Differences in Matrix Operations**: Often due to different CUDA implementations of GEMM operations
2. **Argmax Inconsistencies**: Can occur with duplicate values in reduction operations
3. **TopK Variations**: May happen with ties in sorting operations

### Troubleshooting

If you observe unexpected non-determinism:

1. **Verify Environment Variables**: Ensure all environment variables are set before importing torch
2. **Check CUDA Version**: Different CUDA versions may have different deterministic implementations
3. **Review GPU Compute Capability**: Older GPUs (< 6.0) may have limited deterministic support
4. **Validate Docker Environment**: Use Docker to eliminate host system variations

## Project Structure

```
.
├── main.py                 # Main benchmark script
├── Dockerfile             # Docker environment setup
├── .dockerignore          # Docker build exclusions
├── pyproject.toml         # Python dependencies
├── uv.lock               # Locked dependency versions
├── comparison_report.md   # Example comparison report
└── README.md             # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add new operations or analysis methods
4. Submit a pull request with detailed description

## License

This project is open source. See LICENSE file for details.

## References

- [PyTorch Reproducibility Documentation](https://pytorch.org/docs/stable/notes/randomness.html)
- [CUDA Deterministic Operations](https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility)
- [GPU Architecture Differences in ML](https://developer.nvidia.com/blog/deep-learning-performance-documentation/)
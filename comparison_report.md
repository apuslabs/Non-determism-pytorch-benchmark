# GPU Architecture Cross-Determinism Verification Report

Generated Time: 2025-07-09T07:29:10

## Tested GPU Devices

| Device Name | Compute Capability | PyTorch Version | CUDA Version |
|-------------|-------------------|-----------------|--------------||
| NVIDIA GeForce RTX 4090 | [8, 9] | 2.6.0+cu124 | 12.4 |
| NVIDIA H100 PCIe | [9, 0] | 2.6.0+cu124 | 12.4 |
| Tesla T4 | [7, 5] | 2.6.0+cu124 | 12.4 |

## Analysis Summary

- **Total Comparisons**: 3
- **Operations Analyzed**: topk, argmax, matmul, conv2d, bmm
- **Perfect Matches**: 3
- **Significant Differences**: 12

## Cross-Architecture Difference Analysis

### NVIDIA GeForce RTX 4090_vs_NVIDIA H100 PCIe

| Operation | Max Absolute Difference | Average Absolute Difference | Different Elements (%) |
|-----------|------------------------|------------------------------|----------------------|
| matmul | 2.10e+02 | 3.53e+01 | 99.99% |
| bmm | 1.17e+02 | 1.64e+01 | 90.62% |
| conv2d | 1.98e+02 | 2.70e+01 | 99.99% |
| argmax | DIFFER | DIFFER | 80.20% |
| topk | 0.00e+00 | 0.00e+00 | 0.00% |

### NVIDIA GeForce RTX 4090_vs_Tesla T4

| Operation | Max Absolute Difference | Average Absolute Difference | Different Elements (%) |
|-----------|------------------------|------------------------------|----------------------|
| matmul | 2.27e+02 | 3.60e+01 | 99.99% |
| bmm | 1.20e+02 | 1.79e+01 | 99.99% |
| conv2d | 2.00e+02 | 2.70e+01 | 99.99% |
| argmax | DIFFER | DIFFER | 95.70% |
| topk | 0.00e+00 | 0.00e+00 | 0.00% |

### NVIDIA H100 PCIe_vs_Tesla T4

| Operation | Max Absolute Difference | Average Absolute Difference | Different Elements (%) |
|-----------|------------------------|------------------------------|----------------------|
| matmul | 2.30e+02 | 3.60e+01 | 99.99% |
| bmm | 1.13e+02 | 1.79e+01 | 99.99% |
| conv2d | 1.95e+02 | 2.71e+01 | 99.99% |
| argmax | DIFFER | DIFFER | 95.80% |
| topk | 0.00e+00 | 0.00e+00 | 0.00% |

## Interpretation Guide

**Integer Operations (argmax, etc.):**
- Integer operations should show EXACT_MATCH across all architectures
- Any differences in integer results indicate non-deterministic behavior
- These operations are critical for model reproducibility

**Floating-Point Operations:**
- Small floating-point precision differences (< 1e-6 for float32, < 1e-3 for float16)
- Consistent patterns across multiple runs

**Concerning Differences:**
- Any differences in integer operation results
- Large magnitude differences in floating-point operations (> 1% of typical values)
- Inconsistent patterns between runs
- High percentage of differing elements (> 10%)


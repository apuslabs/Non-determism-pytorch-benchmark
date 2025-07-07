#!/usr/bin/env python3
"""
GPU架构间确定性验证主脚本
文件名: deterministic_test.py
"""

import os
import torch
import numpy as np
import json
import random
from pathlib import Path
from datetime import datetime

def setup_deterministic_environment(seed=1234):
    """设置所有已知的确定性开关"""

    # 基础随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # PyTorch确定性设置
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Tensor Core相关
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # CUDA环境变量 (在脚本开始前设置)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    os.environ['PYTHONHASHSEED'] = '0'

    print(f"✓ Deterministic setup complete with seed {seed}")
    print(f"✓ CUDA Device: {torch.cuda.get_device_name()}")
    print(f"✓ Compute Capability: {torch.cuda.get_device_capability()}")

def generate_test_data(seed=1234):
    """生成固定的测试数据"""
    gen = torch.Generator(device='cuda').manual_seed(seed)

    data = {
        # 矩阵乘法数据
        'matmul_a': torch.randn(1024, 1024, device='cuda', dtype=torch.float16, generator=gen),
        'matmul_b': torch.randn(1024, 1024, device='cuda', dtype=torch.float16, generator=gen),

        # 批量矩阵乘法
        'bmm_a': torch.randn(32, 256, 256, device='cuda', dtype=torch.float16, generator=gen),
        'bmm_b': torch.randn(32, 256, 256, device='cuda', dtype=torch.float16, generator=gen),

        # 卷积数据
        'conv_input': torch.randn(16, 64, 224, 224, device='cuda', dtype=torch.float16, generator=gen),
        'conv_weight': torch.randn(128, 64, 3, 3, device='cuda', dtype=torch.float16, generator=gen),

        # 归约数据
        'reduction_input': torch.randn(1000, 1000, device='cuda', dtype=torch.float32, generator=gen),

        # 带有重复值的排序数据 (更容易触发非确定性)
        'sort_input': torch.cat([
            torch.ones(500, device='cuda') * 0.5,
            torch.randn(500, device='cuda', generator=gen)
        ]),
    }

    return data

def run_test_operations(data, num_repeats=10):
    """执行测试操作并记录结果"""
    results = {}

    # 矩阵乘法测试
    print("Testing matrix multiplication...")
    matmul_results = []
    for i in range(num_repeats):
        result = torch.matmul(data['matmul_a'], data['matmul_b'])
        matmul_results.append(result.cpu().numpy())
        torch.cuda.synchronize()
    results['matmul'] = matmul_results

    # 批量矩阵乘法测试
    print("Testing batch matrix multiplication...")
    bmm_results = []
    for i in range(num_repeats):
        result = torch.bmm(data['bmm_a'], data['bmm_b'])
        bmm_results.append(result.cpu().numpy())
        torch.cuda.synchronize()
    results['bmm'] = bmm_results

    # 卷积测试
    print("Testing convolution...")
    conv_results = []
    for i in range(num_repeats):
        result = torch.nn.functional.conv2d(data['conv_input'], data['conv_weight'])
        conv_results.append(result.cpu().numpy())
        torch.cuda.synchronize()
    results['conv2d'] = conv_results

    # 归约操作测试
    print("Testing argmax...")
    argmax_results = []
    for i in range(num_repeats):
        result = torch.argmax(data['reduction_input'], dim=-1)
        argmax_results.append(result.cpu().numpy())
        torch.cuda.synchronize()
    results['argmax'] = argmax_results

    # TopK测试
    print("Testing topk...")
    topk_results = []
    for i in range(num_repeats):
        values, indices = torch.topk(data['sort_input'], k=100)
        topk_results.append({
            'values': values.cpu().numpy(),
            'indices': indices.cpu().numpy()
        })
        torch.cuda.synchronize()
    results['topk'] = topk_results

    return results

def save_results(results, output_dir):
    """保存结果到文件"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 获取GPU信息
    gpu_info = {
        'device_name': torch.cuda.get_device_name(),
        'compute_capability': torch.cuda.get_device_capability(),
        'timestamp': datetime.now().isoformat(),
        'pytorch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
    }

    # 保存GPU信息
    with open(output_path / 'gpu_info.json', 'w') as f:
        json.dump(gpu_info, f, indent=2)

    # 保存各操作结果
    for op_name, op_results in results.items():
        if op_name == 'topk':
            # TopK结果特殊处理
            for i, result in enumerate(op_results):
                np.savez(output_path / f'{op_name}_{i}.npz', **result)
        else:
            for i, result in enumerate(op_results):
                np.save(output_path / f'{op_name}_{i}.npy', result)

    print(f"Results saved to {output_path}")

def main():
    # 设置确定性环境
    setup_deterministic_environment(seed=42)

    # 生成测试数据
    print("Generating test data...")
    test_data = generate_test_data(seed=42)

    # 执行测试
    print("Running operations...")
    results = run_test_operations(test_data, num_repeats=10)

    # 保存结果
    device_name = torch.cuda.get_device_name().replace(' ', '_').replace('/', '_')
    output_dir = f"results/results_{device_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_results(results, output_dir)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
GPU架构间确定性验证主脚本
文件名: deterministic_test.py
"""

# 必须在导入torch之前设置环境变量
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
os.environ['PYTHONHASHSEED'] = '0'
# 为了完全确定性，禁用某些优化
os.environ['CUDA_CACHE_DISABLE'] = '1'

import torch
import numpy as np
import json
import random
from pathlib import Path
from datetime import datetime
import warnings

def check_cuda_availability():
    """检查CUDA可用性和潜在问题"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用")
    
    print(f"✓ CUDA版本: {torch.version.cuda}")
    print(f"✓ PyTorch版本: {torch.__version__}")
    print(f"✓ GPU设备: {torch.cuda.get_device_name()}")
    print(f"✓ 计算能力: {torch.cuda.get_device_capability()}")
    
    # 检查潜在的确定性问题
    if torch.cuda.get_device_capability()[0] < 6:
        warnings.warn("较老的GPU架构可能存在确定性问题")

def setup_deterministic_environment(seed=1234):
    """设置完整的确定性环境"""
    
    # 检查CUDA
    check_cuda_availability()
    
    # 基础随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # PyTorch确定性设置
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 高精度设置 - 禁用所有可能的优化
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    # 为了确保最高精度，禁用混合精度优化
    if hasattr(torch.backends.cuda.matmul, 'allow_fp16_reduced_precision_reduction'):
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    
    # 设置浮点运算精度
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('highest')

    print(f"✓ 完整确定性设置完成，种子: {seed}")

def generate_test_data(seed=1234):
    """生成固定的测试数据，每个张量使用独立的generator"""
    
    # 为每个张量创建独立的generator，确保完全可重现
    def create_generator(offset=0):
        gen = torch.Generator(device='cuda')
        gen.manual_seed(seed + offset)
        return gen

    data = {
        # 矩阵乘法数据 - 使用独立的generator
        'matmul_a': torch.randn(1024, 1024, device='cuda', dtype=torch.float16, generator=create_generator(0)),
        'matmul_b': torch.randn(1024, 1024, device='cuda', dtype=torch.float16, generator=create_generator(1)),

        # 批量矩阵乘法
        'bmm_a': torch.randn(32, 256, 256, device='cuda', dtype=torch.float16, generator=create_generator(2)),
        'bmm_b': torch.randn(32, 256, 256, device='cuda', dtype=torch.float16, generator=create_generator(3)),

        # 卷积数据
        'conv_input': torch.randn(16, 64, 224, 224, device='cuda', dtype=torch.float16, generator=create_generator(4)),
        'conv_weight': torch.randn(128, 64, 3, 3, device='cuda', dtype=torch.float16, generator=create_generator(5)),

        # 归约数据 - 使用float32以减少精度问题
        'reduction_input': torch.randn(1000, 1000, device='cuda', dtype=torch.float32, generator=create_generator(6)),

        # 带有重复值的排序数据 (更容易触发非确定性)
        'sort_input': torch.cat([
            torch.ones(500, device='cuda') * 0.5,
            torch.randn(500, device='cuda', generator=create_generator(7))
        ]),
    }

    return data

def run_test_operations(data, num_repeats=10):
    """执行测试操作并记录结果，增加错误处理"""
    results = {}

    try:
        # 矩阵乘法测试
        print("Testing matrix multiplication...")
        matmul_results = []
        for i in range(num_repeats):
            torch.cuda.empty_cache()  # 清理缓存确保一致性
            with torch.no_grad():  # 禁用梯度计算
                result = torch.matmul(data['matmul_a'], data['matmul_b'])
                torch.cuda.synchronize()  # 确保操作完成
                # 转换为float32再转numpy，减少精度损失
                matmul_results.append(result.float().cpu().numpy())
        results['matmul'] = matmul_results

        # 批量矩阵乘法测试
        print("Testing batch matrix multiplication...")
        bmm_results = []
        for i in range(num_repeats):
            torch.cuda.empty_cache()
            with torch.no_grad():
                result = torch.bmm(data['bmm_a'], data['bmm_b'])
                torch.cuda.synchronize()
                bmm_results.append(result.float().cpu().numpy())
        results['bmm'] = bmm_results

        # 卷积测试
        print("Testing convolution...")
        conv_results = []
        for i in range(num_repeats):
            torch.cuda.empty_cache()
            with torch.no_grad():
                result = torch.nn.functional.conv2d(data['conv_input'], data['conv_weight'])
                torch.cuda.synchronize()
                conv_results.append(result.float().cpu().numpy())
        results['conv2d'] = conv_results

        # 归约操作测试
        print("Testing argmax...")
        argmax_results = []
        for i in range(num_repeats):
            torch.cuda.empty_cache()
            with torch.no_grad():
                result = torch.argmax(data['reduction_input'], dim=-1)
                torch.cuda.synchronize()
                argmax_results.append(result.cpu().numpy())
        results['argmax'] = argmax_results

        # TopK测试
        print("Testing topk...")
        topk_results = []
        for i in range(num_repeats):
            torch.cuda.empty_cache()
            with torch.no_grad():
                values, indices = torch.topk(data['sort_input'], k=100)
                torch.cuda.synchronize()
                topk_results.append({
                    'values': values.cpu().numpy(),
                    'indices': indices.cpu().numpy()
                })
        results['topk'] = topk_results

    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        raise

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

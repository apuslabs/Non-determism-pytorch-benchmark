#!/usr/bin/env python3
"""
跨架构结果对比脚本
文件名: compare_results.py
"""

import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

def load_results(result_dirs: List[str]) -> Dict:
    """加载多个GPU的结果"""
    all_results = {}

    for result_dir in result_dirs:
        path = Path(result_dir)
        if not path.exists():
            print(f"Warning: {result_dir} not found")
            continue

        # 加载GPU信息
        with open(path / 'gpu_info.json', 'r') as f:
            gpu_info = json.load(f)

        device_name = gpu_info['device_name']

        # 加载操作结果
        results = {}
        for op_name in ['matmul', 'bmm', 'conv2d', 'argmax']:
            op_results = []
            for i in range(10):  # 假设有10次重复
                file_path = path / f'{op_name}_{i}.npy'
                if file_path.exists():
                    op_results.append(np.load(file_path))
            if op_results:
                results[op_name] = op_results

        # 特殊处理TopK
        topk_results = []
        for i in range(10):
            file_path = path / f'topk_{i}.npz'
            if file_path.exists():
                topk_results.append(dict(np.load(file_path)))
        if topk_results:
            results['topk'] = topk_results

        all_results[device_name] = {
            'gpu_info': gpu_info,
            'results': results
        }

    return all_results

def compare_arrays(arr1: np.ndarray, arr2: np.ndarray, name: str) -> Dict:
    """比较两个数组的差异"""
    if arr1.shape != arr2.shape:
        return {
            'error': f"Shape mismatch: {arr1.shape} vs {arr2.shape}",
            'max_abs_diff': float('inf')
        }

    # 基本统计
    abs_diff = np.abs(arr1 - arr2)
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)

    # ULP差异 (仅适用于相同dtype)
    if arr1.dtype == arr2.dtype and np.issubdtype(arr1.dtype, np.floating):
        # 计算ULP差异的简化版本
        ulp_diff = abs_diff / np.finfo(arr1.dtype).eps
        max_ulp_diff = np.max(ulp_diff)
    else:
        max_ulp_diff = None

    # 不同元素百分比
    different_elements = np.sum(arr1 != arr2)
    different_percentage = different_elements / arr1.size * 100

    return {
        'max_abs_diff': float(max_abs_diff),
        'mean_abs_diff': float(mean_abs_diff),
        'max_ulp_diff': float(max_ulp_diff) if max_ulp_diff is not None else None,
        'different_elements': int(different_elements),
        'different_percentage': float(different_percentage),
        'total_elements': int(arr1.size)
    }

def analyze_consistency(all_results: Dict) -> Dict:
    """分析跨架构一致性"""
    device_names = list(all_results.keys())
    analysis = {}

    if len(device_names) < 2:
        print("Need at least 2 GPU results for comparison")
        return analysis

    # 逐对比较
    for i in range(len(device_names)):
        for j in range(i + 1, len(device_names)):
            device1, device2 = device_names[i], device_names[j]
            comparison_key = f"{device1}_vs_{device2}"

            print(f"\\n🔍 Comparing {device1} vs {device2}")
            print("=" * 60)

            comparison_results = {}

            # 比较每个操作
            for op_name in ['matmul', 'bmm', 'conv2d', 'argmax']:
                if (op_name in all_results[device1]['results'] and
                    op_name in all_results[device2]['results']):

                    results1 = all_results[device1]['results'][op_name]
                    results2 = all_results[device2]['results'][op_name]

                    op_comparisons = []
                    for rep_idx in range(min(len(results1), len(results2))):
                        comp = compare_arrays(
                            results1[rep_idx],
                            results2[rep_idx],
                            f"{op_name}_rep_{rep_idx}"
                        )
                        op_comparisons.append(comp)

                    comparison_results[op_name] = op_comparisons

                    # 打印摘要
                    max_diff = max(comp['max_abs_diff'] for comp in op_comparisons)
                    avg_different_pct = np.mean([comp['different_percentage'] for comp in op_comparisons])

                    status = "🟢 IDENTICAL" if max_diff == 0 else "🔴 DIFFERENT"
                    print(f"{op_name:12} | {status:12} | Max Diff: {max_diff:.2e} | Diff %: {avg_different_pct:.2f}%")

            # 特殊处理TopK
            if ('topk' in all_results[device1]['results'] and
                'topk' in all_results[device2]['results']):

                topk1 = all_results[device1]['results']['topk']
                topk2 = all_results[device2]['results']['topk']

                topk_comparisons = []
                for rep_idx in range(min(len(topk1), len(topk2))):
                    # 比较indices（更敏感）
                    indices_comp = compare_arrays(
                        topk1[rep_idx]['indices'],
                        topk2[rep_idx]['indices'],
                        f"topk_indices_rep_{rep_idx}"
                    )
                    topk_comparisons.append(indices_comp)

                comparison_results['topk'] = topk_comparisons

                max_diff = max(comp['different_percentage'] for comp in topk_comparisons)
                print(f"{'topk':12} | {'🔴 DIFFERENT' if max_diff > 0 else '🟢 IDENTICAL':12} | Indices Diff %: {max_diff:.2f}%")

            analysis[comparison_key] = comparison_results

    return analysis

def generate_report(all_results: Dict, analysis: Dict, output_file: str = "comparison_report.md"):
    """生成详细的比较报告"""
    with open(output_file, 'w') as f:
        f.write("# GPU架构间确定性验证报告\\n\\n")
        f.write(f"生成时间: {np.datetime64('now')}\\n\\n")

        # GPU信息表格
        f.write("## 参与测试的GPU设备\\n\\n")
        f.write("| 设备名称 | 计算能力 | PyTorch版本 | CUDA版本 |\\n")
        f.write("|----------|----------|-------------|----------|\\n")

        for device_name, device_data in all_results.items():
            info = device_data['gpu_info']
            f.write(f"| {device_name} | {info['compute_capability']} | {info['pytorch_version']} | {info['cuda_version']} |\\n")

        # 差异分析
        f.write("\\n## 跨架构差异分析\\n\\n")

        for comparison_key, comp_results in analysis.items():
            f.write(f"### {comparison_key}\\n\\n")
            f.write("| 操作 | 最大绝对差异 | 平均绝对差异 | 不同元素百分比 |\\n")
            f.write("|------|--------------|--------------|----------------|\\n")

            for op_name, op_comps in comp_results.items():
                max_abs_diff = max(comp['max_abs_diff'] for comp in op_comps)
                avg_abs_diff = np.mean([comp['mean_abs_diff'] for comp in op_comps])
                avg_diff_pct = np.mean([comp['different_percentage'] for comp in op_comps])

                f.write(f"| {op_name} | {max_abs_diff:.2e} | {avg_abs_diff:.2e} | {avg_diff_pct:.2f}% |\\n")

            f.write("\\n")

    print(f"📄 详细报告已保存至: {output_file}")

def main():
    import sys

    if len(sys.argv) < 3:
        print("Usage: python compare_results.py <result_dir1> <result_dir2> [result_dir3...]")
        sys.exit(1)

    result_dirs = sys.argv[1:]

    print("🔄 Loading results...")
    all_results = load_results(result_dirs)

    if len(all_results) < 2:
        print("❌ Need at least 2 valid result directories")
        sys.exit(1)

    print("🔍 Analyzing consistency...")
    analysis = analyze_consistency(all_results)

    print("📄 Generating report...")
    generate_report(all_results, analysis)

    print("\\n✅ Analysis complete!")

if __name__ == "__main__":
    main()

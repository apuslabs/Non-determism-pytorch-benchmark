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
import warnings

def validate_result_directory(result_dir: str) -> bool:
    """验证结果目录的完整性"""
    path = Path(result_dir)
    if not path.exists():
        print(f"❌ 目录不存在: {result_dir}")
        return False
    
    # 检查必要文件
    gpu_info_file = path / 'gpu_info.json'
    if not gpu_info_file.exists():
        print(f"❌ 缺少GPU信息文件: {gpu_info_file}")
        return False
    
    # 检查操作结果文件
    required_ops = ['matmul', 'bmm', 'conv2d', 'argmax', 'topk']
    missing_files = []
    
    for op_name in required_ops:
        for i in range(10):  # 假设有10次重复
            if op_name == 'topk':
                file_path = path / f'{op_name}_{i}.npz'
            else:
                file_path = path / f'{op_name}_{i}.npy'
            
            if not file_path.exists():
                missing_files.append(str(file_path))
    
    if missing_files:
        print(f"⚠️ 警告: 缺少部分结果文件: {len(missing_files)} 个文件")
        if len(missing_files) <= 5:  # 只显示前5个
            for file in missing_files[:5]:
                print(f"  - {file}")
    
    return True

def load_results(result_dirs: List[str]) -> Dict:
    """加载多个GPU的结果，增加错误处理"""
    all_results = {}

    for result_dir in result_dirs:
        if not validate_result_directory(result_dir):
            continue
            
        path = Path(result_dir)
        
        try:
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
                        try:
                            data = np.load(file_path)
                            # 验证数据完整性
                            if data.size == 0:
                                print(f"⚠️ 警告: 空数据文件 {file_path}")
                                continue
                            if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                                print(f"⚠️ 警告: 检测到NaN或Inf值在 {file_path}")
                            op_results.append(data)
                        except Exception as e:
                            print(f"❌ 加载文件失败 {file_path}: {e}")
                            continue
                            
                if op_results:
                    results[op_name] = op_results
                else:
                    print(f"⚠️ 警告: 没有找到 {op_name} 的有效结果")

            # 特殊处理TopK
            topk_results = []
            for i in range(10):
                file_path = path / f'topk_{i}.npz'
                if file_path.exists():
                    try:
                        topk_data = dict(np.load(file_path))
                        # 验证TopK数据完整性
                        if 'values' not in topk_data or 'indices' not in topk_data:
                            print(f"⚠️ 警告: TopK文件格式不正确 {file_path}")
                            continue
                        topk_results.append(topk_data)
                    except Exception as e:
                        print(f"❌ 加载TopK文件失败 {file_path}: {e}")
                        continue
                        
            if topk_results:
                results['topk'] = topk_results

            all_results[device_name] = {
                'gpu_info': gpu_info,
                'results': results
            }
            
            print(f"✅ 成功加载 {device_name} 的结果 ({len(results)} 个操作)")
            
        except Exception as e:
            print(f"❌ 加载结果目录 {result_dir} 时发生错误: {e}")
            continue

    return all_results

def compare_arrays(arr1: np.ndarray, arr2: np.ndarray, name: str, tolerance: float = 1e-6) -> Dict:
    """比较两个数组的差异，增加容错处理"""
    try:
        if arr1.shape != arr2.shape:
            return {
                'error': f"形状不匹配: {arr1.shape} vs {arr2.shape}",
                'max_abs_diff': float('inf'),
                'comparison_failed': True
            }

        # 检查数据类型兼容性
        if arr1.dtype != arr2.dtype:
            print(f"⚠️ 警告: 数据类型不匹配 {name}: {arr1.dtype} vs {arr2.dtype}")
            # 转换为公共类型
            common_dtype = np.result_type(arr1.dtype, arr2.dtype)
            arr1 = arr1.astype(common_dtype)
            arr2 = arr2.astype(common_dtype)

        # 基本统计
        abs_diff = np.abs(arr1 - arr2)
        max_abs_diff = np.max(abs_diff)
        mean_abs_diff = np.mean(abs_diff)
        
        # 相对误差
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rel_diff = abs_diff / (np.abs(arr1) + 1e-8)  # 避免除零
            max_rel_diff = np.max(rel_diff)

        # ULP差异 (仅适用于浮点数)
        max_ulp_diff = None
        if np.issubdtype(arr1.dtype, np.floating):
            try:
                # 计算ULP差异的简化版本
                ulp_diff = abs_diff / np.finfo(arr1.dtype).eps
                max_ulp_diff = np.max(ulp_diff)
            except:
                pass

        # 不同元素统计
        tolerance_mask = abs_diff > tolerance
        different_elements = np.sum(tolerance_mask)
        different_percentage = different_elements / arr1.size * 100

        # 确定性评估
        is_identical = max_abs_diff == 0
        is_within_tolerance = max_abs_diff <= tolerance

        return {
            'max_abs_diff': float(max_abs_diff),
            'mean_abs_diff': float(mean_abs_diff),
            'max_rel_diff': float(max_rel_diff),
            'max_ulp_diff': float(max_ulp_diff) if max_ulp_diff is not None else None,
            'different_elements': int(different_elements),
            'different_percentage': float(different_percentage),
            'total_elements': int(arr1.size),
            'is_identical': bool(is_identical),
            'is_within_tolerance': bool(is_within_tolerance),
            'tolerance_used': float(tolerance),
            'comparison_failed': False
        }
        
    except Exception as e:
        return {
            'error': f"比较过程中发生错误: {str(e)}",
            'comparison_failed': True
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

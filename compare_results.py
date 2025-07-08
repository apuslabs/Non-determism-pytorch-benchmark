#!/usr/bin/env python3
"""
Cross-Architecture Result Comparison Script

This script analyzes and compares PyTorch operation results across different GPU 
architectures to identify deterministic behavior differences. It provides comprehensive
statistical analysis and generates detailed comparison reports.
"""

import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import matplotlib.pyplot as plt

# Analysis constants
DEFAULT_TOLERANCE = 1e-6
REQUIRED_OPERATIONS = ['matmul', 'bmm', 'conv2d', 'argmax', 'topk']
DEFAULT_REPEAT_COUNT = 10


def validate_result_directory(result_dir: str) -> bool:
    """Validate the integrity of a result directory
    
    Args:
        result_dir: Path to the result directory
        
    Returns:
        True if directory is valid, False otherwise
    """
    path = Path(result_dir)
    if not path.exists():
        print(f"❌ Directory does not exist: {result_dir}")
        return False
    
    # Check for essential files
    gpu_info_file = path / 'gpu_info.json'
    if not gpu_info_file.exists():
        print(f"❌ Missing GPU info file: {gpu_info_file}")
        return False
    
    # Check operation result files
    missing_files = []
    
    for op_name in REQUIRED_OPERATIONS:
        for i in range(DEFAULT_REPEAT_COUNT):
            if op_name == 'topk':
                file_path = path / f'{op_name}_{i}.npz'
            else:
                file_path = path / f'{op_name}_{i}.npy'
            
            if not file_path.exists():
                missing_files.append(str(file_path))
    
    if missing_files:
        print(f"⚠️  Warning: Missing some result files: {len(missing_files)} files")
        if len(missing_files) <= 5:  # Show only first 5
            for file in missing_files[:5]:
                print(f"  - {file}")
                
        # Allow validation to pass if only some files are missing
        if len(missing_files) > len(REQUIRED_OPERATIONS) * DEFAULT_REPEAT_COUNT * 0.5:
            print(f"❌ Too many missing files ({len(missing_files)}), validation failed")
            return False
    
    return True


def load_gpu_info(result_dir: Path) -> Optional[Dict[str, Any]]:
    """Load GPU information from result directory
    
    Args:
        result_dir: Path to result directory
        
    Returns:
        GPU information dictionary or None if failed
    """
    try:
        with open(result_dir / 'gpu_info.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Failed to load GPU info from {result_dir}: {e}")
        return None


def load_operation_results(result_dir: Path, op_name: str) -> List[np.ndarray]:
    """Load results for a specific operation
    
    Args:
        result_dir: Path to result directory
        op_name: Name of the operation
        
    Returns:
        List of numpy arrays containing operation results
    """
    results = []
    
    for i in range(DEFAULT_REPEAT_COUNT):
        if op_name == 'topk':
            file_path = result_dir / f'{op_name}_{i}.npz'
            if file_path.exists():
                try:
                    topk_data = dict(np.load(file_path))
                    if 'values' not in topk_data or 'indices' not in topk_data:
                        print(f"⚠️  Warning: Invalid TopK file format {file_path}")
                        continue
                    results.append(topk_data)
                except Exception as e:
                    print(f"❌ Failed to load TopK file {file_path}: {e}")
                    continue
        else:
            file_path = result_dir / f'{op_name}_{i}.npy'
            if file_path.exists():
                try:
                    data = np.load(file_path)
                    if data.size == 0:
                        print(f"⚠️  Warning: Empty data file {file_path}")
                        continue
                    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                        print(f"⚠️  Warning: NaN or Inf values detected in {file_path}")
                    results.append(data)
                except Exception as e:
                    print(f"❌ Failed to load file {file_path}: {e}")
                    continue
    
    return results


def load_results(result_dirs: List[str]) -> Dict[str, Dict[str, Any]]:
    """Load results from multiple GPU result directories
    
    Args:
        result_dirs: List of result directory paths
        
    Returns:
        Dictionary mapping device names to their results and info
    """
    all_results = {}

    for result_dir in result_dirs:
        if not validate_result_directory(result_dir):
            continue
            
        path = Path(result_dir)
        
        # Load GPU information
        gpu_info = load_gpu_info(path)
        if gpu_info is None:
            continue
            
        device_name = gpu_info['device_name']

        # Load operation results
        results = {}
        for op_name in REQUIRED_OPERATIONS:
            op_results = load_operation_results(path, op_name)
            if op_results:
                results[op_name] = op_results
            else:
                print(f"⚠️  Warning: No valid results found for {op_name}")

        all_results[device_name] = {
            'gpu_info': gpu_info,
            'results': results
        }
        
        print(f"✅ Successfully loaded results for {device_name} ({len(results)} operations)")

    return all_results


def calculate_array_statistics(arr1: np.ndarray, arr2: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive statistics for array comparison
    
    Args:
        arr1: First array
        arr2: Second array
        
    Returns:
        Dictionary containing various statistical measures
    """
    abs_diff = np.abs(arr1 - arr2)
    
    stats = {
        'max_abs_diff': float(np.max(abs_diff)),
        'mean_abs_diff': float(np.mean(abs_diff)),
        'std_abs_diff': float(np.std(abs_diff)),
        'median_abs_diff': float(np.median(abs_diff)),
    }
    
    # Relative error calculation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        denominator = np.maximum(np.abs(arr1), np.abs(arr2))
        rel_diff = np.divide(abs_diff, denominator, 
                           out=np.zeros_like(abs_diff), where=denominator != 0)
        stats['max_rel_diff'] = float(np.max(rel_diff))
        stats['mean_rel_diff'] = float(np.mean(rel_diff))
    
    # ULP difference for floating point types
    if np.issubdtype(arr1.dtype, np.floating):
        try:
            eps = np.finfo(arr1.dtype).eps
            ulp_diff = abs_diff / eps
            stats['max_ulp_diff'] = float(np.max(ulp_diff))
            stats['mean_ulp_diff'] = float(np.mean(ulp_diff))
        except:
            stats['max_ulp_diff'] = None
            stats['mean_ulp_diff'] = None
    else:
        stats['max_ulp_diff'] = None
        stats['mean_ulp_diff'] = None
    
    return stats


def compare_arrays(arr1: np.ndarray, arr2: np.ndarray, name: str, 
                  tolerance: float = DEFAULT_TOLERANCE) -> Dict[str, Any]:
    """Compare two arrays and analyze their differences
    
    Args:
        arr1: First array to compare
        arr2: Second array to compare
        name: Name identifier for the comparison
        tolerance: Tolerance threshold for difference analysis
        
    Returns:
        Dictionary containing comprehensive comparison results
    """
    try:
        if arr1.shape != arr2.shape:
            return {
                'error': f"Shape mismatch: {arr1.shape} vs {arr2.shape}",
                'max_abs_diff': float('inf'),
                'comparison_failed': True
            }

        # Handle data type compatibility
        if arr1.dtype != arr2.dtype:
            print(f"⚠️  Warning: Data type mismatch for {name}: {arr1.dtype} vs {arr2.dtype}")
            common_dtype = np.result_type(arr1.dtype, arr2.dtype)
            arr1 = arr1.astype(common_dtype)
            arr2 = arr2.astype(common_dtype)

        # Special handling for integer arrays (like argmax results)
        is_integer_type = np.issubdtype(arr1.dtype, np.integer)
        is_argmax_operation = 'argmax' in name.lower()
        
        if is_integer_type or is_argmax_operation:
            # For integer arrays, use exact comparison
            different_mask = arr1 != arr2
            different_elements = int(np.sum(different_mask))
            different_percentage = float(different_elements / arr1.size * 100)
            
            # For integer arrays, max_abs_diff is either 0 (identical) or inf (different)
            max_abs_diff = 0.0 if different_elements == 0 else float('inf')
            
            result = {
                'max_abs_diff': max_abs_diff,
                'mean_abs_diff': max_abs_diff,
                'std_abs_diff': 0.0 if different_elements == 0 else float('inf'),
                'median_abs_diff': 0.0 if different_elements == 0 else float('inf'),
                'max_rel_diff': 0.0 if different_elements == 0 else float('inf'),
                'mean_rel_diff': 0.0 if different_elements == 0 else float('inf'),
                'max_ulp_diff': None,
                'mean_ulp_diff': None,
                'different_elements': different_elements,
                'different_percentage': different_percentage,
                'total_elements': int(arr1.size),
                'is_identical': different_elements == 0,
                'is_within_tolerance': different_elements == 0,  # For integers, only exact match is acceptable
                'tolerance_used': tolerance,
                'comparison_failed': False,
                'data_type': str(arr1.dtype),
                'array_shape': arr1.shape,
                'is_integer_comparison': True
            }
            
            return result

        # Calculate comprehensive statistics for floating point arrays
        stats = calculate_array_statistics(arr1, arr2)
        
        # Tolerance-based analysis
        tolerance_mask = np.abs(arr1 - arr2) > tolerance
        different_elements = int(np.sum(tolerance_mask))
        different_percentage = float(different_elements / arr1.size * 100)

        # Determinism assessment
        is_identical = stats['max_abs_diff'] == 0
        is_within_tolerance = stats['max_abs_diff'] <= tolerance

        result = {
            **stats,
            'different_elements': different_elements,
            'different_percentage': different_percentage,
            'total_elements': int(arr1.size),
            'is_identical': is_identical,
            'is_within_tolerance': is_within_tolerance,
            'tolerance_used': tolerance,
            'comparison_failed': False,
            'data_type': str(arr1.dtype),
            'array_shape': arr1.shape,
            'is_integer_comparison': False
        }
        
        return result
        
    except Exception as e:
        return {
            'error': f"Comparison error: {str(e)}",
            'comparison_failed': True
        }


def compare_topk_results(topk1: Dict[str, np.ndarray], topk2: Dict[str, np.ndarray], 
                        name: str) -> Dict[str, Any]:
    """Compare TopK results (values and indices)
    
    Args:
        topk1: First TopK result dictionary
        topk2: Second TopK result dictionary
        name: Name identifier for the comparison
        
    Returns:
        Dictionary containing TopK comparison results
    """
    try:
        indices_comp = compare_arrays(topk1['indices'], topk2['indices'], f"{name}_indices")
        values_comp = compare_arrays(topk1['values'], topk2['values'], f"{name}_values")
        
        return {
            'indices_comparison': indices_comp,
            'values_comparison': values_comp,
            'indices_identical': indices_comp['is_identical'],
            'values_identical': values_comp['is_identical'],
            'comparison_failed': indices_comp['comparison_failed'] or values_comp['comparison_failed']
        }
    except Exception as e:
        return {
            'error': f"TopK comparison error: {str(e)}",
            'comparison_failed': True
        }


def analyze_consistency(all_results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Analyze cross-architecture consistency
    
    Args:
        all_results: Dictionary containing results from all devices
        
    Returns:
        Dictionary containing pairwise comparison analysis
    """
    device_names = list(all_results.keys())
    analysis = {}

    if len(device_names) < 2:
        print("❌ Need at least 2 GPU results for comparison")
        return analysis

    print(f"\n🔍 Analyzing consistency across {len(device_names)} devices")
    print("=" * 80)

    # Pairwise comparisons
    for i in range(len(device_names)):
        for j in range(i + 1, len(device_names)):
            device1, device2 = device_names[i], device_names[j]
            comparison_key = f"{device1}_vs_{device2}"

            print(f"\n📊 Comparing {device1} vs {device2}")
            print("-" * 60)

            comparison_results = {}

            # Compare standard operations
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

                    # Print summary
                    if op_comparisons:
                        valid_comparisons = [comp for comp in op_comparisons if not comp.get('comparison_failed', False)]
                        if valid_comparisons:
                            max_diff = max(comp.get('max_abs_diff', 0) for comp in valid_comparisons)
                            avg_different_pct = np.mean([comp.get('different_percentage', 0) for comp in valid_comparisons])
                            
                            # Check if this is an integer comparison (like argmax)
                            is_integer_comp = any(comp.get('is_integer_comparison', False) for comp in valid_comparisons)
                            
                            if is_integer_comp:
                                status = "🟢 IDENTICAL" if max_diff == 0 else "🔴 DIFFERENT"
                                diff_display = "EXACT" if max_diff == 0 else "DIFFER"
                                print(f"{op_name:12} | {status:12} | Type: INTEGER | Result: {diff_display} | Diff %: {avg_different_pct:.2f}%")
                            else:
                                status = "🟢 IDENTICAL" if max_diff == 0 else "🔴 DIFFERENT"
                                print(f"{op_name:12} | {status:12} | Max Diff: {max_diff:.2e} | Diff %: {avg_different_pct:.2f}%")
                        else:
                            max_diff = float('inf')  # All comparisons failed
                            avg_different_pct = 100.0
                            status = "🟢 IDENTICAL" if max_diff == 0 else "🔴 DIFFERENT"
                            print(f"{op_name:12} | {status:12} | Max Diff: {max_diff:.2e} | Diff %: {avg_different_pct:.2f}%")

            # Handle TopK comparisons
            if ('topk' in all_results[device1]['results'] and
                'topk' in all_results[device2]['results']):

                topk1 = all_results[device1]['results']['topk']
                topk2 = all_results[device2]['results']['topk']

                topk_comparisons = []
                for rep_idx in range(min(len(topk1), len(topk2))):
                    comp = compare_topk_results(
                        topk1[rep_idx],
                        topk2[rep_idx],
                        f"topk_rep_{rep_idx}"
                    )
                    topk_comparisons.append(comp)

                comparison_results['topk'] = topk_comparisons

                if topk_comparisons:
                    indices_identical = all(comp.get('indices_identical', False) for comp in topk_comparisons if not comp.get('comparison_failed', False))
                    status = "🟢 IDENTICAL" if indices_identical else "🔴 DIFFERENT"
                    print(f"{'topk':12} | {status:12} | Indices comparison")

            analysis[comparison_key] = comparison_results

    return analysis


def generate_summary_statistics(analysis: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Generate summary statistics from the analysis
    
    Args:
        analysis: Analysis results from analyze_consistency
        
    Returns:
        Dictionary containing summary statistics
    """
    summary = {
        'total_comparisons': len(analysis),
        'operations_analyzed': set(),
        'perfect_matches': 0,
        'significant_differences': 0,
    }
    
    for comparison_key, comp_results in analysis.items():
        for op_name, op_comps in comp_results.items():
            summary['operations_analyzed'].add(op_name)
            
            if op_name == 'topk':
                # Handle TopK differently
                perfect_match = all(comp.get('indices_identical', False) and comp.get('values_identical', False) 
                                  for comp in op_comps if not comp.get('comparison_failed', False))
            else:
                perfect_match = all(comp.get('is_identical', False) 
                                  for comp in op_comps if not comp.get('comparison_failed', False))
            
            if perfect_match:
                summary['perfect_matches'] += 1
            else:
                # Check for significant differences
                if op_name != 'topk':
                    valid_comps = [comp for comp in op_comps if not comp.get('comparison_failed', False)]
                    if valid_comps:
                        max_diff_pct = max(comp.get('different_percentage', 0) for comp in valid_comps)
                        if max_diff_pct > 10.0:  # More than 10% different
                            summary['significant_differences'] += 1
    
    summary['operations_analyzed'] = list(summary['operations_analyzed'])
    return summary


def generate_report(all_results: Dict[str, Dict[str, Any]], analysis: Dict[str, Dict[str, Any]], 
                   output_file: str = "comparison_report.md") -> None:
    """Generate a detailed comparison report
    
    Args:
        all_results: Results from all devices
        analysis: Analysis results
        output_file: Output file path for the report
    """
    summary = generate_summary_statistics(analysis)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# GPU Architecture Cross-Determinism Verification Report\n\n")
        f.write(f"Generated Time: {np.datetime64('now')}\n\n")

        # GPU information table
        f.write("## Tested GPU Devices\n\n")
        f.write("| Device Name | Compute Capability | PyTorch Version | CUDA Version |\n")
        f.write("|-------------|-------------------|-----------------|--------------||\n")

        for device_name, device_data in all_results.items():
            info = device_data['gpu_info']
            f.write(f"| {device_name} | {info['compute_capability']} | {info['pytorch_version']} | {info['cuda_version']} |\n")

        # Summary statistics
        f.write(f"\n## Analysis Summary\n\n")
        f.write(f"- **Total Comparisons**: {summary['total_comparisons']}\n")
        f.write(f"- **Operations Analyzed**: {', '.join(summary['operations_analyzed'])}\n")
        f.write(f"- **Perfect Matches**: {summary['perfect_matches']}\n")
        f.write(f"- **Significant Differences**: {summary['significant_differences']}\n\n")

        # Detailed difference analysis
        f.write("## Cross-Architecture Difference Analysis\n\n")

        for comparison_key, comp_results in analysis.items():
            f.write(f"### {comparison_key}\n\n")
            f.write("| Operation | Max Absolute Difference | Average Absolute Difference | Different Elements (%) |\n")
            f.write("|-----------|------------------------|------------------------------|----------------------|\n")

            for op_name, op_comps in comp_results.items():
                if op_name == 'topk':
                    # Special handling for TopK
                    indices_identical = all(comp.get('indices_identical', False) for comp in op_comps if not comp.get('comparison_failed', False))
                    status = "0.00e+00" if indices_identical else "DIFFERENT"
                    f.write(f"| {op_name} | {status} | {status} | {'0.00%' if indices_identical else 'Variable'} |\n")
                else:
                    valid_comps = [comp for comp in op_comps if not comp.get('comparison_failed', False)]
                    if valid_comps:
                        max_abs_diff = max(comp.get('max_abs_diff', 0) for comp in valid_comps)
                        avg_abs_diff = np.mean([comp.get('mean_abs_diff', 0) for comp in valid_comps])
                        avg_diff_pct = np.mean([comp.get('different_percentage', 0) for comp in valid_comps])
                        
                        # Check if this is an integer comparison
                        is_integer_comp = any(comp.get('is_integer_comparison', False) for comp in valid_comps)
                        
                        if is_integer_comp:
                            # For integer comparisons, show exact match status
                            exact_status = "EXACT_MATCH" if max_abs_diff == 0 else "DIFFER"
                            f.write(f"| {op_name} | {exact_status} | {exact_status} | {avg_diff_pct:.2f}% |\n")
                        else:
                            f.write(f"| {op_name} | {max_abs_diff:.2e} | {avg_abs_diff:.2e} | {avg_diff_pct:.2f}% |\n")
                    else:
                        f.write(f"| {op_name} | ALL_FAILED | ALL_FAILED | ALL_FAILED |\n")

            f.write("\n")

        # Interpretation guide
        f.write("## Interpretation Guide\n\n")
        f.write("**Integer Operations (argmax, etc.):**\n")
        f.write("- Integer operations should show EXACT_MATCH across all architectures\n")
        f.write("- Any differences in integer results indicate non-deterministic behavior\n")
        f.write("- These operations are critical for model reproducibility\n\n")
        f.write("**Floating-Point Operations:**\n")
        f.write("- Small floating-point precision differences (< 1e-6 for float32, < 1e-3 for float16)\n")
        f.write("- Consistent patterns across multiple runs\n\n")
        f.write("**Concerning Differences:**\n")
        f.write("- Any differences in integer operation results\n")
        f.write("- Large magnitude differences in floating-point operations (> 1% of typical values)\n")
        f.write("- Inconsistent patterns between runs\n")
        f.write("- High percentage of differing elements (> 10%)\n\n")

    print(f"📄 Detailed report saved to: {output_file}")


def main() -> None:
    """Main execution function for result comparison"""
    if len(sys.argv) < 3:
        print("Usage: python compare_results.py <result_dir1> <result_dir2> [result_dir3...]")
        print("\nExample:")
        print("  python compare_results.py results/results_RTX_4090_20250108_120000 results/results_H100_20250108_120100")
        sys.exit(1)

    result_dirs = sys.argv[1:]

    print("🚀 Starting GPU Cross-Architecture Result Comparison")
    print("=" * 60)

    print("🔄 Loading results...")
    all_results = load_results(result_dirs)

    if len(all_results) < 2:
        print("❌ Need at least 2 valid result directories for comparison")
        sys.exit(1)

    print(f"✅ Loaded results from {len(all_results)} devices")

    print("\n🔍 Analyzing consistency...")
    analysis = analyze_consistency(all_results)

    print("\n📄 Generating report...")
    generate_report(all_results, analysis)

    print("\n" + "=" * 60)
    print("✅ Analysis completed successfully!")
    print("📊 Check 'comparison_report.md' for detailed results")


if __name__ == "__main__":
    main()

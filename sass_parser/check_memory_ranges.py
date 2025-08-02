import os
import re
import json
import argparse
from memory_footprint_analyzer import MemoryFootprintAnalyzer
from utils import format_bytes

KERNEL_TRACE_MAP = {
    'write': 'write.txt',
    'vecAdd': 'vectoradd.txt',
    'readWriteKernel': 'readWrite.txt',
    'int_one_hot': 'int_one_hot.txt',
    'copy_element_kernel': 'copy_element.txt',
    'copy_elements_kernel': 'copy_elements.txt',
    'reverse_index_copy_kernel': 'reverse_index_copy.txt',
    'index_select_kernel': 'index_select.txt',
    'simple_cond_kernel': 'simple_cond.txt',
    'simple_cond_kernel_signed': 'simple_cond_signed.txt',
    'simple_cond_kernel_dd': 'simple_cond_dd.txt',
    'complex_cond_kernel': 'complex_cond.txt',
    'composite_conditions_kernel': 'composite_conditions.txt',
    'multiple_branches_for_loop_kernel': 'multiple_branches_for_loop.txt',
    'composite_loop_kernel': 'composite_loop.txt',
    'complex_loop_kernel': 'complex_loop.txt',
    'nested_for_loop_kernel': 'nested_for_loop.txt',
    'simple_loop_kernel': 'simple_loop.txt',
    'simple_loop_kernel_signed': 'simple_loop_signed.txt',
    'simple_loop_kernel_dd': 'simple_loop_dd.txt',
}

SASS_FILES = {
    'write': 'sass_code/write(float*)-71583.sass',
    'vecAdd': 'sass_code/vecAdd(float*, float*, float*, int)-71757.sass',
    'readWriteKernel': 'sass_code/readWriteKernel(float*, float*)-71676.sass',
    'int_one_hot': 'sass_code/int_one_hot(float*, short)-71851.sass',
    'copy_element_kernel': 'sass_code/copy_element_kernel(int*, int*, int)-37636.sass',
    'copy_elements_kernel': 'sass_code/copy_elements_kernel(int*, int*, int*, int)-37933.sass',
    'reverse_index_copy_kernel': 'sass_code/reverse_index_copy_kernel(int*, int*, int)-38146.sass',
    'index_select_kernel': 'sass_code/index_select_kernel(int*, int*, int*)-37465.sass',
    'simple_cond_kernel': 'sass_code/simple_cond_kernel(int*, bool)-11459.sass',
    'simple_cond_kernel_signed': 'sass_code/simple_cond_kernel_signed(int*, int)-12162.sass',
    'simple_cond_kernel_dd': 'sass_code/simple_cond_kernel_dd(int*, bool*)-3690.sass',
    'complex_cond_kernel': 'sass_code/complex_cond_kernel.sass',
    'composite_conditions_kernel': 'sass_code/composite_conditions_kernel(bool, bool, int*)-7031.sass',
    'simple_loop_kernel': 'sass_code/simple_loop_kernel(int*, int)-5775.sass',
    'simple_loop_kernel_dd': 'sass_code/simple_loop_kernel_dd(int*, int*)-5838.sass',
    'simple_loop_kernel_signed': 'sass_code/simple_loop_kernel_signed(int*, int)-5986.sass',
    'multiple_branches_for_loop_kernel': 'sass_code/multiple_branches_for_loop_kernel(int, int*, int*, int*)-5507.sass',
    'composite_loop_kernel': 'sass_code/composite_loop_kernel(int, int, int*)-5340.sass',
    'complex_loop_kernel': 'sass_code/complex_loop_kernel(int*, int)-5244.sass',
    'nested_for_loop_kernel': 'sass_code/nested_for_loop_kernel(int*, int, int)-9957.sass',
}

def parse_trace_file(trace_path):
    addresses = []
    for encoding in ['utf-8', 'latin-1', 'ascii']:
        try:
            with open(trace_path, 'r', encoding=encoding) as f:
                for line in f:
                    line = line.strip()
                    if not line or not '0x' in line:
                        continue
                    
                    if 'LAUNCH' in line:
                        continue
                        
                    if not any(op in line for op in ['LDG', 'STG', 'LDS', 'STS', 'LDL', 'STL']):
                        continue
                        
                    line_parts = line.split(' - ')
                    memory_op_index = -1
                    for i, part in enumerate(line_parts):
                        if any(op in part for op in ['LDG', 'STG', 'LDS', 'STS', 'LDL', 'STL']):
                            memory_op_index = i
                            break
                    
                    if memory_op_index >= 0 and memory_op_index + 1 < len(line_parts):
                        address_part = line_parts[memory_op_index + 1]
                        for addr_str in re.findall(r'0x[0-9a-fA-F]+', address_part):
                            addr = int(addr_str, 16)
                            if addr == 0:
                                continue
                            if addr > 0x1000000000:
                                addresses.append(addr)
            break 
        except UnicodeDecodeError:
            continue 
    return addresses

def extract_kernel_name_from_sass_filename(sass_filename):
    base_name = os.path.basename(sass_filename)
    
    if base_name.endswith('.sass'):
        base_name = base_name[:-5]
    
    if '(' in base_name:
        kernel_name = base_name.split('(')[0]
    elif '-' in base_name:
        kernel_name = base_name.split('-')[0]
    else:
        kernel_name = base_name
    
    return kernel_name

def analyze_single_kernel(analyzer, kernel_name, trace_file, sass_file):
    print(f"\n{'='*60}\nKernel: {kernel_name}\n{'='*60}")
    
    kernel_result = {
        "kernel_name": kernel_name,
        "trace_file": trace_file,
        "sass_file": sass_file,
        "error": None,
        "trace_data": {},
        "predicted_ranges": [],
        "comparison": {},
        "match_status": "unknown"
    }
    
    if not os.path.exists(trace_file):
        error_msg = f"Trace file not found: {trace_file}"
        print(error_msg)
        kernel_result["error"] = error_msg
        return kernel_result

    addresses = parse_trace_file(trace_file)
    unique_addresses = sorted(set(addresses))
    if not unique_addresses:
        error_msg = "No addresses found in trace."
        print(error_msg)
        kernel_result["error"] = error_msg
        return kernel_result

    min_addr = min(unique_addresses)
    max_addr = max(unique_addresses)
    trace_range = max_addr - min_addr + 4
    
    kernel_result["trace_data"] = {
        "unique_addresses_count": len(unique_addresses),
        "total_bytes": len(unique_addresses) * 4,
        "min_address": f"{min_addr:#x}",
        "max_address": f"{max_addr:#x}",
        "range_span": trace_range,
        "addresses": [f"{addr:#x}" for addr in unique_addresses] if len(unique_addresses) < 100 else f"Too many addresses ({len(unique_addresses)}), use trace file"
    }

    print(f"Trace: {trace_file}")
    print(f"  Unique addresses: {len(unique_addresses)}")
    print(f"  Total bytes (unique): {format_bytes(len(unique_addresses) * 4)}")
    
    if unique_addresses:
        print(f"  Trace memory range: {min_addr:#x} - {max_addr + 4:#x} ({format_bytes(trace_range)})")
        print(f"  Trace span: {format_bytes(max_addr - min_addr)} + 4 bytes")

    analysis = analyzer.analyze_kernel_memory(kernel_name, sass_file)
    if 'error' in analysis:
        error_msg = analysis['error']
        print(error_msg)
        kernel_result["error"] = error_msg
        return kernel_result

    memory_ranges = analysis['memory_ranges']['global']
    if not memory_ranges:
        error_msg = "No global memory ranges inferred."
        print(error_msg)
        kernel_result["error"] = error_msg
        return kernel_result

    total_pred_bytes = 0
    print(f"\nPredicted memory ranges:")
    for i, r in enumerate(memory_ranges):
        est = r.get('estimated_range', {})
        if not est:
            continue
        pred_bytes = est.get('total_bytes', 0)
        total_pred_bytes += pred_bytes
        
        range_data = {
            "range_id": i + 1,
            "total_bytes": pred_bytes,
            "access_pattern": est.get('pattern', 'unknown'),
            "base_pointer": est.get('base_pointer', 'unknown')
        }
        
        if 'min_offset' in est and 'max_offset' in est:
            range_data["min_address"] = f"{est['min_offset']:#x}"
            range_data["max_address"] = f"{est['max_offset']:#x}"
            range_data["range_span"] = est['max_offset'] - est['min_offset']
        
        kernel_result["predicted_ranges"].append(range_data)
        
        print(f"  Range {i+1}:")
        print(f"    Total bytes: {format_bytes(pred_bytes)}")
        if 'min_offset' in est and 'max_offset' in est:
            print(f"    Address range: {est['min_offset']:#x} - {est['max_offset']:#x}")
            range_span = est['max_offset'] - est['min_offset']
            print(f"    Range span: {format_bytes(range_span)}")
        if 'pattern' in est:
            print(f"    Access pattern: {est['pattern']}")
        if 'base_pointer' in est:
            print(f"    Base pointer: {est['base_pointer']}")
    
    if unique_addresses and total_pred_bytes > 0:
        trace_min = min(unique_addresses)
        trace_max = max(unique_addresses)
        print(f"\nTrace vs Predicted Comparison:")
        print(f"  Trace range:     {trace_min:#x} - {trace_max + 4:#x} ({format_bytes(trace_max - trace_min + 4)})")
        
        pred_min = float('inf')
        pred_max = 0
        for r in memory_ranges:
            est = r.get('estimated_range', {})
            if 'min_offset' in est and 'max_offset' in est:
                pred_min = min(pred_min, est['min_offset'])
                pred_max = max(pred_max, est['max_offset'])
        
        comparison_data = {
            "trace_min_address": f"{trace_min:#x}",
            "trace_max_address": f"{trace_max + 4:#x}",
            "trace_range_span": trace_max - trace_min + 4,
            "predicted_total_bytes": total_pred_bytes,
            "trace_total_bytes": len(unique_addresses) * 4
        }
        
        if pred_min != float('inf') and pred_max > 0:
            comparison_data["predicted_min_address"] = f"{pred_min:#x}"
            comparison_data["predicted_max_address"] = f"{pred_max:#x}"
            comparison_data["predicted_range_span"] = pred_max - pred_min
            
            print(f"  Predicted range: {pred_min:#x} - {pred_max:#x} ({format_bytes(pred_max - pred_min)})")
            
        kernel_result["comparison"] = comparison_data

    print(f"\nTotal predicted bytes (all ranges): {format_bytes(total_pred_bytes)}")
    print(f"Trace total bytes: {format_bytes(len(unique_addresses) * 4)}")
    
    tolerance = max(16, total_pred_bytes * 0.01)  
    
    bytes_match = abs(total_pred_bytes - len(unique_addresses) * 4) <= tolerance
    
    if bytes_match:
        print(f"  ✅ Total access size matches (±{tolerance:.0f} bytes).")
        kernel_result["match_status"] = "match"
    else:
        print(f"  ❌ Total access size mismatch: Trace {format_bytes(len(unique_addresses) * 4)}, Analyzer {format_bytes(total_pred_bytes)}")
        kernel_result["match_status"] = "mismatch"
    
    return kernel_result

def analyze_all_kernels(analyzer):
    """Analyze all kernels using the predefined mappings"""
    results = {
        "kernels": {},
        "summary": {
            "total_kernels": 0,
            "matching_kernels": 0,
            "mismatched_kernels": 0
        }
    }

    for kernel, sass_path in SASS_FILES.items():
        trace_file = os.path.join('traces', KERNEL_TRACE_MAP[kernel])
        kernel_result = analyze_single_kernel(analyzer, kernel, trace_file, sass_path)
        
        results["kernels"][kernel] = kernel_result
        results["summary"]["total_kernels"] += 1
        
        if kernel_result["match_status"] == "match":
            results["summary"]["matching_kernels"] += 1
        elif kernel_result["match_status"] == "mismatch":
            results["summary"]["mismatched_kernels"] += 1
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description='Compare memory ranges between trace files and SASS analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''Examples:
  # Analyze a single kernel with specific files
  python check_memory_ranges.py -k kernel_data.txt -t trace.txt -s kernel.sass -n kernel_name
  
  # Analyze a single kernel with automatic name extraction
  python check_memory_ranges.py -k kernel_data.txt -t trace.txt -s kernel.sass
  
  # Analyze all kernels using predefined mappings
  python check_memory_ranges.py -a
  
  # Analyze all kernels with custom kernel data file
  python check_memory_ranges.py -a -k custom_kernel_data.txt'''
    )
    
    parser.add_argument('-k', '--kernel-data', 
                        help='Path to kernel data file (default: toy_kernel_data.txt)')
    parser.add_argument('-t', '--trace-file',
                        help='Path to trace file (for single kernel analysis)')
    parser.add_argument('-s', '--sass-file',
                        help='Path to SASS file (for single kernel analysis)')
    parser.add_argument('-n', '--kernel-name',
                        help='Name of the kernel (if not provided, extracted from SASS filename)')
    parser.add_argument('-a', '--all-kernels', action='store_true',
                        help='Analyze all kernels using predefined mappings')
    
    args = parser.parse_args()
    
    if not args.all_kernels and (not args.trace_file or not args.sass_file):
        parser.error("For single kernel analysis, both --trace-file and --sass-file are required")
    
    if args.all_kernels and (args.trace_file or args.sass_file or args.kernel_name):
        parser.error("--all-kernels cannot be used with single kernel options")
    
    kernel_data_file = args.kernel_data or 'toy_kernel_data.txt'
    
    kernel_data_file = os.path.expanduser(os.path.abspath(kernel_data_file))
    
    if not os.path.exists(kernel_data_file):
        print(f"Error: Kernel data file not found: {kernel_data_file}")
        return 1
    
    analyzer = MemoryFootprintAnalyzer()
    analyzer.parse_kernel_data(kernel_data_file)
    
    if args.all_kernels:
        print(f"Analyzing all kernels with kernel data: {kernel_data_file}")
        results = analyze_all_kernels(analyzer)
        
        with open('memory_analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Total kernels analyzed: {results['summary']['total_kernels']}")
        print(f"Matching kernels: {results['summary']['matching_kernels']}")
        print(f"Mismatched kernels: {results['summary']['mismatched_kernels']}")
        print(f"Results saved to: memory_analysis_results.json")
        
    else:
        trace_file = os.path.expanduser(os.path.abspath(args.trace_file))
        sass_file = os.path.expanduser(os.path.abspath(args.sass_file))
        
        if not os.path.exists(trace_file):
            print(f"Error: Trace file not found: {trace_file}")
            return 1
        
        if not os.path.exists(sass_file):
            print(f"Error: SASS file not found: {sass_file}")
            return 1
        
        if args.kernel_name:
            kernel_name = args.kernel_name
        else:
            kernel_name = extract_kernel_name_from_sass_filename(sass_file)
            print(f"Extracted kernel name from filename: {kernel_name}")
        
        print(f"Using kernel data: {kernel_data_file}")
        print(f"Using trace file: {trace_file}")
        print(f"Using SASS file: {sass_file}")
        
        kernel_result = analyze_single_kernel(analyzer, kernel_name, trace_file, sass_file)
        
        results = {
            "kernels": {kernel_name: kernel_result},
            "summary": {
                "total_kernels": 1,
                "matching_kernels": 1 if kernel_result["match_status"] == "match" else 0,
                "mismatched_kernels": 1 if kernel_result["match_status"] == "mismatch" else 0
            }
        }
        
        output_file = f'memory_analysis_{kernel_name}.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        
        if kernel_result["match_status"] == "match":
            print("✅ Memory analysis matches trace data")
        elif kernel_result["match_status"] == "mismatch":
            print("❌ Memory analysis does not match trace data")
        else:
            print("❓ Unable to determine match status")
    
    return 0

if __name__ == "__main__":
    exit(main())
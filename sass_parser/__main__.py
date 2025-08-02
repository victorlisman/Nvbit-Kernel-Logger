from . import MemoryFootprintAnalyzer, format_bytes
import sys
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Analyze CUDA kernel memory footprint')
    parser.add_argument('--kernel-data', '-k', 
                       help='Path to kernel data file (default: toy_kernel_data.txt)',
                       default='toy_kernel_data.txt')
    parser.add_argument('--sass-file', '-s',
                       help='Path to SASS file to analyze')
    parser.add_argument('--kernel-name', '-n',
                       help='Name of the kernel (extracted from SASS filename if not provided)')
    parser.add_argument('--all-kernels', '-a',
                       action='store_true',
                       help='Analyze all predefined kernels (ignores --sass-file)')
    
    args = parser.parse_args()
    
    analyzer = MemoryFootprintAnalyzer()
    
    kernel_data_path = os.path.expanduser(args.kernel_data)
    kernel_data_path = os.path.abspath(kernel_data_path)
    
    if not os.path.exists(kernel_data_path):
        print(f"Error: Kernel data file '{kernel_data_path}' not found")
        sys.exit(1)
    
    analyzer.parse_kernel_data(kernel_data_path)
    
    if args.all_kernels:
        print("Running analysis on all predefined kernels...")
        analyze_all_kernels(analyzer)
    elif args.sass_file:
        sass_file_path = os.path.expanduser(args.sass_file)
        sass_file_path = os.path.abspath(sass_file_path)
        
        print(f"Running analysis on single SASS file: {sass_file_path}")
        
        if not os.path.exists(sass_file_path):
            print(f"Error: SASS file '{sass_file_path}' not found")
            sys.exit(1)
        
        kernel_name = args.kernel_name
        if not kernel_name:
            kernel_name = os.path.splitext(os.path.basename(sass_file_path))[0]
            if '-' in kernel_name and kernel_name.split('-')[-1].isdigit():
                kernel_name = '-'.join(kernel_name.split('-')[:-1])
            
            if '(' in kernel_name:
                kernel_name = kernel_name.split('(')[0].strip()
        print(f"Extracted kernel name: {kernel_name}")
        analyze_single_kernel(analyzer, kernel_name, sass_file_path)
    else:
        print("Error: Must specify either --sass-file or --all-kernels")
        parser.print_help()
        sys.exit(1)

def analyze_single_kernel(analyzer, kernel_name, sass_file):
    print(f"Analyzing single kernel: {kernel_name}")
    print(f"SASS file: {sass_file}")
    print(f"\n{'='*60}")
    print(f"DETAILED ANALYSIS: {kernel_name}")
    print(f"{'='*60}")
    
    analysis = analyzer.analyze_kernel_memory(kernel_name, sass_file)
    print_analysis_results(kernel_name, analysis)

def analyze_all_kernels(analyzer):
    sass_files = {
        'write': 'sass_code/write(float*)-71583.sass',
        'vecAdd': 'sass_code/vecAdd(float*, float*, float*, int)-71757.sass',
        'readWriteKernel': 'sass_code/readWriteKernel(float*, float*)-71676.sass',
        'int_one_hot': 'sass_code/int_one_hot(float*, short)-71851.sass',
        'one_hot_encode_kernel': 'sass_code/one_hot_encode_kernel(short*, float*, int, int)-71936.sass',
        'copy_element_kernel': 'sass_code/copy_element_kernel(int*, int*, int)-37636.sass',
        'copy_elements_kernel': 'sass_code/copy_elements_kernel(int*, int*, int*, int)-37933.sass',
        'reverse_index_copy_kernel': 'sass_code/reverse_index_copy_kernel(int*, int*, int)-38146.sass',
        'index_select_kernel': 'sass_code/index_select_kernel(int*, int*, int*)-37465.sass',
        'simple_cond_kernel': 'sass_code/simple_cond_kernel(int*, bool)-11459.sass',
        'simple_cond_kernel_signed': 'sass_code/simple_cond_kernel_signed(int*, int)-12162.sass',
        'simple_cond_kernel_dd': 'sass_code/simple_cond_kernel_dd(int*, bool*)-3690.sass',
        'complex_cond_kernel': 'sass_code/complex_cond_kernel.sass',
        'composite_conditions_kernel': 'sass_code/composite_conditions_kernel(bool, bool, int*)-7031.sass',
        'multiple_branches_kernel': 'sass_code/multiple_branches_kernel(int, int*, int*, int*)-8079.sass',
        'simple_loop_kernel': 'sass_code/simple_loop_kernel(int*, int)-5775.sass',
        'simple_loop_kernel_dd': 'sass_code/simple_loop_kernel_dd(int*, int*)-5838.sass',
        'simple_loop_kernel_signed': 'sass_code/simple_loop_kernel_signed(int*, int)-5986.sass',
        'multiple_branches_for_loop_kernel': 'sass_code/multiple_branches_for_loop_kernel(int, int*, int*, int*)-5507.sass',
        'composite_loop_kernel': 'sass_code/composite_loop_kernel(int, int, int*)-5340.sass',
        'complex_loop_kernel': 'sass_code/complex_loop_kernel(int*, int)-5244.sass',
        'nested_for_loop_kernel': 'sass_code/nested_for_loop_kernel(int*, int, int)-9957.sass',
    }
    
    for kernel_name, sass_file in sass_files.items():
        print(f"\n{'='*60}")
        print(f"DETAILED ANALYSIS: {kernel_name}")
        print(f"{'='*60}")
        
        analysis = analyzer.analyze_kernel_memory(kernel_name, sass_file)
        print_analysis_results(kernel_name, analysis)

def print_analysis_results(kernel_name, analysis):
    if 'error' not in analysis:
        kernel = analysis['kernel_info']
        print(f"Kernel: {kernel_name}")
        print(f"Grid: {kernel['grid_dim']}, Block: {kernel['block_dim']}")
        print(f"Total threads: {kernel['total_threads']}")
        
        print(f"\nMemory Access Patterns:")
        for i, access in enumerate(analysis['memory_accesses']):
            print(f"  [{i+1}] {access['type'].upper()} {access['scope']} memory:")
            print(f"      Instruction: {access['raw']}")
            print(f"      Size: {access['access_size']} bytes")
            for mem_op in access['memory_operands']:
                print(f"      Address: {mem_op['raw']} -> {mem_op['symbolic_expression']}")
        
        print(f"\nAddressing Expressions:")
        for expr, info in analysis['addressing_expressions'].items():
            print(f"  {expr}: used {info['count']} times ({info['access_type']} {info['scope']})")
        
        print(f"\nMemory Ranges:")
        for scope, ranges in analysis['memory_ranges'].items():
            if ranges:
                print(f"  {scope.upper()} memory:")
                for range_info in ranges:
                    print(f"    Expression: {range_info['expression']}")
                    if 'expression' in range_info:
                        print(f"    Symbolic address: {range_info['expression']}")
                    if 'estimated_range' in range_info:
                        est = range_info['estimated_range']
                        print(f"    Base pointer: {est.get('base_pointer', '0x0')}")
                        print(f"    Estimated range: {est['min_offset']:#x} - {est['max_offset']:#x}")
                        print(f"    Total bytes: {format_bytes(est['total_bytes'])}")
                        if 'pattern' in est:
                            print(f"    Access pattern: {est['pattern']}")
                        if 'condition_value' in est and 'condition_type' in est:
                            print(f"    Condition: {est['condition_value']} ({est['condition_type']})")
                        if 'kernel_type' in est:
                            print(f"    Kernel type: {est['kernel_type']}")
                        if 'address_set' in est:
                            print(f"    Accessed addresses: {', '.join(est['address_set'])}")
    else:
        print(f"Error analyzing kernel {kernel_name}: {analysis['error']}")

if __name__ == "__main__":
    main()
    sys.exit(0)
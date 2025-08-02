from typing import List, Dict, Optional
import re

try:
    from .sass_instruction import SassInstruction
    from .kernel_info import KernelInfo
    from .memory_analysis import KernelMemoryAnalysis
    from .utils import parse_sass_line
except ImportError:
    from sass_instruction import SassInstruction
    from kernel_info import KernelInfo
    from memory_analysis import KernelMemoryAnalysis
    from utils import parse_sass_line

class MemoryFootprintAnalyzer:
    def __init__(self):
        self.kernels = {}
        self.sass_instructions = {}
        self.kernel_analyses = {}
        self.type_sizes = {
            'char': 1, 'char*': 8, 'short': 2, 'short*': 8,
            'int': 4, 'int*': 8, 'long': 8, 'long*': 8,
            'float': 4, 'float*': 8, 'double': 8, 'double*': 8,
            'void*': 8, 'half': 2, 'half*': 8, '__half': 2, '__half*': 8,
        }
        
    def get_data_type_size(self, data_type: str) -> int:
        clean_type = data_type.strip().replace('const ', '').replace('volatile ', '')
        return self.type_sizes.get(clean_type, 4)
    
    def analyze_kernel_memory(self, kernel_name: str, sass_file: str) -> Dict:
        instructions = self.parse_sass_file(sass_file)
        self.sass_instructions[kernel_name] = instructions
        
        kernel_info = self.kernels.get(kernel_name)
        if not kernel_info:
            return {'error': f'No kernel info found for {kernel_name}'}
        
        analysis = KernelMemoryAnalysis(kernel_name, instructions, kernel_info)
        self.kernel_analyses[kernel_name] = analysis
        
        memory_ranges = analysis.get_memory_ranges()
        
        footprint = self.calculate_memory_footprint(kernel_name)
        
        return {
            'kernel_name': kernel_name,
            'kernel_info': {
                'total_threads': kernel_info.total_threads,
                'grid_dim': kernel_info.grid_dim,
                'block_dim': kernel_info.block_dim,
                'memory_args': kernel_info.get_memory_args()
            },
            'memory_accesses': analysis.memory_accesses,
            'addressing_expressions': analysis.address_expressions,
            'memory_ranges': memory_ranges,
            'memory_footprint': footprint
        }
    
    def calculate_memory_footprint(self, kernel_name: str) -> Dict:
        kernel_info = self.kernels.get(kernel_name)
        if not kernel_info:
            return {'error': f'No kernel info found for {kernel_name}'}
        
        instructions = self.sass_instructions.get(kernel_name, [])
        
        access_counts = {
            'global_loads': 0, 'global_stores': 0, 'shared_loads': 0,
            'shared_stores': 0, 'local_loads': 0, 'local_stores': 0
        }
        
        total_bytes_per_thread = {
            'global_load_bytes': 0, 'global_store_bytes': 0,
            'shared_load_bytes': 0, 'shared_store_bytes': 0,
            'local_load_bytes': 0, 'local_store_bytes': 0
        }
        
        for inst in instructions:
            mem_info = inst.get_memory_access_info()
            if mem_info:
                scope = mem_info['scope']
                access_type = mem_info['type']
                access_size = mem_info['access_size']
                
                key = f"{scope}_{access_type}s"
                if key in access_counts:
                    access_counts[key] += 1
                    
                byte_key = f"{scope}_{access_type}_bytes"
                if byte_key in total_bytes_per_thread:
                    total_bytes_per_thread[byte_key] += access_size
        
        total_memory_footprint = {}
        for key, bytes_per_thread in total_bytes_per_thread.items():
            total_memory_footprint[key] = bytes_per_thread * kernel_info.total_threads
        
        arg_memory_usage = []
        total_arg_memory = 0
        
        for arg in kernel_info.get_memory_args():
            if 'size' in arg or arg.get('access_type') in ['input', 'output', 'input_output']:
                base_type = arg['type'].replace('*', '')
                element_size = self.get_data_type_size(base_type)
                estimated_elements = kernel_info.total_threads
                estimated_size = estimated_elements * element_size
                
                arg_memory_usage.append({
                    'arg_num': arg['arg_num'], 'type': arg['type'],
                    'access_type': arg['access_type'], 'element_size': element_size,
                    'estimated_elements': estimated_elements, 'estimated_size_bytes': estimated_size
                })
                total_arg_memory += estimated_size
        
        return {
            'kernel_name': kernel_name, 'total_threads': kernel_info.total_threads,
            'grid_dim': kernel_info.grid_dim, 'block_dim': kernel_info.block_dim,
            'access_counts': access_counts, 'bytes_per_thread': total_bytes_per_thread,
            'total_memory_footprint': total_memory_footprint,
            'argument_memory_usage': arg_memory_usage,
            'total_argument_memory_bytes': total_arg_memory,
            'total_instruction_memory_bytes': sum(total_memory_footprint.values()),
            'combined_total_bytes': total_arg_memory + sum(total_memory_footprint.values())
        }
        
    def parse_kernel_data(self, kernel_data_file: str):
        with open(kernel_data_file, 'r') as f:
            content = f.read()
            
        kernel_blocks = content.split('Intercepted kernel launch:')[1:]
        
        for block in kernel_blocks:
            lines = block.strip().split('\n')
            if not lines:
                continue
                
            name_line = lines[0].strip()
            kernel_name = name_line.split('(')[0].strip()
            
            grid_dim = None
            block_dim = None
            args = []
            
            for line in lines[1:]:
                line = line.strip()
                if line.startswith('gridDim'):
                    grid_match = re.search(r'gridDim = \((\d+), (\d+), (\d+)\)', line)
                    if grid_match:
                        grid_dim = tuple(map(int, grid_match.groups()))
                elif line.startswith('blockDim'):
                    block_match = re.search(r'blockDim = \((\d+), (\d+), (\d+)\)', line)
                    if block_match:
                        block_dim = tuple(map(int, block_match.groups()))
                elif line.startswith('[') and 'Arg' in line:
                    arg_info = self.parse_argument_line(line)
                    if arg_info:
                        args.append(arg_info)
                        
            if grid_dim and block_dim:
                self.kernels[kernel_name] = KernelInfo(kernel_name, grid_dim, block_dim, args)
    
    def parse_argument_line(self, line: str) -> Optional[Dict]:
        arg_match = re.search(r'\[(.*?)\]\s+Arg\s+(\d+)\s+\((.*?)\):\s+(.*?)(?:,|$)', line)
        if arg_match:
            access_type, arg_num, data_type, value = arg_match.groups()
            arg_info = {
                'access_type': access_type, 'arg_num': int(arg_num),
                'type': data_type, 'value': value.strip()
            }
            indexes_match = re.search(r'indexes:\s*\{([^\}]*)\}', line)
            if indexes_match:
                indexes = [int(idx.strip()) for idx in indexes_match.group(1).split(',') if idx.strip()]
                arg_info['indexes'] = indexes
            return arg_info
        return None
    
    def parse_sass_file(self, sass_file: str) -> List[SassInstruction]:
        instructions = []
        with open(sass_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('//'):
                    instruction = parse_sass_line(line)
                    if instruction:
                        instructions.append(instruction)
        return instructions

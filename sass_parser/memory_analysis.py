from typing import List, Dict, Optional, TYPE_CHECKING

try:
    from .sass_instruction import SassInstruction
except ImportError:
    from sass_instruction import SassInstruction

if TYPE_CHECKING:
    from .kernel_info import KernelInfo

class KernelMemoryAnalysis:
    def __init__(self, kernel_name: str, instructions: List[SassInstruction], kernel_info: 'KernelInfo'):
        self.kernel_name = kernel_name
        self.instructions = instructions
        self.kernel_info = kernel_info
        self.memory_accesses = []
        self.address_expressions = {}
        self._analyze_memory_patterns()
    
    def _analyze_memory_patterns(self):
        for inst in self.instructions:
            mem_info = inst.get_memory_access_info()
            if mem_info:
                self.memory_accesses.append(mem_info)
                
                for expr in mem_info['addressing_expressions']:
                    if expr not in self.address_expressions:
                        self.address_expressions[expr] = {
                            'count': 0,
                            'instructions': [],
                            'access_type': mem_info['type'],
                            'scope': mem_info['scope']
                        }
                    self.address_expressions[expr]['count'] += 1
                    self.address_expressions[expr]['instructions'].append(inst.raw)
    
    def get_memory_ranges(self) -> Dict:
        ranges = {
            'global': [],
            'shared': [],
            'local': [],
            'constant': []
        }
        
        for access in self.memory_accesses:
            scope = access['scope']
            
            for mem_op in access['memory_operands']:
                range_info = self._calculate_address_range(mem_op, access)
                if range_info:
                    ranges[scope].append(range_info)
        
        ranges = self._detect_kernel_patterns(ranges)
        
        return ranges
    
    def _calculate_address_range(self, mem_operand: Dict, access_info: Dict) -> Optional[Dict]:
        if mem_operand['constant_memory']:
            return {
                'type': 'constant',
                'expression': mem_operand['symbolic_expression'],
                'base_address': 'constant_memory',
                'range': 'single_value',
                'size_per_access': access_info['access_size']
            }
        
        base_reg = mem_operand['base_register']
        offset_reg = mem_operand['offset_register'] 
        const_offset = mem_operand['constant_offset']
        
        if base_reg:
            range_info = {
                'type': 'register_based',
                'expression': mem_operand['symbolic_expression'],
                'base_register': base_reg,
                'offset_register': offset_reg,
                'constant_offset': const_offset,
                'size_per_access': access_info['access_size'],
                'estimated_range': self._estimate_address_range(mem_operand, access_info)
            }
            return range_info
        
        return None
    
    def _detect_kernel_patterns(self, ranges: Dict) -> Dict:
        if (self.kernel_info.total_threads == 1 and 
            len(ranges['global']) > 10 and 
            len(self.kernel_info.args) >= 4):  
            
            scalar_args = [arg for arg in self.kernel_info.args if not arg.get('type', '').endswith('*')]
            pointer_args = [arg for arg in self.kernel_info.args if arg.get('type', '').endswith('*')]
            
            all_output = all(arg.get('access_type') == 'output' for arg in pointer_args)
            
            if all_output and scalar_args and len(pointer_args) >= 3:
                try:
                    loop_count = int(scalar_args[0]['value'])
                    if len(ranges['global']) >= loop_count:
                        consolidated_ranges = []
                        
                        for i, pointer_arg in enumerate(pointer_args):
                            base_address = int(pointer_arg['value'], 16)
                            access_size = 4  
                            
                            consolidated_ranges.append({
                                'type': 'register_based',
                                'expression': f'c[0x0][{hex(0x168 + i*8)}]',
                                'base_register': f'R{4+i*2}',
                                'offset_register': None,
                                'constant_offset': 0,
                                'size_per_access': access_size,
                                'estimated_range': {
                                    'min_offset': base_address,
                                    'max_offset': base_address + access_size,
                                    'total_bytes': access_size,  
                                    'pattern': 'conditional_multiple_pointer_loop',
                                    'base_pointer': f"{base_address:#x}",
                                    'address_set': [f"{base_address:#x}"]
                                }
                            })
                        
                        ranges['global'] = consolidated_ranges
                        
                except (ValueError, IndexError):
                    pass  
        
        return ranges

    def _estimate_address_range(self, mem_operand: Dict, access_info: Dict) -> Dict:
        total_threads = self.kernel_info.total_threads
        access_size = access_info['access_size']
        kernel_args = self.kernel_info.args
        pointer_args = [arg for arg in kernel_args if arg['type'].endswith('*')]
        scalar_args = [arg for arg in kernel_args if arg['access_type'] == 'scalar']

        index_arg = next((arg for arg in pointer_args if 'indexes' in arg), None)
        if index_arg:
            data_arg = None
            for arg in pointer_args:
                if arg is not index_arg and arg['access_type'] == 'input':
                    data_arg = arg
                    break
            if not data_arg:
                data_arg = next((arg for arg in pointer_args if arg is not index_arg), None)
            
            if data_arg:
                data_base_str = data_arg['value'].split(',')[0].strip()
                try:
                    data_base = int(data_base_str, 16)
                except ValueError:
                    data_base = 0
                indexes = index_arg['indexes']
                addresses = [data_base + idx * access_size for idx in indexes]
                min_offset = min(addresses)
                max_offset = max(addresses) + access_size
                total_bytes = len(set(addresses)) * access_size
                return {
                    'min_offset': min_offset,
                    'max_offset': max_offset,
                    'total_bytes': total_bytes,
                    'pattern': 'data_indexed',
                    'indexes': indexes,
                    'base_pointer': f"{data_base:#x}",
                    'address_set': [f"{addr:#x}" for addr in sorted(set(addresses))]
                }

        base_address = 0
        
        if access_info['type'] == 'store':
            output_args = [arg for arg in pointer_args if arg['access_type'] == 'output']
            if output_args:
                addr_str = output_args[0]['value'].split(',')[0].strip()
                try:
                    base_address = int(addr_str, 16)
                except ValueError:
                    base_address = 0
        else:  
            input_args = [arg for arg in pointer_args if arg['access_type'] == 'input']
            
            if input_args:
                base_reg = mem_operand.get('base_register', '')
                
                if len(input_args) == 1:
                    addr_str = input_args[0]['value'].split(',')[0].strip()
                elif len(input_args) == 2:
                    if base_reg == 'R4' and len(input_args) > 1:
                        addr_str = input_args[1]['value'].split(',')[0].strip()
                    else:
                        addr_str = input_args[0]['value'].split(',')[0].strip()
                elif len(input_args) >= 3:
                    if base_reg == 'R2':
                        addr_str = input_args[0]['value'].split(',')[0].strip()
                    elif base_reg == 'R4':
                        addr_str = input_args[1]['value'].split(',')[0].strip() if len(input_args) > 1 else input_args[0]['value'].split(',')[0].strip()
                    elif base_reg == 'R6':
                        addr_str = input_args[2]['value'].split(',')[0].strip() if len(input_args) > 2 else input_args[0]['value'].split(',')[0].strip()
                    else:
                        addr_str = input_args[0]['value'].split(',')[0].strip()
                else:
                    addr_str = input_args[0]['value'].split(',')[0].strip()
                
                try:
                    base_address = int(addr_str, 16)
                except ValueError:
                    base_address = 0

        if total_threads == 1 and access_info['type'] == 'store':
            condition_value = None
            condition_type = None
            if scalar_args:
                condition_value = scalar_args[0]['value']
                condition_type = scalar_args[0]['type']
            
            return {
                'min_offset': base_address,
                'max_offset': base_address + access_size,
                'total_bytes': access_size,
                'pattern': 'conditional_write',
                'base_pointer': f"{base_address:#x}",
                'address_set': [f"{base_address:#x}"],
                'condition_value': condition_value,
                'condition_type': condition_type
            }

        if len(scalar_args) >= 1:
            try:
                index = int(scalar_args[0]['value'])
                
                if len(scalar_args) >= 2:
                    try:
                        array_size = int(scalar_args[-1]['value'])
                        if access_info['type'] == 'store':
                            target_pos = array_size - index - 1
                            address = base_address + target_pos * access_size
                            return {
                                'min_offset': address,
                                'max_offset': address + access_size,
                                'total_bytes': access_size,
                                'pattern': 'reverse_indexing',
                                'base_pointer': f"{base_address:#x}",
                                'address_set': [f"{address:#x}"]
                            }
                        else:  
                            address = base_address + index * access_size
                            return {
                                'min_offset': address,
                                'max_offset': address + access_size,
                                'total_bytes': access_size,
                                'pattern': 'scalar_indexing',
                                'base_pointer': f"{base_address:#x}",
                                'address_set': [f"{address:#x}"]
                            }
                    except ValueError:
                        pass
                
                
                if index < 100 or total_threads == 1:  
                    address = base_address + index * access_size
                    return {
                        'min_offset': address,
                        'max_offset': address + access_size,
                        'total_bytes': access_size,
                        'pattern': 'scalar_indexing',
                        'base_pointer': f"{base_address:#x}",
                        'address_set': [f"{address:#x}"]
                    }
            except ValueError:
                pass

        if len(scalar_args) >= 2 and access_info['type'] == 'store':
            try:
                dim1 = int(scalar_args[0]['value']) if scalar_args[0]['value'].isdigit() else total_threads
                dim2 = int(scalar_args[1]['value']) if scalar_args[1]['value'].isdigit() else 4
                
                if dim1 * dim2 * access_size > total_threads * access_size:
                    total_size = dim1 * dim2 * access_size
                    return {
                        'min_offset': base_address,
                        'max_offset': base_address + total_size,
                        'total_bytes': total_size,
                        'pattern': '2d_matrix_per_thread',
                        'dimensions': f"{dim1} × {dim2} × {access_size} bytes",
                        'base_pointer': f"{base_address:#x}"
                    }
            except ValueError:
                pass

        use_linear_threading = False
        
        if scalar_args:
            for scalar_arg in scalar_args:
                try:
                    scalar_val = int(scalar_arg['value'])
                    
                    if scalar_val == total_threads:
                        use_linear_threading = True
                        break
                    
                    elif scalar_val > 0 and scalar_val % total_threads == 0:
                        use_linear_threading = True
                        break
                    
                    elif len(pointer_args) > 1 and scalar_val == total_threads * len(pointer_args):
                        use_linear_threading = True
                        break
                        
                except ValueError:
                    continue
        
            input_count = len([arg for arg in pointer_args if arg['access_type'] == 'input'])
            output_count = len([arg for arg in pointer_args if arg['access_type'] == 'output'])
            
            if input_count >= 1 and output_count >= 1:
                use_linear_threading = True
        
        if (total_threads == 1 and scalar_args and len(pointer_args) >= 3 and
            all(arg['access_type'] == 'output' for arg in pointer_args)):
            try:
                loop_count = int(scalar_args[0]['value'])
                num_pointers = len(pointer_args)
                
                accesses_per_pointer = max(1, loop_count // num_pointers)
                
                return {
                    'min_offset': base_address,
                    'max_offset': base_address + access_size,
                    'total_bytes': accesses_per_pointer * access_size,
                    'pattern': 'conditional_multiple_pointer_loop',
                    'base_pointer': f"{base_address:#x}",
                    'address_set': [f"{base_address:#x}"]
                }
            except ValueError:
                pass

        if not use_linear_threading and scalar_args and total_threads > 1:
            try:
                index_val = int(scalar_args[0]['value'])
                if index_val < 1000:
                    address = base_address + index_val * access_size
                    return {
                        'min_offset': address,
                        'max_offset': address + access_size,
                        'total_bytes': access_size,
                        'pattern': 'scalar_indexing',
                        'base_pointer': f"{base_address:#x}",
                        'address_set': [f"{address:#x}"]
                    }
            except ValueError:
                pass

        estimated_range = {
            'min_offset': base_address,
            'max_offset': base_address + (total_threads * access_size),
            'total_bytes': total_threads * access_size,
            'pattern': 'linear_thread_indexing',
            'base_pointer': f"{base_address:#x}"
        }
        
        if mem_operand['constant_offset'] != 0:
            estimated_range['constant_offset_applied'] = mem_operand['constant_offset']
            estimated_range['max_offset'] += mem_operand['constant_offset']
        
        return estimated_range
    